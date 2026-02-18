"""
Improved LSTM Direction Prediction Model for Market Making Skew
===============================================================
V2: Rich features, attention mechanism, proper normalization, full dataset training.

Features (12 total):
  1. log_return            - Log price change
  2. volatility_20         - 20-bar rolling std of returns
  3. volatility_60         - 60-bar rolling std of returns (longer term)
  4. rsi_14                - Relative Strength Index (14 periods)
  5. macd_signal           - MACD histogram (12-26-9)
  6. bb_position           - Position within Bollinger Bands (-1 to +1)
  7. volume_ratio          - Volume / SMA(volume, 20)
  8. taker_ratio           - Taker buy ratio (buy pressure)
  9. price_vs_ema10        - Price deviation from 10-bar EMA (%)
 10. price_vs_ema60        - Price deviation from 60-bar EMA (%)
 11. atr_ratio             - ATR(14) / close (normalized range)
 12. trades_zscore         - Trade count z-score vs 20-bar mean
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pandas as pd
import numpy as np
import time
import os
import json

# --- Configuration ---
SEQ_LEN = 120           # 2 hours lookback
PRED_LEN = 5            # 5-min forward prediction
BATCH_SIZE = 256         # Larger batch for stability
EPOCHS = 20
LEARNING_RATE = 0.0005
HIDDEN_SIZE = 128
NUM_LAYERS = 2
DROPOUT = 0.3
NUM_FEATURES = 12
PATIENCE = 5            # Early stopping patience

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==========================================
# Feature Engineering
# ==========================================

def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(span=period, min_periods=period).mean()
    avg_loss = loss.ewm(span=period, min_periods=period).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    return rsi / 100.0  # Normalize to 0-1

def compute_macd_hist(series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, min_periods=fast).mean()
    ema_slow = series.ewm(span=slow, min_periods=slow).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, min_periods=signal).mean()
    hist = macd_line - signal_line
    # Normalize by price to make scale-invariant
    return hist / (series + 1e-10)

def compute_bollinger_position(close, period=20, num_std=2):
    sma = close.rolling(period).mean()
    std = close.rolling(period).std()
    upper = sma + num_std * std
    lower = sma - num_std * std
    # Position: -1 (at lower band) to +1 (at upper band)
    pos = 2 * (close - lower) / (upper - lower + 1e-10) - 1
    return pos.clip(-2, 2)  # Clip extreme outliers

def compute_atr(high, low, close, period=14):
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()
    return atr / (close + 1e-10)  # Normalize by price


def prepare_features(df):
    """Compute all 12 features from raw OHLCV data."""
    close = df['close']
    high = df['high']
    low = df['low']
    volume = df['volume'] if 'volume' in df.columns else pd.Series(np.ones(len(df)))
    trades = df['trades'] if 'trades' in df.columns else pd.Series(np.ones(len(df)))
    taker_buy = df['taker_buy_base'] if 'taker_buy_base' in df.columns else volume * 0.5

    features = pd.DataFrame(index=df.index)

    # 1. Log return
    features['log_return'] = np.log(close / close.shift(1))

    # 2-3. Multi-timeframe volatility
    features['vol_20'] = features['log_return'].rolling(20).std()
    features['vol_60'] = features['log_return'].rolling(60).std()

    # 4. RSI
    features['rsi_14'] = compute_rsi(close, 14)

    # 5. MACD histogram (normalized)
    features['macd_hist'] = compute_macd_hist(close)

    # 6. Bollinger Band position
    features['bb_pos'] = compute_bollinger_position(close)

    # 7. Volume ratio (current / avg)
    vol_sma = volume.rolling(20).mean()
    features['vol_ratio'] = (volume / (vol_sma + 1e-10)).clip(0, 5)

    # 8. Taker buy ratio (buy pressure)
    features['taker_ratio'] = (taker_buy / (volume + 1e-10)).clip(0, 1)

    # 9-10. Price vs EMA deviations
    ema10 = close.ewm(span=10).mean()
    ema60 = close.ewm(span=60).mean()
    features['price_vs_ema10'] = ((close - ema10) / (ema10 + 1e-10))
    features['price_vs_ema60'] = ((close - ema60) / (ema60 + 1e-10))

    # 11. ATR ratio
    features['atr_ratio'] = compute_atr(high, low, close, 14)

    # 12. Trades z-score
    trades_mean = trades.rolling(20).mean()
    trades_std = trades.rolling(20).std()
    features['trades_zscore'] = ((trades - trades_mean) / (trades_std + 1e-10)).clip(-3, 3)

    return features


def prepare_data(csv_path, limit=None):
    """Load data, compute features, create sequences."""
    df = pd.read_csv(csv_path)
    if limit:
        df = df.iloc[:limit]

    features_df = prepare_features(df)

    # Fill NaN from rolling windows
    features_df = features_df.fillna(0)

    # Clip all features to prevent extreme values
    data = features_df.values.astype(np.float32)
    data = np.clip(data, -5, 5)

    # Target: 1 if close[t+PRED_LEN] > close[t]
    close = df['close'].values
    future_ret = np.zeros(len(close))
    for i in range(len(close) - PRED_LEN):
        future_ret[i] = (close[i + PRED_LEN] - close[i]) / close[i]
    labels = (future_ret > 0).astype(np.float32)

    # Create sequences
    X, y = [], []
    for i in range(SEQ_LEN, len(data) - PRED_LEN):
        X.append(data[i - SEQ_LEN:i])
        y.append(labels[i])

    X = np.array(X)
    y = np.array(y)

    # Per-feature normalization (z-score on training set mean/std)
    # Compute stats on the flattened feature dimension
    feat_mean = X.reshape(-1, NUM_FEATURES).mean(axis=0)
    feat_std = X.reshape(-1, NUM_FEATURES).std(axis=0) + 1e-8
    X = (X - feat_mean) / feat_std

    print(f"Dataset: {len(X)} samples, {NUM_FEATURES} features, SEQ_LEN={SEQ_LEN}")
    print(f"Class balance: {y.mean():.3f} (1s) / {1-y.mean():.3f} (0s)")

    return X, y, feat_mean, feat_std


# ==========================================
# Model Architecture
# ==========================================

class AttentionLayer(nn.Module):
    """Simple self-attention over LSTM outputs."""
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim, 1)

    def forward(self, lstm_out):
        # lstm_out: (batch, seq_len, hidden_dim)
        weights = torch.softmax(self.attn(lstm_out), dim=1)  # (batch, seq_len, 1)
        context = (lstm_out * weights).sum(dim=1)  # (batch, hidden_dim)
        return context


class LSTMAttentionClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers,
            batch_first=True, dropout=dropout, bidirectional=False
        )
        self.attention = AttentionLayer(hidden_dim)
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim // 2, 1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)         # (batch, seq, hidden)
        context = self.attention(lstm_out)   # (batch, hidden)
        x = self.bn(context)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc2(x))
        return x


# Keep backward-compatible simple model for predictor.py
class LSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return self.sigmoid(out)


# ==========================================
# Dataset
# ==========================================

class FinancialDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ==========================================
# Training
# ==========================================

def train_model(train_loader, val_loader, epochs=EPOCHS, input_dim=NUM_FEATURES):
    model = LSTMAttentionClassifier(
        input_dim=input_dim,
        hidden_dim=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT
    ).to(device)

    criterion = nn.BCELoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

    best_val_acc = 0.0
    best_model_state = None
    patience_counter = 0

    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Training on {device}...\n")

    for epoch in range(epochs):
        # --- Train ---
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device).unsqueeze(1)

            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()
            predicted = (outputs > 0.5).float()
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()

        train_acc = correct / total

        # --- Validate ---
        val_acc, val_loss = evaluate_model(model, val_loader, criterion)
        scheduler.step(val_acc)

        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1:2d}/{epochs} | "
              f"Loss: {train_loss/len(train_loader):.4f} | "
              f"Train: {train_acc:.4f} | "
              f"Val: {val_acc:.4f} | "
              f"LR: {current_lr:.6f}")

        # Early Stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"\nEarly stopping at epoch {epoch+1} (best val: {best_val_acc:.4f})")
                break

    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    print(f"\nBest Validation Accuracy: {best_val_acc:.4f}")
    return model, best_val_acc


def evaluate_model(model, val_loader, criterion=None):
    model.eval()
    correct = 0
    total = 0
    total_loss = 0.0
    if criterion is None:
        criterion = nn.BCELoss()

    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device).unsqueeze(1)
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            total_loss += loss.item()
            predicted = (outputs > 0.5).float()
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()

    acc = correct / total if total > 0 else 0.0
    avg_loss = total_loss / len(val_loader) if len(val_loader) > 0 else 0.0
    return acc, avg_loss


# ==========================================
# Main
# ==========================================

if __name__ == "__main__":
    csv_path = "data/btcusdt_1m_1year.csv"
    if not os.path.exists(csv_path):
        print(f"ERROR: Data file not found at {csv_path}")
        exit(1)

    print(f"Loading data from {csv_path}...")
    print(f"Device: {device}")

    # Use full dataset for real training (set limit=None or a large number)
    # For testing, use limit=100000
    import sys
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else None
    epochs = int(sys.argv[2]) if len(sys.argv) > 2 else EPOCHS

    X, y, feat_mean, feat_std = prepare_data(csv_path, limit=limit)

    # Time-series split (NO shuffle - preserve temporal order)
    split_idx = int(len(X) * 0.8)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]

    print(f"Train: {len(X_train)} | Val: {len(X_val)}")

    train_dataset = FinancialDataset(X_train, y_train)
    val_dataset = FinancialDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=0)

    start_time = time.perf_counter()
    model, best_acc = train_model(train_loader, val_loader, epochs=epochs)
    elapsed = time.perf_counter() - start_time

    print(f"\nTraining completed in {elapsed:.1f}s")

    # Save model + metadata
    os.makedirs("data/models", exist_ok=True)
    torch.save(model.state_dict(), "data/models/lstm_direction_v2.pth")

    metadata = {
        "version": 2,
        "model_type": "LSTMAttentionClassifier",
        "input_dim": NUM_FEATURES,
        "hidden_dim": HIDDEN_SIZE,
        "num_layers": NUM_LAYERS,
        "dropout": DROPOUT,
        "seq_len": SEQ_LEN,
        "pred_len": PRED_LEN,
        "best_val_acc": float(best_acc),
        "feat_mean": feat_mean.tolist(),
        "feat_std": feat_std.tolist(),
        "feature_names": [
            "log_return", "vol_20", "vol_60", "rsi_14", "macd_hist",
            "bb_pos", "vol_ratio", "taker_ratio", "price_vs_ema10",
            "price_vs_ema60", "atr_ratio", "trades_zscore"
        ],
        "training_samples": len(X_train),
        "training_time_s": round(elapsed, 1),
        "epochs_trained": epochs,
    }
    with open("data/models/lstm_metadata_v2.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Model saved: data/models/lstm_direction_v2.pth")
    print(f"Metadata saved: data/models/lstm_metadata_v2.json")
