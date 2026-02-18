
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
import json
import sys

# Add project root to path for imports if needed
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.lstm_model import LSTMAttentionClassifier, AttentionLayer, prepare_features, SEQ_LEN, NUM_FEATURES

class MLPredictor:
    def __init__(self, model_dir="data/models", device=None):
        self.device = device if device else torch.device("cpu")
        self.model_path = os.path.join(model_dir, "lstm_direction_v2.pth")
        self.meta_path = os.path.join(model_dir, "lstm_metadata_v2.json")
        self.loaded = False
        
        self._load_model()
        
    def _load_model(self):
        if not os.path.exists(self.model_path) or not os.path.exists(self.meta_path):
            print(f"Warning: Model/Meta not found at {self.model_path}")
            return

        try:
            with open(self.meta_path, 'r') as f:
                self.meta = json.load(f)
                
            self.feat_mean = np.array(self.meta['feat_mean'], dtype=np.float32)
            self.feat_std = np.array(self.meta['feat_std'], dtype=np.float32)
            
            # Initialize model architecture from metadata
            self.model = LSTMAttentionClassifier(
                input_dim=self.meta['input_dim'],
                hidden_dim=self.meta['hidden_dim'],
                num_layers=self.meta['num_layers'],
                dropout=self.meta['dropout']
            ).to(self.device)
            
            state_dict = torch.load(self.model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            self.model.eval()
            self.loaded = True
            print(f"MLPredictor loaded V2 model from {self.model_path}")
            
        except Exception as e:
            print(f"Failed to load ML model: {e}")
            self.loaded = False

    def predict(self, df_buffer):
        """
        Predict probability of Up move based on recent history.
        buffer_df: DataFrame containing at least (SEQ_LEN + 60) rows with columns:
                   ['open', 'high', 'low', 'close', 'volume', 'trades', 'taker_buy_base']
        """
        if not self.loaded:
            return 0.5
            
        # We need enough data for feature engineering (rolling windows)
        # Max lookback needed for features is 60 (vol_60, ema60). 
        # Plus normalization window? No, normalization is global.
        # So we need SEQ_LEN (120) + 60 = 180 rows minimum.
        
        if len(df_buffer) < SEQ_LEN + 60:
            return 0.5
            
        # Compute features
        # We pass the full buffer to prepare_features
        # It returns a DataFrame with feature columns
        try:
            features = prepare_features(df_buffer)
            
            # We need the last SEQ_LEN rows
            # But wait, prepare_features returns aligned dataframe. 
            # We take the LAST 120 rows of calculated features.
            
            # Check for NaNs (due to rolling windows at start)
            if features.iloc[-1].isnull().any():
                return 0.5
                
            seq_features = features.iloc[-SEQ_LEN:].values.astype(np.float32)
            
            # Normalize using Training Mean/Std
            seq_features = (seq_features - self.feat_mean) / self.feat_std
            
            # Clip
            seq_features = np.clip(seq_features, -5, 5)
            
            # Create tensor (batch_size=1)
            x_tensor = torch.FloatTensor(seq_features).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                prob = self.model(x_tensor).item()
                
            return prob
            
        except Exception as e:
            print(f"Prediction error: {e}")
            return 0.5

if __name__ == "__main__":
    # Test
    pred = MLPredictor()
    
    # Create dummy data
    dates = pd.date_range(start='2024-01-01', periods=300, freq='1min')
    data = {
        'open': np.linspace(100, 105, 300),
        'high': np.linspace(101, 106, 300),
        'low': np.linspace(99, 104, 300),
        'close': np.linspace(100, 105, 300), # Uptrend
        'volume': np.random.rand(300) * 100,
        'trades': np.random.rand(300) * 10,
        'taker_buy_base': np.random.rand(300) * 50
    }
    df = pd.DataFrame(data, index=dates)
    
    if pred.loaded:
        prob = pred.predict(df)
        print(f"Prediction for uptrend: {prob:.4f}")
        
        # Downtrend
        df['close'] = np.linspace(105, 100, 300)
        prob_down = pred.predict(df)
        print(f"Prediction for downtrend: {prob_down:.4f}")
