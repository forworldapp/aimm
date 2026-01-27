"""
HMM vs Baseline Backtest Comparison
- Tests v3.8.0 baseline (no HMM)
- Tests v3.8.0 with HMM regime detection
- Memory-safe chunked processing
"""
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
import gc
import sys
import os
import warnings
warnings.filterwarnings('ignore')

# Add parent directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.hmm_regime_detector import HMMRegimeDetector

def fetch_month_data(symbol: str, year: int, month: int, interval: str = "1m") -> pd.DataFrame:
    """Fetch 1 month of data"""
    start_date = datetime(year, month, 1)
    if month == 12:
        end_date = datetime(year + 1, 1, 1)
    else:
        end_date = datetime(year, month + 1, 1)
    
    start_time = int(start_date.timestamp() * 1000)
    end_time = int(end_date.timestamp() * 1000)
    
    all_klines = []
    current_start = start_time
    
    while current_start < end_time:
        url = "https://api.binance.com/api/v3/klines"
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": current_start,
            "endTime": end_time,
            "limit": 1000
        }
        
        try:
            response = requests.get(url, params=params, timeout=30)
            data = response.json()
            if not data or isinstance(data, dict):
                break
            all_klines.extend(data)
            if interval == "1h":
                current_start = data[-1][0] + 3600000
            else:
                current_start = data[-1][0] + 60000
        except:
            break
    
    if not all_klines:
        return pd.DataFrame()
    
    df = pd.DataFrame(all_klines, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_volume', 'trades', 'taker_buy_base',
        'taker_buy_quote', 'ignore'
    ])
    
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = df[col].astype(float)
    
    return df

def simulate_baseline(df: pd.DataFrame, spread_pct: float = 0.002, 
                       order_usd: float = 100, max_pos_usd: float = 5000) -> dict:
    """Simulate v3.8.0 baseline without HMM"""
    n = len(df)
    mid = df['close'].values
    high = df['high'].values
    low = df['low'].values
    
    # Fixed parameters (no HMM)
    gamma = 1.0
    kappa = 1000
    grid_layers = 7
    
    prev_mid = np.roll(mid, 1)
    prev_mid[0] = mid[0]
    
    bid_level = prev_mid * (1 - spread_pct/2)
    ask_level = prev_mid * (1 + spread_pct/2)
    
    bid_fill = (low <= bid_level).astype(float)
    ask_fill = (high >= ask_level).astype(float)
    
    position = 0.0
    balance = 10000.0
    entry_price = 0.0
    pnl_history = []
    
    for i in range(1, n):
        # Check position limit
        pos_usd = abs(position) * mid[i]
        
        if bid_fill[i] and pos_usd < max_pos_usd:
            qty = order_usd / bid_level[i]
            if position >= 0:
                entry_price = (position * entry_price + qty * bid_level[i]) / (position + qty) if position > 0 else bid_level[i]
                position += qty
            else:
                close_qty = min(qty, abs(position))
                pnl = (entry_price - bid_level[i]) * close_qty
                balance += pnl
                position += close_qty
        
        if ask_fill[i] and pos_usd < max_pos_usd:
            qty = order_usd / ask_level[i]
            if position <= 0:
                entry_price = (abs(position) * entry_price + qty * ask_level[i]) / (abs(position) + qty) if position < 0 else ask_level[i]
                position -= qty
            else:
                close_qty = min(qty, position)
                pnl = (ask_level[i] - entry_price) * close_qty
                balance += pnl
                position -= close_qty
        
        unrealized = (mid[i] - entry_price) * position if position > 0 else (entry_price - mid[i]) * abs(position) if position < 0 else 0
        pnl_history.append(balance + unrealized)
    
    final_pnl = pnl_history[-1] - 10000 if pnl_history else 0
    return {'pnl': final_pnl, 'params': {'gamma': gamma, 'kappa': kappa, 'layers': grid_layers}}


def simulate_with_hmm(df: pd.DataFrame, hmm_detector: HMMRegimeDetector,
                       order_usd: float = 100, max_pos_usd: float = 5000) -> dict:
    """Simulate v3.8.0 with HMM regime detection"""
    n = len(df)
    mid = df['close'].values
    high = df['high'].values
    low = df['low'].values
    
    # HMM regime parameters
    regime_params = {
        'low_vol': {'gamma': 0.1, 'kappa': 500, 'spread': 0.001, 'layers': 10},
        'high_vol': {'gamma': 0.3, 'kappa': 200, 'spread': 0.003, 'layers': 5},
        'trend_up': {'gamma': 0.2, 'kappa': 300, 'spread': 0.002, 'layers': 7},
        'trend_down': {'gamma': 0.2, 'kappa': 300, 'spread': 0.002, 'layers': 7}
    }
    
    position = 0.0
    balance = 10000.0
    entry_price = 0.0
    pnl_history = []
    regime_changes = 0
    
    # Get regime predictions using hourly data
    # Sample every hour for regime prediction
    regime_window = 100
    last_regime = 'low_vol'
    
    for i in range(1, n):
        # Update regime every 60 candles (simulating hourly update on minute data)
        if i >= regime_window and i % 60 == 0:
            try:
                window_df = df.iloc[max(0, i-regime_window):i].copy()
                regime = hmm_detector.predict(window_df)
                if regime and regime != last_regime:
                    regime_changes += 1
                    last_regime = regime
            except:
                pass
        
        params = regime_params.get(last_regime, regime_params['low_vol'])
        spread_pct = params['spread']
        
        prev_mid = mid[i-1]
        bid_level = prev_mid * (1 - spread_pct/2)
        ask_level = prev_mid * (1 + spread_pct/2)
        
        pos_usd = abs(position) * mid[i]
        
        if low[i] <= bid_level and pos_usd < max_pos_usd:
            qty = order_usd / bid_level
            if position >= 0:
                entry_price = (position * entry_price + qty * bid_level) / (position + qty) if position > 0 else bid_level
                position += qty
            else:
                close_qty = min(qty, abs(position))
                pnl = (entry_price - bid_level) * close_qty
                balance += pnl
                position += close_qty
        
        if high[i] >= ask_level and pos_usd < max_pos_usd:
            qty = order_usd / ask_level
            if position <= 0:
                entry_price = (abs(position) * entry_price + qty * ask_level) / (abs(position) + qty) if position < 0 else ask_level
                position -= qty
            else:
                close_qty = min(qty, position)
                pnl = (ask_level - entry_price) * close_qty
                balance += pnl
                position -= close_qty
        
        unrealized = (mid[i] - entry_price) * position if position > 0 else (entry_price - mid[i]) * abs(position) if position < 0 else 0
        pnl_history.append(balance + unrealized)
    
    final_pnl = pnl_history[-1] - 10000 if pnl_history else 0
    return {'pnl': final_pnl, 'regime_changes': regime_changes}


def run_comparison(months_back: int = 6):
    """Run comparison backtest"""
    print("=" * 70)
    print("  📊 HMM vs BASELINE BACKTEST COMPARISON")
    print("=" * 70)
    
    # Load HMM model
    print("\n📥 Loading HMM model...")
    hmm = HMMRegimeDetector(model_path="data/regime_model_hmm.pkl")
    if not hmm.is_fitted:
        print("❌ HMM model not found! Run ml/train_hmm.py first")
        return
    print("✅ HMM model loaded")
    
    baseline_results = []
    hmm_results = []
    
    now = datetime.now()
    
    for i in range(months_back, 0, -1):
        target = now - timedelta(days=30*i)
        year, month = target.year, target.month
        month_name = f"{year}-{month:02d}"
        
        print(f"\n[{months_back - i + 1}/{months_back}] {month_name}...")
        
        # Fetch minute data for accurate MM simulation
        df = fetch_month_data("BTCUSDT", year, month, "1m")
        if df.empty or len(df) < 1000:
            print(f"  ⚠️ Skipped (no data)")
            continue
        
        print(f"  📥 {len(df):,} candles")
        
        # Baseline test (using config values: $5000 max pos, $200 order)
        baseline = simulate_baseline(df, spread_pct=0.002, order_usd=200, max_pos_usd=5000)
        baseline_results.append({'month': month_name, 'pnl': baseline['pnl']})
        
        # HMM test (using config values)
        hmm_result = simulate_with_hmm(df, hmm, order_usd=200, max_pos_usd=5000)
        hmm_results.append({'month': month_name, 'pnl': hmm_result['pnl'], 'regimes': hmm_result['regime_changes']})
        
        print(f"  Baseline: ${baseline['pnl']:+.2f} | HMM: ${hmm_result['pnl']:+.2f}")
        
        del df
        gc.collect()
    
    # Summary
    print("\n" + "=" * 70)
    print("  📊 MONTHLY COMPARISON")
    print("=" * 70)
    
    print(f"\n  {'Month':<12} {'Baseline':>12} {'HMM':>12} {'Diff':>12} {'Winner':>10}")
    print("  " + "-" * 60)
    
    total_baseline = 0
    total_hmm = 0
    baseline_wins = 0
    hmm_wins = 0
    
    for b, h in zip(baseline_results, hmm_results):
        diff = h['pnl'] - b['pnl']
        winner = "HMM" if diff > 0 else "Baseline"
        if diff > 0:
            hmm_wins += 1
        else:
            baseline_wins += 1
        
        total_baseline += b['pnl']
        total_hmm += h['pnl']
        
        print(f"  {b['month']:<12} ${b['pnl']:>10.2f} ${h['pnl']:>10.2f} ${diff:>+10.2f} {winner:>10}")
    
    print("  " + "-" * 60)
    diff = total_hmm - total_baseline
    winner = "HMM 🏆" if diff > 0 else "Baseline 🏆"
    print(f"  {'TOTAL':<12} ${total_baseline:>10.2f} ${total_hmm:>10.2f} ${diff:>+10.2f} {winner:>10}")
    
    print(f"\n  📊 승률: Baseline {baseline_wins}/{len(baseline_results)} | HMM {hmm_wins}/{len(hmm_results)}")
    
    print("\n" + "=" * 70)
    if diff > 0:
        print(f"  🏆 HMM이 ${diff:.2f} 더 수익!")
    else:
        print(f"  🏆 Baseline이 ${-diff:.2f} 더 수익!")
    print("=" * 70)

if __name__ == "__main__":
    run_comparison(months_back=12)
