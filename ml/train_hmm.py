"""
HMM Model Training Script
- Fetches 1-year hourly BTC data from Binance
- Trains HMM regime detector
- Saves to data/regime_model_hmm.pkl
"""
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.hmm_regime_detector import HMMRegimeDetector

def fetch_binance_hourly_1year(symbol: str = "BTCUSDT") -> pd.DataFrame:
    """Fetch 1 year of hourly data from Binance"""
    print(f"📥 Fetching 1 year of hourly data for {symbol}...")
    
    end_time = datetime.now()
    start_time = end_time - timedelta(days=365)
    
    all_klines = []
    current_start = int(start_time.timestamp() * 1000)
    end_ts = int(end_time.timestamp() * 1000)
    
    while current_start < end_ts:
        url = "https://api.binance.com/api/v3/klines"
        params = {
            "symbol": symbol,
            "interval": "1h",
            "startTime": current_start,
            "endTime": end_ts,
            "limit": 1000
        }
        
        try:
            response = requests.get(url, params=params, timeout=30)
            data = response.json()
            if not data or isinstance(data, dict):
                break
            all_klines.extend(data)
            print(f"  Fetched {len(all_klines):,} candles...", end='\r')
            current_start = data[-1][0] + 3600000  # 1 hour in ms
        except Exception as e:
            print(f"Error: {e}")
            break
    
    print(f"\n✅ Total: {len(all_klines):,} hourly candles")
    
    # Convert to DataFrame
    df = pd.DataFrame(all_klines, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_volume', 'trades', 'taker_buy_base',
        'taker_buy_quote', 'ignore'
    ])
    
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = df[col].astype(float)
    
    return df

def train_hmm_model():
    """Main training function"""
    print("=" * 60)
    print("  🧠 HMM Regime Detector Training")
    print("=" * 60)
    
    # Fetch data
    df = fetch_binance_hourly_1year("BTCUSDT")
    
    if len(df) < 1000:
        print("❌ Not enough data to train (need at least 1000 candles)")
        return False
    
    # Initialize HMM Detector
    detector = HMMRegimeDetector(model_path="data/regime_model_hmm.pkl")
    
    # Train
    print("\n🔧 Training HMM model with 4 states...")
    detector.fit(df, n_components=4)
    
    # Save
    print("\n💾 Saving model to data/regime_model_hmm.pkl...")
    detector.save_model()
    
    # Verify
    print("\n📊 Verifying model...")
    if detector.is_fitted:
        print("✅ Model trained successfully!")
        
        # Test prediction
        test_regime = detector.predict_live("BTCUSDT")
        print(f"📌 Current Regime: {test_regime}")
        
        probs = detector.predict_live_proba("BTCUSDT")
        if probs:
            print(f"📊 Regime Probabilities:")
            for regime, prob in probs.items():
                print(f"   {regime}: {prob:.1%}")
        
        return True
    else:
        print("❌ Model training failed")
        return False

if __name__ == "__main__":
    success = train_hmm_model()
    if success:
        print("\n" + "=" * 60)
        print("  ✅ HMM MODEL ACTIVATED!")
        print("  Restart the bot to use HMM regime detection")
        print("=" * 60)
