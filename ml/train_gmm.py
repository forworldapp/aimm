"""
Train GMM Regime Detection Model with Extended Data
Fetches 3000+ hourly candles from Binance for comprehensive training
"""
import requests
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.regime_detector import RegimeDetector

def fetch_binance_candles(symbol="BTCUSDT", interval="1h", limit=1000, total=3000):
    """
    Fetch extended historical candles from Binance API.
    Fetches in batches to get more than 1000 candles.
    """
    all_data = []
    end_time = None
    
    while len(all_data) < total:
        url = "https://api.binance.com/api/v3/klines"
        params = {
            "symbol": symbol,
            "interval": interval,
            "limit": limit
        }
        if end_time:
            params["endTime"] = end_time
        
        response = requests.get(url, params=params)
        data = response.json()
        
        if not data:
            break
            
        all_data = data + all_data  # Prepend older data
        end_time = data[0][0] - 1  # Move to earlier time
        
        print(f"Fetched {len(all_data)} candles...")
        
        if len(data) < limit:
            break
    
    # Convert to DataFrame
    df = pd.DataFrame(all_data, columns=[
        'open_time', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_volume', 'trades', 'taker_buy_base',
        'taker_buy_quote', 'ignore'
    ])
    
    # Select and convert types
    df = df[['open_time', 'open', 'high', 'low', 'close', 'volume']]
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = pd.to_numeric(df[col])
    
    return df

def main():
    print("=" * 60)
    print("GMM Regime Detector Training")
    print("=" * 60)
    
    # Fetch extended data (1 year = 24 * 365 = 8760)
    print("\n1. Fetching 8760 hourly candles (1 year) from Binance...")
    df = fetch_binance_candles(total=8760)
    print(f"   Total candles: {len(df)}")
    print(f"   Date range: {pd.to_datetime(df['open_time'].min(), unit='ms')} to {pd.to_datetime(df['open_time'].max(), unit='ms')}")
    
    # Save data for reference
    df.to_csv("data/btc_hourly_1year.csv", index=False)
    print(f"   Saved to data/btc_hourly_1year.csv")
    
    # Train GMM model
    print("\n2. Training GMM model...")
    detector = RegimeDetector(model_path="data/regime_model.pkl")
    labels = detector.fit(df, n_clusters=4)
    
    # Save model
    detector.save_model()
    print(f"\n3. Model saved to data/regime_model.pkl")
    
    # Test prediction
    print("\n4. Testing prediction...")
    regime = detector.predict(df.tail(100))
    print(f"   Current regime: {regime}")
    
    # Regime distribution
    regime_names = {v: k for k, v in detector.cluster_to_regime.items()}
    label_counts = pd.Series(labels).value_counts()
    print("\n5. Training data regime distribution:")
    for cluster, count in label_counts.items():
        regime_name = detector.cluster_to_regime.get(cluster, "unknown")
        pct = count / len(labels) * 100
        print(f"   {regime_name}: {count} ({pct:.1f}%)")
    
    print("\n" + "=" * 60)
    print("GMM Training Complete!")
    print("=" * 60)

if __name__ == "__main__":
    main()
