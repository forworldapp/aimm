import sys
import os
import pandas as pd
import numpy as np

# Adjust path for local imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.regime_detector import RegimeDetector

def train_1m_model():
    print("=" * 60)
    print("GMM Regime Detector Training (1-Minute)")
    print("=" * 60)
    
    # Load 1m data
    data_file = "data/btcusdt_1m_1year.csv"
    if not os.path.exists(data_file):
        print(f"Error: Data file {data_file} not found. Run fetch_1m_data.py first.")
        return

    print(f"\n1. Loading 1m data from {data_file}...")
    df = pd.read_csv(data_file)
    print(f"   Total candles: {len(df)}")
    
    # Train GMM model
    print("\n2. Training GMM model...")
    # Initialize detector with new model path
    detector = RegimeDetector(model_path="data/regime_model_1m.pkl")
    
    # Fit model
    detector.fit(df)
    
    # Test prediction
    last_candle = df.tail(50)
    current_regime = detector.predict(last_candle)
    print(f"\n3. Testing prediction on latest data...")
    print(f"   Current regime: {current_regime}")
    
    print("\n============================================================")
    print("GMM 1m Training Complete!")
    print("============================================================")

if __name__ == "__main__":
    train_1m_model()
