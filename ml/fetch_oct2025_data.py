"""
Fetch October 2025 1-minute candle data from Binance
High volatility period testing
"""

import requests
import pandas as pd
import time
from datetime import datetime

def fetch_1m_candles(symbol="BTCUSDT", start_date="2025-10-01", end_date="2025-10-31"):
    """
    Fetch 1-minute candles from Binance for specified date range.
    """
    url = "https://api.binance.com/api/v3/klines"
    
    start_ts = int(datetime.strptime(start_date, "%Y-%m-%d").timestamp() * 1000)
    end_ts = int(datetime.strptime(end_date, "%Y-%m-%d").timestamp() * 1000)
    
    all_candles = []
    current_ts = start_ts
    
    print(f"Fetching 1m candles from {start_date} to {end_date}...")
    
    while current_ts < end_ts:
        params = {
            "symbol": symbol,
            "interval": "1m",
            "startTime": current_ts,
            "limit": 1000
        }
        
        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            if not data:
                break
                
            all_candles.extend(data)
            current_ts = data[-1][0] + 60000  # Next minute
            
            print(f"  Fetched {len(all_candles)} candles so far...")
            time.sleep(0.2)  # Rate limiting
            
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(1)
            continue
    
    # Convert to DataFrame
    df = pd.DataFrame(all_candles, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_volume', 'trades', 'taker_buy_base',
        'taker_buy_quote', 'ignore'
    ])
    
    # Clean up
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = pd.to_numeric(df[col])
    
    df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
    
    return df


if __name__ == "__main__":
    # Fetch October 2025 data
    df = fetch_1m_candles(start_date="2025-10-01", end_date="2025-10-31")
    
    print(f"\nTotal candles: {len(df)}")
    print(f"Date range: {pd.to_datetime(df['timestamp'].iloc[0], unit='ms')} to {pd.to_datetime(df['timestamp'].iloc[-1], unit='ms')}")
    
    # Calculate volatility stats
    df['returns'] = df['close'].pct_change()
    volatility = df['returns'].std() * 100
    max_change = df['returns'].abs().max() * 100
    
    print(f"\nVolatility Stats:")
    print(f"  Std Dev of Returns: {volatility:.4f}%")
    print(f"  Max Single-Candle Move: {max_change:.2f}%")
    print(f"  Price Range: ${df['low'].min():.0f} - ${df['high'].max():.0f}")
    
    # Save
    output_file = "data/btcusdt_1m_oct2025.csv"
    df.to_csv(output_file, index=False)
    print(f"\nSaved to {output_file}")
