import pandas as pd
import requests
import time
from datetime import datetime, timedelta
import os

def fetch_1m_candles(symbol="BTCUSDT", years=1):
    """
    Fetch 1-minute candles for the specified duration using pagination.
    Binance API limit is 1000 candles per request.
    1 Year = 525,600 minutes -> ~526 requests.
    """
    base_url = "https://api.binance.com/api/v3/klines"
    all_data = []
    
    # Calculate start time
    end_time = datetime.now()
    start_time_limit = end_time - timedelta(days=365 * years)
    
    print(f"Fetching 1m candles for {symbol} from {start_time_limit} to {end_time}...")
    
    # We fetch backwards from NOW
    current_end = int(end_time.timestamp() * 1000)
    limit_ts = int(start_time_limit.timestamp() * 1000)
    
    req_count = 0
    
    while current_end > limit_ts:
        params = {
            "symbol": symbol,
            "interval": "1m",
            "limit": 1000,
            "endTime": current_end
        }
        
        try:
            response = requests.get(base_url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if not data:
                print("No more data returned.")
                break
                
            # Convert to DataFrame
            df = pd.DataFrame(data, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                'taker_buy_quote', 'ignore'
            ])
            
            # Numeric conversion
            cols = ['open', 'high', 'low', 'close', 'volume']
            df[cols] = df[cols].apply(pd.to_numeric)
            
            # Store data
            all_data.append(df)
            
            # Update current_end for next batch (set to timestamp of first candle in this batch - 1ms)
            first_ts = df['timestamp'].iloc[0]
            current_end = first_ts - 1
            
            req_count += 1
            print(f"Request {req_count}: Fetched {len(df)} candles. Current date: {pd.to_datetime(first_ts, unit='ms')}")
            
            # Rate limit respect
            time.sleep(0.1)
            
        except Exception as e:
            print(f"Error fetching data: {e}")
            time.sleep(5) # Retry delay
            continue

    if not all_data:
        print("No data fetched!")
        return None
        
    # Concatenate and sort
    final_df = pd.concat(all_data).sort_values('timestamp').reset_index(drop=True)
    
    # Filter duplicates just in case
    final_df = final_df.drop_duplicates(subset=['timestamp'])
    
    print(f"\nDownload Complete!")
    print(f"Total Candles: {len(final_df)}")
    print(f"Date Range: {pd.to_datetime(final_df['timestamp'].min(), unit='ms')} to {pd.to_datetime(final_df['timestamp'].max(), unit='ms')}")
    
    # Save
    if not os.path.exists("data"):
        os.makedirs("data")
        
    filename = f"data/{symbol.lower()}_1m_1year.csv"
    final_df.to_csv(filename, index=False)
    print(f"Saved to {filename}")
    return final_df

if __name__ == "__main__":
    fetch_1m_candles()
