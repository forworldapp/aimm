import ccxt
import csv
import os
import time
from datetime import datetime

DATA_DIR = "data"
FILENAME = f"binance_data_{int(time.time())}.csv"
FILEPATH = os.path.join(DATA_DIR, FILENAME)

def fetch_binance_data(symbol='BTC/USDT', timeframe='1m', limit=1000):
    print(f"Fetching {limit} candles of {symbol} from Binance...")
    exchange = ccxt.binance()
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
    
    print(f"Fetched {len(ohlcv)} candles. Converting to tick format...")
    
    os.makedirs(DATA_DIR, exist_ok=True)
    
    with open(FILEPATH, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["timestamp", "symbol", "best_bid", "best_ask", "bid_qty", "ask_qty", "spread"])
        
        for candle in ohlcv:
            # timestamp, open, high, low, close, volume
            ts = candle[0]
            open_price = candle[1]
            high = candle[2]
            low = candle[3]
            close = candle[4]
            volume = candle[5]
            
            # Simulate ticks within the candle (Open -> High -> Low -> Close)
            # This is a simplification, but better than nothing.
            # We create 4 ticks per minute candle.
            
            ticks = [open_price, high, low, close]
            
            for price in ticks:
                # Simulate spread (0.01%)
                spread = price * 0.0001
                best_bid = price - (spread / 2)
                best_ask = price + (spread / 2)
                
                # Simulate qty (randomized based on volume)
                qty = (volume / 4) * 0.1 # fraction of volume
                
                iso_ts = datetime.fromtimestamp(ts / 1000).isoformat()
                
                writer.writerow([iso_ts, "BTC-USDT", best_bid, best_ask, qty, qty, spread])
                
                # Increment timestamp slightly for next tick
                ts += 15000 # +15 seconds

    print(f"Data saved to {FILEPATH}")
    return FILEPATH

if __name__ == "__main__":
    fetch_binance_data(limit=1440) # 1440 mins = 24 hours
