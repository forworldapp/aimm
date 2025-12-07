import csv
import os
import time
import random
import math
import numpy as np
from datetime import datetime, timedelta

DATA_DIR = "data"
FILENAME = f"ticker_data_realistic_{int(time.time())}.csv"
FILEPATH = os.path.join(DATA_DIR, FILENAME)

def generate_data(num_ticks=10000):
    print(f"Generating {num_ticks} ticks of realistic market data...")
    os.makedirs(DATA_DIR, exist_ok=True)

    # Initial Price
    price = 100000.0
    
    # Simulation Params
    volatility = 5.0      # Base volatility
    trend_strength = 0.0  # Current trend strength
    trend_duration = 0    # How long the trend lasts
    
    with open(FILEPATH, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["timestamp", "symbol", "best_bid", "best_ask", "bid_qty", "ask_qty", "spread"])
        
        current_time = datetime.now()
        
        for i in range(num_ticks):
            # 1. Update Trend (Regime Switching)
            if trend_duration <= 0:
                # Switch regime: 
                # 0: Ranging (Mean Reversion) - 50% chance
                # 1: Uptrend - 25% chance
                # 2: Downtrend - 25% chance
                regime = random.choices([0, 1, 2], weights=[0.5, 0.25, 0.25])[0]
                
                if regime == 0: # Ranging
                    trend_strength = 0.0
                    trend_duration = random.randint(100, 500)
                elif regime == 1: # Uptrend
                    trend_strength = random.uniform(0.5, 2.0)
                    trend_duration = random.randint(50, 200)
                else: # Downtrend
                    trend_strength = random.uniform(-2.0, -0.5)
                    trend_duration = random.randint(50, 200)
            
            trend_duration -= 1
            
            # 2. Price Movement
            # Random Noise + Trend
            noise = np.random.normal(0, volatility)
            change = noise + trend_strength
            
            # Mean Reversion effect (if ranging)
            if trend_strength == 0:
                # Pull back to moving average (simulated simply here)
                # Let's just say price resists moving too far from start in ranging mode
                # This is a simplification.
                pass 

            price += change
            
            # 3. Spread & Orderbook
            # Volatility affects spread
            current_vol = abs(change)
            spread = max(1.0, current_vol * 0.5 + random.uniform(0, 2))
            
            best_bid = price - (spread / 2)
            best_ask = price + (spread / 2)
            
            # Quantities
            bid_qty = random.uniform(0.1, 5.0)
            ask_qty = random.uniform(0.1, 5.0)
            
            # Timestamp
            timestamp = current_time.isoformat()
            current_time += timedelta(milliseconds=100) # 10 ticks per second
            
            writer.writerow([timestamp, "BTC-USDT", best_bid, best_ask, bid_qty, ask_qty, spread])
            
            if i % 1000 == 0:
                print(f"Generated {i}/{num_ticks} ticks...")

    print(f"Data saved to {FILEPATH}")
    return FILEPATH

if __name__ == "__main__":
    generate_data(20000) # Generate 20,000 ticks
