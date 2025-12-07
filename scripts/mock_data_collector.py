import asyncio
import csv
import os
import time
import logging
import random
from datetime import datetime

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger("MockDataCollector")

DATA_DIR = "data"
FILENAME = f"ticker_data_{int(time.time())}.csv"
FILEPATH = os.path.join(DATA_DIR, FILENAME)

async def collect_data():
    logger.info(f"Starting MOCK Data Collection...")
    logger.info(f"Saving to {FILEPATH}")

    os.makedirs(DATA_DIR, exist_ok=True)

    # Initial Price
    price = 100000.0
    
    with open(FILEPATH, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["timestamp", "symbol", "best_bid", "best_ask", "bid_qty", "ask_qty", "spread"])
        
        try:
            while True:
                # Random Walk
                change = random.uniform(-10, 10)
                price += change
                
                spread = random.uniform(5, 10)
                best_bid = price - (spread / 2)
                best_ask = price + (spread / 2)
                
                bid_qty = random.uniform(0.1, 2.0)
                ask_qty = random.uniform(0.1, 2.0)
                
                timestamp = datetime.now().isoformat()
                
                writer.writerow([timestamp, "BTC-USDT", best_bid, best_ask, bid_qty, ask_qty, spread])
                file.flush()
                
                # logger.info(f"Recorded: {best_bid:.2f} / {best_ask:.2f}")
                
                await asyncio.sleep(1)
                
        except KeyboardInterrupt:
            logger.info("Data collection stopped by user.")
        except Exception as e:
            logger.error(f"Error: {e}")

if __name__ == "__main__":
    try:
        asyncio.run(collect_data())
    except KeyboardInterrupt:
        pass
