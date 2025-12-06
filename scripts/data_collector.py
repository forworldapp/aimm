import asyncio
import csv
import os
import time
import logging
from datetime import datetime
from core.config import Config
from core.grvt_exchange import GrvtExchange

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger("DataCollector")

DATA_DIR = "data"
FILENAME = f"ticker_data_{int(time.time())}.csv"
FILEPATH = os.path.join(DATA_DIR, FILENAME)

async def collect_data():
    # Load Config
    Config.load("config.yaml")
    
    # Initialize Exchange
    exchange = GrvtExchange(
        api_key=Config.GRVT_API_KEY,
        private_key=Config.GRVT_PRIVATE_KEY,
        env=Config.get("exchange", "env")
    )
    await exchange.connect()
    
    symbol = Config.get("exchange", "symbol", "BTC-USDT")
    logger.info(f"Starting Data Collection for {symbol}...")
    logger.info(f"Saving to {FILEPATH}")

    # Ensure data directory exists
    os.makedirs(DATA_DIR, exist_ok=True)

    # Open CSV file
    with open(FILEPATH, mode='w', newline='') as file:
        writer = csv.writer(file)
        # Header
        writer.writerow(["timestamp", "symbol", "best_bid", "best_ask", "bid_qty", "ask_qty", "spread"])
        
        try:
            while True:
                start_time = time.time()
                
                # Fetch Orderbook
                orderbook = await exchange.get_orderbook(symbol)
                
                if orderbook and 'bids' in orderbook and 'asks' in orderbook:
                    best_bid = orderbook['bids'][0][0] if orderbook['bids'] else 0
                    best_ask = orderbook['asks'][0][0] if orderbook['asks'] else 0
                    bid_qty = orderbook['bids'][0][1] if orderbook['bids'] else 0
                    ask_qty = orderbook['asks'][0][1] if orderbook['asks'] else 0
                    
                    spread = best_ask - best_bid
                    timestamp = datetime.now().isoformat()
                    
                    # Write to CSV
                    writer.writerow([timestamp, symbol, best_bid, best_ask, bid_qty, ask_qty, spread])
                    file.flush() # Ensure data is written immediately
                    
                    # logger.info(f"Recorded: {best_bid} / {best_ask}")
                
                # Wait for next tick (approx 1 sec)
                # Adjust sleep to maintain roughly 1s interval
                elapsed = time.time() - start_time
                sleep_time = max(0.1, 1.0 - elapsed)
                await asyncio.sleep(sleep_time)
                
        except KeyboardInterrupt:
            logger.info("Data collection stopped by user.")
        except Exception as e:
            logger.error(f"Error: {e}")

if __name__ == "__main__":
    try:
        asyncio.run(collect_data())
    except KeyboardInterrupt:
        pass
