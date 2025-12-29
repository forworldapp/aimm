import asyncio
import logging
from dotenv import load_dotenv
load_dotenv()

from core.config import Config
from strategies.market_maker import MarketMaker

# --- Logging Setup ---
import os
import logging
from logging.handlers import RotatingFileHandler

def setup_logging():
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log_file = os.path.join(log_dir, "bot.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            RotatingFileHandler(log_file, maxBytes=5*1024*1024, backupCount=5),
            logging.StreamHandler()
        ]
    )
    logging.info("Logging initialized (System Time)")

setup_logging()
logger = logging.getLogger("Main")

# Suppress noisy logs
logging.getLogger("pysdk").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)

async def main():
    logger.info("Starting GRVT Bot...")
    
    # 4. Run Strategy (with restart loop)
    while True:
        # 0. Load Config (Reload on restart)
        Config.load("config.yaml")
        
        # 1. Initialize Exchange (Re-init for new symbol/settings)
        mode = Config.get("exchange", "mode", "paper")
        if mode == "paper":
            from core.paper_exchange import PaperGrvtExchange
            exchange = PaperGrvtExchange() # Will load new symbol from config
        else:
            from core.grvt_exchange import GrvtExchange
            # Graceful disconnect old exchange if possible? 
            # Currently GrvtExchange in core doesn't need explicit close but good practice.
            exchange = GrvtExchange()
            
        await exchange.connect()
        
        # 2. Run Strategy
        strategy = MarketMaker(exchange)
        logger.info(f"Starting Strategy on {exchange.symbol if mode=='paper' else 'Real'}")
        
        exit_code = await strategy.run()
        
        if exit_code == 'restart':
            logger.info("Restarting Bot Process...")
            await asyncio.sleep(2) # Cooldown
            continue # Loop again
        else:
            logger.info("Bot Stopped.")
            break

if __name__ == "__main__":
    import socket
    import sys

    # Singleton Pattern: Bind a specific port to prevent multiple instances
    def get_lock():
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            s.bind(("127.0.0.1", 45433)) # Changed port to avoid conflict
            return s
        except socket.error:
            return None

    lock_socket = get_lock()
    if not lock_socket:
        logger.error("FATAL: Another instance of GRVT Bot is already running.")
        print("Error: Bot is already running! Check terminal or task manager.")
        sys.exit(1)

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Bot stopped by user.")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
    finally:
        if lock_socket:
            lock_socket.close()
