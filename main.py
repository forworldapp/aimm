import asyncio
import logging
from core.config import Config
from strategies.market_maker import MarketMaker

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("Main")

async def main():
    logger.info("Starting GRVT Bot...")
    
    # 0. Load Config
    Config.load("config.yaml")

    # 1. Initialize Exchange
    mode = Config.get("exchange", "mode", "paper")
    
    if mode == "paper":
        from core.paper_exchange import PaperGrvtExchange
        exchange = PaperGrvtExchange()
        logger.info("Initialized Paper Exchange")
    else:
        from core.grvt_exchange import GrvtExchange
        exchange = GrvtExchange()
        logger.info("Initialized Real Exchange")
    
    # 2. Connect
    await exchange.connect()
    
    # 3. Simple Check (Fetch Balance)
    balance = await exchange.get_balance()
    logger.info(f"Initial Balance: {balance}")
    
    # 4. Run Strategy
    strategy = MarketMaker(exchange)
    await strategy.run()

if __name__ == "__main__":
    import socket
    import sys

    # Singleton Pattern: Bind a specific port to prevent multiple instances
    def get_lock():
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            s.bind(("127.0.0.1", 45432))
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
        logger.error(f"Fatal error: {e}")
    finally:
        if lock_socket:
            lock_socket.close()
