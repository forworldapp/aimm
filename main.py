import asyncio
import logging
from core.config import Config
from core.grvt_exchange import GrvtExchange
from strategies.market_maker import MarketMaker

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("Main")

async def main():
    logger.info("Starting GRVT Bot...")
    
    # 1. Initialize Exchange
    exchange = GrvtExchange(
        api_key=Config.GRVT_API_KEY,
        private_key=Config.GRVT_PRIVATE_KEY,
        env=Config.GRVT_ENV
    )
    
    # 2. Connect
    await exchange.connect()
    
    # 3. Simple Check (Fetch Balance)
    balance = await exchange.get_balance()
    logger.info(f"Initial Balance: {balance}")
    
    # 4. Run Strategy
    # Initialize Market Maker Strategy
    # Parameters can be moved to Config later
    strategy = MarketMaker(
        exchange=exchange,
        symbol=Config.TRADING_PAIR,
        spread=Config.SPREAD_PCT,
        amount=Config.ORDER_AMOUNT,
        refresh_interval=5 # 5 seconds loop
    )
    
    # Run the strategy
    await strategy.run()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Bot stopped by user.")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
