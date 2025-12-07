import asyncio
import logging
from core.config import Config
from core.grvt_exchange import GrvtExchange

# Setup Logging
logging.basicConfig(level=logging.INFO)

async def main():
    Config.load("config.yaml")
    
    api_key = Config.GRVT_API_KEY
    private_key = Config.GRVT_PRIVATE_KEY
    env = Config.get("exchange", "env", "prod")
    
    print(f"Initializing Exchange (Env: {env})...")
    exchange = GrvtExchange(api_key, private_key, env)
    
    # Check if fetch_order_book is async
    import inspect
    is_async = inspect.iscoroutinefunction(exchange.exchange.fetch_order_book)
    print(f"Is fetch_order_book async? {is_async}")
    
    # Try to load markets
    try:
        print("Loading markets...")
        # Note: load_markets might be sync or async
        if inspect.iscoroutinefunction(exchange.exchange.load_markets):
            markets = await exchange.exchange.load_markets()
        else:
            markets = exchange.exchange.load_markets()
            
        print(f"Markets loaded: {len(markets)}")
        print("Available symbols:")
        for symbol in markets.keys():
            print(f" - {symbol}")
            
    except Exception as e:
        print(f"Error loading markets: {e}")

if __name__ == "__main__":
    asyncio.run(main())
