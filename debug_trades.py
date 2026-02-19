
import os
import sys
import yaml
import asyncio
import logging
from pprint import pprint

# Setup paths
sys.path.append(os.getcwd())

from core.grvt_exchange import GrvtExchange

async def main():
    logging.basicConfig(level=logging.INFO)
    
    # Load config
    try:
        with open('config.yaml', 'r') as f:
            conf = yaml.safe_load(f)
            # Set env vars for GrvtExchange to pick up
            if 'exchange' in conf and 'grvt_api_key' in conf['exchange']:
                # The config structure usually has these under 'exchange' or similar, 
                # but based on previous context, they might be in .env. 
                # Let's try to find where they are.
                # Actually, GrvtExchange reads from os.environ.
                pass
    except:
        pass

    # HARDCODED FALLBACK (for debugging only, if needed, but better to read .env if possible)
    # But wait, the bot runs fine, so .env probably exists or env vars are set in the terminal session?
    # No, the bot loads from .env using python-dotenv or similar.
    
    # Let's try to load .env manually
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    print("Initializing Exchange...")
    # Ensure env vars are present
    if not os.environ.get('GRVT_TRADING_ACCOUNT_ID'):
        print("GRVT_TRADING_ACCOUNT_ID not found in env. Attempting to load from .env file...")
        try:
             with open('.env', 'r') as f:
                 for line in f:
                     if '=' in line:
                         k, v = line.strip().split('=', 1)
                         os.environ[k] = v.strip('"').strip("'")
        except:
            print("Could not load .env")

    exchange = GrvtExchange(env='prod')
    
    if not exchange.exchange:
        print("Failed to init exchange")
        return

    print("Fetching trades...")
    try:
        # We'll use the GrvtExchange wrapper's internal exchange object directly for inspection
        response = exchange.exchange.fetch_my_trades('BTC_USDT_Perp', limit=10)
        
        print(f"\nType: {type(response)}")
        
        if isinstance(response, dict) and 'result' in response:
            trades = response['result']
        else:
            trades = response
            
        print(f"Trades count: {len(trades)}")
        
        print("\n--- Trade Structure Preview (Last 3) ---")
        if trades and len(trades) > 0:
            for t in trades[-3:]:
                print(f"Time: {t.get('event_time')} | ID: {t.get('trade_id')} | CID: {t.get('client_order_id')} | Side: {t.get('is_buyer')} | Price: {t.get('price')} | Size: {t.get('size')}")
        else:
            print("No trades found.")
            
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(main())
