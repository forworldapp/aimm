#!/usr/bin/env python
"""Close BTC position via market order."""
import os
from dotenv import load_dotenv
load_dotenv()

from pysdk.grvt_ccxt import GrvtCcxt
from pysdk.grvt_ccxt_env import GrvtEnv

api = GrvtCcxt(
    env=GrvtEnv.PROD,
    parameters={
        'api_key': os.environ.get('GRVT_API_KEY'),
        'private_key': os.environ.get('GRVT_PRIVATE_KEY'),
        'trading_account_id': os.environ.get('GRVT_TRADING_ACCOUNT_ID')
    }
)

# Check current position
positions = api.fetch_positions(['BTC_USDT_Perp'])
for pos in positions:
    if pos.get('instrument') == 'BTC_USDT_Perp':
        size = float(pos.get('size', 0))
        print(f"Current Position: {size} BTC @ {pos.get('entry_price')}")
        
        if abs(size) > 0:
            # Close position with market order (opposite side)
            side = 'sell' if size > 0 else 'buy'
            close_size = abs(size)
            
            print(f"Closing position: {side} {close_size} BTC...")
            order = api.create_order(
                symbol='BTC_USDT_Perp',
                order_type='market',
                side=side,
                amount=close_size
            )
            print(f"Close order placed: {order.get('order_id')}")
        else:
            print("No position to close.")
        break
else:
    print("No BTC_USDT_Perp position found.")
