#!/usr/bin/env python
"""Debug fetch_my_trades response format."""
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

try:
    trades = api.fetch_my_trades('BTC_USDT_Perp', limit=5)
    print(f"Type: {type(trades)}")
    print(f"Length: {len(trades) if trades else 0}")
    if trades:
        if isinstance(trades, list):
            print("First item type:", type(trades[0]))
            print("First item:", trades[0])
        elif isinstance(trades, dict):
            first_key = list(trades.keys())[0]
            print("First key:", first_key)
            print("First value type:", type(trades[first_key]))
            print("First value:", trades[first_key])
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
