import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    GRVT_API_KEY = os.getenv("GRVT_API_KEY")
    GRVT_PRIVATE_KEY = os.getenv("GRVT_PRIVATE_KEY")
    GRVT_ENV = os.getenv("GRVT_ENV", "testnet") # testnet or prod
    
    # Trading Parameters
    TRADING_PAIR = "BTC-USDT" # Example
    ORDER_AMOUNT = 0.001
    SPREAD_PCT = 0.001 # 0.1%
