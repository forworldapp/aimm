import os
import asyncio
import logging
from typing import Dict, List, Optional
from .exchange_interface import ExchangeInterface

# Try importing the official SDK. 
# If not installed, we'll log a warning (for dev environment safety).
try:
    # Based on pip show, the package uses 'pysdk' as top-level
    from pysdk.grvt_ccxt import GrvtCcxt
    from pysdk.grvt_ccxt_env import GrvtEnv
except ImportError:
    try:
        # Fallback in case it changes or I misread
        from grvt_pysdk.exchange.grvt_ccxt import GrvtCcxt
        from grvt_pysdk.exchange.grvt_ccxt_env import GrvtEnv
    except ImportError:
        GrvtCcxt = None
        GrvtEnv = None

class GrvtExchange(ExchangeInterface):
    """
    Concrete implementation of ExchangeInterface for GRVT Exchange.
    Uses grvt-pysdk for underlying communication.
    """

    def __init__(self, api_key: str, private_key: str, env: str = 'testnet'):
        self.logger = logging.getLogger(__name__)
        self.api_key = api_key
        self.private_key = private_key
        self.env = env
        self.exchange = None
        
        if GrvtCcxt is None:
            self.logger.error("grvt-pysdk not installed. Please install it via pip.")
            return

        # Initialize the CCXT-compatible wrapper
        # Correct usage: env must be GrvtEnv Enum, creds in parameters
        
        target_env = GrvtEnv.TESTNET
        if self.env == 'prod' or self.env == 'mainnet':
            target_env = GrvtEnv.PROD
            
        self.exchange = GrvtCcxt(
            env=target_env,
            parameters={
                'apiKey': self.api_key,
                'secret': self.private_key,
            }
        )

    async def connect(self):
        """
        GRVT SDK handles connection lazily usually, but we can verify creds here.
        """
        if not self.exchange:
            raise RuntimeError("GRVT SDK not initialized")
        
        self.logger.info(f"Connected to GRVT ({self.env})")
        # Optional: Load markets to cache symbol info
        # await self.exchange.load_markets() 

    async def get_balance(self) -> Dict[str, float]:
        if not self.exchange: return {}
        try:
            balance = self.exchange.fetch_balance()
            return balance
        except Exception as e:
            self.logger.error(f"Error fetching balance: {e}")
            return {}

    async def place_limit_order(self, symbol: str, side: str, price: float, quantity: float) -> Optional[str]:
        """
        Places a limit order using the SDK with retry logic.
        """
        if not self.exchange: return None
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # CCXT standard: create_order(symbol, type, side, amount, price)
                order = self.exchange.create_order(
                    symbol=symbol,
                    type='limit',
                    side=side,
                    amount=quantity,
                    price=price
                )
                self.logger.info(f"Order placed: {order['id']}")
                return order['id']
            except Exception as e:
                err_msg = str(e).lower()
                if "rate limit" in err_msg:
                    wait_time = 1.0 * (attempt + 1)
                    self.logger.warning(f"Rate limited. Waiting {wait_time}s...")
                    await asyncio.sleep(wait_time)
                    continue
                elif "insufficient" in err_msg:
                    self.logger.error(f"Insufficient balance: {e}")
                    return None # Don't retry
                else:
                    self.logger.error(f"Order failed (attempt {attempt+1}/{max_retries}): {e}")
                    if attempt == max_retries - 1:
                        return None
                    await asyncio.sleep(0.5)
        return None

    async def cancel_order(self, symbol: str, order_id: str):
        if not self.exchange: return
        try:
            self.exchange.cancel_order(order_id, symbol)
            self.logger.info(f"Order canceled: {order_id}")
        except Exception as e:
            self.logger.error(f"Failed to cancel order {order_id}: {e}")

    async def get_orderbook(self, symbol: str) -> Dict:
        if not self.exchange: return {}
        try:
            # limit=10 for top 10 bids/asks
            orderbook = self.exchange.fetch_order_book(symbol, limit=10)
            return orderbook
        except Exception as e:
            self.logger.error(f"Error fetching orderbook: {e}")
            return {}

    async def get_open_orders(self, symbol: str) -> List[Dict]:
        if not self.exchange: return []
        try:
            orders = self.exchange.fetch_open_orders(symbol)
            return orders
        except Exception as e:
            self.logger.error(f"Error fetching open orders: {e}")
            return []

    async def get_position(self, symbol: str) -> Dict:
        if not self.exchange: return {}
        try:
            # CCXT fetch_positions usually returns a list
            positions = self.exchange.fetch_positions([symbol])
            for pos in positions:
                if pos['symbol'] == symbol:
                    return {
                        'amount': float(pos['contracts']) if pos['contracts'] else 0.0,
                        'entryPrice': float(pos['entryPrice']) if pos['entryPrice'] else 0.0,
                        'unrealizedPnL': float(pos['unrealizedPnl']) if pos['unrealizedPnl'] else 0.0
                    }
            return {'amount': 0.0, 'entryPrice': 0.0, 'unrealizedPnL': 0.0}
        except Exception as e:
            self.logger.error(f"Error fetching position: {e}")
            return {'amount': 0.0, 'entryPrice': 0.0, 'unrealizedPnL': 0.0}
