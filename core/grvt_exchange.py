import os
import asyncio
import logging
from typing import Dict, List, Optional
from .exchange_interface import ExchangeInterface

# Try importing the official SDK. 
# If not installed, we'll log a warning (for dev environment safety).
try:
    from grvt_pysdk.exchange.grvt_ccxt import GrvtCcxt
except ImportError:
    GrvtCcxt = None

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
        # Note: 'sandbox' param determines testnet vs mainnet in many CCXT implementations
        self.exchange = GrvtCcxt({
            'apiKey': self.api_key,
            'secret': self.private_key,
            'sandbox': (self.env == 'testnet')
        })

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
            balance = await self.exchange.fetch_balance()
            return balance
        except Exception as e:
            self.logger.error(f"Error fetching balance: {e}")
            return {}

    async def place_limit_order(self, symbol: str, side: str, price: float, quantity: float) -> str:
        """
        Places a limit order using the SDK.
        """
        if not self.exchange: return ""
        try:
            # CCXT standard: create_order(symbol, type, side, amount, price)
            order = await self.exchange.create_order(
                symbol=symbol,
                type='limit',
                side=side,
                amount=quantity,
                price=price
            )
            self.logger.info(f"Order placed: {order['id']}")
            return order['id']
        except Exception as e:
            self.logger.error(f"Failed to place order: {e}")
            raise

    async def cancel_order(self, symbol: str, order_id: str):
        if not self.exchange: return
        try:
            await self.exchange.cancel_order(order_id, symbol)
            self.logger.info(f"Order canceled: {order_id}")
        except Exception as e:
            self.logger.error(f"Failed to cancel order {order_id}: {e}")

    async def get_orderbook(self, symbol: str) -> Dict:
        if not self.exchange: return {}
        try:
            # limit=10 for top 10 bids/asks
            orderbook = await self.exchange.fetch_order_book(symbol, limit=10)
            return orderbook
        except Exception as e:
            self.logger.error(f"Error fetching orderbook: {e}")
            return {}

    async def get_open_orders(self, symbol: str) -> List[Dict]:
        if not self.exchange: return []
        try:
            orders = await self.exchange.fetch_open_orders(symbol)
            return orders
        except Exception as e:
            self.logger.error(f"Error fetching open orders: {e}")
            return []

    async def get_position(self, symbol: str) -> Dict:
        if not self.exchange: return {}
        try:
            # CCXT fetch_positions usually returns a list
            positions = await self.exchange.fetch_positions([symbol])
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
