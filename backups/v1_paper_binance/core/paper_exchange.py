import json
import os
import time
import logging
import asyncio
import random
import ccxt.async_support as ccxt
from typing import Dict, List
from core.grvt_exchange import GrvtExchange
from core.config import Config

class PaperGrvtExchange(GrvtExchange):
    """
    Paper Trading Exchange.
    Uses BINANCE data for simulation (proxy for GRVT), bypassing SDK errors.
    """
    def __init__(self):
        # Do NOT call super().__init__() to avoid SDK initialization error
        self.logger = logging.getLogger("PaperExchange")
        self.api_key = "dummy"
        self.private_key = "dummy"
        self.env = "paper"
        
        # Initialize Binance for Data
        self.exchange = ccxt.binance()
        
        # Paper Trading State
        self.paper_balance = {'USDT': 10000.0, 'BTC': 0.0}
        self.paper_orders = {} 
        self.paper_position = {'amount': 0.0, 'entryPrice': 0.0, 'unrealizedPnL': 0.0}
        self.paper_order_id_counter = 0
        
        self.monitor_task = None
        self.status_file = os.path.join("data", "paper_status.json")
        os.makedirs("data", exist_ok=True)

    async def connect(self):
        # Connect to Binance
        await self.exchange.load_markets()
        self.logger.info("Connected to Binance (Data Source for Paper Trading)")
        
        # Start fill checker loop
        self.monitor_task = asyncio.create_task(self._monitor_fills())
        self.logger.info("Paper Trading Mode Initialized. Balance: $10000")
        self._save_status()

    async def get_orderbook(self, symbol: str) -> Dict:
        # Map symbol if needed (GRVT 'BTC-USDT' -> Binance 'BTC/USDT')
        binance_symbol = symbol.replace("-", "/")
        try:
            ob = await self.exchange.fetch_order_book(binance_symbol, limit=10)
            return ob
        except Exception as e:
            self.logger.error(f"Error fetching orderbook from Binance: {e}")
            return {}

    def _save_status(self):
        """Save current paper trading status to JSON for dashboard."""
        try:
            status = {
                "timestamp": time.time(),
                "balance": self.paper_balance,
                "position": self.paper_position,
                "open_orders": len([o for o in self.paper_orders.values() if o['status'] == 'open'])
            }
            with open(self.status_file, "w") as f:
                json.dump(status, f)
        except Exception as e:
            self.logger.error(f"Failed to save paper status: {e}")

    async def _monitor_fills(self):
        """Continuously check if paper orders should be filled based on real market data."""
        while True:
            try:
                await self._check_paper_fills()
                # Periodically save status even if no trade (to update unrealized PnL if we implemented it fully)
                # For now, we save on trade.
            except Exception as e:
                self.logger.error(f"Error in paper fill monitor: {e}")
            await asyncio.sleep(0.1) # Check every 100ms

    async def _check_paper_fills(self):
        # Get Real Market Data
        symbol = Config.get("exchange", "symbol", "BTC-USDT")
        orderbook = await self.get_orderbook(symbol)
        if not orderbook or not orderbook['bids'] or not orderbook['asks']:
            return

        best_bid = orderbook['bids'][0][0]
        best_ask = orderbook['asks'][0][0]
        
        # Check Fills
        for order_id, order in list(self.paper_orders.items()):
            if order['status'] != 'open': continue
            
            filled = False
            fill_price = order['price']
            
            # Simple Fill Logic (Same as MockExchange)
            if order['side'] == 'buy':
                if best_ask <= order['price']: # Cross
                    filled = True
                elif best_bid <= order['price']: # Touch
                    if random.random() < 0.1: # 10% chance on touch in real-time
                        filled = True
                        
            elif order['side'] == 'sell':
                if best_bid >= order['price']: # Cross
                    filled = True
                elif best_ask >= order['price']: # Touch
                    if random.random() < 0.1:
                        filled = True
            
            if filled:
                self._execute_paper_trade(order, fill_price)

    def _execute_paper_trade(self, order, price):
        qty = order['quantity']
        cost = qty * price
        
        # Fee/Rebate (0.1bps rebate)
        rebate = cost * 0.00001
        
        if order['side'] == 'buy':
            self.paper_balance['USDT'] -= (cost - rebate)
            self.paper_balance['BTC'] += qty
            
            # Update Position
            old_qty = self.paper_position['amount']
            new_qty = old_qty + qty
            if new_qty != 0:
                self.paper_position['entryPrice'] = ((old_qty * self.paper_position['entryPrice']) + cost) / new_qty
            self.paper_position['amount'] = new_qty
            
        elif order['side'] == 'sell':
            self.paper_balance['USDT'] += (cost + rebate)
            self.paper_balance['BTC'] -= qty
            
            old_qty = self.paper_position['amount']
            new_qty = old_qty - qty
            self.paper_position['amount'] = new_qty

        order['status'] = 'filled'
        self.logger.info(f"PAPER TRADE FILLED: {order['side']} {qty} @ {price}")
        self._save_status()

    # --- Overridden Methods ---

    async def place_limit_order(self, symbol: str, side: str, price: float, quantity: float) -> str:
        self.paper_order_id_counter += 1
        order_id = f"paper_{self.paper_order_id_counter}"
        
        self.paper_orders[order_id] = {
            'id': order_id,
            'symbol': symbol,
            'side': side,
            'price': price,
            'quantity': quantity,
            'status': 'open'
        }
        # self.logger.info(f"Paper Order Placed: {side} {quantity} @ {price}")
        return order_id

    async def cancel_order(self, symbol: str, order_id: str):
        if order_id in self.paper_orders:
            self.paper_orders[order_id]['status'] = 'canceled'

    async def get_open_orders(self, symbol: str) -> List[Dict]:
        return [o for o in self.paper_orders.values() if o['status'] == 'open']

    async def get_position(self, symbol: str) -> Dict:
        # Update Unrealized PnL based on Real Price
        orderbook = await self.get_orderbook(symbol)
        if orderbook and orderbook['bids'] and orderbook['asks']:
            mid_price = (orderbook['bids'][0][0] + orderbook['asks'][0][0]) / 2
            pos = self.paper_position
            if pos['amount'] != 0:
                pos['unrealizedPnL'] = (mid_price - pos['entryPrice']) * pos['amount']
        
        return self.paper_position

    async def get_balance(self) -> Dict[str, float]:
        return self.paper_balance
