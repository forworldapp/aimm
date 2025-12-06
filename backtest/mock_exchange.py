import logging
import pandas as pd
from typing import Dict, List, Optional
from core.exchange_interface import ExchangeInterface

class MockExchange(ExchangeInterface):
    """
    Simulates an exchange for backtesting.
    """
    def __init__(self, data: pd.DataFrame, initial_balance: float = 10000.0):
        self.logger = logging.getLogger("MockExchange")
        self.data = data
        self.current_index = 0
        self.balance = {'USDT': initial_balance, 'BTC': 0.0}
        self.orders = {} # {order_id: {symbol, side, price, quantity, status}}
        self.order_id_counter = 0
        self.position = {'amount': 0.0, 'entryPrice': 0.0, 'unrealizedPnL': 0.0}
        self.trade_history = []

    def next_tick(self):
        """Move to the next time step."""
        if self.current_index < len(self.data) - 1:
            self.current_index += 1
            self._check_fills()
            return True
        return False

    def _get_current_row(self):
        return self.data.iloc[self.current_index]

    def _check_fills(self):
        """Check if open orders match current market data."""
        row = self._get_current_row()
        best_bid = row['best_bid']
        best_ask = row['best_ask']
        
        # Simple Fill Logic for Maker Orders:
        # Buy Limit filled if Market Best Ask drops <= Limit Price (Price crashed to our bid)
        # Sell Limit filled if Market Best Bid rises >= Limit Price (Price pumped to our ask)
        # Note: This is a conservative approximation for backtesting on quote data.
        
        for order_id, order in list(self.orders.items()):
            if order['status'] != 'open': continue
            
            filled = False
            fill_price = order['price']
            
            if order['side'] == 'buy':
                if best_ask <= order['price']:
                    filled = True
            elif order['side'] == 'sell':
                if best_bid >= order['price']:
                    filled = True
            
            if filled:
                self._execute_trade(order, fill_price)

    def _execute_trade(self, order, price):
        qty = order['quantity']
        cost = qty * price
        
        if order['side'] == 'buy':
            self.balance['USDT'] -= cost
            self.balance['BTC'] += qty
            
            # Update Position (Weighted Average Price)
            old_qty = self.position['amount']
            new_qty = old_qty + qty
            if new_qty != 0:
                self.position['entryPrice'] = ((old_qty * self.position['entryPrice']) + cost) / new_qty
            self.position['amount'] = new_qty
            
        elif order['side'] == 'sell':
            self.balance['USDT'] += cost
            self.balance['BTC'] -= qty
            
            # Update Position
            old_qty = self.position['amount']
            new_qty = old_qty - qty
            # Entry price doesn't change on reduction, only realized PnL happens (tracked in balance)
            self.position['amount'] = new_qty

        order['status'] = 'filled'
        self.trade_history.append({
            'timestamp': self._get_current_row()['timestamp'],
            'side': order['side'],
            'price': price,
            'qty': qty,
            'pnl': 0 # Realized PnL calc is complex, skipping for MVP
        })
        # self.logger.info(f"Trade Filled: {order['side']} {qty} @ {price}")

    # --- Interface Implementation ---

    async def connect(self):
        pass

    async def get_balance(self) -> Dict[str, float]:
        return self.balance

    async def place_limit_order(self, symbol: str, side: str, price: float, quantity: float) -> str:
        self.order_id_counter += 1
        order_id = str(self.order_id_counter)
        self.orders[order_id] = {
            'id': order_id,
            'symbol': symbol,
            'side': side,
            'price': price,
            'quantity': quantity,
            'status': 'open'
        }
        return order_id

    async def cancel_order(self, symbol: str, order_id: str):
        if order_id in self.orders:
            self.orders[order_id]['status'] = 'canceled'

    async def get_orderbook(self, symbol: str) -> Dict:
        row = self._get_current_row()
        return {
            'bids': [[row['best_bid'], 1.0]], # Dummy qty
            'asks': [[row['best_ask'], 1.0]]
        }

    async def get_open_orders(self, symbol: str) -> List[Dict]:
        return [o for o in self.orders.values() if o['status'] == 'open']

    async def get_position(self, symbol: str) -> Dict:
        # Update Unrealized PnL
        row = self._get_current_row()
        mid_price = (row['best_bid'] + row['best_ask']) / 2
        
        pos = self.position
        if pos['amount'] != 0:
            # Long: (Current - Entry) * Qty
            # Short: (Entry - Current) * Qty (if amount is negative, logic handles it?)
            # Here amount is signed? Let's assume signed.
            pos['unrealizedPnL'] = (mid_price - pos['entryPrice']) * pos['amount']
            
        return pos
