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
        
        # Relaxed Fill Logic for Backtesting:
        # If our order is at the best price (or better), assume a fill probability.
        # In real MM, we are the best bid/ask.
        
        for order_id, order in list(self.orders.items()):
            if order['status'] != 'open': continue
            
            filled = False
            fill_price = order['price']
            
            if order['side'] == 'buy':
                # If Market Ask drops to our price (Cross) -> Definite Fill
                if best_ask <= order['price']:
                    filled = True
                # If we are at the Best Bid (Touch) -> Probabilistic Fill (e.g. 50%)
                elif best_bid <= order['price']: 
                    # Simulating Taker Sell hitting our Bid
                    import random
                    if random.random() < 0.5: 
                        filled = True
                        
            elif order['side'] == 'sell':
                # If Market Bid rises to our price (Cross) -> Definite Fill
                if best_bid >= order['price']:
                    filled = True
                # If we are at the Best Ask (Touch) -> Probabilistic Fill
                elif best_ask >= order['price']:
                    import random
                    if random.random() < 0.5:
                        filled = True
            
            if filled:
                self._execute_trade(order, fill_price)

    def _execute_trade(self, order, price):
        qty = order['quantity']
        cost = qty * price
        
        # Maker Rebate (Fee Level 4: -0.001%)
        # 0.001% = 0.00001
        rebate_rate = 0.00001 
        rebate = cost * rebate_rate
        
        if order['side'] == 'buy':
            # Buy: Pay cost, but get rebate (reduce cost)
            self.balance['USDT'] -= (cost - rebate)
            self.balance['BTC'] += qty
            
            # Update Position (Weighted Average Price)
            old_qty = self.position['amount']
            new_qty = old_qty + qty
            if new_qty != 0:
                self.position['entryPrice'] = ((old_qty * self.position['entryPrice']) + cost) / new_qty
            self.position['amount'] = new_qty
            
        elif order['side'] == 'sell':
            # Sell: Receive cost + rebate
            self.balance['USDT'] += (cost + rebate)
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

    async def cancel_all_orders(self, symbol: str):
        """Cancel all open orders."""
        for order_id, order in self.orders.items():
            if order['status'] == 'open':
                order['status'] = 'canceled'

    async def get_account_summary(self) -> Dict:
        """Return account summary for strategy."""
        row = self._get_current_row()
        mid_price = (row['best_bid'] + row['best_ask']) / 2
        total_equity = self.balance['USDT'] + (self.position['amount'] * mid_price)
        return {
            'total_equity': total_equity,
            'balance': self.balance,
            'position': self.position
        }

    async def close_position(self, symbol: str):
        """Close all positions at market price."""
        if self.position['amount'] != 0:
            row = self._get_current_row()
            close_price = row['best_bid'] if self.position['amount'] > 0 else row['best_ask']
            pnl = (close_price - self.position['entryPrice']) * self.position['amount']
            self.balance['USDT'] += pnl
            self.position = {'amount': 0.0, 'entryPrice': 0.0, 'unrealizedPnL': 0.0}
