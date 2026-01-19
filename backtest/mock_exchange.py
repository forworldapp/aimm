import logging
import pandas as pd
from typing import Dict, List, Optional
from core.exchange_interface import ExchangeInterface

print("DEBUG: PaperExchange Module Loaded - V3 (Overflow Fix)")

class MockExchange(ExchangeInterface):
    """
    Simulates an exchange for backtesting.
    V3: Fixed overflow issues and added position limits.
    """
    def __init__(self, data: pd.DataFrame, initial_balance: float = 10000.0):
        self.logger = logging.getLogger("MockExchange")
        self.data = data
        self.current_index = 0
        self.initial_balance = initial_balance
        self.balance = {'USDT': initial_balance, 'BTC': 0.0}
        self.orders = {} # {order_id: {symbol, side, price, quantity, status}}
        self.order_id_counter = 0
        self.position = {'amount': 0.0, 'entryPrice': 0.0, 'unrealizedPnL': 0.0}
        self.trade_history = []
        
        # Safety limits to prevent overflow
        self.max_position_btc = 1.0  # Max 1 BTC position
        self.max_orders = 100  # Max open orders

    def next_tick(self):
        """Move to the next time step."""
        if self.current_index < len(self.data) - 1:
            self.current_index += 1
            self._check_fills()
            return True
        return False

    def _get_current_row(self):
        return self.data.iloc[self.current_index]
    
    def get_current_price(self) -> float:
        """Get current mid price."""
        row = self._get_current_row()
        return (row['best_bid'] + row['best_ask']) / 2
    
    def get_equity(self) -> float:
        """Calculate total equity (USDT + BTC value)."""
        row = self._get_current_row()
        mid_price = (row['best_bid'] + row['best_ask']) / 2
        
        # Clamp BTC value to prevent overflow
        btc_balance = max(-1.0, min(1.0, self.balance['BTC']))
        btc_value = btc_balance * mid_price
        
        total = self.balance['USDT'] + btc_value
        
        # Sanity check - equity should be within reasonable range
        total = max(0, min(total, self.initial_balance * 100))
        
        return total

    def _check_fills(self):
        """Check if open orders match current market data."""
        row = self._get_current_row()
        
        # Simulating fills using OHLC High/Low
        market_high = row['high']
        market_low = row['low']
        
        for order_id, order in list(self.orders.items()):
            if order['status'] != 'open': 
                continue
            
            filled = False
            fill_price = order['price']
            
            if order['side'] == 'buy':
                if market_low <= order['price']:
                    filled = True
                        
            elif order['side'] == 'sell':
                if market_high >= order['price']:
                    filled = True
            
            if filled:
                self._execute_trade(order, fill_price)

    def _execute_trade(self, order, price):
        qty = order['quantity']
        cost = qty * price
        
        # Check position limits BEFORE executing
        if order['side'] == 'buy':
            new_btc = self.balance['BTC'] + qty
            if new_btc > self.max_position_btc:
                order['status'] = 'rejected'
                return
        else:
            new_btc = self.balance['BTC'] - qty
            if new_btc < -self.max_position_btc:
                order['status'] = 'rejected'
                return
        
        # Maker Rebate (Fee Level 4: -0.001%)
        rebate_rate = 0.00001 
        rebate = cost * rebate_rate
        
        if order['side'] == 'buy':
            # Buy: Pay cost, but get rebate
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
            self.position['amount'] = new_qty

        order['status'] = 'filled'
        self.trade_history.append({
            'timestamp': self._get_current_row()['timestamp'],
            'side': order['side'],
            'price': price,
            'qty': qty,
            'pnl': 0
        })

    # --- Interface Implementation ---

    async def connect(self):
        pass

    async def get_balance(self) -> Dict[str, float]:
        return self.balance

    async def place_limit_order(self, symbol: str, side: str, price: float, quantity: float) -> str:
        # Limit check - reject if too many orders
        open_orders = sum(1 for o in self.orders.values() if o['status'] == 'open')
        if open_orders >= self.max_orders:
            return "rejected_max_orders"
        
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
            'bids': [[row['best_bid'], 1.0]],
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
        
        # Use clamped BTC to prevent overflow
        btc_clamped = max(-1.0, min(1.0, self.balance['BTC']))
        total_equity = self.balance['USDT'] + (btc_clamped * mid_price)
        
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
            self.balance['BTC'] = 0.0
            self.position = {'amount': 0.0, 'entryPrice': 0.0, 'unrealizedPnL': 0.0}

