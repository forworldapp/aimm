import asyncio
import logging
import json
import os
import csv
import time
import random
import ccxt.async_support as ccxt
from typing import Dict, List, Optional
from core.grvt_exchange import GrvtExchange
from core.config import Config

class PaperGrvtExchange(GrvtExchange):
    """
    Simulated Exchange using Real Mainnet Data (GRVT).
    
    Mechanism:
    - **Data Feed**: Connects to Real GRVT Mainnet to fetch live Orderbook & Prices.
    - **Execution**: Simulates order matching locally.
      - **Maker Logic**: Orders rest in `paper_orders`. They fill if market moves past them.
      - **Taker Logic**: Immediate fills if crossing the spread (disabled by strategy Post-Only).
    - **State Management**:
      - Tracks distinct Paper Balance (USDT) and Position (BTC).
      - Saves status to `data/paper_status.json` every second for Dashboard visibility.
      - Appends equity history to `data/pnl_history.csv` for Charting.
    """
    def __init__(self):
        # Do NOT call super().__init__() to avoid SDK initialization error
        self.logger = logging.getLogger("PaperExchange")
        self.api_key = "dummy"
        self.private_key = "dummy"
        self.env = "paper"
        
        # Initialize Real GRVT Exchange for Data
        Config.load()
        # We access the internal ccxt wrapper directly
        real_exchange = GrvtExchange(Config.GRVT_API_KEY, Config.GRVT_PRIVATE_KEY, Config.get("exchange", "env", "prod"))
        self.exchange = real_exchange.exchange
        
        # Paper Trading State
        self.paper_balance = {'USDT': 10000.0}
        self.paper_orders = {} 
        self.paper_position = {'amount': 0.0, 'entryPrice': 0.0, 'unrealizedPnL': 0.0}
        self.paper_order_id_counter = 0
        self.last_mid_price = 0.0
        
        self.monitor_task = None
        self.status_file = os.path.join("data", "paper_status.json")
        self.history_file = os.path.join("data", "pnl_history.csv")
        self.trade_file = os.path.join("data", "trade_history.csv")
        os.makedirs("data", exist_ok=True)
        
        # Initialize History Files
        if not os.path.exists(self.history_file):
            with open(self.history_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["timestamp", "total_usdt_value", "realized_pnl", "price"])

        if not os.path.exists(self.trade_file):
            with open(self.trade_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["timestamp", "symbol", "side", "price", "quantity", "cost", "fee", "realized_pnl", "action"])

    async def connect(self):
        # Prevent multiple monitor loops
        if self.monitor_task is not None and not self.monitor_task.done():
            self.logger.warning("Monitor task, already running.")
            return

        # Start fill checker loop
        self.monitor_task = asyncio.create_task(self._monitor_fills())
        self.logger.info("Paper Trading Mode Initialized. Balance: $10000")
        self._save_status()
    
    # ... (monitor_fills and check_fills remain same) ...

    # We need to preserve _monitor_fills and _check_paper_fills logic, just targeting __init__ and _execute_trade
    # Since replace_file_content target context is small, I will split edits.

    # First edit: __init__ adds file init
    # Second edit: _execute_paper_trade adds CSV write
    pass


    # --- Data & Execution Loop ---

    async def _monitor_fills(self):
        """Continuously check fills and update PnL."""
        last_history_save = 0
        while True:
            try:
                await self._check_paper_fills()
                self._update_unrealized_pnl()
                
                # Save History & Status every 1 second
                if time.time() - last_history_save >= 1.0:
                    self._save_history()
                    self._save_status() # Force update dashboard status (Price, PnL)
                    last_history_save = time.time()
                    
            except Exception as e:
                self.logger.error(f"Error in paper fill monitor: {e}")
            await asyncio.sleep(0.1) # High frequency check

    async def _check_paper_fills(self):
        # Get Real Market Data
        symbol = Config.get("exchange", "symbol", "BTC_USDT_Perp")
        try:
            # SDK fetch_order_book limit usually supports 10, 20...
            orderbook = self.exchange.fetch_order_book(symbol, limit=10)
        except Exception as e:
            # self.logger.error(f"Error fetching orderbook: {e}")
            return # Skip this tick
            
        if not orderbook or 'bids' not in orderbook or 'asks' not in orderbook:
            return

        try:
            # Parse Prices
            bids = orderbook['bids']
            asks = orderbook['asks']
            if not bids or not asks: return

            best_bid = float(bids[0]['price']) if isinstance(bids[0], dict) else float(bids[0][0])
            best_ask = float(asks[0]['price']) if isinstance(asks[0], dict) else float(asks[0][0])
            mid_price = (best_bid + best_ask) / 2
            self.last_mid_price = mid_price
        except:
            return
        
        # Check Fills
        for order_id, order in list(self.paper_orders.items()):
            if order['status'] != 'open': continue
            
            filled = False
            fill_price = order['price']
            
            # Simple Fill Logic
            if order['side'] == 'buy':
                if best_ask <= order['price']: # Cross
                    filled = True
                elif best_bid <= order['price']: # Touch
                    if random.random() < 0.10: # 10% likelihood (Realistic)
                        filled = True
                        
            elif order['side'] == 'sell':
                if best_bid >= order['price']: # Cross
                    filled = True
                elif best_ask >= order['price']: # Touch
                    if random.random() < 0.10:
                        filled = True
            
            if filled:
                self._execute_paper_trade(order, fill_price)

    def _update_unrealized_pnl(self):
        """Update unrealized PnL based on last mid price."""
        pos = self.paper_position
        if abs(pos['amount']) < 1e-9: # Handle floating point errors
            pos['amount'] = 0.0
            pos['unrealizedPnL'] = 0.0
            return

        price = self.last_mid_price
        if price <= 0: return # No price data yet

        if pos['amount'] > 0: # Long
            pos['unrealizedPnL'] = (price - pos['entryPrice']) * pos['amount']
        else: # Short
            pos['unrealizedPnL'] = (pos['entryPrice'] - price) * abs(pos['amount'])

    def _execute_paper_trade(self, order, price):
        qty = order['quantity'] # Always positive
        side = order['side']
        
        cost = qty * price
        rebate = cost * 0.00001
        fee = 0 # Currently using rebate model, fee is negative rebate
        
        # Current Position
        old_pos = self.paper_position['amount']
        old_entry = self.paper_position['entryPrice']
        
        realized_pnl = 0.0
        new_pos = 0.0
        new_entry = old_entry
        
        # --- Logic: Split into Opening (Increase Size) vs Closing (Reduce Size) ---
        
        # Determine movement direction
        # Buy -> +qty, Sell -> -qty
        signed_qty = qty if side == 'buy' else -qty
        new_pos = old_pos + signed_qty
        
        # Check if we are increasing position (moving away from 0) or reducing (moving towards 0)
        is_opening = False
        is_closing = False
        
        if old_pos == 0:
            is_opening = True
        elif (old_pos > 0 and signed_qty > 0) or (old_pos < 0 and signed_qty < 0):
            # Same sign: Increasing position
            is_opening = True
        elif (old_pos > 0 and signed_qty < 0) or (old_pos < 0 and signed_qty > 0):
            # Opposite sign: Reducing or Flipping
            if abs(signed_qty) <= abs(old_pos):
                is_closing = True
            else:
                # Flip position (Close All + Open New)
                # Treating complexities by splitting trade is best, but for simple paper trading:
                # 1. Realize PnL on full old closure
                # 2. Open remainder as new position
                is_closing = True # Logic below handles full close part
                is_opening = True # Logic below needs to handle remainder
        
        # --- Update Balance (Fees/Rebates) ---
        # NOTE: In USD-Margined Perps, we only track USDT balance.
        # Rebates are added directly to USDT balance.
        self.paper_balance['USDT'] += rebate

        # --- Update Position & PnL ---
        
        if old_pos == 0:
            # Simple Open
            new_entry = price
        
        elif is_opening and not is_closing:
            # Pure Increase (Weighted Average Entry)
            # Entry = (OldVal + NewVal) / TotalQty
            old_val = abs(old_pos) * old_entry
            new_val = qty * price
            total_qty = abs(new_pos)
            new_entry = (old_val + new_val) / total_qty
            
        elif is_closing:
            # Reducing Position -> Realize PnL
            # PnL = (Exit - Entry) * Qty * Sign
            # Long (Old>0), Sell (Price): (Price - Entry) * Qty
            # Short (Old<0), Buy (Price): (Entry - Price) * Qty
            
            close_qty = min(abs(old_pos), qty)
            
            if old_pos > 0: # Closing Long
                pnl = (price - old_entry) * close_qty
            else: # Closing Short
                pnl = (old_entry - price) * close_qty
                
            realized_pnl = pnl
            self.paper_balance['USDT'] += realized_pnl
            
            # Entry Price does NOT change when reducing position
            new_entry = old_entry
            
            # Handle Flip (if qty > old_pos)
            if abs(signed_qty) > abs(old_pos):
                # We crossed 0. The remainder is a new open.
                remainder = abs(signed_qty) - abs(old_pos)
                new_entry = price # New entry for the flipped portion
        
        # Update State
        self.paper_position['amount'] = round(new_pos, 8)
        self.paper_position['entryPrice'] = new_entry if abs(self.paper_position['amount']) > 1e-8 else 0.0
        
        # Reset Unrealized if closed
        if new_pos == 0:
            self.paper_position['unrealizedPnL'] = 0.0

        # Determine Action Label
        action_label = "Trade"
        if old_pos == 0:
            action_label = f"Open {side.title()}"
        elif is_opening and not is_closing:
            action_label = f"Increase {side.title()}"
        elif is_closing:
             action_label = f"Reduce {'Long' if old_pos > 0 else 'Short'}"

        order['status'] = 'filled'
        self.logger.info(f"PAPER TRADE: {action_label} | {qty} @ {price} | PnL: {realized_pnl:.2f}")
        
        # Save Trade History
        try:
            with open(self.trade_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    time.time(),
                    order.get('symbol', 'BTC_USDT_Perp'),
                    side,
                    price,
                    qty,
                    cost,
                    rebate,
                    realized_pnl,
                    action_label
                ])
        except Exception as e:
            self.logger.error(f"Failed to save trade history: {e}")

        self._save_status()

    # --- State Saving ---

    def _save_status(self):
        """Save current snapshot for dashboard."""
        try:
            status = {
                "timestamp": time.time(),
                "balance": self.paper_balance,
                "position": self.paper_position,
                "mid_price": self.last_mid_price,
                "open_orders": len([o for o in self.paper_orders.values() if o['status'] == 'open'])
            }
            with open(self.status_file, "w") as f:
                json.dump(status, f)
        except Exception as e:
            self.logger.error(f"Failed to save paper status: {e}")

    def _save_history(self):
        """Append current equity to history CSV."""
        try:
            # Calculate Total Equity (USDT + BTC Value + Unrealized PnL)
            # Actually, paper_balance['USDT'] includes realized PnL already.
            # We just need to add Unrealized PnL of current position.
            
            pos = self.paper_position
            pos_val = 0
            unrealized = 0
            
            if self.last_mid_price > 0 and abs(pos['amount']) > 1e-8:
                unrealized = (self.last_mid_price - pos['entryPrice']) * pos['amount']
            
            total_equity = self.paper_balance['USDT'] + unrealized
            
            # Simple Append
            with open(self.history_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    time.time(),
                    round(total_equity, 2),
                    0, # Realized PnL tracking needs complexity, skipping for total equity focus
                    round(self.last_mid_price, 2)
                ])
                
        except Exception as e:
            self.logger.error(f"Failed to save history: {e}")

    # --- Overridden Methods for Bot Interaction ---

    async def get_orderbook(self, symbol: str) -> Dict:
        # Re-implement wrapper to call sync SDK method asynchronously if needed, or just call directly
        # Since we use self.exchange (GrvtCcxt) directly in _check_paper_fills, this is for the bot strategy
        try:
            # SDK fetch_order_book is sync
            return self.exchange.fetch_order_book(symbol, limit=10)
        except:
            return {}

    async def place_limit_order(self, symbol: str, side: str, price: float, quantity: float) -> str:
        self.paper_order_id_counter += 1
        order_id = f"paper_{self.paper_order_id_counter}"
        self.paper_orders[order_id] = {
            'id': order_id, 'symbol': symbol, 'side': side,
            'price': price, 'quantity': quantity, 'status': 'open'
        }
        return order_id

    async def cancel_order(self, symbol: str, order_id: str):
        if order_id == "all":
            await self.cancel_all_orders(symbol)
            return
        if order_id in self.paper_orders:
            self.paper_orders[order_id]['status'] = 'canceled'

    async def get_open_orders(self, symbol: str) -> List[Dict]:
        return [o for o in self.paper_orders.values() if o['status'] == 'open']

    async def get_position(self, symbol: str) -> Dict:
        # Update unrealized PnL before returning
        pos = self.paper_position
        if self.last_mid_price > 0 and pos['amount'] != 0:
             pos['unrealizedPnL'] = (self.last_mid_price - pos['entryPrice']) * pos['amount']
        return pos

    async def cancel_all_orders(self, symbol: str):
        for oid in self.paper_orders:
            if self.paper_orders[oid]['status'] == 'open':
                self.paper_orders[oid]['status'] = 'canceled'
        self._save_status()

    async def close_position(self, symbol: str):
        pos = self.paper_position
        if pos['amount'] == 0: return
        
        # Force close at last price
        price = self.last_mid_price
        qty = abs(pos['amount'])
        cost = qty * price
        rebate = cost * 0.00001
        
        if pos['amount'] > 0: # Closing Long
            pnl = (price - pos['entryPrice']) * qty
        else: # Closing Short
            pnl = (pos['entryPrice'] - price) * qty
            
        realized_pnl = pnl + rebate # Add rebate
        self.paper_balance['USDT'] += realized_pnl
            
        self.paper_position = {'amount': 0.0, 'entryPrice': 0.0, 'unrealizedPnL': 0.0}
        
        action_label = "Manual Close"
        self.logger.info(f"Position Closed via Dashboard. PnL: {realized_pnl:.2f}")

        # Save Trade History
        try:
            with open(self.trade_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    time.time(),
                    symbol,
                    'sell' if pos['amount'] > 0 else 'buy', # Counter trade
                    price,
                    qty,
                    cost,
                    rebate,
                    realized_pnl,
                    action_label
                ])
        except Exception as e:
            self.logger.error(f"Failed to save trade history: {e}")

        self._save_status()

    async def get_balance(self) -> Dict[str, float]:
        return self.paper_balance
