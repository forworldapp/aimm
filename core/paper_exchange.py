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

print("DEBUG: PaperExchange Module Loaded - V2 (Direct Write)")

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
        real_exchange = GrvtExchange(Config.get("exchange", "env", "prod"))
        self.exchange = real_exchange.exchange
        
        # Paper Trading State
        self.paper_balance = {'USDT': 10000.0}
        self.paper_orders = {} 
        self.paper_position = {'amount': 0.0, 'entryPrice': 0.0, 'unrealizedPnL': 0.0}
        self.paper_order_id_counter = 0
        self.last_mid_price = 0.0
        
        # Grid Profit Tracking (v1.9.2 - FIFO)
        self.cumulative_grid_profit = 0.0
        self.increase_price_queue = []  # FIFO queue: [(price, qty), ...]
        self.last_increase_price = 0.0  # Keep for dashboard display
        
        self.symbol = Config.get("exchange", "symbol", "BTC_USDT_Perp")
        self.monitor_task = None
        self.status_file = os.path.join("data", f"paper_status_{self.symbol}.json")
        self.history_file = os.path.join("data", f"pnl_history_{self.symbol}.csv")
        self.trade_file = os.path.join("data", f"trade_history_{self.symbol}.csv")
        os.makedirs("data", exist_ok=True)
        
        # Initialize History Files
        if not os.path.exists(self.history_file):
            with open(self.history_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["timestamp", "total_usdt_value", "realized_pnl", "price"])

        if not os.path.exists(self.trade_file):
            with open(self.trade_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["timestamp", "symbol", "side", "price", "amount", "cost", "rebate", "realized_pnl", "grid_profit", "note"])

        # Persistence: Try to load previous state from JSON to survive restarts
        # This prevents the bot from losing track of positions and equity.
        try:
            if os.path.exists(self.status_file):
                with open(self.status_file, 'r') as f:
                    saved_status = json.load(f)
                    
                    # Restore Balance
                    if 'balance' in saved_status:
                        self.paper_balance = saved_status['balance']
                        
                    # Restore Position
                    if 'position' in saved_status:
                        # Dashboard uses 'amount', we use 'amount' internally too.
                        # Ensure fields match our internal structure
                        pos_data = saved_status['position']
                        self.paper_position = {
                            'amount': float(pos_data.get('amount', 0.0)),
                            'entryPrice': float(pos_data.get('entryPrice', 0.0)),
                            'unrealizedPnL': float(pos_data.get('unrealizedPnL', 0.0))
                        }
                    
                    self.logger.info(f"Restored Paper State. Bal: {self.paper_balance}, Pos: {self.paper_position}")
        except Exception as e:
            self.logger.error(f"Failed to restore paper state: {e}")

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
        # Maker Rebate: 0.01% (1 bps) - User specific
        rebate = cost * 0.0001 
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
        # Initialize variables that may not be set in all code paths
        grid_profit = 0.0
        realized_pnl = 0.0
        
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
            
            # Grid Profit Calculation (v1.9.2 - FIFO)
            # Match reduce trades with oldest increase trades first
            grid_profit = 0.0
            remaining_close_qty = close_qty
            
            while remaining_close_qty > 0 and self.increase_price_queue:
                inc_price, inc_qty = self.increase_price_queue[0]
                
                # How much can we match from this increase?
                match_qty = min(remaining_close_qty, inc_qty)
                
                # Calculate Grid Profit for this portion
                if old_pos > 0:  # Long -> Sell to close
                    portion_profit = (price - inc_price) * match_qty
                else:  # Short -> Buy to close
                    portion_profit = (inc_price - price) * match_qty
                
                grid_profit += portion_profit
                remaining_close_qty -= match_qty
                
                # Update or remove the queue entry
                if match_qty >= inc_qty:
                    self.increase_price_queue.pop(0)  # Fully matched, remove
                else:
                    # Partially matched, reduce quantity
                    self.increase_price_queue[0] = (inc_price, inc_qty - match_qty)
            
            self.cumulative_grid_profit += grid_profit
            
            # Entry Price does NOT change when reducing position
            new_entry = old_entry
            
            # Handle Flip (if qty > old_pos)
            if abs(signed_qty) > abs(old_pos):
                # We crossed 0. The remainder is a new open.
                remainder = abs(signed_qty) - abs(old_pos)
                # v1.9.3: Entry for flipped portion only (not the full qty)
                new_entry = price
                # Clear queue on position flip (new direction)
                self.increase_price_queue = []
                # Add the flip remainder to FIFO queue
                self.increase_price_queue.append((price, remainder))
                self.last_increase_price = price
        
        # Track increase prices for FIFO Grid Profit (only for pure increase, not flip)
        if is_opening and not is_closing:
            # Pure Increase: Calculate weighted average Entry (already done in Line 260-266)
            self.increase_price_queue.append((price, qty))
            self.last_increase_price = price  # Keep for dashboard
        
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
        grid_profit_str = f" | Grid: ${grid_profit:.4f}" if grid_profit != 0 else ""
        self.logger.info(f"PAPER TRADE: {action_label} | {qty} @ {price} | PnL: {realized_pnl:.2f}{grid_profit_str}")
        
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
                    grid_profit,
                    action_label
                ])
        except Exception as e:
            self.logger.error(f"Failed to save trade history: {e}")

        self._save_status()

    # --- State Saving ---

    def set_market_regime(self, regime: str):
        self.market_regime = regime
        self._save_status() # Force save so dashboard updates immediately

    def set_as_metrics(self, metrics: dict):
        """Set Avellaneda-Stoikov model metrics for dashboard display."""
        self.as_metrics = metrics
        # Don't force save here - will be saved in next cycle

    def _save_status(self):
        """Save current snapshot for dashboard. Direct write to avoid Windows lock issues."""
        status = {
            "timestamp": time.time(),
            "balance": self.paper_balance,
            "position": self.paper_position,
            "mid_price": self.last_mid_price,
            "open_orders_list": [
                {'side': o['side'], 'price': o['price'], 'amount': o['quantity']}
                for o in self.paper_orders.values() if o['status'] == 'open'
            ],
            "open_orders": len([o for o in self.paper_orders.values() if o['status'] == 'open']),
            "market_regime": getattr(self, 'market_regime', 'unknown'),
            "regime": getattr(self, 'market_regime', 'unknown'),
            "cumulative_grid_profit": self.cumulative_grid_profit,
            "last_increase_price": self.last_increase_price,
            # A&S Model Metrics (populated by strategy)
            "as_metrics": getattr(self, 'as_metrics', {
                "reservation_price": 0.0,
                "optimal_spread": 0.0,
                "volatility_sigma": 0.0,
                "gamma": 0.0,
                "kappa": 0.0
            })
        }

        try:
            # Direct write - simpler and avoids Windows replace() lock issues
            with open(self.status_file, "w") as f:
                json.dump(status, f)
        except (PermissionError, OSError) as e:
            self.logger.warning(f"Failed to save status: {e}")


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
        try:
            self._save_status() # Force save for dashboard visibility
        except Exception:
            pass # Non-critical: don't block order placement if save fails
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
        
    async def get_account_summary(self):
        """Simulate account summary for Risk Manager."""
        # Calculate Unrealized PnL
        pos = self.paper_position
        unrealized = 0.0
        if self.last_mid_price > 0 and pos['amount'] != 0:
             unrealized = (self.last_mid_price - pos['entryPrice']) * pos['amount']
             
        # Mock summary structure
        return {
            'total_equity': self.paper_balance['USDT'] + unrealized,
            'available_balance': self.paper_balance['USDT'],
            'initial_margin': 0.0,
            'maintenance_margin': 0.0
        }

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
