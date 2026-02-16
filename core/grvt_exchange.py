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

    def __init__(self, env: str = None):
        self.logger = logging.getLogger(__name__)
        # Load credentials from environment variables
        self.env = env or os.environ.get('GRVT_ENV', 'testnet')
        self.api_key = os.environ.get('GRVT_API_KEY')
        self.private_key = os.environ.get('GRVT_PRIVATE_KEY')
        self.trading_account_id = os.environ.get('GRVT_TRADING_ACCOUNT_ID')
        self.exchange = None
        self._bot_order_ids = set()  # Track order IDs placed by this bot
        self._last_fill_info = None  # Last fill info for Telegram notification
        
        if GrvtCcxt is None:
            self.logger.error("grvt-pysdk not installed. Please install it via pip.")
            return

        # Initialize the CCXT-compatible wrapper
        target_env = GrvtEnv.TESTNET
        if self.env == 'prod' or self.env == 'mainnet':
            target_env = GrvtEnv.PROD
        
        # SDK requires numeric sub_account_id for EIP712 signing
        self.exchange = GrvtCcxt(
            env=target_env,
            parameters={
                'api_key': self.api_key,
                'private_key': self.private_key,
                'trading_account_id': self.trading_account_id
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
                # GRVT SDK: create_order(symbol, order_type, side, amount, price)
                order = self.exchange.create_order(
                    symbol=symbol,
                    order_type='limit',
                    side=side,
                    amount=quantity,
                    price=price
                )
                order_id = order.get('order_id', order.get('id'))
                if order_id:
                    self._bot_order_ids.add(order_id)
                self.logger.info(f"Order placed: {order_id}")
                return order_id
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
                # GRVT uses 'instrument' instead of 'symbol'
                pos_symbol = pos.get('instrument', pos.get('symbol', ''))
                if pos_symbol == symbol:
                    # GRVT uses 'size' and 'entry_price' keys
                    size = pos.get('size', pos.get('contracts', 0))
                    entry = pos.get('entry_price', pos.get('entryPrice', 0))
                    upnl = pos.get('unrealized_pnl', pos.get('unrealizedPnl', 0))
                    return {
                        'amount': float(size) if size else 0.0,
                        'entryPrice': float(entry) if entry else 0.0,
                        'unrealizedPnL': float(upnl) if upnl else 0.0
                    }
            return {'amount': 0.0, 'entryPrice': 0.0, 'unrealizedPnL': 0.0}
        except Exception as e:
            self.logger.error(f"Error fetching position: {e}")
            return {'amount': 0.0, 'entryPrice': 0.0, 'unrealizedPnL': 0.0}

    async def get_account_summary(self) -> Dict:
        """Fetch account summary (equity including unrealized PnL) from GRVT."""
        if not self.exchange: return {}
        try:
            # Use get_account_summary for total_equity with unrealized PnL
            summary = self.exchange.get_account_summary()
            total_equity = float(summary.get('total_equity', 0))
            available = float(summary.get('available_balance', 0))
            unrealized_pnl = float(summary.get('unrealized_pnl', 0))
            return {
                'total_equity': total_equity,
                'available_balance': available,
                'unrealized_pnl': unrealized_pnl,
                'initial_margin': float(summary.get('initial_margin', 0)),
                'maintenance_margin': float(summary.get('maintenance_margin', 0))
            }
        except Exception as e:
            self.logger.error(f"Error fetching account summary: {e}")
            return {'total_equity': 0.0, 'available_balance': 0.0, 'unrealized_pnl': 0.0}

    async def cancel_all_orders(self, symbol: str):
        """Cancel all open orders for a symbol."""
        if not self.exchange: return
        try:
            orders = await self.get_open_orders(symbol)
            for order in orders:
                order_id = order.get('order_id', order.get('id'))
                if order_id:
                    await self.cancel_order(symbol, order_id)
            self.logger.info(f"Cancelled all orders for {symbol}")
        except Exception as e:
            self.logger.error(f"Error cancelling all orders: {e}")

    async def close_position(self, symbol: str):
        """Close the current position by placing a market order in the opposite direction."""
        if not self.exchange: return
        try:
            position = await self.get_position(symbol)
            size = position.get('amount', 0.0)
            if size == 0:
                self.logger.info(f"No position to close for {symbol}")
                return
            
            # Determine the closing side
            side = 'sell' if size > 0 else 'buy'
            abs_size = abs(size)
            
            self.logger.info(f"Closing position: {side} {abs_size} {symbol}")
            
            # Place market order to close
            order = self.exchange.create_order(
                symbol=symbol,
                order_type='market',
                side=side,
                amount=abs_size
            )
            order_id = order.get('order_id', order.get('id'))
            self.logger.info(f"Position closed with market order: {order_id}")
            return order_id
        except Exception as e:
            self.logger.error(f"Market close failed for {symbol}: {e}")
            # Fallback: try limit order at aggressive price
            try:
                position = await self.get_position(symbol)
                size = position.get('amount', 0.0)
                if size == 0:
                    return
                side = 'sell' if size > 0 else 'buy'
                abs_size = abs(size)
                
                # Get current price and add/subtract 0.5% for aggressive fill
                ob = await self.get_orderbook(symbol)
                if side == 'sell' and ob.get('bids'):
                    price = float(ob['bids'][0][0]) * 0.995
                elif side == 'buy' and ob.get('asks'):
                    price = float(ob['asks'][0][0]) * 1.005
                else:
                    self.logger.error("Cannot determine closing price")
                    return
                
                self.logger.info(f"Fallback: closing via limit order at {price}")
                return await self.place_limit_order(symbol, side, price, abs_size)
            except Exception as e2:
                self.logger.error(f"Fallback close also failed: {e2}")

    def save_live_status(self, symbol: str, mid_price: float, regime: str, 
                         position: Dict, open_orders: List, equity: float):
        """Save current status for dashboard (Live mode equivalent of _save_status)."""
        import json
        import os
        import time
        import csv
        
        # Calculate cumulative grid profit from trade history
        trade_file = os.path.join("data", f"trade_history_{symbol.replace('/', '_')}.csv")
        cumulative_grid_profit = 0.0
        if os.path.exists(trade_file):
            try:
                with open(trade_file, 'r') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        cumulative_grid_profit += float(row.get('grid_profit', 0))
            except Exception:
                pass
        
        status_file = os.path.join("data", f"paper_status_{symbol.replace('/', '_')}.json")
        status = {
            "timestamp": time.time(),
            "balance": {"USDT": equity},
            "position": position,
            "mid_price": mid_price,
            "open_orders_list": [
                {
                    'side': 'buy' if o.get('legs', [{}])[0].get('is_buying_asset') else 'sell',
                    'price': float(o.get('legs', [{}])[0].get('limit_price', 0)),
                    'amount': float(o.get('legs', [{}])[0].get('size', 0))
                }
                for o in open_orders if o.get('legs')
            ],
            "open_orders": len(open_orders),
            "market_regime": regime,
            "regime": regime,
            "cumulative_grid_profit": round(cumulative_grid_profit, 4),
            "last_increase_price": 0.0
        }
        
        try:
            tmp_file = status_file + ".tmp"
            with open(tmp_file, "w") as f:
                json.dump(status, f)
            os.replace(tmp_file, status_file)
            
            # Also save history for charts
            history_file = os.path.join("data", f"pnl_history_{symbol.replace('/', '_')}.csv")
            with open(history_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    time.time(),
                    round(equity, 2),
                    round(cumulative_grid_profit, 2),  # Realized PnL from trades
                    round(mid_price, 2)
                ])
        except Exception as e:
            self.logger.warning(f"Failed to save live status: {e}")

    def fetch_and_save_trades(self, symbol: str):
        """Fetch recent trades from GRVT and save ONLY bot-placed trades to history CSV.
        
        Filters out manual trades by matching trade's order_id against
        self._bot_order_ids (orders placed by this bot instance).
        Uses FIFO matching for grid profit calculation.
        """
        import csv
        import os
        import time
        from collections import deque
        
        if not self.exchange:
            return
            
        try:
            # Fetch recent trades for this symbol only
            response = self.exchange.fetch_my_trades(symbol, limit=50)
            if not response:
                return
            
            trades_list = response.get('result', []) if isinstance(response, dict) else response
            if not trades_list:
                return
                
            trade_file = os.path.join("data", f"trade_history_{symbol.replace('/', '_')}.csv")
            
            # Read existing trade IDs to avoid duplicates
            existing_ids = set()
            buy_queue = deque()
            sell_queue = deque()
            
            if os.path.exists(trade_file):
                with open(trade_file, 'r') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        note = row.get('note', '')
                        if note.startswith('Fill '):
                            trade_id = note.replace('Fill ', '')
                            existing_ids.add(trade_id)
                            
                            direction = row.get('direction', '')
                            side = row.get('side', '')
                            if direction == 'increase':
                                price = float(row.get('price', 0))
                                amount = float(row.get('amount', 0))
                                if side == 'buy':
                                    buy_queue.append((price, amount, trade_id))
                                else:
                                    sell_queue.append((price, amount, trade_id))
            
            # Sort trades by time (oldest first for correct FIFO)
            trades_list = sorted(trades_list, key=lambda x: x.get('event_time', 0))
            
            # Process new trades - ONLY those placed by this bot
            new_rows = []
            skipped_manual = 0
            for trade in trades_list:
                trade_id = trade.get('trade_id', '')
                order_id = trade.get('order_id', '')
                if not trade_id or trade_id in existing_ids:
                    continue
                
                # *** FILTER: Only include trades from bot-placed orders ***
                if order_id not in self._bot_order_ids:
                    skipped_manual += 1
                    continue
                    
                # GRVT trade format
                is_buyer = trade.get('is_buyer')
                side = 'buy' if is_buyer else 'sell'
                price = float(trade.get('price', 0))
                size = float(trade.get('size', 0))
                fill_time = int(trade.get('event_time', time.time() * 1e9)) / 1e9
                realized_pnl = float(trade.get('realized_pnl', 0))
                fee = float(trade.get('fee', 0))
                
                # Direction: increase (open) vs reduce (close)
                direction = 'increase' if realized_pnl == 0 else 'reduce'
                
                # FIFO grid profit calculation
                grid_profit = 0.0
                if direction == 'increase':
                    if side == 'buy':
                        buy_queue.append((price, size, trade_id))
                    else:
                        sell_queue.append((price, size, trade_id))
                else:
                    if side == 'buy' and sell_queue:
                        entry_price, _, _ = sell_queue.popleft()
                        grid_profit = (entry_price - price) * size
                    elif side == 'sell' and buy_queue:
                        entry_price, _, _ = buy_queue.popleft()
                        grid_profit = (price - entry_price) * size
                
                new_rows.append([
                    fill_time,
                    symbol.replace('/', '_'),
                    side,
                    direction,
                    price,
                    size,
                    round(price * size, 2),
                    abs(fee),
                    realized_pnl,
                    round(grid_profit, 4),
                    f"Fill {trade_id[:12]}"
                ])
                # Store last fill for Telegram notification
                self._last_fill_info = {
                    'side': side,
                    'price': price,
                    'size': size,
                    'grid_profit': grid_profit
                }
                existing_ids.add(trade_id)
            
            if new_rows:
                file_exists = os.path.exists(trade_file) and os.path.getsize(trade_file) > 0
                with open(trade_file, 'a', newline='') as f:
                    writer = csv.writer(f)
                    if not file_exists:
                        writer.writerow(['timestamp', 'symbol', 'side', 'direction', 'price', 'amount', 'cost', 'rebate', 'realized_pnl', 'grid_profit', 'note'])
                    writer.writerows(new_rows)
                self.logger.info(f"Saved {len(new_rows)} bot trades (skipped {skipped_manual} manual)")
            elif skipped_manual > 0:
                self.logger.debug(f"No new bot trades (skipped {skipped_manual} manual)")
                
        except Exception as e:
            self.logger.warning(f"Failed to fetch trades: {e}")

    def _get_trade_count(self, symbol: str) -> int:
        """Get count of recorded bot trades."""
        import os, csv
        trade_file = os.path.join("data", f"trade_history_{symbol.replace('/', '_')}.csv")
        if not os.path.exists(trade_file):
            return 0
        try:
            with open(trade_file, 'r') as f:
                return sum(1 for _ in csv.DictReader(f))
        except Exception:
            return 0

    def get_bot_pnl(self, symbol: str, current_price: float) -> dict:
        """Calculate bot-only P&L from trade_history CSV.
        
        Returns:
            dict with keys:
                - bot_net_qty: net position size from bot trades only
                - bot_avg_entry: weighted average entry price
                - realized_pnl: sum of grid_profit (closed trades)
                - unrealized_pnl: (current_price - avg_entry) * net_qty
                - total_pnl: realized + unrealized
                - trade_count: number of bot trades
                - bot_cost_basis: total capital used by bot trades
        """
        import os
        import pandas as pd
        
        result = {
            'bot_net_qty': 0.0,
            'bot_avg_entry': 0.0,
            'realized_pnl': 0.0,
            'unrealized_pnl': 0.0,
            'total_pnl': 0.0,
            'trade_count': 0,
            'bot_cost_basis': 0.0
        }
        
        trade_file = os.path.join("data", f"trade_history_{symbol.replace('/', '_')}.csv")
        if not os.path.exists(trade_file):
            return result
        
        try:
            df = pd.read_csv(trade_file)
            if df.empty:
                return result
            
            result['trade_count'] = len(df)
            
            # Realized P&L from grid profits
            if 'grid_profit' in df.columns:
                result['realized_pnl'] = round(df['grid_profit'].sum(), 4)
            
            # Calculate bot net position and average entry price
            net_qty = 0.0
            cost_basis = 0.0  # total cost of current position
            
            for _, row in df.iterrows():
                side = row.get('side', '')
                price = float(row.get('price', 0))
                amount = float(row.get('amount', 0))
                
                if side == 'buy':
                    # Adding to long / reducing short
                    if net_qty >= 0:
                        # Adding to long: update cost basis
                        cost_basis += price * amount
                    else:
                        # Reducing short
                        if amount >= abs(net_qty):
                            # Flipping to long
                            remaining = amount - abs(net_qty)
                            cost_basis = price * remaining
                        else:
                            # Partially reducing short
                            cost_basis = cost_basis * (abs(net_qty) - amount) / abs(net_qty) if net_qty != 0 else 0
                    net_qty += amount
                elif side == 'sell':
                    # Adding to short / reducing long
                    if net_qty <= 0:
                        # Adding to short: update cost basis
                        cost_basis += price * amount
                    else:
                        # Reducing long
                        if amount >= net_qty:
                            # Flipping to short
                            remaining = amount - net_qty
                            cost_basis = price * remaining
                        else:
                            # Partially reducing long
                            cost_basis = cost_basis * (net_qty - amount) / net_qty if net_qty != 0 else 0
                    net_qty -= amount
            
            result['bot_net_qty'] = round(net_qty, 6)
            result['bot_cost_basis'] = round(cost_basis, 2)
            
            # Average entry price
            if net_qty != 0:
                result['bot_avg_entry'] = round(cost_basis / abs(net_qty), 2)
                # Unrealized P&L
                result['unrealized_pnl'] = round((current_price - result['bot_avg_entry']) * net_qty, 4)
            
            result['total_pnl'] = round(result['realized_pnl'] + result['unrealized_pnl'], 4)
            
            return result
        except Exception as e:
            self.logger.warning(f"Failed to calculate bot P&L: {e}")
            return result
