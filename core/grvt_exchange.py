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
        
        if GrvtCcxt is None:
            self.logger.error("grvt-pysdk not installed. Please install it via pip.")
            return

        # Initialize the CCXT-compatible wrapper
        target_env = GrvtEnv.TESTNET
        if self.env == 'prod' or self.env == 'mainnet':
            target_env = GrvtEnv.PROD
        
        # SDK requires ALL credentials in parameters dict
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
        """Fetch recent trades from GRVT and save to trade history CSV with FIFO grid profit."""
        import csv
        import os
        import time
        from collections import deque
        
        if not self.exchange:
            return
            
        try:
            # Fetch recent trades
            response = self.exchange.fetch_my_trades(symbol, limit=50)
            if not response:
                return
            
            # Response is dict with 'result' key containing list of trades
            trades_list = response.get('result', []) if isinstance(response, dict) else response
            if not trades_list:
                return
                
            trade_file = os.path.join("data", f"trade_history_{symbol.replace('/', '_')}.csv")
            
            # Read existing trades for FIFO matching
            existing_ids = set()
            buy_queue = deque()  # FIFO queue for buys: [(price, size, trade_id), ...]
            sell_queue = deque()  # FIFO queue for sells: [(price, size, trade_id), ...]
            
            if os.path.exists(trade_file):
                with open(trade_file, 'r') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        note = row.get('note', '')
                        if note.startswith('Fill '):
                            trade_id = note.replace('Fill ', '')
                            existing_ids.add(trade_id)
                            
                            # Build FIFO queues from existing trades
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
            
            # Process new trades
            new_rows = []
            for trade in trades_list:
                trade_id = trade.get('trade_id', trade.get('order_id', ''))
                if not trade_id or trade_id in existing_ids:
                    continue
                    
                # GRVT trade format
                is_buyer = trade.get('is_buyer')
                side = 'buy' if is_buyer else 'sell'
                price = float(trade.get('price', 0))
                size = float(trade.get('size', 0))
                fill_time = int(trade.get('event_time', time.time() * 1e9)) / 1e9
                realized_pnl = float(trade.get('realized_pnl', 0))
                fee = float(trade.get('fee', 0))
                
                # Direction based on realized_pnl
                direction = 'increase' if realized_pnl == 0 else 'reduce'
                
                # Calculate FIFO grid profit
                grid_profit = 0.0
                if direction == 'increase':
                    # Add to queue
                    if side == 'buy':
                        buy_queue.append((price, size, trade_id))
                    else:
                        sell_queue.append((price, size, trade_id))
                else:
                    # Match with opposite queue (FIFO)
                    if side == 'buy':
                        # Buying to close short - match with oldest sell
                        if sell_queue:
                            entry_price, _, _ = sell_queue.popleft()
                            grid_profit = (entry_price - price) * size  # Short profit
                    else:
                        # Selling to close long - match with oldest buy
                        if buy_queue:
                            entry_price, _, _ = buy_queue.popleft()
                            grid_profit = (price - entry_price) * size  # Long profit
                
                new_rows.append([
                    fill_time,
                    'BTC_USDT_Perp',
                    side,
                    direction,
                    price,
                    size,
                    round(price * size, 2),
                    abs(fee),
                    realized_pnl,
                    round(grid_profit, 4),  # FIFO calculated grid profit
                    f"Fill {trade_id[:12]}"
                ])
                existing_ids.add(trade_id)
            
            # Write new trades
            if new_rows:
                file_exists = os.path.exists(trade_file) and os.path.getsize(trade_file) > 0
                with open(trade_file, 'a', newline='') as f:
                    writer = csv.writer(f)
                    if not file_exists:
                        writer.writerow(['timestamp', 'symbol', 'side', 'direction', 'price', 'amount', 'cost', 'rebate', 'realized_pnl', 'grid_profit', 'note'])
                    writer.writerows(new_rows)
                self.logger.info(f"Saved {len(new_rows)} new trades to history")
                
        except Exception as e:
            self.logger.warning(f"Failed to fetch trades: {e}")
