import asyncio
import logging
import statistics
import os
import json
from typing import List, Dict
from core.exchange_interface import ExchangeInterface
from core.config import Config
from core.risk_manager import RiskManager
from strategies.base_strategy import BaseStrategy
from utils.calculations import round_tick_size

class MarketMaker(BaseStrategy):
    """
    Enhanced Market Maker Strategy for GRVT.
    
    Key Features:
    1. **Inventory Skew**: Adjusts bid/ask prices based on current position to neutralize risk.
       - If Long: Skew asks down to sell faster.
       - If Short: Skew bids up to buy back faster.
    2. **Trend & Volatility Adjustment**:
       - Widens spread during high volatility.
       - Shifts prices slightly in direction of trend.
    3. **Post-Only Enforcement**:
       - Strictly clamps order prices to match Best Bid/Ask to avoid Taker fees.
       - Ensures the bot always acts as a liquidity provider.
    4. **Profit Protection**:
       - Prevents closing positions at a loss during normal/low volatility conditions.
       - Uses 'Hold for Profit' logic to place exit orders above entry price.
       
    Architecture:
    - Runs in a loop with `refresh_interval` (default 3s).
    - Fetches live orderbook and position data.
    - Calculates target prices and replaces all open orders.
    - Listens for external commands (Start/Stop) via `command.json`.
    """

    def __init__(self, exchange: ExchangeInterface):
        # Load params from Config
        symbol = Config.get("exchange", "symbol", "BTC_USDT_Perp")
        super().__init__(exchange, symbol)
        
        self.logger = logging.getLogger("MarketMaker")
        self.risk_manager = RiskManager()
        
        # Strategy Params
        self.base_spread = float(Config.get("strategy", "spread_pct", 0.0002))
        self.amount = float(Config.get("strategy", "order_amount", 0.001))
        self.refresh_interval = int(Config.get("strategy", "refresh_interval", 5))
        self.skew_factor = float(Config.get("risk", "inventory_skew_factor", 0.05))
        
        # Exchange Info
        self.tick_size = 0.1 
        
        # Trend & Volatility State
        self.price_history = []
        self.history_max_len = 50
        
        # Command Control
        self.command_file = os.path.join("data", "command.json")
        self.is_active = True # Default running

    async def run(self):
        self.is_running = True
        self.logger.info(f"Starting Enhanced Market Maker on {self.symbol}")

        while self.is_running:
            try:
                # 0. Check Commands
                await self.check_command()
                
                if not self.is_active:
                    self.logger.info("Bot is PAUSED. Waiting for start command...", extra={'throttle': True})
                    await asyncio.sleep(2)
                    continue

                await self.cycle()
            except Exception as e:
                self.logger.error(f"Error in strategy cycle: {e}")
            
            await asyncio.sleep(self.refresh_interval)

    async def check_command(self):
        """Check for external commands from dashboard."""
        if os.path.exists(self.command_file):
            try:
                with open(self.command_file, "r") as f:
                    cmd_data = json.load(f)
                
                command = cmd_data.get("command")
                if command == "stop_close" and self.is_active:
                    self.logger.warning("RECEIVED STOP & CLOSE COMMAND.")
                    self.is_active = False
                    
                    # Execute Close Logic
                    if hasattr(self.exchange, "cancel_all_orders"):
                        await self.exchange.cancel_all_orders(self.symbol)
                    if hasattr(self.exchange, "close_position"):
                        await self.exchange.close_position(self.symbol)
                    
                    # Remove command file
                    os.remove(self.command_file)
                    
                elif command == "start" and not self.is_active:
                    self.logger.info("RECEIVED START COMMAND.")
                    self.is_active = True
                    os.remove(self.command_file)
                    
            except Exception as e:
                self.logger.error(f"Error reading command: {e}")

    def _update_history(self, price):
        self.price_history.append(price)
        if len(self.price_history) > self.history_max_len:
            self.price_history.pop(0)

    def _get_trend_skew(self):
        if len(self.price_history) < 20:
            return 0.0
        short_ma = statistics.mean(self.price_history[-10:])
        long_ma = statistics.mean(self.price_history[-20:])
        if short_ma > long_ma: return 0.0005 
        elif short_ma < long_ma: return -0.0005
        return 0.0

    def _get_volatility_multiplier(self):
        if len(self.price_history) < 20:
            return 1.0
        stdev = statistics.stdev(self.price_history[-20:])
        mean_price = statistics.mean(self.price_history[-20:])
        vol_pct = stdev / mean_price
        base_vol = 0.0001 
        multiplier = max(1.0, vol_pct / base_vol)
        return min(multiplier, 5.0)

    async def cycle(self):
        # 1. Get Data
        orderbook = await self.exchange.get_orderbook(self.symbol)
        position = await self.exchange.get_position(self.symbol)
        
        if not orderbook or 'bids' not in orderbook:
            return

        # Handle dict structure
        try:
            bids = orderbook['bids']
            asks = orderbook['asks']
            best_bid = float(bids[0]['price']) if isinstance(bids[0], dict) else float(bids[0][0])
            best_ask = float(asks[0]['price']) if isinstance(asks[0], dict) else float(asks[0][0])
            mid_price = (best_bid + best_ask) / 2
        except:
            return
        
        self._update_history(mid_price)
        current_pos_qty = position.get('amount', 0.0)
        
        # 2. Calculate Parameters
        max_skew_qty = self.amount * 20 
        inventory_ratio = max(-1.0, min(1.0, current_pos_qty / max_skew_qty))
        inventory_skew = inventory_ratio * self.skew_factor * -1 
        trend_skew = self._get_trend_skew()
        total_skew = inventory_skew + trend_skew
        
        vol_mult = self._get_volatility_multiplier()
        final_spread = self.base_spread * vol_mult
        
        skewed_mid = mid_price * (1 + total_skew)
        target_bid = round_tick_size(skewed_mid * (1 - final_spread / 2), self.tick_size)
        target_ask = round_tick_size(skewed_mid * (1 + final_spread / 2), self.tick_size)

        # --- 4. Take Profit / Break-Even Logic ---
        # Don't sell below entry (Long) or buy above entry (Short) unless stop-loss logic triggers (handled by risk manager or huge skew)
        # Here we add a gentle force to ensure we quote profitable exits if close to market
        
        min_profit = mid_price * 0.0005 # 0.05% Min Profit
        entry_price = position.get('entryPrice', 0.0)
        
        if current_pos_qty > 0: # Long Position -> Selling (Ask)
            min_ask = entry_price + min_profit
            # If our target ask follows the market down below our entry, we lock in a loss.
            # Instead, keep the ask at Break-Even+Profit, effectively waiting for rebound.
            # Only if price drops massively (Stop Loss) should we follow down (Risk Manager handles that, or we implement SL here).
            # For now, let's hold for profit.
            target_ask = max(target_ask, min_ask)
            
        elif current_pos_qty < 0: # Short Position -> Buying (Bid)
            max_bid = entry_price - min_profit
            # If target bid follows market up above entry, we lock in loss.
            target_bid = min(target_bid, max_bid)

        # --- 5. Post-Only Enforcement (Critical Fix) ---
        # Prevent the Strategy from placing Taker orders (crossing the spread).
        # Even if Skew says "Buy eagerly", we must cap at Best Bid to remain a Maker.
        
        # Ensure we don't cross the market
        # If orderbook is empty, skip clamping (shouldn't happen here)
        if best_bid > 0 and best_ask > 0:
            original_bid = target_bid
            original_ask = target_ask
            
            # Post-Only Clamp
            target_bid = min(target_bid, best_bid)
            target_ask = max(target_ask, best_ask)
            
            # If clamping changed the price significantly, log it
            if target_bid != original_bid or target_ask != original_ask:
                self.logger.debug(f"Post-Only Clamped: Bid {original_bid}->{target_bid}, Ask {original_ask}->{target_ask}")

        # Ensure Spread is maintained (sanity check)
        if target_bid >= target_ask:
            # This can happen if spread is tight and tick rounding messes up
            # Force minimal spread
            target_bid = round_tick_size(mid_price - self.tick_size, self.tick_size)
            target_ask = round_tick_size(mid_price + self.tick_size, self.tick_size)

        # Log status
        self.logger.info(f"Pos: {current_pos_qty:.4f} | Mid: {mid_price:.2f} | Bid: {target_bid} | Ask: {target_ask} | Skew: {total_skew*100:.2f}%")

        # 3. Manage Orders (Cancel All & Replace)
        open_orders = await self.exchange.get_open_orders(self.symbol)
        await self.cancel_all(open_orders)
        
        await self.place_limit_order('buy', target_bid)
        await self.place_limit_order('sell', target_ask)

    async def cancel_all(self, orders: List[Dict]):
        tasks = [self.exchange.cancel_order(self.symbol, o['id']) for o in orders]
        if tasks: await asyncio.gather(*tasks)

    async def place_limit_order(self, side: str, price: float):
        try:
            await self.exchange.place_limit_order(self.symbol, side, price, self.amount)
        except Exception as e:
            self.logger.error(f"Failed to place {side}: {e}")
