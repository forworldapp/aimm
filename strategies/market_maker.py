"""
GRVT Market Maker Strategy V1.4
-------------------------------
Features:
- Bollinger Bands Mean Reversion (Buy Low/Sell High at Band Edges).
- Dynamic Spread (ATR-based volatility adaptation).
- Grid Spacing Optimization (Prevent simultaneous fills).
- RSI Safety Filter (Overbought/Oversold protection).
- Aggressive Entry Mode (Maximize fill rate on signals).

Author: Antigravity
Version: 1.4.1 (Hotfix)
Last Updated: 2025-12-18
Changelog:
- Fixed Persistence: Properly restoring Paper Exchange state on restart.
- Fixed Inventory Logic: Added missing inventory initialization and sync for exit logic.
- Fixed Import Error: Corrected filter module imports.
"""

import sys
import os
import time
import json
import logging
import asyncio
import statistics
import pandas as pd
from datetime import datetime

# Adjust path for local imports if needed
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.config import Config
from core.risk_manager import RiskManager
from core.paper_exchange import PaperGrvtExchange # Assuming this is needed based on the instruction's import list, though not used in the provided snippet.

# New Filters Import
from strategies.filters import RSIFilter, MAFilter, ADXFilter, ATRFilter, BollingerFilter, ComboFilter, ChopFilter


def round_tick_size(price, tick_size):
    return round(price / tick_size) * tick_size

class MarketMaker:
    """
    Enhanced Market Maker Strategy for GRVT.
    Supports Adaptive Regime Detection using various Technical Filters.
    """

    def __init__(self, exchange):
        self.exchange = exchange
        self.symbol = Config.get("exchange", "symbol", "BTC_USDT_Perp")
        
        self.logger = logging.getLogger("MarketMaker")
        
        # Exchange Info
        self.tick_size = 0.1 
        
        # Trend & Volatility State
        self.price_history = []
        self.history_max_len = 50
        
        # Candle Data for Advanced Filters (OHLC)
        self.candles = pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close'])
        self.current_candle = None
        self.last_candle_time = 0
        
        # Command Control
        self.command_file = os.path.join("data", "command.json")
        self.is_active = False 
        self.is_running = True
        
        # Risk State
        self.initial_equity = None
        self.inventory = 0.0 # Position tracking

        # Load Params & Initialize Filter
        self._load_params()
        
    def _load_params(self):
        """Load strategy parameters from config.yaml"""
        Config.load("config.yaml") # Force reload
        
        self.base_spread = float(Config.get("strategy", "spread_pct", 0.0002))
        self.order_size_usd = float(Config.get("strategy", "order_size_usd", 100.0))
        self.amount = 0.0 # Deprecated, auto-calc
        # self.amount = float(Config.get("strategy", "order_amount", 0.001))
        self.refresh_interval = int(Config.get("strategy", "refresh_interval", 3))
        self.skew_factor = float(Config.get("risk", "inventory_skew_factor", 0.05))
        self.max_position_usd = float(Config.get("risk", "max_position_usd", 500.0))  # Position limit
        self.max_loss_usd = float(Config.get("risk", "max_loss_usd", 50.0))  # Circuit breaker
        self.grid_layers = int(Config.get("strategy", "grid_layers", 3))
        self.entry_anchor_mode = Config.get("strategy", "entry_anchor_mode", False)
        
        # Strategy Selector
        self.trend_strategy = Config.get("strategy", "trend_strategy", "bollinger")
        self.latched_regime = None # Memory for signal latch
        
        self.filter_strategy = self._initialize_filter(self.trend_strategy)
        self.rsi_filter = RSIFilter() # Auxiliary filter
        
        self.logger.info(f"Loaded Params: Layers={self.grid_layers}, MaxPos=${self.max_position_usd}, MaxLoss=${self.max_loss_usd}")
        
        self.risk_manager = RiskManager()
        self.risk_manager.max_drawdown = float(Config.get("risk", "max_drawdown_pct", 0.10))

    def _initialize_filter(self, name):
        name = str(name).lower()
        if name == 'adx': return ADXFilter()
        if name == 'atr': return ATRFilter()
        if name == 'chop': return ChopFilter()
        if name == 'combo': return ComboFilter()
        if name == 'rsi':
             conf = Config.get("strategy", "rsi", {})
             return RSIFilter(conf.get('period', 14), conf.get('overbought', 70), conf.get('oversold', 30))
        if name == 'bollinger':
             conf = Config.get("strategy", "bollinger", {})
             return BollingerFilter(conf.get('period', 20), conf.get('std_dev', 2.0))
             conf = Config.get("strategy", "rsi", {})
             return RSIFilter(conf.get('period', 14), conf.get('overbought', 70), conf.get('oversold', 30))
        if name == 'ma_trend' or name == 'adaptive': return MAFilter() 
        return None # 'off'

    async def check_command(self):
        """Check for external commands from dashboard."""
        if os.path.exists(self.command_file):
            try:
                with open(self.command_file, "r") as f:
                    data = json.load(f)
                
                command = data.get("command")
                if command:
                    self.logger.info(f"Received command: {command}")
                    
                    if command == "start":
                        self.is_active = True
                        self.initial_equity = None # Reset drawdown baseline
                        self.logger.info("Bot STARTED.")
                    elif command == "stop":
                        self.is_active = False
                        await self.exchange.cancel_all_orders(self.symbol)
                        self.logger.info("Bot PAUSED.")
                    elif command == "stop_close":
                        self.is_active = False
                        await self.exchange.cancel_all_orders(self.symbol)
                        await self.exchange.close_position(self.symbol) # Market Close
                        self.logger.info("Bot STOPPED & CLOSED.")
                    elif command == "reload_config":
                        self._load_params()
                        self.logger.info("Configuration Reloaded.")
                    elif command == "shutdown":
                        self.is_active = False
                        self.is_running = False
                        self.logger.info("Shutdown sequence initiated.")

                    elif command == "restart":
                        self.logger.info("Received command: restart")
                        self.is_running = False # Break main loop
                        os.remove(self.command_file)
                        return 'restart'

                    # Clear command file
                    os.remove(self.command_file)
                    
            except Exception as e:
                self.logger.error(f"Error reading command: {e}")

    def _update_history(self, price):
        self.price_history.append(price)
        if len(self.price_history) > self.history_max_len:
            self.price_history.pop(0)

    def _update_candle(self, price, timestamp):
        """Update 1-minute OHLC candles."""
        dt = datetime.fromtimestamp(timestamp)
        current_minute = dt.replace(second=0, microsecond=0)
        
        if self.current_candle is None:
            self.current_candle = {
                'timestamp': current_minute,
                'open': price, 'high': price, 'low': price, 'close': price
            }
        elif self.current_candle['timestamp'] != current_minute:
            new_row = pd.DataFrame([self.current_candle])
            self.candles = pd.concat([self.candles, new_row], ignore_index=True)
            if len(self.candles) > 100:
                self.candles = self.candles.iloc[-100:]
            self.current_candle = {
                'timestamp': current_minute,
                'open': price, 'high': price, 'low': price, 'close': price
            }
        else:
            self.current_candle['high'] = max(self.current_candle['high'], price)
            self.current_candle['low'] = min(self.current_candle['low'], price)
            self.current_candle['close'] = price

    def _detect_market_regime(self):
        """Use the selected Filter Strategy to detect regime and append RSI status."""
        if not self.filter_strategy:
            if hasattr(self.exchange, "set_market_regime"):
                self.exchange.set_market_regime('OFF')
            return 'ranging'
            
        if self.current_candle:
            df = pd.concat([self.candles, pd.DataFrame([self.current_candle])], ignore_index=True)
        else:
            df = self.candles
            
        regime = self.filter_strategy.analyze(df)
        
        if self.rsi_filter:
            rsi_val = self.rsi_filter.analyze(df)
            rsi_num = getattr(self.rsi_filter, 'last_rsi', 50.0)
            
            # [Safety Filter] Block signal if RSI does not confirm
            if 'buy_signal' in regime and rsi_val != 'oversold':
                regime = 'neutral' # Blocked by RSI
            elif 'sell_signal' in regime and rsi_val != 'overbought':
                regime = 'neutral' # Blocked by RSI

            if rsi_val != 'neutral' and rsi_val != 'waiting':
                 regime += f" | {rsi_val.upper()} ({rsi_num:.1f})"
            elif rsi_val == 'neutral':
                 regime += f" | RSI: {rsi_num:.1f}"
        
        if self.filter_strategy and 'BB' in self.filter_strategy.name:
            pct_b = getattr(self.filter_strategy, 'last_pct_b', 0.5)
            regime += f" | BB%: {pct_b:.2f}"

        # Shorten display: e.g. "BUY (BB)" instead of "BUY_SIGNAL (BB(20, 2.0))"
        short_name = self.filter_strategy.name.split('(')[0] # "BB"
        status_str = f"{regime.upper()} ({short_name})"
        
        if hasattr(self.exchange, "set_market_regime"):
            self.exchange.set_market_regime(status_str)
            
        return regime

    def _get_trend_skew(self):
        """Calculate skew based on selected filter strategy."""
        
        # 0. Update Candle (Called in cycle, but ensure data exists)
        if not self.filter_strategy:
             if hasattr(self.exchange, "set_market_regime"):
                self.exchange.set_market_regime('OFF (Pure Grid)')
             return 0.0

        # 1. Detect Regime
        regime = self._detect_market_regime()
        
        if regime == 'waiting':
             return 0.0 
             
        if regime == 'ranging':
            return 0.0 
        
        if len(self.price_history) < 20: return 0.0
        short_ma = statistics.mean(self.price_history[-10:])
        long_ma = statistics.mean(self.price_history[-20:])
        
        if long_ma == 0: return 0.0
        diff_pct = (short_ma - long_ma) / long_ma
        
        return max(-0.001, min(0.001, diff_pct * 10))

    def _get_volatility_multiplier(self):
        if len(self.price_history) < 20:
            return 1.0
        stdev = statistics.stdev(self.price_history[-20:])
        mean_price = statistics.mean(self.price_history[-20:])
        vol_pct = stdev / mean_price
        base_vol = 0.0001 
        multiplier = max(1.0, vol_pct / base_vol)
        return min(multiplier, 5.0)

    def _calculate_dynamic_spread(self):
        """Calculate spread based on ATR (Volatility) with USD limits."""
        conf = Config.get("strategy", "dynamic_spread", {})
        if not conf.get('enabled', False):
            return self.base_spread
            
        if len(self.candles) < 20:
            return self.base_spread
            
        # Current Price (Approx from close)
        current_price = self.candles['close'].iloc[-1]
        
        # Calculate ATR(14)
        high = self.candles['high']
        low = self.candles['low']
        close = self.candles['close']
        
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(14).mean().iloc[-1]
        
        # ATR Logic
        ref_atr = conf.get('reference_atr', 100.0)
        multiplier = atr / ref_atr
        multiplier = max(0.5, min(multiplier, 3.0)) # 0.5x ~ 3.0x
        
        raw_spread_pct = self.base_spread * multiplier
        raw_spread_usd = current_price * raw_spread_pct
        
        # Enforce User Limits ($30 ~ $200) -> Adjusted to $100 min per observation
        min_usd = 100.0
        max_usd = 200.0
        
        final_usd = max(min_usd, min(max_usd, raw_spread_usd))
        final_pct = final_usd / current_price if current_price > 0 else self.base_spread
        
        return final_pct
    
    async def check_drawdown(self):
        """Check Max Drawdown safety stop."""
        status = await self.exchange.get_account_summary()
        current_equity = status.get('total_equity', 0.0)
        
        if self.initial_equity is None:
            self.initial_equity = current_equity
            self.logger.info(f"Initial Equity Set: {self.initial_equity}")
            return True
            
        # Drawdown check
        dd_pct = (self.initial_equity - current_equity) / self.initial_equity
        if dd_pct > self.risk_manager.max_drawdown:
            self.logger.critical(f"MAX DRAWDOWN REACHED! {dd_pct*100:.2f}% >= {self.risk_manager.max_drawdown*100:.2f}%")
            self.logger.critical("STOPPING BOT & CLOSING POSITIONS.")
            
            # Close all
            await self.exchange.cancel_all_orders(self.symbol)
            await self.exchange.close_position(self.symbol)
            
            self.is_active = False # Stop Bot
            return False
            
        return True

    async def cycle(self):
        # 1. Get Data
        orderbook = await self.exchange.get_orderbook(self.symbol)
        position = await self.exchange.get_position(self.symbol)
        
        if not orderbook or 'bids' not in orderbook:
            return

        try:
            bids = orderbook['bids']
            asks = orderbook['asks']
            best_bid = float(bids[0]['price']) if isinstance(bids[0], dict) else float(bids[0][0])
            best_ask = float(asks[0]['price']) if isinstance(asks[0], dict) else float(asks[0][0])
            mid_price = (best_bid + best_ask) / 2
        except:
            return
        
        self._update_history(mid_price)
        self._update_candle(mid_price, time.time())
        current_pos_qty = position.get('amount', 0.0)
        self.inventory = current_pos_qty
        
        # --- Circuit Breaker Check ---
        unrealized_pnl = position.get('unrealizedPnL', 0.0)
        if unrealized_pnl < -self.max_loss_usd:
            self.logger.critical(f"🚨 CIRCUIT BREAKER: Loss ${abs(unrealized_pnl):.2f} exceeds max ${self.max_loss_usd:.2f}")
            self.logger.critical("Cancelling all orders and stopping bot!")
            await self.exchange.cancel_all_orders(self.symbol)
            self.is_active = False
            return
        
        # Log Status
        rsi_status = self.rsi_filter.analyze(self.candles)
        # 1. Detect Regime & Apply Latch
        current_regime = self._detect_market_regime()
        rsi_status = self.rsi_filter.analyze(self.candles) # Update last_rsi
        last_rsi = getattr(self.rsi_filter, 'last_rsi', 50.0)
        
        # Reset Latch if flat
        if current_pos_qty == 0:
            self.latched_regime = None
        
        # Reset Latch based on RSI thresholds (Hysteresis)
        # User Rule: Keep Buy Latch if RSI <= 40 / Keep Sell Latch if RSI >= 60
        if self.latched_regime == 'buy_signal' and last_rsi > 40:
             self.latched_regime = None # Release to Neutral
        elif self.latched_regime == 'sell_signal' and last_rsi < 60:
             self.latched_regime = None # Release to Neutral

        # Update Latch if new signal appears (Overrides Reset)
        if 'buy_signal' in current_regime or 'sell_signal' in current_regime:
            self.latched_regime = current_regime
            
        # Use Latched Regime if Current is Neutral but we have a Latch
        effective_regime = current_regime
        if current_regime == 'neutral' and self.latched_regime:
             effective_regime = self.latched_regime
             # Add visual indicator
             effective_regime += " (Latched)"
             
        # Log Status
        self.logger.info(f"Pos: {current_pos_qty:.4f} | Mid: {mid_price:.2f} | Regime: {effective_regime} | RSI: {last_rsi:.1f} | Equity: {position.get('unrealizedPnL', 0):.2f}")

        # Sync to Paper Exchange
        if hasattr(self.exchange, 'set_market_regime'):
            self.exchange.set_market_regime(effective_regime)

        # 2. Calculate Parameters
        # Fix Division by Zero: Use calculated qty based on price
        estimated_qty = (self.order_size_usd / mid_price) if mid_price > 0 else self.amount
        if estimated_qty <= 0: estimated_qty = 0.001 # Fallback
        
        max_skew_qty = estimated_qty * 20 
        inventory_ratio = max(-1.0, min(1.0, current_pos_qty / max_skew_qty))
        inventory_skew = inventory_ratio * self.skew_factor * -1 
        trend_skew = self._get_trend_skew()
        total_skew = inventory_skew + trend_skew
        
        # Determine Spread based on EFFECTIVE regime
        # If Signal (or Latched Signal), use aggressive tight spread? 
        # Actually logic says: if signal, tight spread.
        if 'buy_signal' in effective_regime or 'sell_signal' in effective_regime:
            # Aggressive Entry: Tight spread (10% of base) for execution
            final_spread = self.base_spread * 0.1 
        else:
            final_spread = self._calculate_dynamic_spread()
        
        skewed_mid = mid_price * (1 + total_skew)
        target_bid = round_tick_size(skewed_mid * (1 - final_spread / 2), self.tick_size)
        target_ask = round_tick_size(skewed_mid * (1 + final_spread / 2), self.tick_size)

        # Dropdown Check (Pass)

        # --- 3. Entry Guard / Profit Protection (Anchor) ---
        entry_price = position.get('entryPrice', 0.0)
        
        if self.entry_anchor_mode and current_pos_qty != 0 and entry_price > 0:
            # Regime-Based Stop Loss Logic
            loss_tolerance = 0.0 # Default: Zero Tolerance (Strictly Profit Only)
            
            # If Signal detected (Trend) OR Latched, loosen stop loss to allow escape
            if 'buy_signal' in effective_regime or 'sell_signal' in effective_regime:
                 loss_tolerance = 0.005 # Allow 0.5% loss to cut bad positions in trend
            
            # Neutral: Strict "Hold" (loss_tolerance ~ 0)
            
            if current_pos_qty > 0: # Long
                 # DCA: Buy if price < Entry
                 target_bid = min(target_bid, entry_price * 0.9995)
                 # Anchor: Limit Sell Price
                 limit_price = entry_price * (1.0 - loss_tolerance) + (entry_price * 0.0005) # Adjust slightly
                 target_ask = max(target_ask, entry_price * (1 - loss_tolerance)) 
                 
            elif current_pos_qty < 0: # Short
                 # DCA: Sell if price > Entry
                 target_ask = max(target_ask, entry_price * 1.0005)
                 # Anchor: Limit Buy Price
                 target_bid = min(target_bid, entry_price * (1 + loss_tolerance))

        # --- 4. Permission Flags ---
        allow_buy = rsi_status != 'overbought'
        allow_sell = rsi_status != 'oversold'
        
        if self.filter_strategy and 'BB' in self.filter_strategy.name:
            # Signal Logic
            if 'buy_signal' not in effective_regime:
                allow_buy = False # Default block unless neutral logic overrides
            if 'sell_signal' not in effective_regime:
                allow_sell = False

            # Neutral Logic: Allow Grid Trading (Accumulation)
            # Re-enabled per user request (Step 3329)
            if 'neutral' in effective_regime:
                allow_buy = True
                allow_sell = True
        
        # --- 4.1 Max Position Limit ---
        # Block further accumulation when position exceeds max_position_usd
        position_usd = abs(current_pos_qty) * mid_price
        if position_usd >= self.max_position_usd:
            if current_pos_qty > 0:
                allow_buy = False  # Long position at limit → block buying
                self.logger.debug(f"Max position reached: ${position_usd:.0f} >= ${self.max_position_usd:.0f}, blocking BUY")
            elif current_pos_qty < 0:
                allow_sell = False  # Short position at limit → block selling
                self.logger.debug(f"Max position reached: ${position_usd:.0f} >= ${self.max_position_usd:.0f}, blocking SELL")
                
        # --- 5. Generate Grid Orders ---
        buy_orders = []
        sell_orders = []
        
        # Grid Spacing: 0.12% (Adjusted per user request: ~$100 @ 87k)
        layer_spacing = max(final_spread, 0.0012) 

        for i in range(self.grid_layers):
            # Linearly spaced grid
            bid_p = round_tick_size(target_bid * (1 - (layer_spacing * i)), self.tick_size)
            ask_p = round_tick_size(target_ask * (1 + (layer_spacing * i)), self.tick_size)
            
            # Amount distribution (USD Based)
            if self.order_size_usd > 0:
                raw_qty = self.order_size_usd / mid_price
                # Round to 3 decimals for GRVT (0.001 lot size)
                qty = round(raw_qty, 3)
                # Enforce min qty (0.001 BTC for GRVT)
                qty = max(qty, 0.001)
            else:
                qty = self.amount # Old fixed quantity fallback
            
            if allow_buy:
                buy_orders.append((bid_p, qty))
            if allow_sell:
                sell_orders.append((ask_p, qty))
            
        # --- 6. Smart Order Management (v1.9.1) ---
        # Only update orders if prices changed significantly (>0.1% tolerance)
        PRICE_TOLERANCE = 0.001  # 0.1%
        
        # Get existing orders (GRVT format: legs[0] contains order details)
        existing_orders = await self.exchange.get_open_orders(self.symbol)
        existing_buys = {o.get('order_id', o.get('id')): float(o.get('legs', [{}])[0].get('limit_price', 0)) 
                         for o in existing_orders if o.get('legs') and o['legs'][0].get('is_buying_asset')}
        existing_sells = {o.get('order_id', o.get('id')): float(o.get('legs', [{}])[0].get('limit_price', 0)) 
                          for o in existing_orders if o.get('legs') and not o['legs'][0].get('is_buying_asset')}
        
        new_buy_prices = set(p for p, q in buy_orders)
        new_sell_prices = set(p for p, q in sell_orders)
        
        # Check if orders need update (use 0.5% tolerance to avoid constant churn)
        PRICE_TOLERANCE = 0.005  # 0.5% tolerance
        
        def prices_match(existing_prices, new_prices, tolerance):
            if len(existing_prices) != len(new_prices):
                return False
            if not existing_prices or not new_prices:
                return len(existing_prices) == len(new_prices)
            existing_set = set(existing_prices.values())
            for new_p in new_prices:
                matched = any(abs(new_p - old_p) / old_p < tolerance for old_p in existing_set if old_p > 0)
                if not matched:
                    return False
            return True
        
        buys_need_update = not prices_match(existing_buys, new_buy_prices, PRICE_TOLERANCE)
        sells_need_update = not prices_match(existing_sells, new_sell_prices, PRICE_TOLERANCE)
        
        # Skip update if no new orders to place (avoid cancel+empty replace loop)
        if not buy_orders and not sell_orders:
            pass  # Keep existing orders
        elif buys_need_update or sells_need_update:
            # Cancel and replace only if needed
            await self.exchange.cancel_all_orders(self.symbol)
            
            for p, q in buy_orders:
                await self.exchange.place_limit_order(self.symbol, 'buy', p, q)
            for p, q in sell_orders:
                await self.exchange.place_limit_order(self.symbol, 'sell', p, q)
        # else: Keep existing orders (no action needed)
        
        # Save status for dashboard (Live mode)
        if hasattr(self.exchange, 'save_live_status'):
            open_orders = await self.exchange.get_open_orders(self.symbol)
            status = await self.exchange.get_account_summary()
            self.exchange.save_live_status(
                symbol=self.symbol,
                mid_price=mid_price,
                regime=effective_regime,
                position=position,
                open_orders=open_orders,
                equity=status.get('total_equity', 0.0)
            )
            # Also fetch and save trade history for dashboard
            if hasattr(self.exchange, 'fetch_and_save_trades'):
                self.exchange.fetch_and_save_trades(self.symbol)

    async def run(self):
        self.logger.info("Strategy Started")
        self.is_running = True
        self.is_active = True # Force Auto-Start
        while self.is_running:
            try:
                cmd_res = await self.check_command()
                if cmd_res == 'restart':
                    return 'restart' # Signal main to restart

                if not self.is_active:
                    await asyncio.sleep(2)
                    continue
                
                if not await self.check_drawdown():
                    continue

                await self.cycle()
            except Exception as e:
                self.logger.error(f"Error in strategy cycle: {e}")
            
            await asyncio.sleep(self.refresh_interval)
        return 'stop'
