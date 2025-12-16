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
    """

    def __init__(self, exchange: ExchangeInterface):
        # Load params from Config
        symbol = Config.get("exchange", "symbol", "BTC_USDT_Perp")
        super().__init__(exchange, symbol)
        
        self.logger = logging.getLogger("MarketMaker")
        self.risk_manager = RiskManager()
        
        
        # Strategy Params (Initialized in _load_params)
        self._load_params()
        
        # Exchange Info
        self.tick_size = 0.1 
        
        # Trend & Volatility State
        self.price_history = []
        self.history_max_len = 50
        
        # Command Control
        self.command_file = os.path.join("data", "command.json")
        self.is_active = False # Default PAUSED (Must act safety first)
        
        # Risk State
        self.initial_equity = None

    def _load_params(self):
        """Load strategy parameters from config.yaml"""
        Config.load("config.yaml")
        self.base_spread = float(Config.get("strategy", "spread_pct", 0.0002))
        self.amount = float(Config.get("strategy", "order_amount", 0.001))
        self.refresh_interval = int(Config.get("strategy", "refresh_interval", 5))
        self.skew_factor = float(Config.get("risk", "inventory_skew_factor", 0.05))
        self.grid_layers = int(Config.get("strategy", "grid_layers", 3))
        self.entry_anchor_mode = Config.get("strategy", "entry_anchor_mode", False)
        
        # Load Strategy Mode (Default to 'adaptive' if missing, or handle old bool)
        self.trend_strategy = Config.get("strategy", "trend_strategy", 'adaptive')
        if str(self.trend_strategy).lower() == 'true': self.trend_strategy = 'ma_trend'
        if str(self.trend_strategy).lower() == 'false': self.trend_strategy = 'off'
        
        self.logger.info(f"Loaded Params: Layers={self.grid_layers}, Anchor={self.entry_anchor_mode}, Strategy={self.trend_strategy}")
        
        # Re-initialize RiskManager to update its config values
        self.risk_manager = RiskManager()
        # Ensure we sync local max_drawdown with RiskManager or Config
        self.risk_manager.max_drawdown = float(Config.get("risk", "max_drawdown_pct", 0.10))

    async def run(self):
        """
        메인 실행 루프
        """
        self.is_running = True
        self.logger.info(f"Starting Enhanced Market Maker on {self.symbol}")

        while self.is_running:
            try:
                # 0. 커맨드 확인
                await self.check_command()
                
                # 봇이 PAUSED 상태이면 대기
                if not self.is_active:
                    self.logger.info("Bot is PAUSED. Waiting for start command...", extra={'throttle': True})
                    await asyncio.sleep(2)
                    continue

                # 매매 사이클 실행
                if not await self.check_drawdown():
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
                
                if command == "reload_config":
                    self.logger.info("RECEIVED RELOAD CONFIG COMMAND.")
                    self._load_params()
                    os.remove(self.command_file)

                elif command == "stop_close" and self.is_active:
                    self.logger.warning("RECEIVED STOP & CLOSE COMMAND.")
                    self.is_active = False
                    
                    # Execute Close Logic
                    if hasattr(self.exchange, "cancel_all_orders"):
                        await self.exchange.cancel_all_orders(self.symbol)
                        await asyncio.sleep(0.5) # Wait for processing
                        
                    if hasattr(self.exchange, "close_position"):
                        await self.exchange.close_position(self.symbol)
                        
                    # Double check to ensure no residual orders
                    if hasattr(self.exchange, "cancel_all_orders"):
                        await self.exchange.cancel_all_orders(self.symbol)
                    
                    # Remove command file
                    os.remove(self.command_file)
                    
                elif command == "start" and not self.is_active:
                    self.logger.info("RECEIVED START COMMAND.")
                    self.is_active = True
                    self.initial_equity = None # Reset Drawdown Baseline
                    os.remove(self.command_file)
                    
                elif command == "shutdown":
                    self.logger.critical("RECEIVED SHUTDOWN COMMAND. TERMINATING PROCESS...")
                    self.is_active = False
                    self.is_running = False # This breaks the while loop
                    # Clean up
                    if os.path.exists(self.command_file):
                        os.remove(self.command_file)
                    
            except Exception as e:
                self.logger.error(f"Error reading command: {e}")

    def _update_history(self, price):
        self.price_history.append(price)
        if len(self.price_history) > self.history_max_len:
            self.price_history.pop(0)

    def _detect_market_regime(self, short_ma, long_ma):
        """
        Determine if market is Ranging or Trending.
        Logic: If ShortMA and LongMA are very close (< 0.03% diff), it's Ranging.
        """
        if long_ma == 0: return 'ranging'
        
        divergence = abs(short_ma - long_ma) / long_ma
        threshold = 0.0003 # 0.03%
        
        regime = 'trending'
        if divergence < threshold:
            regime = 'ranging'
            
        # Update Exchange Status for Dashboard Visibility
        if hasattr(self.exchange, "set_market_regime"):
            self.exchange.set_market_regime(regime)
            
        return regime

    def _get_trend_skew(self):
        """Calculate skew based on selected strategy."""
        
        # 1. Check Strategy OFF
        if hasattr(self, 'trend_strategy') and self.trend_strategy == 'off':
            if hasattr(self.exchange, "set_market_regime"):
                self.exchange.set_market_regime('off')
            return 0.0

        # 2. Check Insufficient Data (Reduced to 30)
        min_history = 30
        if len(self.price_history) < min_history:
            if hasattr(self.exchange, "set_market_regime"):
                self.exchange.set_market_regime(f'waiting ({len(self.price_history)}/{min_history})')
            return 0.0
            
        short_ma = statistics.mean(self.price_history[-10:])
        long_ma = statistics.mean(self.price_history[-min_history:]) 
        
        # 3. Adaptive Logic
        if hasattr(self, 'trend_strategy') and self.trend_strategy == 'adaptive':
            regime = self._detect_market_regime(short_ma, long_ma)
            if regime == 'ranging':
                return 0.0 # Suppress Skew in ranging market
        
        # 4. Default / MA Trend Logic
        if hasattr(self.exchange, "set_market_regime"):
             # If manual 'ma_trend', we are always trending effectively, or just show 'manual'
             if self.trend_strategy == 'ma_trend':
                 self.exchange.set_market_regime('manual_trend')

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
        min_profit = mid_price * 0.0005 # 0.05% Min Profit
        entry_price = position.get('entryPrice', 0.0)
        
        if current_pos_qty > 0: # Long Position -> Selling (Ask)
            min_ask = entry_price + min_profit
            target_ask = max(target_ask, min_ask)
            
        elif current_pos_qty < 0: # Short Position -> Buying (Bid)
            max_bid = entry_price - min_profit
            target_bid = min(target_bid, max_bid)

        # --- 5. Post-Only Enforcement ---
        if best_bid > 0 and best_ask > 0:
            original_bid = target_bid
            original_ask = target_ask
            
            # Post-Only Clamp
            target_bid = min(target_bid, best_bid)
            target_ask = max(target_ask, best_ask)
            
            if target_bid != original_bid or target_ask != original_ask:
                self.logger.debug(f"Post-Only Clamped: Bid {original_bid}->{target_bid}, Ask {original_ask}->{target_ask}")

        # Ensure Spread is maintained
        if target_bid >= target_ask:
            target_bid = round_tick_size(mid_price - self.tick_size, self.tick_size)
            target_ask = round_tick_size(mid_price + self.tick_size, self.tick_size)

        # Log status
        self.logger.info(f"Pos: {current_pos_qty:.4f} | Mid: {mid_price:.2f} | SkewedMid: {skewed_mid:.2f} | Bid: {target_bid} | Ask: {target_ask}")

        # --- 1-3. Grid Strategy with Smart Update (1-2) ---
        layers = self.grid_layers
        
        # Asymmetric Layers Logic: Reduce layers if inventory is heavy
        buy_layers = layers
        sell_layers = layers
        
        if inventory_ratio > 0.4: # Holding too many Longs -> Reduce Buy Orders (Don't add more risk)
            buy_layers = max(1, int(layers * 0.5))
        elif inventory_ratio < -0.4: # Holding too many Shorts -> Reduce Sell Orders
            sell_layers = max(1, int(layers * 0.5))

        layer_spread_inc = 0.0005 
        
        desired_orders = []
        
        # 1. Generate BUY Orders
        for i in range(buy_layers):
            spread_mult = 1 + (i * 0.5)
            layer_amount = self.amount * (1 - i * 0.2)
            if layer_amount < 0.0001: layer_amount = 0.0001
            
            p_bid = round_tick_size(skewed_mid * (1 - (final_spread / 2) * spread_mult), self.tick_size)
            if i == 0: p_bid = target_bid
            
            # v1.2 Idea: Only Buy if Price < Entry (Don't Pyramid Up)
            if self.entry_anchor_mode and current_pos_qty > 0:
                avg_entry = float(self.exchange.paper_position.get('entryPrice', 0))
                if p_bid > avg_entry:
                    continue # Skip this buy layer

            desired_orders.append({'side': 'buy', 'price': p_bid, 'amount': layer_amount})

        # 2. Generate SELL Orders
        for i in range(sell_layers):
            spread_mult = 1 + (i * 0.5)
            layer_amount = self.amount * (1 - i * 0.2)
            if layer_amount < 0.0001: layer_amount = 0.0001
            
            p_ask = round_tick_size(skewed_mid * (1 + (final_spread / 2) * spread_mult), self.tick_size)
            if i == 0: p_ask = target_ask
            
            # v1.2 Idea: Only Sell if Price > Entry (Don't Sell Low)
            if self.entry_anchor_mode and current_pos_qty < 0:
                avg_entry = float(self.exchange.paper_position.get('entryPrice', 0))
                if p_ask < avg_entry:
                    continue # Skip this sell layer

            desired_orders.append({'side': 'sell', 'price': p_ask, 'amount': layer_amount})

        # Smart Update Logic
        open_orders = await self.exchange.get_open_orders(self.symbol)
        
        # 1. Cancel orders that are NOT in desired list
        # Simple matching by price and side (ignoring amount small diffs)
        # We use a set of signatures for desired orders
        desired_sigs = set((o['side'], o['price']) for o in desired_orders)
        
        for order in open_orders:
            sig = (order['side'], float(order['price']))
            # Check if this existing order matches any desired order
            match = False
            for d_sig in desired_sigs:
                if d_sig[0] == sig[0] and abs(d_sig[1] - sig[1]) < self.tick_size / 2:
                    match = True
                    break
            
            if not match:
                await self.exchange.cancel_order(self.symbol, order['id'])
        
        # 2. Place orders that are missing
        # We check current open orders (refetching might be safer but for speed we track "kept" orders)
        # Actually, let's just track what we kept from above loop if we optimized.
        # But simpler: check if desired order exists in open_orders
        
        current_sigs = set((o['side'], float(o['price'])) for o in open_orders)
        # Note: The above set includes orders we just sent cancel for? 
        # No, canceling is async. But we should assume they are gone or track better.
        # For v1.1 simplicity, we trust the 'match' logic above kept the valid ones.
        
        # Re-evaluate open_orders state or just track locally?
        # Let's simple check: If a desired order signature is NOT in current_sigs, place it.
        # (This implies if we canceled it above, it's removed from 'logic' view? No.)
        # Correct logic: We need to know which existing orders are KEEPING.
        
        kept_orders_sigs = set()
        for order in open_orders:
            sig = (order['side'], float(order['price']))
            for d_sig in desired_sigs:
                if d_sig[0] == sig[0] and abs(d_sig[1] - sig[1]) < self.tick_size / 2:
                    kept_orders_sigs.add(d_sig)
                    break

        for obj in desired_orders:
            sig = (obj['side'], obj['price'])
            if sig not in kept_orders_sigs:
                await self.place_limit_order(obj['side'], obj['price'], mid_price)

    async def cancel_all(self, orders: List[Dict]):
        tasks = [self.exchange.cancel_order(self.symbol, o['id']) for o in orders]
        if tasks: await asyncio.gather(*tasks)

    async def place_limit_order(self, side: str, price: float, current_mid: float):
        # v1.1: Risk Check
        try:
            position = await self.exchange.get_position(self.symbol)
            pos_amt = position.get('amount', 0.0)
            current_pos_usd = abs(pos_amt * current_mid)
            new_order_usd = self.amount * price
            
            if not self.risk_manager.check_trade_allowed(current_pos_usd, new_order_usd):
                self.logger.warning(f"Order BLOCKED by RiskManager: exposure {current_pos_usd:.2f} + {new_order_usd:.2f} > Limit")
                return

            await self.exchange.place_limit_order(self.symbol, side, price, self.amount)
        except Exception as e:
            self.logger.error(f"Failed to place {side}: {e}")

    async def check_drawdown(self):
        """v1.1: Max Drawdown Protection"""
        try:
            balance = await self.exchange.get_balance()
            
            # Estimate Total Equity (USDT + Unrealized PnL)
            # Note: In real production, use exchange's total equity field if available
            usdt = balance.get('USDT', 0.0)
            
            pos = await self.exchange.get_position(self.symbol)
            unrealized = pos.get('unrealizedPnL', 0.0)
            
            current_equity = usdt + unrealized
            
            if self.initial_equity is None:
                self.initial_equity = current_equity
                return True
                
            drawdown_pct = (self.initial_equity - current_equity) / self.initial_equity
            
            if drawdown_pct >= self.risk_manager.max_drawdown:
                self.logger.critical(f"🚨 MAX DRAWDOWN REACHED: {drawdown_pct*100:.2f}% >= {self.risk_manager.max_drawdown*100}%")
                self.logger.critical("🚨 ACTIVATING EMERGENCY STOP & CLOSE")
                self.is_active = False
                
                # Emergency Close
                if hasattr(self.exchange, "cancel_all_orders"):
                    await self.exchange.cancel_all_orders(self.symbol)
                if hasattr(self.exchange, "close_position"):
                    await self.exchange.close_position(self.symbol)
                    
                return False
                
        except Exception as e:
            self.logger.error(f"Drawdown check failed: {e}")
            
        return True
