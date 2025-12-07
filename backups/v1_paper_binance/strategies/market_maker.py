import asyncio
import logging
import statistics
from typing import List, Dict
from core.exchange_interface import ExchangeInterface
from core.config import Config
from core.risk_manager import RiskManager
from strategies.base_strategy import BaseStrategy
from utils.calculations import round_tick_size

class MarketMaker(BaseStrategy):
    """
    Enhanced Market Maker with Trend Following & Volatility Adjustment.
    """

    def __init__(self, exchange: ExchangeInterface):
        # Load params from Config
        symbol = Config.get("exchange", "symbol", "BTC-USDT")
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

    async def run(self):
        self.is_running = True
        self.logger.info(f"Starting Enhanced Market Maker on {self.symbol}")

        while self.is_running:
            try:
                await self.cycle()
            except Exception as e:
                self.logger.error(f"Error in strategy cycle: {e}")
            
            await asyncio.sleep(self.refresh_interval)

    def _update_history(self, price):
        self.price_history.append(price)
        if len(self.price_history) > self.history_max_len:
            self.price_history.pop(0)

    def _get_trend_skew(self):
        """
        Calculate skew based on simple Moving Average trend.
        Returns a value to adjust mid price.
        """
        if len(self.price_history) < 20:
            return 0.0
            
        short_ma = statistics.mean(self.price_history[-10:])
        long_ma = statistics.mean(self.price_history[-20:])
        
        # If Short MA > Long MA (Uptrend) -> Positive Skew (Shift Mid UP to buy more/sell higher)
        # If Short MA < Long MA (Downtrend) -> Negative Skew (Shift Mid DOWN to sell more/buy lower)
        
        if short_ma > long_ma:
            return 0.0005 # +5 bps skew for uptrend
        elif short_ma < long_ma:
            return -0.0005 # -5 bps skew for downtrend
        return 0.0

    def _get_volatility_multiplier(self):
        """
        Calculate spread multiplier based on volatility (Standard Deviation).
        """
        if len(self.price_history) < 20:
            return 1.0
            
        stdev = statistics.stdev(self.price_history[-20:])
        mean_price = statistics.mean(self.price_history[-20:])
        
        # Volatility as percentage of price
        vol_pct = stdev / mean_price
        
        # Base volatility threshold (e.g., 0.01%). If higher, widen spread.
        base_vol = 0.0001 
        
        multiplier = max(1.0, vol_pct / base_vol)
        return min(multiplier, 5.0) # Cap at 5x spread

    async def cycle(self):
        # 1. Get Data
        orderbook = await self.exchange.get_orderbook(self.symbol)
        position = await self.exchange.get_position(self.symbol)
        
        if not orderbook or 'bids' not in orderbook:
            return

        best_bid = orderbook['bids'][0][0]
        best_ask = orderbook['asks'][0][0]
        mid_price = (best_bid + best_ask) / 2
        
        self._update_history(mid_price)
        
        current_pos_qty = position.get('amount', 0.0)
        
        # 2. Calculate Parameters
        
        # A. Inventory Skew
        # Using RiskManager's logic or simple logic here
        # Let's use a simple linear skew for now based on config
        # Max pos for skew = 10x order amount
        max_skew_qty = self.amount * 20 
        inventory_ratio = max(-1.0, min(1.0, current_pos_qty / max_skew_qty))
        inventory_skew = inventory_ratio * self.skew_factor * -1 
        # Note: If Long (ratio > 0), we want to sell -> Lower Price -> Negative Skew
        
        # B. Trend Skew
        trend_skew = self._get_trend_skew()
        
        # Total Skew
        total_skew = inventory_skew + trend_skew
        
        # C. Dynamic Spread
        vol_mult = self._get_volatility_multiplier()
        final_spread = self.base_spread * vol_mult
        
        # Apply Skew
        skewed_mid = mid_price * (1 + total_skew)
        
        # Calculate Targets
        target_bid = round_tick_size(skewed_mid * (1 - final_spread / 2), self.tick_size)
        target_ask = round_tick_size(skewed_mid * (1 + final_spread / 2), self.tick_size)

        self.logger.info(f"Pos: {current_pos_qty:.4f} | Mid: {mid_price:.2f} | VolMult: {vol_mult:.2f} | Trend: {trend_skew*10000:.1f}bps | Bid: {target_bid} | Ask: {target_ask}")

        # 3. Manage Orders (Cancel All & Replace)
        open_orders = await self.exchange.get_open_orders(self.symbol)
        await self.cancel_all(open_orders)
        
        # Place new orders
        # Simple risk check: Don't exceed max position
        # (Real implementation would be more complex)
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
