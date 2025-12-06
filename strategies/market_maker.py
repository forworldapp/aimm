import asyncio
import logging
from typing import List, Dict
from core.exchange_interface import ExchangeInterface
from core.config import Config
from core.risk_manager import RiskManager
from strategies.base_strategy import BaseStrategy
from utils.calculations import round_tick_size, round_step_size

class MarketMaker(BaseStrategy):
    """
    Market Making Strategy with Inventory Management.
    """

    def __init__(self, exchange: ExchangeInterface):
        # Load params from Config
        symbol = Config.get("exchange", "symbol", "BTC-USDT")
        super().__init__(exchange, symbol)
        
        self.logger = logging.getLogger("MarketMaker")
        self.risk_manager = RiskManager()
        
        # Strategy Params
        self.spread = float(Config.get("strategy", "spread_pct", 0.001))
        self.amount = float(Config.get("strategy", "order_amount", 0.001))
        self.refresh_interval = int(Config.get("strategy", "refresh_interval", 5))
        self.skew_factor = float(Config.get("risk", "inventory_skew_factor", 0.0))
        
        # Exchange Info (Hardcoded for MVP)
        self.tick_size = 0.1 
        self.step_size = 0.001

    async def run(self):
        self.is_running = True
        self.logger.info(f"Starting Market Maker on {self.symbol} (Spread: {self.spread*100}%)")

        while self.is_running:
            try:
                await self.cycle()
            except Exception as e:
                self.logger.error(f"Error in strategy cycle: {e}")
            
            await asyncio.sleep(self.refresh_interval)

    async def cycle(self):
        # 1. Get Data
        orderbook = await self.exchange.get_orderbook(self.symbol)
        position = await self.exchange.get_position(self.symbol)
        
        if not orderbook or 'bids' not in orderbook:
            return

        best_bid = orderbook['bids'][0][0]
        best_ask = orderbook['asks'][0][0]
        mid_price = (best_bid + best_ask) / 2
        
        current_pos_qty = position.get('amount', 0.0)
        
        # 2. Calculate Skew
        # Max position for skew calc (approximate, e.g., 10x order amount)
        max_skew_qty = self.amount * 10 
        skew_adjust = self.risk_manager.calculate_skew(current_pos_qty, max_skew_qty, self.skew_factor)
        
        # Apply Skew: If we have Long pos, skew_adjust is negative -> Lower prices -> Sell easier
        skewed_mid = mid_price * (1 + skew_adjust)
        
        # 3. Calculate Targets
        target_bid = round_tick_size(skewed_mid * (1 - self.spread), self.tick_size)
        target_ask = round_tick_size(skewed_mid * (1 + self.spread), self.tick_size)

        self.logger.info(f"Pos: {current_pos_qty} | Mid: {mid_price:.2f} | SkewedMid: {skewed_mid:.2f} | Bid: {target_bid} | Ask: {target_ask}")

        # 4. Risk Check (Pre-Trade)
        # Check if we can add more exposure
        can_buy = self.risk_manager.check_trade_allowed(current_pos_qty * mid_price, self.amount * mid_price)
        can_sell = self.risk_manager.check_trade_allowed(current_pos_qty * mid_price, self.amount * mid_price) 
        # Note: Selling reduces long exposure, but increases short exposure. 
        # For simplicity, check_trade_allowed checks absolute exposure.
        
        # 5. Manage Orders
        open_orders = await self.exchange.get_open_orders(self.symbol)
        
        # Cancel if necessary
        should_reset = False
        for order in open_orders:
            price = float(order['price'])
            side = order['side']
            
            if side == 'buy' and abs(price - target_bid) > self.tick_size:
                should_reset = True
            if side == 'sell' and abs(price - target_ask) > self.tick_size:
                should_reset = True
                
        if should_reset or not open_orders:
            await self.cancel_all(open_orders)
            
            # Place new orders
            if can_buy:
                await self.place_limit_order('buy', target_bid)
            if can_sell:
                await self.place_limit_order('sell', target_ask)

    async def cancel_all(self, orders: List[Dict]):
        tasks = [self.exchange.cancel_order(self.symbol, o['id']) for o in orders]
        if tasks: await asyncio.gather(*tasks)

    async def place_limit_order(self, side: str, price: float):
        try:
            await self.exchange.place_limit_order(self.symbol, side, price, self.amount)
        except Exception as e:
            self.logger.error(f"Failed to place {side}: {e}")
