import asyncio
import logging
from typing import List, Dict
from core.exchange_interface import ExchangeInterface
from strategies.base_strategy import BaseStrategy
from utils.calculations import round_tick_size, round_step_size

class MarketMaker(BaseStrategy):
    """
    Simple Market Making Strategy.
    - Places a Bid and Ask around the Mid Price.
    - Cancels and replaces orders if the price moves significantly.
    """

    def __init__(self, exchange: ExchangeInterface, symbol: str, 
                 spread: float = 0.001, amount: float = 0.001, 
                 refresh_interval: int = 5):
        """
        :param spread: Percentage spread from mid price (e.g., 0.001 = 0.1%)
        :param amount: Order quantity per side
        :param refresh_interval: Time in seconds to wait between loops
        """
        super().__init__(exchange, symbol)
        self.logger = logging.getLogger("MarketMaker")
        self.spread = spread
        self.amount = amount
        self.refresh_interval = refresh_interval
        
        # Hardcoded for now, should be fetched from exchange info
        self.tick_size = 0.1 
        self.step_size = 0.001

    async def run(self):
        self.is_running = True
        self.logger.info(f"Starting Market Maker on {self.symbol}...")

        while self.is_running:
            try:
                await self.cycle()
            except Exception as e:
                self.logger.error(f"Error in strategy cycle: {e}")
            
            await asyncio.sleep(self.refresh_interval)

    async def cycle(self):
        """
        Single execution cycle.
        """
        # 1. Get Market Data
        orderbook = await self.exchange.get_orderbook(self.symbol)
        if not orderbook or 'bids' not in orderbook or 'asks' not in orderbook:
            self.logger.warning("Empty orderbook, skipping cycle.")
            return

        best_bid = orderbook['bids'][0][0] if orderbook['bids'] else 0
        best_ask = orderbook['asks'][0][0] if orderbook['asks'] else 0
        
        if best_bid == 0 or best_ask == 0:
            self.logger.warning("Invalid orderbook prices.")
            return

        mid_price = (best_bid + best_ask) / 2
        
        # 2. Calculate Target Prices
        target_bid = round_tick_size(mid_price * (1 - self.spread), self.tick_size)
        target_ask = round_tick_size(mid_price * (1 + self.spread), self.tick_size)

        self.logger.info(f"Mid: {mid_price} | Target Bid: {target_bid} | Target Ask: {target_ask}")

        # 3. Manage Orders
        open_orders = await self.exchange.get_open_orders(self.symbol)
        
        # Simple Logic: If any order exists that is NOT at our target price, Cancel All & Replace.
        # (Optimization: Only cancel the specific wrong order)
        
        should_reset = False
        if not open_orders:
            should_reset = True
        else:
            for order in open_orders:
                price = float(order['price'])
                side = order['side']
                
                # Check if price deviation is too large (e.g., > 1 tick)
                if side == 'buy' and abs(price - target_bid) > self.tick_size:
                    should_reset = True
                    break
                if side == 'sell' and abs(price - target_ask) > self.tick_size:
                    should_reset = True
                    break

        if should_reset:
            self.logger.info("Price moved. Resetting orders...")
            await self.cancel_all(open_orders)
            await self.place_orders(target_bid, target_ask)
        else:
            self.logger.info("Orders are within range. Holding.")

    async def cancel_all(self, orders: List[Dict]):
        """
        Cancel all provided orders.
        """
        tasks = []
        for order in orders:
            tasks.append(self.exchange.cancel_order(self.symbol, order['id']))
        if tasks:
            await asyncio.gather(*tasks)

    async def place_orders(self, bid_price: float, ask_price: float):
        """
        Place both Bid and Ask orders.
        """
        # Place Bid
        try:
            await self.exchange.place_limit_order(
                self.symbol, 'buy', bid_price, self.amount
            )
        except Exception as e:
            self.logger.error(f"Failed to place Buy: {e}")

        # Place Ask
        try:
            await self.exchange.place_limit_order(
                self.symbol, 'sell', ask_price, self.amount
            )
        except Exception as e:
            self.logger.error(f"Failed to place Sell: {e}")
