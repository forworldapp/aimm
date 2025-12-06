from abc import ABC, abstractmethod
import asyncio
from core.exchange_interface import ExchangeInterface

class BaseStrategy(ABC):
    """
    Abstract Base Class for all trading strategies.
    """
    def __init__(self, exchange: ExchangeInterface, symbol: str):
        self.exchange = exchange
        self.symbol = symbol
        self.is_running = False

    @abstractmethod
    async def run(self):
        """
        Main loop of the strategy.
        """
        pass

    async def stop(self):
        """
        Stop the strategy loop.
        """
        self.is_running = False
