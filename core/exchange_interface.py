from abc import ABC, abstractmethod
from typing import Dict, Optional, List

class ExchangeInterface(ABC):
    """
    Abstract Base Class for Exchange interactions.
    Ensures modularity by defining a standard interface for any exchange.
    """

    @abstractmethod
    async def connect(self):
        """Establish connection to the exchange (WebSocket/HTTP)."""
        pass

    @abstractmethod
    async def get_balance(self) -> Dict[str, float]:
        """Fetch current account balance."""
        pass

    @abstractmethod
    async def place_limit_order(self, symbol: str, side: str, price: float, quantity: float) -> str:
        """
        Place a limit order.
        :param symbol: Trading pair (e.g., 'BTC-USDT')
        :param side: 'buy' or 'sell'
        :param price: Limit price
        :param quantity: Order size
        :return: Order ID
        """
        pass

    @abstractmethod
    async def cancel_order(self, symbol: str, order_id: str):
        """Cancel a specific order."""
        pass

    @abstractmethod
    async def get_orderbook(self, symbol: str) -> Dict:
        """Fetch current orderbook (L2)."""
        pass

    @abstractmethod
    async def get_open_orders(self, symbol: str) -> List[Dict]:
        """Fetch all open orders."""
        pass
