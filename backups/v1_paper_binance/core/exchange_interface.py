from abc import abstractmethod
from typing import Dict, List

class ExchangeInterface:
    # ... (Previous methods) ...
    
    @abstractmethod
    async def connect(self): pass

    @abstractmethod
    async def get_balance(self) -> Dict[str, float]: pass

    @abstractmethod
    async def place_limit_order(self, symbol: str, side: str, price: float, quantity: float) -> str: pass

    @abstractmethod
    async def cancel_order(self, symbol: str, order_id: str): pass

    @abstractmethod
    async def get_orderbook(self, symbol: str) -> Dict: pass

    @abstractmethod
    async def get_open_orders(self, symbol: str) -> List[Dict]: pass
    
    @abstractmethod
    async def get_position(self, symbol: str) -> Dict:
        """
        Fetch current position for a symbol.
        Should return {'amount': float, 'entryPrice': float, 'unrealizedPnL': float}
        """
        pass
