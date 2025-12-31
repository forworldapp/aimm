import aiohttp
import asyncio
import logging
from typing import List, Dict, Any
from .config import GAMMA_API_URL, CLOB_API_URL

logger = logging.getLogger(__name__)

class MarketMonitor:
    def __init__(self):
        self.session = None

    async def start(self):
        self.session = aiohttp.ClientSession()

    async def stop(self):
        if self.session:
            await self.session.close()

    async def get_top_markets(self, limit: int = 20) -> List[Dict[str, Any]]:
        """
        Fetches top volume markets from Gamma API.
        """
        if not self.session:
            await self.start()
            
        url = f"{GAMMA_API_URL}/events"
        params = {
            "limit": limit,
            "active": "true",
            "archived": "false",
            "closed": "false",
            "order": "volume" # Sort by volume
        }
        
        try:
            async with self.session.get(url, params=params) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    # Gamma API structure can vary, usually returns a list or list wrapper
                    # The /events endpoint returns events which contain markets.
                    # We need to extract markets from events.
                    markets = []
                    for event in data:
                        if "markets" in event:
                            markets.extend(event["markets"])
                    return markets
                else:
                    logger.error(f"Failed to fetch markets: {resp.status} {await resp.text()}")
                    return []
        except Exception as e:
            logger.error(f"Error fetching top markets: {e}")
            return []

    async def get_orderbook(self, token_id: str) -> Dict[str, Any]:
        """
        Fetches orderbook for a specific token from CLOB API.
        This is needed to get the REAL actionable liquidity, not just 'last price'.
        """
        if not self.session:
            await self.start()

        # CLOB usage usually requires token_id.
        # Endpoint: /book?token_id=...
        url = f"{CLOB_API_URL}/book"
        params = {"token_id": token_id}

        try:
            async with self.session.get(url, params=params) as resp:
                if resp.status == 200:
                    return await resp.json()
                else:
                    # 404 might mean no liquidity or invalid token
                    return {}
        except Exception as e:
            logger.error(f"Error fetching orderbook for {token_id}: {e}")
            return {}
