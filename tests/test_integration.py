import unittest
from unittest.mock import MagicMock
import sys
import os

# Adjust path
sys.path.append(os.getcwd())

from strategies.market_maker import MarketMaker
from core.config import Config

class TestMarketMakerIntegration(unittest.TestCase):
    def test_order_flow_initialization(self):
        # Mock Config
        def config_side_effect(section, key, default=None):
            if section == "order_flow_analysis" and key == "enabled":
                return True
            if section == "order_flow_analysis":
                return {} # return empty dict for other keys (like 'obi')
            if section == "strategy":
                return 0.001
            return default
            
        Config.get = MagicMock(side_effect=config_side_effect)
        
        # Mock Exchange
        exchange = MagicMock()
        
        # Init Strategy
        mm = MarketMaker(exchange)
        
        # Check if order_flow is initialized
        self.assertIsNotNone(mm.order_flow)
        print("OrderFlowAnalyzer initialized successfully in MarketMaker")

if __name__ == '__main__':
    unittest.main()
