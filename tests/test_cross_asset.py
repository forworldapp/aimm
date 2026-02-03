"""
Unit tests for Cross-Asset Hedging module
"""
import unittest
import numpy as np

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.cross_asset_hedger import CorrelationAnalyzer, CrossAssetHedger, CrossAssetHedgeIntegrator


class TestCorrelationAnalyzer(unittest.TestCase):
    def setUp(self):
        self.analyzer = CorrelationAnalyzer({'window_minutes': 100, 'min_correlation': 0.7})
    
    def test_high_correlation(self):
        # Simulate correlated prices with same random factor
        np.random.seed(42)
        for i in range(100):
            common_factor = np.random.normal(0, 1)
            btc = 50000 * (1 + common_factor * 0.001)
            eth = 3000 * (1 + common_factor * 0.001 * 0.8)  # ETH tracks BTC with 0.8 beta
            self.analyzer.update(eth, btc)
        
        corr = self.analyzer.calculate_correlation()
        self.assertGreater(corr, 0.5)  # Should be positively correlated
    
    def test_uncorrelated(self):
        # Random prices
        np.random.seed(42)
        for i in range(100):
            btc = 50000 + np.random.normal(0, 100)
            eth = 3000 + np.random.normal(0, 50)
            self.analyzer.update(eth, btc)
        
        corr = self.analyzer.calculate_correlation()
        self.assertLess(abs(corr), 0.5)


class TestCrossAssetHedger(unittest.TestCase):
    def setUp(self):
        self.hedger = CrossAssetHedger({
            'correlation': {'window_minutes': 100, 'min_correlation': 0.5},
            'hedge': {'ratio_beta': 0.8, 'min_position_usd': 50}
        })
    
    def test_hedge_calculation(self):
        # Simulate correlated prices
        np.random.seed(42)
        for i in range(100):
            btc = 50000 + i * 10
            eth = 3000 + i * 0.6
            self.hedger.update_prices(eth, btc)
        
        # Long 1 ETH = +$3000 exposure
        result = self.hedger.calculate_hedge_position(
            eth_inventory=1.0,
            eth_price=3000,
            btc_price=50000
        )
        
        # Should recommend short BTC to hedge
        self.assertLess(result['hedge_btc'], 0)
        self.assertIn(result['action'], ['none', 'rebalance'])


class TestCrossAssetHedgeIntegrator(unittest.TestCase):
    def setUp(self):
        self.integrator = CrossAssetHedgeIntegrator({
            'enabled': True,
            'correlation': {'window_minutes': 50, 'min_correlation': 0.5},
            'hedge': {'ratio_beta': 0.8}
        })
    
    def test_disabled(self):
        disabled = CrossAssetHedgeIntegrator({'enabled': False})
        result = disabled.update_and_analyze(3000, 50000, 1.0)
        
        self.assertFalse(result['enabled'])
        self.assertEqual(result['action'], 'none')
    
    def test_update_and_analyze(self):
        # Populate history
        for i in range(60):
            self.integrator.update_and_analyze(
                eth_price=3000 + i * 0.5,
                btc_price=50000 + i * 8,
                eth_inventory=0
            )
        
        # Now with inventory
        result = self.integrator.update_and_analyze(
            eth_price=3030,
            btc_price=50480,
            eth_inventory=1.0
        )
        
        self.assertTrue(result['enabled'])
        self.assertIn('correlation', result)


if __name__ == '__main__':
    unittest.main()
