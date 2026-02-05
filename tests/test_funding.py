"""
Unit tests for Funding Rate Arbitrage module
"""
import unittest
from datetime import datetime, timedelta, timezone

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.funding_monitor import FundingRateMonitor, FundingIntegratedMM

class TestFundingRateMonitor(unittest.TestCase):
    def setUp(self):
        self.monitor = FundingRateMonitor()
    
    def test_annualized_rate(self):
        # 0.01% per 8h = ~8.76% annual
        rate = self.monitor.calculate_annualized_rate(0.0001, 8)
        self.assertAlmostEqual(rate, 10.95, delta=0.1)
    
    def test_analyze_positive_funding(self):
        # High positive funding = short opportunity
        self.monitor.update(0.0005)  # 0.05%
        analysis = self.monitor.analyze_opportunity()
        
        self.assertTrue(analysis['opportunity'])
        self.assertEqual(analysis['direction'], 'short')
        self.assertGreater(analysis['annual_yield'], 50)
    
    def test_analyze_negative_funding(self):
        # High negative funding = long opportunity
        self.monitor.update(-0.0005)  # -0.05%
        analysis = self.monitor.analyze_opportunity()
        
        self.assertTrue(analysis['opportunity'])
        self.assertEqual(analysis['direction'], 'long')
    
    def test_analyze_neutral(self):
        # Low funding = no opportunity
        self.monitor.update(0.00001)  # 0.001%
        analysis = self.monitor.analyze_opportunity()
        
        self.assertFalse(analysis['opportunity'])
        self.assertEqual(analysis['direction'], 'neutral')


class TestFundingIntegratedMM(unittest.TestCase):
    def setUp(self):
        self.integrator = FundingIntegratedMM()
    
    def test_short_bias_adjustment(self):
        # Short opportunity should reduce bid, increase ask
        analysis = {
            'opportunity': True,
            'direction': 'short',
            'confidence': 0.8,
            'hours_to_funding': 2.0,
            'annual_yield': 50.0
        }
        
        adj = self.integrator.get_adjustment(analysis)
        
        self.assertLess(adj['bid_size_mult'], 1.0)
        self.assertGreater(adj['ask_size_mult'], 1.0)
        self.assertLess(adj['target_inventory_pct'], 0)
    
    def test_long_bias_adjustment(self):
        # Long opportunity should increase bid, reduce ask
        analysis = {
            'opportunity': True,
            'direction': 'long',
            'confidence': 0.8,
            'hours_to_funding': 2.0,
            'annual_yield': 50.0
        }
        
        adj = self.integrator.get_adjustment(analysis)
        
        self.assertGreater(adj['bid_size_mult'], 1.0)
        self.assertLess(adj['ask_size_mult'], 1.0)
        self.assertGreater(adj['target_inventory_pct'], 0)
    
    def test_freeze_before_funding(self):
        # Should freeze 30min before funding
        analysis = {
            'opportunity': True,
            'direction': 'short',
            'confidence': 0.8,
            'hours_to_funding': 0.4,  # 24 min
            'annual_yield': 50.0
        }
        
        adj = self.integrator.get_adjustment(analysis)
        
        self.assertTrue(adj['freeze_orders'])


if __name__ == '__main__':
    unittest.main()
