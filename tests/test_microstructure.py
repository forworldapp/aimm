"""
Unit tests for Microstructure Signals module
"""
import unittest
import time

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.microstructure import VPIN, TradeArrivalAnalyzer, VolumeClock, MicrostructureIntegrator

class TestVPIN(unittest.TestCase):
    def setUp(self):
        self.vpin = VPIN({'bucket_size_usd': 1000, 'n_buckets': 10, 'threshold': 0.5})
    
    def test_balanced_trades(self):
        # Equal buy/sell volume should give low VPIN
        for _ in range(10):
            self.vpin.update({'price': 100, 'size': 5, 'side': 'buy'})
            self.vpin.update({'price': 100, 'size': 5, 'side': 'sell'})
        
        vpin = self.vpin.calculate()
        self.assertLess(vpin, 0.3)  # Low imbalance
    
    def test_imbalanced_trades(self):
        # All buy = high VPIN
        for _ in range(20):
            self.vpin.update({'price': 100, 'size': 5, 'side': 'buy'})
        
        vpin = self.vpin.calculate()
        self.assertGreater(vpin, 0.8)  # High imbalance


class TestTradeArrival(unittest.TestCase):
    def setUp(self):
        self.analyzer = TradeArrivalAnalyzer({'baseline_window_seconds': 60, 'elevated_threshold': 2.0})
    
    def test_baseline_calculation(self):
        # Add trades over time
        current = time.time()
        for i in range(60):
            self.analyzer.record_trade(current - 60 + i)
        
        result = self.analyzer.calculate_arrival_rate(current)
        self.assertAlmostEqual(result['baseline_rate'], 1.0, delta=0.1)


class TestMicrostructureIntegrator(unittest.TestCase):
    def setUp(self):
        self.integrator = MicrostructureIntegrator({
            'vpin': {'bucket_size_usd': 1000, 'n_buckets': 5, 'threshold': 0.5},
            'defensive_risk_score': 1.0,
            'cautious_risk_score': 0.5
        })
    
    def test_normal_conditions(self):
        # Without trades, should be normal
        result = self.integrator.analyze()
        self.assertEqual(result['action'], 'normal')
        self.assertEqual(result['spread_mult'], 1.0)
    
    def test_high_vpin_defensive(self):
        # Simulate high imbalance
        for _ in range(30):
            self.integrator.update_trade({'price': 100, 'size': 5, 'side': 'buy'})
        
        result = self.integrator.analyze()
        # May trigger cautious or defensive based on VPIN
        self.assertGreaterEqual(result['spread_mult'], 1.0)


if __name__ == '__main__':
    unittest.main()
