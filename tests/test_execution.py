"""
Unit tests for Execution Algorithms module
"""
import unittest
import time

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.execution_algo import TWAPExecutor, VWAPExecutor, AdaptiveExecutor, ExecutionIntegrator


class TestTWAPExecutor(unittest.TestCase):
    def setUp(self):
        self.twap = TWAPExecutor({'slice_interval_seconds': 10})
    
    def test_create_schedule(self):
        # 100 units over 5 minutes = 30 slices (5*60/10)
        schedule = self.twap.create_schedule(100, 5, 'buy')
        
        self.assertEqual(len(schedule), 30)
        self.assertEqual(sum(s['quantity'] for s in schedule), 100)
        self.assertTrue(all(s['side'] == 'buy' for s in schedule))
    
    def test_get_next_slice(self):
        self.twap.create_schedule(100, 1, 'sell')
        
        # Should get first slice immediately
        next_slice = self.twap.get_next_slice()
        self.assertIsNotNone(next_slice)
        self.assertEqual(next_slice['slice_id'], 0)
    
    def test_mark_executed(self):
        self.twap.create_schedule(100, 1, 'buy')
        self.twap.mark_executed(0, 50000.0)
        
        summary = self.twap.get_execution_summary()
        self.assertEqual(summary['slices_done'], 1)


class TestVWAPExecutor(unittest.TestCase):
    def setUp(self):
        self.vwap = VWAPExecutor()
    
    def test_volume_profile(self):
        # Should have entries for all 24 hours
        for hour in range(24):
            pct = self.vwap.get_target_participation(hour)
            self.assertGreater(pct, 0)
            self.assertLess(pct, 0.15)
    
    def test_calculate_order_size(self):
        size = self.vwap.calculate_order_size(
            remaining_qty=100,
            remaining_time_hours=4,
            current_hour=15,  # High volume hour
            current_volume=1000
        )
        
        self.assertGreater(size, 0)
        self.assertLessEqual(size, 100)


class TestAdaptiveExecutor(unittest.TestCase):
    def setUp(self):
        self.adaptive = AdaptiveExecutor()
    
    def test_analyze_conditions_volatile(self):
        style = self.adaptive.analyze_market_conditions(spread=0.001, volatility=0.05)
        self.assertEqual(style.value, 'twap')
    
    def test_urgency_calculation(self):
        # On schedule
        urgency = self.adaptive.calculate_urgency(0.5, 0.5)
        self.assertAlmostEqual(urgency, 1.0, delta=0.1)
        
        # Behind schedule
        urgency = self.adaptive.calculate_urgency(0.8, 0.4)
        self.assertGreater(urgency, 1.0)
    
    def test_adjusted_size(self):
        # Behind schedule = larger
        size = self.adaptive.get_adjusted_size(10, 1.8, 0.001)
        self.assertGreater(size, 10)


class TestExecutionIntegrator(unittest.TestCase):
    def setUp(self):
        self.integrator = ExecutionIntegrator({'enabled': True})
    
    def test_start_execution(self):
        result = self.integrator.start_execution(100, 'buy', 10)
        
        self.assertEqual(result['status'], 'started')
        self.assertIn('task_id', result)
        self.assertGreater(result['slices'], 0)
    
    def test_disabled(self):
        disabled = ExecutionIntegrator({'enabled': False})
        result = disabled.start_execution(100, 'buy', 10)
        
        self.assertEqual(result['status'], 'disabled')


if __name__ == '__main__':
    unittest.main()
