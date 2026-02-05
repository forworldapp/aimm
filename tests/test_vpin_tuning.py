"""
Tests for VPIN Retuning Components (v5.2.1)
"""
import unittest
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.microstructure import VPIN, AdaptiveVPIN, MicrostructureIntegrator
from core.kill_switch import KillSwitch, ModulePerformance


class TestAdaptiveVPIN(unittest.TestCase):
    """Test AdaptiveVPIN threshold calculations"""
    
    def setUp(self):
        self.vpin = AdaptiveVPIN({
            'base_threshold': 0.5,
            'volatility_adjustment': 0.1,
            'volume_adjustment': 0.1,
            'trend_adjustment': 0.05,
        })
    
    def test_base_threshold(self):
        """Base threshold with no market conditions"""
        threshold = self.vpin.calculate_adaptive_threshold()
        self.assertEqual(threshold, 0.5)
    
    def test_high_volatility_adjustment(self):
        """Threshold increases with high volatility"""
        self.vpin.update_market_conditions(volatility=0.03, volume_ratio=1.0)
        threshold = self.vpin.calculate_adaptive_threshold()
        self.assertAlmostEqual(threshold, 0.6, places=2)
    
    def test_volume_surge_adjustment(self):
        """Threshold increases with volume surge"""
        self.vpin.update_market_conditions(volatility=0.01, volume_ratio=2.5)
        threshold = self.vpin.calculate_adaptive_threshold()
        self.assertAlmostEqual(threshold, 0.6, places=2)
    
    def test_combined_adjustments(self):
        """All conditions combined"""
        self.vpin.update_market_conditions(volatility=0.025, volume_ratio=2.5, is_trending=True)
        threshold = self.vpin.calculate_adaptive_threshold()
        self.assertAlmostEqual(threshold, 0.75, places=2)
    
    def test_max_threshold_cap(self):
        """Threshold capped at maximum"""
        self.vpin.max_threshold = 0.7
        self.vpin.update_market_conditions(volatility=0.05, volume_ratio=3.0, is_trending=True)
        threshold = self.vpin.calculate_adaptive_threshold()
        self.assertEqual(threshold, 0.7)


class TestKillSwitch(unittest.TestCase):
    """Test per-module kill switch"""
    
    def setUp(self):
        self.ks = KillSwitch({
            'enabled': True,
            'microstructure_signals_limit': 200,
        })
    
    def test_register_module(self):
        """Module registration"""
        self.ks.register_module('microstructure_signals')
        self.assertIn('microstructure_signals', self.ks.modules)
        self.assertTrue(self.ks.is_enabled('microstructure_signals'))
    
    def test_record_pnl(self):
        """PnL recording"""
        self.ks.register_module('microstructure_signals')
        self.ks.record_pnl('microstructure_signals', -50)
        self.assertEqual(self.ks.modules['microstructure_signals'].pnl, -50)
    
    def test_auto_disable_on_loss(self):
        """Module disabled when loss exceeds threshold"""
        self.ks.register_module('microstructure_signals')
        
        # Record losses incrementally
        for _ in range(5):
            self.ks.record_pnl('microstructure_signals', -50)
        
        # Should be disabled after $250 loss (limit is $200)
        self.assertFalse(self.ks.is_enabled('microstructure_signals'))
        self.assertIsNotNone(self.ks.modules['microstructure_signals'].disabled_reason)
    
    def test_reset_module(self):
        """Module can be reset"""
        self.ks.register_module('microstructure_signals')
        self.ks.record_pnl('microstructure_signals', -250)
        self.assertFalse(self.ks.is_enabled('microstructure_signals'))
        
        self.ks.reset_module('microstructure_signals')
        self.assertTrue(self.ks.is_enabled('microstructure_signals'))
        self.assertEqual(self.ks.modules['microstructure_signals'].pnl, 0)
    
    def test_get_status(self):
        """Status report"""
        self.ks.register_module('microstructure_signals')
        self.ks.record_pnl('microstructure_signals', -100)
        
        status = self.ks.get_status()
        self.assertIn('microstructure_signals', status)
        self.assertEqual(status['microstructure_signals']['pnl'], -100)


class TestMicrostructureIntegrator(unittest.TestCase):
    """Test integrator uses AdaptiveVPIN"""
    
    def test_uses_adaptive_vpin_by_default(self):
        """Integrator uses AdaptiveVPIN when configured"""
        integrator = MicrostructureIntegrator({
            'use_adaptive_vpin': True,
            'vpin': {'base_threshold': 0.6}
        })
        self.assertIsInstance(integrator.vpin, AdaptiveVPIN)
    
    def test_uses_standard_vpin_when_disabled(self):
        """Integrator uses standard VPIN when adaptive disabled"""
        integrator = MicrostructureIntegrator({
            'use_adaptive_vpin': False,
            'vpin': {'threshold': 0.7}
        })
        self.assertIsInstance(integrator.vpin, VPIN)
        self.assertNotIsInstance(integrator.vpin, AdaptiveVPIN)


if __name__ == '__main__':
    unittest.main(verbosity=2)
