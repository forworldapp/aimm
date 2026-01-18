"""
Funding Rate & Liquidation Prediction - Phase 3.2 & 3.3
Predicts funding rate direction and detects liquidation cascades.

Author: Antigravity
Version: 1.0.0
"""

import numpy as np
from typing import Tuple, List
from collections import deque
import logging


class FundingPredictor:
    """
    Funding Rate Prediction for inventory bias optimization.
    
    Strategy:
    - Positive funding → shorts receive → prefer short inventory
    - Negative funding → longs receive → prefer long inventory
    """
    
    def __init__(self, history_size: int = 24):
        """
        Args:
            history_size: Number of funding periods to track (default 24 = 3 days at 8h)
        """
        self.funding_history: deque = deque(maxlen=history_size)
        self.logger = logging.getLogger("FundingPredictor")
    
    def record_funding(self, rate: float):
        """Record a funding rate observation (e.g., 0.0001 = 0.01%)."""
        self.funding_history.append(rate)
    
    def predict_next_funding(self) -> Tuple[float, float]:
        """
        Predict next funding rate and confidence.
        
        Returns:
            Tuple of (predicted_rate, confidence)
        """
        if len(self.funding_history) < 4:
            return 0.0, 0.0
        
        recent = list(self.funding_history)
        
        # Simple prediction: weighted average + trend
        avg_rate = np.mean(recent[-8:]) if len(recent) >= 8 else np.mean(recent)
        
        # Trend: recent 4 vs previous 4
        if len(recent) >= 8:
            recent_avg = np.mean(recent[-4:])
            prev_avg = np.mean(recent[-8:-4])
            trend = recent_avg - prev_avg
        else:
            trend = 0
        
        predicted = avg_rate + trend * 0.3
        
        # Confidence based on consistency
        std = np.std(recent)
        confidence = 1 / (1 + std * 1000)  # Higher std = lower confidence
        
        return float(predicted), float(np.clip(confidence, 0, 1))
    
    def get_inventory_bias(self) -> float:
        """
        Get recommended inventory bias based on funding prediction.
        
        Returns:
            Bias from -1 (prefer short) to +1 (prefer long)
        """
        predicted, confidence = self.predict_next_funding()
        
        if confidence < 0.3:
            return 0.0  # Low confidence → neutral
        
        # Positive funding → shorts receive → bias short (-1)
        # Negative funding → longs receive → bias long (+1)
        bias = -np.sign(predicted) * min(abs(predicted) * 5000, 1) * confidence
        
        return float(np.clip(bias, -1, 1))
    
    def get_stats(self) -> dict:
        """Get predictor statistics."""
        if not self.funding_history:
            return {'samples': 0, 'avg_rate': 0, 'bias': 0}
        
        return {
            'samples': len(self.funding_history),
            'avg_rate': float(np.mean(self.funding_history)),
            'last_rate': float(self.funding_history[-1]),
            'bias': self.get_inventory_bias()
        }


class LiquidationDetector:
    """
    Liquidation Cascade Detection.
    
    Detects potential liquidation cascades from:
    - Open Interest (OI) drops
    - Rapid price movements
    - Volume spikes
    
    When detected, strategy should:
    - Widen spreads
    - Reduce position sizes
    - Potentially pause trading
    """
    
    def __init__(self, 
                 oi_threshold: float = 0.05,
                 price_threshold: float = 0.02,
                 volume_threshold: float = 2.0):
        """
        Args:
            oi_threshold: OI change threshold for cascade (5%)
            price_threshold: Price change threshold (2%)
            volume_threshold: Volume spike ratio threshold (2x average)
        """
        self.oi_threshold = oi_threshold
        self.price_threshold = price_threshold
        self.volume_threshold = volume_threshold
        
        self.oi_history: deque = deque(maxlen=100)
        self.price_history: deque = deque(maxlen=100)
        self.volume_history: deque = deque(maxlen=100)
        
        self.cascade_active = False
        self.cascade_direction = 'none'
        self.cascade_severity = 0.0
        
        self.logger = logging.getLogger("LiquidationDetector")
    
    def update(self, oi: float, price: float, volume: float):
        """
        Update with latest market data.
        
        Args:
            oi: Open Interest value
            price: Current price
            volume: Recent volume
        """
        self.oi_history.append(oi)
        self.price_history.append(price)
        self.volume_history.append(volume)
        
        # Check for cascade
        self._detect_cascade()
    
    def _detect_cascade(self):
        """Detect liquidation cascade from history."""
        if len(self.oi_history) < 10:
            self.cascade_active = False
            return
        
        # Calculate changes
        oi_now = self.oi_history[-1]
        oi_prev = self.oi_history[-10]
        oi_change = (oi_now - oi_prev) / max(oi_prev, 1)
        
        price_now = self.price_history[-1]
        price_prev = self.price_history[-10]
        price_change = (price_now - price_prev) / price_prev
        
        avg_volume = np.mean(list(self.volume_history)[:-5]) if len(self.volume_history) > 5 else 1
        recent_volume = np.mean(list(self.volume_history)[-5:])
        volume_spike = recent_volume / max(avg_volume, 1)
        
        # Detect long liquidation cascade:
        # OI dropping + price dropping + volume spike
        is_long_liq = (oi_change < -self.oi_threshold and 
                       price_change < -self.price_threshold and 
                       volume_spike > self.volume_threshold)
        
        # Detect short liquidation cascade:
        # OI dropping + price rising + volume spike
        is_short_liq = (oi_change < -self.oi_threshold and 
                        price_change > self.price_threshold and 
                        volume_spike > self.volume_threshold)
        
        if is_long_liq:
            self.cascade_active = True
            self.cascade_direction = 'long_liquidation'
            self.cascade_severity = abs(oi_change) + abs(price_change) * 5 + volume_spike * 0.2
        elif is_short_liq:
            self.cascade_active = True
            self.cascade_direction = 'short_liquidation'
            self.cascade_severity = abs(oi_change) + abs(price_change) * 5 + volume_spike * 0.2
        else:
            self.cascade_active = False
            self.cascade_direction = 'none'
            self.cascade_severity = 0.0
    
    def detect(self, oi_change: float, price_change: float, volume_spike: float) -> Tuple[bool, float, str]:
        """
        Quick detection from pre-calculated metrics.
        
        Args:
            oi_change: OI change ratio (e.g., -0.05 = -5%)
            price_change: Price change ratio
            volume_spike: Volume vs average ratio
            
        Returns:
            Tuple of (is_cascade, severity, direction)
        """
        is_long_liq = (oi_change < -self.oi_threshold and 
                       price_change < -self.price_threshold and 
                       volume_spike > self.volume_threshold)
        
        is_short_liq = (oi_change < -self.oi_threshold and 
                        price_change > self.price_threshold and 
                        volume_spike > self.volume_threshold)
        
        if is_long_liq:
            severity = abs(oi_change) + abs(price_change) * 5 + volume_spike * 0.2
            return True, severity, 'long_liquidation'
        elif is_short_liq:
            severity = abs(oi_change) + abs(price_change) * 5 + volume_spike * 0.2
            return True, severity, 'short_liquidation'
        
        return False, 0.0, 'none'
    
    def get_adjustment(self) -> dict:
        """
        Get strategy adjustments for current cascade state.
        
        Returns:
            Dict with spread_mult, size_mult, inventory_limit
        """
        if not self.cascade_active:
            return {'spread_mult': 1.0, 'size_mult': 1.0, 'inventory_limit': 1.0}
        
        severity = self.cascade_severity
        
        return {
            'spread_mult': 1 + severity,  # Widen spread
            'size_mult': 1 / (1 + severity),  # Reduce size
            'inventory_limit': max(0.2, 0.8 - severity * 0.3),  # Reduce max position
            'direction': self.cascade_direction,
            'severity': severity
        }
    
    def get_stats(self) -> dict:
        """Get detector statistics."""
        return {
            'cascade_active': self.cascade_active,
            'direction': self.cascade_direction,
            'severity': self.cascade_severity,
            'oi_samples': len(self.oi_history)
        }


# Unit tests
if __name__ == "__main__":
    print("=" * 60)
    print("FUNDING PREDICTOR TESTS")
    print("=" * 60)
    
    funding = FundingPredictor()
    
    # Simulate positive funding trend
    rates = [0.0001, 0.00015, 0.0002, 0.00018, 0.00022, 0.00025, 0.0003, 0.0002]
    for r in rates:
        funding.record_funding(r)
    
    pred, conf = funding.predict_next_funding()
    bias = funding.get_inventory_bias()
    print(f"\nPositive Funding Scenario:")
    print(f"  Predicted: {pred:.4%}, Confidence: {conf:.1%}")
    print(f"  Inventory Bias: {bias:.2f} (negative = prefer short)")
    
    print("\n" + "=" * 60)
    print("LIQUIDATION DETECTOR TESTS")
    print("=" * 60)
    
    detector = LiquidationDetector()
    
    # Test direct detection
    test_cases = [
        (-0.08, -0.03, 3.0, "Long Liquidation"),
        (-0.08, +0.03, 3.0, "Short Liquidation"),
        (-0.02, -0.01, 1.5, "Normal Market"),
        (-0.10, -0.05, 5.0, "Severe Long Liq"),
    ]
    
    for oi, price, vol, desc in test_cases:
        is_casc, sev, direction = detector.detect(oi, price, vol)
        print(f"\n{desc}:")
        print(f"  OI:{oi:.1%}, Price:{price:.1%}, Vol:{vol:.1f}x")
        print(f"  Cascade: {is_casc}, Direction: {direction}, Severity: {sev:.2f}")
