"""
Fill Probability Model - Phase 3.1
Predicts order fill probability based on spread, volatility, and market conditions.

Author: Antigravity
Version: 1.0.0
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Optional
import logging

try:
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    GradientBoostingRegressor = None
    StandardScaler = None


class FillProbabilityModel:
    """
    Predicts order fill probability.
    
    Use cases:
    1. Optimize spread ↔ fill rate tradeoff
    2. Calculate optimal spread for target fill rate
    3. Estimate expected fills for given quote
    
    Features:
    - spread_bps: Quote spread in basis points
    - volatility: Current market volatility
    - book_imbalance: Order book bid/ask imbalance
    - queue_position: Estimated queue position (0-1)
    """
    
    def __init__(self):
        self.model = None
        self.scaler = None
        self.is_fitted = False
        self.order_history: List[dict] = []
        self.min_training_samples = 50
        self.logger = logging.getLogger("FillProbability")
    
    def record_order(self, order: dict):
        """
        Record an order outcome for training.
        
        Args:
            order: Dict with keys:
                - spread_bps: Spread used
                - volatility: Market volatility at quote time
                - book_imbalance: Order book imbalance (-1 to 1)
                - queue_position: Estimated position (0=front, 1=back)
                - filled_ratio: Actual fill ratio (0 to 1)
        """
        self.order_history.append(order)
        
        # Retrain periodically
        if len(self.order_history) >= self.min_training_samples:
            if len(self.order_history) % 50 == 0:
                self._train_model()
    
    def _train_model(self):
        """Train the fill probability model."""
        if not SKLEARN_AVAILABLE:
            self.logger.warning("sklearn not available")
            return
        
        if len(self.order_history) < self.min_training_samples:
            return
        
        # Prepare training data
        df = pd.DataFrame(self.order_history)
        
        feature_cols = ['spread_bps', 'volatility', 'book_imbalance', 'queue_position']
        
        # Check all columns exist
        for col in feature_cols:
            if col not in df.columns:
                self.logger.warning(f"Missing column: {col}")
                return
        
        X = df[feature_cols].values
        y = df['filled_ratio'].values
        
        # Scale features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Train gradient boosting regressor
        self.model = GradientBoostingRegressor(
            n_estimators=50,
            max_depth=4,
            random_state=42
        )
        self.model.fit(X_scaled, y)
        self.is_fitted = True
        
        self.logger.info(f"Fill model trained on {len(self.order_history)} samples")
    
    def predict_fill_prob(self,
                          spread_bps: float,
                          volatility: float,
                          book_imbalance: float = 0,
                          queue_position: float = 0.5) -> float:
        """
        Predict fill probability for given conditions.
        
        Returns:
            Fill probability (0 to 1)
        """
        if self.is_fitted and self.model is not None:
            features = np.array([[spread_bps, volatility, book_imbalance, queue_position]])
            features_scaled = self.scaler.transform(features)
            prob = self.model.predict(features_scaled)[0]
            return float(np.clip(prob, 0, 1))
        
        # Heuristic fallback
        return self._heuristic_fill_prob(spread_bps, volatility)
    
    def _heuristic_fill_prob(self, spread_bps: float, volatility: float) -> float:
        """
        Heuristic fill probability when model not trained.
        
        Assumptions:
        - Tighter spread = higher fill probability
        - Higher volatility = higher fill probability
        """
        # Baseline: 5bps spread → ~50% fill
        base_prob = 0.5
        
        # Spread adjustment: each bps reduces fill by ~3%
        spread_effect = -0.03 * (spread_bps - 5)
        
        # Volatility adjustment: higher = more fills
        vol_effect = volatility * 50  # vol=1% → +0.5
        
        prob = base_prob + spread_effect + vol_effect
        return float(np.clip(prob, 0.05, 0.95))
    
    def optimal_spread_for_target_fill(self,
                                       target_fill: float,
                                       volatility: float,
                                       book_imbalance: float = 0,
                                       min_spread: float = 2,
                                       max_spread: float = 30) -> float:
        """
        Find optimal spread to achieve target fill rate.
        Uses binary search.
        
        Args:
            target_fill: Target fill probability (0 to 1)
            volatility: Current market volatility
            book_imbalance: Order book imbalance
            min_spread: Minimum spread to consider (bps)
            max_spread: Maximum spread to consider (bps)
            
        Returns:
            Optimal spread in basis points
        """
        low, high = min_spread, max_spread
        
        for _ in range(20):
            mid = (low + high) / 2
            fill_prob = self.predict_fill_prob(mid, volatility, book_imbalance)
            
            if fill_prob < target_fill:
                high = mid  # Need tighter spread
            else:
                low = mid  # Can widen spread
        
        return float(mid)
    
    def get_expected_pnl(self,
                         spread_bps: float,
                         volatility: float,
                         order_size_usd: float = 200) -> float:
        """
        Calculate expected PnL per order cycle.
        
        Expected PnL = fill_prob × spread revenue - (costs)
        """
        fill_prob = self.predict_fill_prob(spread_bps, volatility)
        
        # Revenue: half spread per fill (bid+ask average)
        revenue_per_fill = order_size_usd * (spread_bps / 20000)
        
        # Cost: transaction fee (assuming 5bps)
        cost_per_fill = order_size_usd * 0.0005
        
        expected_pnl = fill_prob * 2 * (revenue_per_fill - cost_per_fill)
        return float(expected_pnl)
    
    def get_stats(self) -> dict:
        """Get model statistics."""
        return {
            'samples': len(self.order_history),
            'is_fitted': self.is_fitted,
            'min_samples_needed': self.min_training_samples
        }


# Unit tests
if __name__ == "__main__":
    model = FillProbabilityModel()
    
    print("=" * 60)
    print("FILL PROBABILITY MODEL TESTS")
    print("=" * 60)
    
    # Test heuristic (before model training)
    test_cases = [
        (3, 0.005, "Tight spread, low vol"),
        (5, 0.005, "Default spread, low vol"),
        (10, 0.005, "Wide spread, low vol"),
        (5, 0.02, "Default spread, HIGH vol"),
        (15, 0.02, "Wide spread, HIGH vol"),
    ]
    
    print("\nHeuristic Fill Predictions:")
    for spread, vol, desc in test_cases:
        prob = model.predict_fill_prob(spread, vol)
        pnl = model.get_expected_pnl(spread, vol)
        print(f"  {desc}: {prob:.1%} fill, ${pnl:.3f} expected PnL")
    
    print("\nOptimal Spread for 60% fill rate:")
    for vol in [0.005, 0.01, 0.02]:
        optimal = model.optimal_spread_for_target_fill(0.6, vol)
        print(f"  Vol={vol:.1%}: {optimal:.1f} bps")
