"""
Adverse Selection Detection - Phase 1.2
Detects informed traders exploiting stale quotes and adjusts strategy accordingly.

Author: Antigravity
Version: 1.0.0
"""

import numpy as np
from collections import deque
from typing import Tuple, Dict, List, Optional
import logging
import time

try:
    from sklearn.ensemble import RandomForestClassifier
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    RandomForestClassifier = None


class AdverseSelectionDetector:
    """
    Adverse Selection Detection and Response System.
    
    Detects informed traders by analyzing:
    1. Price movement after fills (adverse = price moves against us)
    2. Trade velocity (fast consecutive fills = likely informed)
    3. Trade size patterns (large sizes = likely informed)
    4. Order book imbalance (unidirectional flow)
    
    Response Levels:
    - AS Prob < 50%: Normal operation
    - AS Prob 50-70%: +3bps spread
    - AS Prob 70-85%: +7bps spread, -30% size
    - AS Prob > 85%: Pause quoting 2 seconds
    """
    
    def __init__(self, 
                 history_size: int = 500,
                 label_delay_seconds: float = 5.0,
                 adverse_threshold: float = 0.0005):
        """
        Args:
            history_size: Max trades to keep in history
            label_delay_seconds: Time to wait before labeling trade as adverse
            adverse_threshold: Price change threshold for adverse classification (0.05%)
        """
        self.history_size = history_size
        self.label_delay_seconds = label_delay_seconds
        self.adverse_threshold = adverse_threshold
        
        self.trade_history: deque = deque(maxlen=history_size)
        self.pending_labels: Dict[str, dict] = {}
        self.training_data: List[dict] = []
        
        # ML Model
        self.model = None
        self.is_fitted = False
        self.min_training_samples = 50
        
        # Statistics
        self.total_trades = 0
        self.adverse_trades = 0
        
        # Heuristic parameters (used before ML is trained)
        self.size_zscore_threshold = 2.0
        self.velocity_threshold = 0.5  # seconds
        
        self.logger = logging.getLogger("AdverseSelection")
    
    def extract_features(self, trade: dict) -> np.ndarray:
        """
        Extract features for adverse selection prediction.
        
        Features:
        1. trade_size: Normalized trade size
        2. time_since_last: Seconds since last trade
        3. book_imbalance: Order book imbalance (-1 to 1)
        4. trade_velocity: Number of trades in last 10 seconds
        5. size_zscore: Z-score of trade size vs history
        6. directional_flow: Net direction of recent trades (-1 to 1)
        """
        # Calculate derived features
        time_since_last = self._calc_time_since_last()
        trade_velocity = self._calc_trade_velocity()
        size_zscore = self._calc_size_zscore(trade.get('size', 0))
        directional_flow = self._calc_directional_flow()
        
        features = np.array([
            trade.get('size', 0),
            time_since_last,
            trade.get('book_imbalance', 0),
            trade_velocity,
            size_zscore,
            directional_flow
        ])
        
        return features
    
    def _calc_time_since_last(self) -> float:
        """Calculate time since last trade."""
        if len(self.trade_history) < 2:
            return 10.0  # Default: long time
        
        last_trade = self.trade_history[-1]
        prev_trade = self.trade_history[-2]
        
        return last_trade.get('timestamp', 0) - prev_trade.get('timestamp', 0)
    
    def _calc_trade_velocity(self) -> int:
        """Count trades in last 10 seconds."""
        if not self.trade_history:
            return 0
        
        now = self.trade_history[-1].get('timestamp', time.time())
        count = 0
        for trade in reversed(self.trade_history):
            if now - trade.get('timestamp', 0) <= 10:
                count += 1
            else:
                break
        return count
    
    def _calc_size_zscore(self, size: float) -> float:
        """Calculate z-score of trade size."""
        if len(self.trade_history) < 10:
            return 0.0
        
        sizes = [t.get('size', 0) for t in self.trade_history]
        mean = np.mean(sizes)
        std = np.std(sizes)
        
        if std < 1e-10:
            return 0.0
        
        return (size - mean) / std
    
    def _calc_directional_flow(self) -> float:
        """Calculate net direction of recent trades (-1 to 1)."""
        if len(self.trade_history) < 5:
            return 0.0
        
        recent = list(self.trade_history)[-10:]
        buys = sum(1 for t in recent if t.get('side') == 'buy')
        sells = len(recent) - buys
        
        if len(recent) == 0:
            return 0.0
        
        return (buys - sells) / len(recent)
    
    def predict(self, trade: dict) -> Tuple[bool, float]:
        """
        Predict if trade shows adverse selection.
        
        Args:
            trade: Trade dict with size, side, timestamp, book_imbalance
            
        Returns:
            Tuple of (is_adverse, probability)
        """
        features = self.extract_features(trade)
        
        # Use ML model if trained
        if self.is_fitted and self.model is not None:
            prob = self.model.predict_proba(features.reshape(1, -1))[0, 1]
            return prob > 0.5, prob
        
        # Fallback: Heuristic prediction
        return self._heuristic_predict(features)
    
    def _heuristic_predict(self, features: np.ndarray) -> Tuple[bool, float]:
        """
        Heuristic prediction when ML model not available.
        """
        size_zscore = features[4]
        time_since_last = features[1]
        trade_velocity = features[3]
        
        # Simple heuristic: high score if large size + fast velocity
        risk_score = 0.0
        
        # Large trade size
        if size_zscore > self.size_zscore_threshold:
            risk_score += 0.3
        elif size_zscore > self.size_zscore_threshold * 0.5:
            risk_score += 0.15
        
        # Fast consecutive trades
        if time_since_last < self.velocity_threshold:
            risk_score += 0.25
        elif time_since_last < self.velocity_threshold * 2:
            risk_score += 0.1
        
        # High trade velocity
        if trade_velocity > 5:
            risk_score += 0.2
        elif trade_velocity > 3:
            risk_score += 0.1
        
        risk_score = min(risk_score, 1.0)
        is_adverse = risk_score > 0.5
        
        return is_adverse, risk_score
    
    def get_adjustment(self, probability: float) -> dict:
        """
        Get strategy adjustment based on AS probability.
        
        Returns dict with:
        - spread_add_bps: Additional spread in basis points
        - size_mult: Size multiplier
        - pause_seconds: Quoting pause duration
        """
        if probability < 0.5:
            return {'spread_add_bps': 0, 'size_mult': 1.0, 'pause_seconds': 0}
        elif probability < 0.7:
            return {'spread_add_bps': 3, 'size_mult': 1.0, 'pause_seconds': 0}
        elif probability < 0.85:
            return {'spread_add_bps': 7, 'size_mult': 0.7, 'pause_seconds': 0}
        else:
            return {'spread_add_bps': 10, 'size_mult': 0.5, 'pause_seconds': 2}
    
    def record_trade(self, trade_id: str, trade: dict):
        """
        Record a trade for analysis.
        
        Args:
            trade_id: Unique trade identifier
            trade: Trade dict with size, side, price, timestamp
        """
        self.total_trades += 1
        self.trade_history.append(trade)
        
        # Add to pending labels (will be labeled after 5 seconds)
        features = self.extract_features(trade)
        self.pending_labels[trade_id] = {
            'features': features,
            'price': trade.get('price', 0),
            'side': trade.get('side', 'buy'),
            'timestamp': trade.get('timestamp', time.time())
        }
    
    def label_trade(self, trade_id: str, price_after: float):
        """
        Label a trade as adverse or not based on price movement.
        
        Args:
            trade_id: Trade ID from record_trade
            price_after: Price observed after label_delay_seconds
        """
        if trade_id not in self.pending_labels:
            return
        
        info = self.pending_labels.pop(trade_id)
        entry_price = info['price']
        side = info['side']
        
        # Calculate price change
        price_change = (price_after - entry_price) / entry_price
        
        # Adverse selection:
        # - We bought, price dropped → adverse
        # - We sold, price rose → adverse
        if side == 'buy':
            is_adverse = price_change < -self.adverse_threshold
        else:
            is_adverse = price_change > self.adverse_threshold
        
        if is_adverse:
            self.adverse_trades += 1
        
        # Store for training
        self.training_data.append({
            'features': info['features'],
            'label': int(is_adverse)
        })
        
        # Retrain model periodically
        if len(self.training_data) >= self.min_training_samples:
            if len(self.training_data) % 20 == 0:
                self._train_model()
    
    def _train_model(self):
        """Train the ML model on collected data."""
        if not SKLEARN_AVAILABLE:
            self.logger.warning("sklearn not available, using heuristics only")
            return
        
        if len(self.training_data) < self.min_training_samples:
            return
        
        X = np.array([d['features'] for d in self.training_data])
        y = np.array([d['label'] for d in self.training_data])
        
        # Need both classes
        if len(set(y)) < 2:
            self.logger.warning("Not enough class diversity for training")
            return
        
        self.model = RandomForestClassifier(
            n_estimators=50,
            max_depth=5,
            random_state=42
        )
        self.model.fit(X, y)
        self.is_fitted = True
        
        self.logger.info(f"AS Model trained on {len(self.training_data)} samples, "
                        f"Adverse rate: {self.adverse_trades/max(1,self.total_trades):.1%}")
    
    def get_stats(self) -> dict:
        """Get detector statistics."""
        return {
            'total_trades': self.total_trades,
            'adverse_trades': self.adverse_trades,
            'adverse_rate': self.adverse_trades / max(1, self.total_trades),
            'is_ml_fitted': self.is_fitted,
            'training_samples': len(self.training_data)
        }


# Unit tests
if __name__ == "__main__":
    detector = AdverseSelectionDetector()
    
    print("=" * 60)
    print("ADVERSE SELECTION DETECTOR TESTS")
    print("=" * 60)
    
    # Simulate trades
    trades = [
        {'size': 0.1, 'side': 'buy', 'price': 100000, 'timestamp': 1.0, 'book_imbalance': 0},
        {'size': 0.1, 'side': 'buy', 'price': 100010, 'timestamp': 2.0, 'book_imbalance': 0},
        {'size': 0.5, 'side': 'buy', 'price': 100020, 'timestamp': 2.3, 'book_imbalance': 0.3},  # Large, fast
        {'size': 0.1, 'side': 'sell', 'price': 99990, 'timestamp': 5.0, 'book_imbalance': -0.2},
        {'size': 2.0, 'side': 'buy', 'price': 99950, 'timestamp': 5.2, 'book_imbalance': 0.5},  # Very large
    ]
    
    for i, trade in enumerate(trades):
        detector.record_trade(f"trade_{i}", trade)
        is_as, prob = detector.predict(trade)
        adj = detector.get_adjustment(prob)
        
        print(f"\nTrade {i}: size={trade['size']}, side={trade['side']}")
        print(f"  AS Probability: {prob:.1%}")
        print(f"  Adjustment: +{adj['spread_add_bps']}bps, {adj['size_mult']:.0%} size")
    
    print("\n" + "=" * 60)
    print("Stats:", detector.get_stats())
