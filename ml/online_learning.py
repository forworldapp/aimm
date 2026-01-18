"""
Online Learning Module - Phase 4.2
Enables incremental model updates without full retraining.

Author: Antigravity
Version: 1.0.0
"""

import numpy as np
from typing import Any, Optional, Callable
import logging


class OnlineModelUpdater:
    """
    Online Learning wrapper for ML models.
    
    Enables incremental updates without full batch retraining.
    
    Features:
    - Learning rate decay
    - Drift detection
    - Buffer for mini-batch updates
    """
    
    def __init__(self, 
                 model: Any,
                 base_learning_rate: float = 0.01,
                 decay_rate: float = 0.001,
                 buffer_size: int = 32):
        """
        Args:
            model: sklearn-compatible model with partial_fit or custom update
            base_learning_rate: Initial learning rate
            decay_rate: Learning rate decay per update
            buffer_size: Mini-batch size for updates
        """
        self.model = model
        self.base_lr = base_learning_rate
        self.decay_rate = decay_rate
        self.buffer_size = buffer_size
        
        self.n_updates = 0
        self.current_lr = base_learning_rate
        
        # Buffer for mini-batch updates
        self.X_buffer = []
        self.y_buffer = []
        
        # Performance tracking
        self.recent_errors = []
        self.drift_detected = False
        
        self.logger = logging.getLogger("OnlineUpdater")
    
    def update(self, X: np.ndarray, y: np.ndarray, force: bool = False):
        """
        Add observation to buffer and potentially trigger update.
        
        Args:
            X: Feature vector(s)
            y: Target value(s)
            force: Force immediate update regardless of buffer size
        """
        # Add to buffer
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if np.isscalar(y):
            y = np.array([y])
        
        self.X_buffer.extend(X)
        self.y_buffer.extend(y)
        
        # Check if buffer full or forced
        if len(self.X_buffer) >= self.buffer_size or force:
            self._perform_update()
    
    def _perform_update(self):
        """Perform actual model update."""
        if not self.X_buffer:
            return
        
        X = np.array(self.X_buffer)
        y = np.array(self.y_buffer)
        
        # Decay learning rate
        self.current_lr = self.base_lr / (1 + self.n_updates * self.decay_rate)
        
        # Check if model supports partial_fit
        if hasattr(self.model, 'partial_fit'):
            self.model.partial_fit(X, y)
        elif hasattr(self.model, 'set_params') and hasattr(self.model, 'fit'):
            # For models without partial_fit, use warm start if available
            try:
                self.model.set_params(warm_start=True)
                self.model.fit(X, y)
            except:
                self.logger.warning("Model doesn't support warm_start")
        else:
            self.logger.warning("Model doesn't support online learning")
        
        self.n_updates += 1
        
        # Clear buffer
        self.X_buffer = []
        self.y_buffer = []
        
        self.logger.debug(f"Online update #{self.n_updates}, lr={self.current_lr:.6f}")
    
    def detect_drift(self, error: float, threshold: float = 2.0) -> bool:
        """
        Detect concept drift from prediction errors.
        
        Uses simple running statistics comparison.
        
        Args:
            error: Current prediction error
            threshold: Number of standard deviations for drift
            
        Returns:
            True if drift detected
        """
        self.recent_errors.append(error)
        
        # Need enough history
        if len(self.recent_errors) < 100:
            return False
        
        # Keep last 1000 errors
        if len(self.recent_errors) > 1000:
            self.recent_errors = self.recent_errors[-1000:]
        
        # Compare recent vs historical
        recent = self.recent_errors[-20:]
        historical = self.recent_errors[-100:-20]
        
        recent_mean = np.mean(recent)
        hist_mean = np.mean(historical)
        hist_std = np.std(historical)
        
        if hist_std < 1e-10:
            return False
        
        z_score = abs(recent_mean - hist_mean) / hist_std
        
        self.drift_detected = z_score > threshold
        
        if self.drift_detected:
            self.logger.warning(f"Drift detected! z-score={z_score:.2f}")
            # Reset learning rate for faster adaptation
            self.current_lr = self.base_lr * 2
        
        return self.drift_detected
    
    def reset_learning_rate(self, new_lr: Optional[float] = None):
        """Reset learning rate (e.g., after drift)."""
        if new_lr is not None:
            self.current_lr = new_lr
        else:
            self.current_lr = self.base_lr
        self.n_updates = 0
    
    def get_stats(self) -> dict:
        """Get updater statistics."""
        return {
            'n_updates': self.n_updates,
            'current_lr': float(self.current_lr),
            'buffer_size': len(self.X_buffer),
            'drift_detected': self.drift_detected,
            'recent_avg_error': float(np.mean(self.recent_errors[-20:])) if self.recent_errors else 0
        }


class IncrementalMean:
    """
    Incremental mean and variance calculation.
    
    Uses Welford's online algorithm for numerical stability.
    """
    
    def __init__(self):
        self.n = 0
        self.mean = 0.0
        self.M2 = 0.0  # Sum of squared differences
    
    def update(self, x: float):
        """Add new observation."""
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self.M2 += delta * delta2
    
    @property
    def variance(self) -> float:
        """Get current variance."""
        if self.n < 2:
            return 0.0
        return self.M2 / (self.n - 1)
    
    @property
    def std(self) -> float:
        """Get current standard deviation."""
        return np.sqrt(self.variance)
    
    def get_stats(self) -> dict:
        return {
            'n': self.n,
            'mean': float(self.mean),
            'std': float(self.std)
        }


class ExponentialMovingStats:
    """
    Exponential moving average and variance.
    
    More weight on recent observations.
    """
    
    def __init__(self, alpha: float = 0.1):
        """
        Args:
            alpha: Smoothing factor (0.1 = 10% weight on new, 90% on old)
        """
        self.alpha = alpha
        self.mean = None
        self.var = None
        self.n = 0
    
    def update(self, x: float):
        """Add new observation."""
        self.n += 1
        
        if self.mean is None:
            self.mean = x
            self.var = 0.0
        else:
            diff = x - self.mean
            self.mean += self.alpha * diff
            self.var = (1 - self.alpha) * (self.var + self.alpha * diff ** 2)
    
    @property
    def std(self) -> float:
        return np.sqrt(self.var) if self.var else 0.0
    
    def get_stats(self) -> dict:
        return {
            'n': self.n,
            'mean': float(self.mean) if self.mean else 0,
            'std': float(self.std)
        }


# Unit tests
if __name__ == "__main__":
    print("=" * 60)
    print("ONLINE LEARNING MODULE TESTS")
    print("=" * 60)
    
    # Test IncrementalMean
    print("\n1. IncrementalMean Test")
    stats = IncrementalMean()
    data = [10, 12, 15, 11, 13, 14, 12]
    
    for x in data:
        stats.update(x)
    
    print(f"   Data: {data}")
    print(f"   Incremental Mean: {stats.mean:.2f} (actual: {np.mean(data):.2f})")
    print(f"   Incremental Std: {stats.std:.2f} (actual: {np.std(data, ddof=1):.2f})")
    
    # Test ExponentialMovingStats
    print("\n2. ExponentialMovingStats Test")
    ema = ExponentialMovingStats(alpha=0.2)
    
    for x in data:
        ema.update(x)
    
    print(f"   EMA: {ema.mean:.2f}, EMA Std: {ema.std:.2f}")
    
    # Test Drift Detection
    print("\n3. Drift Detection Test")
    
    class DummyModel:
        def partial_fit(self, X, y):
            pass
    
    updater = OnlineModelUpdater(DummyModel(), buffer_size=10)
    
    # Normal errors for a while
    for i in range(80):
        updater.update(np.random.randn(5), np.random.randn())
        updater.detect_drift(np.random.randn() * 0.5)
    
    # Suddenly larger errors (drift)
    for i in range(25):
        updater.update(np.random.randn(5), np.random.randn())
        drift = updater.detect_drift(np.random.randn() * 2 + 3)  # Shifted mean
    
    print(f"   Drift detected: {updater.drift_detected}")
    print(f"   Stats: {updater.get_stats()}")
