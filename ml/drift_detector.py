"""
Drift Detector for Model Performance Monitoring
v4.0 - PSI-based Concept Drift Detection

This module detects when the model's predictions become unreliable
due to changes in the underlying data distribution (concept drift).
"""

from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime
import numpy as np
import pandas as pd


@dataclass
class DriftConfig:
    """Configuration for drift detection."""
    method: str = 'psi'  # 'psi', 'ks', 'js'
    psi_threshold: float = 0.25  # PSI > 0.25 = significant drift
    psi_warning: float = 0.10   # PSI > 0.10 = minor drift
    check_frequency_hours: int = 4
    n_bins: int = 10
    reference_window: int = 10000  # Reference distribution size
    comparison_window: int = 1000  # Recent data window


class DriftDetector:
    """
    Population Stability Index (PSI) based drift detector.
    
    PSI measures the difference between two distributions:
    - Reference: Training data or historical baseline
    - Current: Recent predictions/features
    
    PSI Interpretation:
    - < 0.10: No significant drift
    - 0.10 - 0.25: Minor drift, monitor closely
    - > 0.25: Significant drift, consider retraining
    
    Usage:
        detector = DriftDetector(config)
        detector.set_reference(training_features)
        
        # During inference
        psi, has_drift = detector.check_drift(recent_features)
        if has_drift:
            print("Concept drift detected!")
    """
    
    def __init__(self, config: Optional[DriftConfig] = None):
        self.config = config or DriftConfig()
        self.reference_distributions: Dict[str, np.ndarray] = {}
        self.bin_edges: Dict[str, np.ndarray] = {}
        self.feature_names: List[str] = []
        self.last_check_time: Optional[datetime] = None
        self.drift_history: List[Dict[str, Any]] = []
    
    def set_reference(
        self,
        X: pd.DataFrame,
        feature_subset: Optional[List[str]] = None
    ) -> None:
        """
        Set reference distribution from training data.
        
        Args:
            X: Training features DataFrame
            feature_subset: Optional subset of features to monitor
        """
        if feature_subset:
            X = X[feature_subset]
        
        self.feature_names = X.columns.tolist()
        
        for col in self.feature_names:
            values = X[col].dropna().values
            
            # Create bins
            _, edges = np.histogram(values, bins=self.config.n_bins)
            self.bin_edges[col] = edges
            
            # Store reference distribution
            counts, _ = np.histogram(values, bins=edges)
            self.reference_distributions[col] = counts / len(values)
    
    def calculate_psi(
        self,
        reference: np.ndarray,
        current: np.ndarray
    ) -> float:
        """
        Calculate Population Stability Index between two distributions.
        
        PSI = Σ (current% - reference%) * ln(current% / reference%)
        """
        # Avoid division by zero
        eps = 1e-10
        reference = np.clip(reference, eps, 1.0)
        current = np.clip(current, eps, 1.0)
        
        psi = np.sum((current - reference) * np.log(current / reference))
        return float(psi)
    
    def check_drift(
        self,
        X: pd.DataFrame,
        return_details: bool = False
    ) -> Tuple[float, bool, Optional[Dict[str, float]]]:
        """
        Check for drift in current data against reference.
        
        Args:
            X: Current features DataFrame
            return_details: Whether to return per-feature PSI
            
        Returns:
            (total_psi, has_drift, feature_psi_dict)
        """
        if not self.reference_distributions:
            raise ValueError("Reference not set. Call set_reference first.")
        
        feature_psi = {}
        total_psi = 0.0
        
        for col in self.feature_names:
            if col not in X.columns:
                continue
            
            values = X[col].dropna().values
            if len(values) == 0:
                continue
            
            # Calculate current distribution
            counts, _ = np.histogram(values, bins=self.bin_edges[col])
            current_dist = counts / len(values)
            
            # Calculate PSI
            psi = self.calculate_psi(
                self.reference_distributions[col],
                current_dist
            )
            feature_psi[col] = psi
            total_psi += psi
        
        # Average PSI across features
        avg_psi = total_psi / len(feature_psi) if feature_psi else 0.0
        has_drift = avg_psi > self.config.psi_threshold
        
        # Log drift event
        self.drift_history.append({
            'timestamp': datetime.now(),
            'psi': avg_psi,
            'has_drift': has_drift,
            'n_features': len(feature_psi),
        })
        
        self.last_check_time = datetime.now()
        
        if return_details:
            return avg_psi, has_drift, feature_psi
        return avg_psi, has_drift, None
    
    def get_drifted_features(
        self,
        X: pd.DataFrame,
        threshold: Optional[float] = None
    ) -> List[Tuple[str, float]]:
        """
        Get list of features with significant drift.
        
        Returns:
            List of (feature_name, psi_value) tuples sorted by PSI
        """
        threshold = threshold or self.config.psi_threshold
        _, _, feature_psi = self.check_drift(X, return_details=True)
        
        if feature_psi is None:
            return []
        
        drifted = [(name, psi) for name, psi in feature_psi.items() 
                   if psi > threshold]
        return sorted(drifted, key=lambda x: x[1], reverse=True)
    
    def should_check(self) -> bool:
        """Check if enough time has passed since last check."""
        if self.last_check_time is None:
            return True
        
        hours_since = (datetime.now() - self.last_check_time).total_seconds() / 3600
        return hours_since >= self.config.check_frequency_hours
    
    def get_drift_summary(self, window: int = 100) -> Dict[str, Any]:
        """Get summary of recent drift history."""
        recent = self.drift_history[-window:] if self.drift_history else []
        
        if not recent:
            return {'n_checks': 0}
        
        psi_values = [d['psi'] for d in recent]
        drift_counts = sum(1 for d in recent if d['has_drift'])
        
        return {
            'n_checks': len(recent),
            'drift_rate': drift_counts / len(recent),
            'mean_psi': np.mean(psi_values),
            'max_psi': np.max(psi_values),
            'last_check': recent[-1]['timestamp'],
        }


class AccuracyMonitor:
    """
    Monitor prediction accuracy over time.
    
    Tracks rolling accuracy and detects performance degradation.
    """
    
    def __init__(
        self,
        window_size: int = 1000,
        accuracy_threshold: float = 0.48,
        consecutive_miss_threshold: int = 10
    ):
        self.window_size = window_size
        self.accuracy_threshold = accuracy_threshold
        self.consecutive_miss_threshold = consecutive_miss_threshold
        
        self.predictions: List[Tuple[str, str]] = []  # (predicted, actual)
        self.consecutive_misses = 0
        self.degradation_alerts: List[datetime] = []
    
    def update(self, predicted: str, actual: str) -> bool:
        """
        Update with new prediction result.
        
        Returns:
            True if prediction was correct
        """
        is_correct = predicted == actual
        self.predictions.append((predicted, actual))
        
        # Maintain window size
        if len(self.predictions) > self.window_size:
            self.predictions.pop(0)
        
        # Track consecutive misses
        if is_correct:
            self.consecutive_misses = 0
        else:
            self.consecutive_misses += 1
        
        return is_correct
    
    def get_accuracy(self, window: Optional[int] = None) -> float:
        """Get rolling accuracy."""
        if not self.predictions:
            return 0.0
        
        recent = self.predictions[-window:] if window else self.predictions
        correct = sum(1 for p, a in recent if p == a)
        return correct / len(recent)
    
    def is_degraded(self) -> bool:
        """Check if model performance is degraded."""
        if len(self.predictions) < 100:
            return False
        
        return self.get_accuracy() < self.accuracy_threshold
    
    def has_consecutive_misses(self) -> bool:
        """Check if consecutive miss threshold exceeded."""
        return self.consecutive_misses >= self.consecutive_miss_threshold
    
    def get_class_accuracies(self) -> Dict[str, float]:
        """Get per-class accuracy."""
        if not self.predictions:
            return {}
        
        class_correct: Dict[str, int] = {}
        class_total: Dict[str, int] = {}
        
        for predicted, actual in self.predictions:
            class_total[actual] = class_total.get(actual, 0) + 1
            if predicted == actual:
                class_correct[actual] = class_correct.get(actual, 0) + 1
        
        return {
            cls: class_correct.get(cls, 0) / total
            for cls, total in class_total.items()
        }
    
    def get_summary(self) -> Dict[str, Any]:
        """Get monitoring summary."""
        return {
            'total_predictions': len(self.predictions),
            'accuracy': self.get_accuracy(),
            'accuracy_100': self.get_accuracy(100),
            'consecutive_misses': self.consecutive_misses,
            'is_degraded': self.is_degraded(),
            'class_accuracies': self.get_class_accuracies(),
        }


class PerformanceMonitor:
    """
    Combined performance monitoring with drift detection and accuracy tracking.
    """
    
    def __init__(
        self,
        drift_config: Optional[DriftConfig] = None,
        window_size: int = 1000,
        accuracy_threshold: float = 0.48
    ):
        self.drift_detector = DriftDetector(drift_config)
        self.accuracy_monitor = AccuracyMonitor(
            window_size=window_size,
            accuracy_threshold=accuracy_threshold
        )
        self.pnl_contribution: List[float] = []
    
    def set_reference(self, X: pd.DataFrame) -> None:
        """Set reference distribution for drift detection."""
        self.drift_detector.set_reference(X)
    
    def update_prediction(self, predicted: str, actual: str) -> None:
        """Update with prediction result."""
        self.accuracy_monitor.update(predicted, actual)
    
    def update_pnl(self, pnl: float) -> None:
        """Update PnL contribution."""
        self.pnl_contribution.append(pnl)
        if len(self.pnl_contribution) > 1000:
            self.pnl_contribution.pop(0)
    
    def check_health(self, current_features: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Comprehensive health check.
        
        Returns:
            Health status dict with recommendations
        """
        health = {
            'is_healthy': True,
            'issues': [],
            'recommendations': [],
        }
        
        # Check accuracy
        acc_summary = self.accuracy_monitor.get_summary()
        health['accuracy'] = acc_summary
        
        if acc_summary['is_degraded']:
            health['is_healthy'] = False
            health['issues'].append('Model accuracy degraded')
            health['recommendations'].append('Consider retraining model')
        
        if self.accuracy_monitor.has_consecutive_misses():
            health['is_healthy'] = False
            health['issues'].append('Consecutive prediction misses')
            health['recommendations'].append('Reduce model weight temporarily')
        
        # Check drift
        if current_features is not None and self.drift_detector.should_check():
            psi, has_drift, _ = self.drift_detector.check_drift(current_features)
            health['drift'] = {'psi': psi, 'has_drift': has_drift}
            
            if has_drift:
                health['is_healthy'] = False
                health['issues'].append(f'Concept drift detected (PSI={psi:.3f})')
                health['recommendations'].append('Retrain model with recent data')
        
        # Check PnL
        if self.pnl_contribution:
            total_pnl = sum(self.pnl_contribution)
            health['pnl'] = {
                'total': total_pnl,
                'recent_100': sum(self.pnl_contribution[-100:]) if len(self.pnl_contribution) >= 100 else None
            }
            
            if total_pnl < 0:
                health['issues'].append('Negative PnL contribution')
        
        return health
