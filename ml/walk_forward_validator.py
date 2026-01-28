"""
Walk-Forward Validator for Time Series Cross-Validation
v4.0 - No Future Data Leakage

This module provides walk-forward validation which is essential for
evaluating ML models on time series data without future information leakage.
"""

from typing import Iterator, List, Tuple, Optional, Generator
from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass
class WalkForwardConfig:
    """Configuration for walk-forward validation."""
    n_splits: int = 5
    train_period_days: int = 60
    test_period_days: int = 7
    gap_days: int = 1  # Gap between train and test to prevent leakage
    min_train_size: int = 1000  # Minimum training samples


class WalkForwardValidator:
    """
    Walk-Forward (Rolling Window) Cross-Validator for Time Series.
    
    This validator ensures no future information leakage by:
    1. Always training on past data only
    2. Adding a gap between train and test sets
    3. Rolling forward through time
    
    Example usage:
        validator = WalkForwardValidator(n_splits=5, train_period_days=60)
        for train_idx, test_idx in validator.split(X):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            model.fit(X_train, y_train)
            score = model.score(X_test, y_test)
    """
    
    def __init__(
        self,
        n_splits: int = 5,
        train_period_days: int = 60,
        test_period_days: int = 7,
        gap_days: int = 1,
        periods_per_day: int = 1440,  # For 1-minute bars
        min_train_size: int = 1000
    ):
        """
        Initialize walk-forward validator.
        
        Args:
            n_splits: Number of validation splits
            train_period_days: Training window size in days
            test_period_days: Test window size in days
            gap_days: Gap between train and test to prevent leakage
            periods_per_day: Number of periods per day (1440 for 1-min bars)
            min_train_size: Minimum required training samples
        """
        self.n_splits = n_splits
        self.train_period = train_period_days * periods_per_day
        self.test_period = test_period_days * periods_per_day
        self.gap = gap_days * periods_per_day
        self.periods_per_day = periods_per_day
        self.min_train_size = min_train_size
    
    def split(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
        groups: Optional[np.ndarray] = None
    ) -> Generator[Tuple[List[int], List[int]], None, None]:
        """
        Generate train/test indices for each split.
        
        Args:
            X: Feature DataFrame (used for size only)
            y: Target Series (ignored, for sklearn compatibility)
            groups: Group labels (ignored, for sklearn compatibility)
            
        Yields:
            Tuple of (train_indices, test_indices)
        """
        n_samples = len(X)
        
        # Calculate total size needed for splits
        total_test = self.n_splits * self.test_period
        min_required = self.min_train_size + self.gap + total_test
        
        if n_samples < min_required:
            raise ValueError(
                f"Not enough samples: {n_samples} < {min_required} "
                f"(min_train={self.min_train_size}, gap={self.gap}, "
                f"test={self.test_period}*{self.n_splits})"
            )
        
        # Calculate where each test fold ends
        # Work backwards from the end of data
        for i in range(self.n_splits):
            # Test end is at most at the end of data, but we need room for all folds
            test_end = n_samples - (self.n_splits - 1 - i) * self.test_period
            test_start = test_end - self.test_period
            
            # Gap between train and test
            train_end = test_start - self.gap
            train_start = max(0, train_end - self.train_period)
            
            # Skip if not enough training data
            if train_end - train_start < self.min_train_size:
                continue
            
            # Ensure valid ranges
            if train_start >= train_end or test_start >= test_end:
                continue
                
            train_indices = list(range(train_start, train_end))
            test_indices = list(range(test_start, min(test_end, n_samples)))
            
            yield train_indices, test_indices
    
    def get_n_splits(
        self,
        X: Optional[pd.DataFrame] = None,
        y: Optional[pd.Series] = None,
        groups: Optional[np.ndarray] = None
    ) -> int:
        """Get number of splits (sklearn compatibility)."""
        return self.n_splits
    
    def get_split_info(self, n_samples: int) -> List[dict]:
        """
        Get detailed information about each split for debugging.
        
        Args:
            n_samples: Total number of samples
            
        Returns:
            List of dicts with split information
        """
        info = []
        dummy_X = pd.DataFrame(index=range(n_samples))
        
        for i, (train_idx, test_idx) in enumerate(self.split(dummy_X)):
            info.append({
                'fold': i,
                'train_start': train_idx[0],
                'train_end': train_idx[-1],
                'train_size': len(train_idx),
                'test_start': test_idx[0],
                'test_end': test_idx[-1],
                'test_size': len(test_idx),
                'gap': test_idx[0] - train_idx[-1] - 1
            })
        
        return info


class ExpandingWindowValidator:
    """
    Expanding Window Cross-Validator for Time Series.
    
    Unlike walk-forward, this validator uses all available past data
    for training, expanding the window as we move forward.
    
    Useful when you want to use as much historical data as possible.
    """
    
    def __init__(
        self,
        n_splits: int = 5,
        test_period_days: int = 7,
        gap_days: int = 1,
        periods_per_day: int = 1440,
        min_train_size: int = 1000
    ):
        self.n_splits = n_splits
        self.test_period = test_period_days * periods_per_day
        self.gap = gap_days * periods_per_day
        self.periods_per_day = periods_per_day
        self.min_train_size = min_train_size
    
    def split(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
        groups: Optional[np.ndarray] = None
    ) -> Generator[Tuple[List[int], List[int]], None, None]:
        """Generate train/test indices with expanding training window."""
        n_samples = len(X)
        
        for i in range(self.n_splits):
            test_end = n_samples - (self.n_splits - 1 - i) * self.test_period
            test_start = test_end - self.test_period
            
            # Expanding window: train from beginning up to gap before test
            train_end = test_start - self.gap
            train_start = 0
            
            if train_end - train_start < self.min_train_size:
                continue
            
            train_indices = list(range(train_start, train_end))
            test_indices = list(range(test_start, min(test_end, n_samples)))
            
            yield train_indices, test_indices
    
    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return self.n_splits


class PurgedKFoldValidator:
    """
    Purged K-Fold Cross-Validator for Time Series.
    
    This is a k-fold validator with purging and embargo periods
    to prevent information leakage in overlapping label scenarios.
    
    Recommended for cases where labels span multiple periods
    (e.g., 15-minute return labels on 1-minute data).
    """
    
    def __init__(
        self,
        n_splits: int = 5,
        purge_periods: int = 10,  # Periods to purge around test set
        embargo_periods: int = 5,  # Periods to embargo after test set
        periods_per_day: int = 1440
    ):
        self.n_splits = n_splits
        self.purge = purge_periods
        self.embargo = embargo_periods
        self.periods_per_day = periods_per_day
    
    def split(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
        groups: Optional[np.ndarray] = None
    ) -> Generator[Tuple[List[int], List[int]], None, None]:
        """Generate purged train/test indices."""
        n_samples = len(X)
        indices = np.arange(n_samples)
        
        fold_sizes = np.full(self.n_splits, n_samples // self.n_splits, dtype=int)
        fold_sizes[:n_samples % self.n_splits] += 1
        
        current = 0
        for fold_size in fold_sizes:
            test_start = current
            test_end = current + fold_size
            
            # Create purged training indices
            train_mask = np.ones(n_samples, dtype=bool)
            
            # Purge: remove data points too close to test set
            purge_start = max(0, test_start - self.purge)
            purge_end = min(n_samples, test_end + self.embargo)
            train_mask[purge_start:purge_end] = False
            
            train_indices = indices[train_mask].tolist()
            test_indices = indices[test_start:test_end].tolist()
            
            current = test_end
            
            yield train_indices, test_indices
    
    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return self.n_splits


def validate_no_leakage(
    train_idx: List[int],
    test_idx: List[int],
    gap_required: int = 1
) -> bool:
    """
    Validate that there is no time leakage between train and test sets.
    
    Args:
        train_idx: Training indices
        test_idx: Test indices
        gap_required: Minimum required gap between sets
        
    Returns:
        True if no leakage, raises ValueError otherwise
    """
    train_max = max(train_idx)
    test_min = min(test_idx)
    
    actual_gap = test_min - train_max - 1
    
    if actual_gap < gap_required:
        raise ValueError(
            f"Potential data leakage: gap={actual_gap} < required={gap_required}. "
            f"Train max={train_max}, Test min={test_min}"
        )
    
    # Check for overlap
    overlap = set(train_idx) & set(test_idx)
    if overlap:
        raise ValueError(f"Train/test overlap detected: {len(overlap)} samples")
    
    return True


def create_validator(
    method: str = 'walk_forward',
    **kwargs
) -> WalkForwardValidator:
    """
    Factory function to create appropriate validator.
    
    Args:
        method: 'walk_forward', 'expanding', or 'purged_kfold'
        **kwargs: Arguments passed to validator constructor
        
    Returns:
        Validator instance
    """
    validators = {
        'walk_forward': WalkForwardValidator,
        'expanding': ExpandingWindowValidator,
        'purged_kfold': PurgedKFoldValidator,
    }
    
    if method not in validators:
        raise ValueError(f"Unknown method: {method}. Available: {list(validators.keys())}")
    
    return validators[method](**kwargs)
