"""
LightGBM Direction Predictor
v4.0 - Modular ML Predictor for Multi-Exchange Portability

This module implements the LightGBM direction predictor with:
- Exchange-agnostic design
- Risk-aware prediction with confidence thresholds
- Fallback mechanisms
- Performance monitoring hooks
"""

from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from pathlib import Path
import pickle
import json
import numpy as np
import pandas as pd
import lightgbm as lgb
from datetime import datetime


@dataclass
class PredictorConfig:
    """Configuration for LightGBM predictor."""
    # Model settings
    model_path: str = 'data/direction_model_lgb.pkl'
    
    # Prediction settings
    prediction_horizon: int = 1  # minutes ahead
    confidence_threshold: float = 0.55  # Min confidence for directional signal
    neutral_zone: Tuple[float, float] = (0.45, 0.55)
    
    # Skew settings
    skew_multiplier: float = 1.0
    size_adjustment: bool = True
    layer_asymmetry: bool = True
    
    # Risk limits
    max_size_multiplier: float = 1.5
    min_size_multiplier: float = 0.5
    max_layer_asymmetry: Tuple[int, int] = (7, 3)
    
    # Fallback settings
    fallback_on_error: str = 'neutral'  # 'neutral' or 'hmm_only'
    fallback_on_low_confidence: str = 'neutral'


@dataclass
class Prediction:
    """Container for prediction results."""
    direction: str  # 'UP', 'DOWN', 'NEUTRAL'
    confidence: float  # 0.0 to 1.0
    probabilities: Dict[str, float]  # {'UP': 0.4, 'NEUTRAL': 0.3, 'DOWN': 0.3}
    
    # Derived trading signals
    size_multiplier: float = 1.0
    bid_layers: int = 5
    ask_layers: int = 5
    spread_adjustment: float = 0.0  # Percentage adjustment
    
    # Metadata
    timestamp: Optional[datetime] = None
    model_version: Optional[str] = None
    is_fallback: bool = False


class LightGBMPredictor:
    """
    LightGBM Direction Predictor for Market Making.
    
    Predicts short-term price direction (UP/DOWN/NEUTRAL) with confidence
    scores and translates predictions into trading signals.
    
    Usage:
        predictor = LightGBMPredictor(config)
        predictor.load_model('data/direction_model_lgb.pkl')
        
        features = feature_engineer.compute_features(ohlcv)
        prediction = predictor.predict(features.iloc[-1])
        
        print(f"Direction: {prediction.direction} ({prediction.confidence:.1%})")
        print(f"Size multiplier: {prediction.size_multiplier:.2f}")
    """
    
    # Class label mapping
    LABEL_MAP = {0: 'DOWN', 1: 'NEUTRAL', 2: 'UP'}
    REVERSE_MAP = {'DOWN': 0, 'NEUTRAL': 1, 'UP': 2}
    
    def __init__(self, config: Optional[PredictorConfig] = None):
        """Initialize predictor with configuration."""
        self.config = config or PredictorConfig()
        self.model: Optional[lgb.Booster] = None
        self.feature_names: List[str] = []
        self.model_version: Optional[str] = None
        self.metadata: Dict[str, Any] = {}
        
        # Performance tracking
        self._prediction_count = 0
        self._correct_predictions = 0
        self._recent_predictions: List[Tuple[str, str]] = []  # (predicted, actual)
    
    def load_model(self, path: Optional[str] = None) -> bool:
        """
        Load trained model from disk.
        
        Args:
            path: Path to model pickle file
            
        Returns:
            True if successful, False otherwise
        """
        model_path = Path(path or self.config.model_path)
        
        try:
            with open(model_path, 'rb') as f:
                data = pickle.load(f)
            
            if isinstance(data, dict):
                # New format with metadata
                self.model = data['model']
                self.feature_names = data.get('feature_names', [])
                self.model_version = data.get('version', 'unknown')
                self.metadata = data.get('metadata', {})
            else:
                # Legacy format: just the model
                self.model = data
                self.feature_names = []
                self.model_version = 'legacy'
            
            return True
            
        except FileNotFoundError:
            print(f"Model not found at {model_path}")
            return False
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def save_model(self, path: Optional[str] = None) -> bool:
        """
        Save model to disk with metadata.
        
        Args:
            path: Path to save model
            
        Returns:
            True if successful
        """
        if self.model is None:
            raise ValueError("No model to save")
        
        model_path = Path(path or self.config.model_path)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            'model': self.model,
            'feature_names': self.feature_names,
            'version': self.model_version or datetime.now().strftime('%Y%m%d_%H%M%S'),
            'metadata': self.metadata,
        }
        
        with open(model_path, 'wb') as f:
            pickle.dump(data, f)
        
        return True
    
    def predict(
        self,
        features: Union[pd.Series, pd.DataFrame, np.ndarray],
        hmm_regime: Optional[str] = None
    ) -> Prediction:
        """
        Make direction prediction for given features.
        
        Args:
            features: Feature vector or DataFrame row
            hmm_regime: Optional HMM regime for conflict resolution
            
        Returns:
            Prediction object with direction, confidence, and trading signals
        """
        if self.model is None:
            return self._fallback_prediction("Model not loaded")
        
        try:
            # Prepare features
            X = self._prepare_features(features)
            
            # Get probabilities
            probs = self.model.predict(X)[0]
            
            # Convert to dict
            prob_dict = {
                'DOWN': probs[0],
                'NEUTRAL': probs[1],
                'UP': probs[2]
            }
            
            # Determine direction and confidence
            direction = max(prob_dict, key=prob_dict.get)
            confidence = prob_dict[direction]
            
            # Apply confidence threshold
            if confidence < self.config.confidence_threshold:
                direction = 'NEUTRAL'
                confidence = prob_dict['NEUTRAL']
            
            # Resolve conflict with HMM if provided
            if hmm_regime and direction != 'NEUTRAL':
                direction, confidence = self._resolve_hmm_conflict(
                    hmm_regime, direction, confidence
                )
            
            # Calculate trading signals
            size_mult, bid_layers, ask_layers, spread_adj = self._calculate_signals(
                direction, confidence
            )
            
            return Prediction(
                direction=direction,
                confidence=confidence,
                probabilities=prob_dict,
                size_multiplier=size_mult,
                bid_layers=bid_layers,
                ask_layers=ask_layers,
                spread_adjustment=spread_adj,
                timestamp=datetime.now(),
                model_version=self.model_version,
                is_fallback=False
            )
            
        except Exception as e:
            return self._fallback_prediction(str(e))
    
    def predict_batch(
        self,
        features: pd.DataFrame,
        hmm_regimes: Optional[List[str]] = None
    ) -> List[Prediction]:
        """Make predictions for multiple samples."""
        predictions = []
        regimes = hmm_regimes or [None] * len(features)
        
        for i, (_, row) in enumerate(features.iterrows()):
            pred = self.predict(row, hmm_regimes[i] if hmm_regimes else None)
            predictions.append(pred)
        
        return predictions
    
    def update_accuracy(self, predicted: str, actual: str) -> None:
        """
        Update accuracy tracking with actual outcome.
        
        Args:
            predicted: Predicted direction
            actual: Actual direction that occurred
        """
        self._prediction_count += 1
        if predicted == actual:
            self._correct_predictions += 1
        
        # Track recent predictions (rolling window)
        self._recent_predictions.append((predicted, actual))
        if len(self._recent_predictions) > 1000:
            self._recent_predictions.pop(0)
    
    def get_accuracy(self, window: Optional[int] = None) -> float:
        """
        Get prediction accuracy.
        
        Args:
            window: Optional window size for recent accuracy
            
        Returns:
            Accuracy as float (0.0 to 1.0)
        """
        if window and self._recent_predictions:
            recent = self._recent_predictions[-window:]
            correct = sum(1 for p, a in recent if p == a)
            return correct / len(recent) if recent else 0.0
        
        if self._prediction_count == 0:
            return 0.0
        return self._correct_predictions / self._prediction_count
    
    def get_consecutive_misses(self) -> int:
        """Get count of consecutive wrong predictions."""
        count = 0
        for predicted, actual in reversed(self._recent_predictions):
            if predicted != actual:
                count += 1
            else:
                break
        return count
    
    def _prepare_features(
        self,
        features: Union[pd.Series, pd.DataFrame, np.ndarray]
    ) -> np.ndarray:
        """Prepare features for prediction."""
        if isinstance(features, pd.DataFrame):
            if len(features) == 1:
                X = features.values
            else:
                X = features.values
        elif isinstance(features, pd.Series):
            X = features.values.reshape(1, -1)
        else:
            X = np.asarray(features).reshape(1, -1)
        
        return X
    
    def _resolve_hmm_conflict(
        self,
        hmm_regime: str,
        lgb_direction: str,
        lgb_confidence: float
    ) -> Tuple[str, float]:
        """
        Resolve conflict between HMM regime and LGB prediction.
        
        Follows the conflict matrix from implementation plan.
        """
        # Define conflict cases
        conflict_matrix = {
            ('trend_up', 'DOWN'): {
                'high': ('UP', 0.3),    # 70%+
                'med': ('UP', 0.5),     # 60-70%
                'low': ('UP', 0.7),     # <60%
            },
            ('trend_down', 'UP'): {
                'high': ('DOWN', 0.3),
                'med': ('DOWN', 0.5),
                'low': ('DOWN', 0.7),
            },
            ('low_vol', 'UP'): {
                'high': ('UP', 0.5),
                'med': ('UP', 0.3),
                'low': ('NEUTRAL', 1.0),
            },
            ('low_vol', 'DOWN'): {
                'high': ('DOWN', 0.5),
                'med': ('DOWN', 0.3),
                'low': ('NEUTRAL', 1.0),
            },
            ('high_vol', 'UP'): {
                'high': ('UP', 0.4),
                'med': ('NEUTRAL', 0.5),
                'low': ('NEUTRAL', 1.0),
            },
            ('high_vol', 'DOWN'): {
                'high': ('DOWN', 0.4),
                'med': ('NEUTRAL', 0.5),
                'low': ('NEUTRAL', 1.0),
            },
        }
        
        # Determine confidence level
        if lgb_confidence >= 0.70:
            level = 'high'
        elif lgb_confidence >= 0.60:
            level = 'med'
        else:
            level = 'low'
        
        key = (hmm_regime, lgb_direction)
        if key in conflict_matrix:
            direction, multiplier = conflict_matrix[key][level]
            return direction, lgb_confidence * multiplier
        
        # No conflict
        return lgb_direction, lgb_confidence
    
    def _calculate_signals(
        self,
        direction: str,
        confidence: float
    ) -> Tuple[float, int, int, float]:
        """
        Calculate trading signals from prediction.
        
        Returns:
            (size_multiplier, bid_layers, ask_layers, spread_adjustment)
        """
        base_layers = 5
        
        if direction == 'NEUTRAL' or not self.config.size_adjustment:
            return 1.0, base_layers, base_layers, 0.0
        
        # Size multiplier based on confidence
        # Higher confidence = more aggressive sizing
        excess_conf = confidence - self.config.confidence_threshold
        max_excess = 1.0 - self.config.confidence_threshold
        
        if excess_conf > 0:
            # Scale from 1.0 to max_multiplier
            scale = excess_conf / max_excess
            size_mult = 1.0 + scale * (self.config.max_size_multiplier - 1.0)
        else:
            size_mult = 1.0
        
        # Clamp to limits
        size_mult = max(self.config.min_size_multiplier,
                       min(self.config.max_size_multiplier, size_mult))
        
        # Layer asymmetry
        if self.config.layer_asymmetry and confidence >= 0.60:
            max_bid, max_ask = self.config.max_layer_asymmetry
            
            if direction == 'UP':
                bid_layers = max_bid
                ask_layers = max_ask
            else:  # DOWN
                bid_layers = max_ask
                ask_layers = max_bid
        else:
            bid_layers = base_layers
            ask_layers = base_layers
        
        # Spread adjustment
        # High confidence -> tighter spread (more aggressive)
        # Low confidence -> wider spread (more conservative)
        if confidence >= 0.70:
            spread_adj = -10.0  # 10% tighter
        elif confidence >= 0.60:
            spread_adj = -5.0
        elif confidence < 0.50:
            spread_adj = +10.0  # 10% wider
        else:
            spread_adj = 0.0
        
        return size_mult, bid_layers, ask_layers, spread_adj
    
    def _fallback_prediction(self, reason: str) -> Prediction:
        """Create fallback prediction when model fails."""
        return Prediction(
            direction='NEUTRAL',
            confidence=0.5,
            probabilities={'UP': 0.33, 'NEUTRAL': 0.34, 'DOWN': 0.33},
            size_multiplier=1.0,
            bid_layers=5,
            ask_layers=5,
            spread_adjustment=0.0,
            timestamp=datetime.now(),
            model_version=self.model_version,
            is_fallback=True
        )


class PredictorManager:
    """
    Manager for LightGBM predictor with risk controls.
    
    Handles:
    - Consecutive miss detection
    - Loss limit enforcement
    - Model degradation detection
    - Automatic fallback to HMM-only mode
    """
    
    def __init__(
        self,
        predictor: LightGBMPredictor,
        consecutive_miss_threshold: int = 5,
        severe_miss_threshold: int = 10,
        cooldown_minutes: int = 30,
        accuracy_min: float = 0.48
    ):
        self.predictor = predictor
        self.consecutive_miss_threshold = consecutive_miss_threshold
        self.severe_miss_threshold = severe_miss_threshold
        self.cooldown_minutes = cooldown_minutes
        self.accuracy_min = accuracy_min
        
        self._is_disabled = False
        self._cooldown_until: Optional[datetime] = None
        self._skew_reduction = 1.0  # 1.0 = full, 0.5 = 50% reduced
    
    @property
    def is_active(self) -> bool:
        """Check if predictor is active (not disabled or in cooldown)."""
        if self._is_disabled:
            return False
        
        if self._cooldown_until and datetime.now() < self._cooldown_until:
            return False
        
        return True
    
    @property
    def skew_reduction(self) -> float:
        """Get current skew reduction factor."""
        return self._skew_reduction
    
    def predict(
        self,
        features: Union[pd.Series, pd.DataFrame],
        hmm_regime: Optional[str] = None
    ) -> Prediction:
        """
        Make prediction with risk controls.
        
        Returns neutral prediction if predictor is disabled or in cooldown.
        """
        if not self.is_active:
            return self.predictor._fallback_prediction("Predictor disabled")
        
        prediction = self.predictor.predict(features, hmm_regime)
        
        # Apply skew reduction if active
        if self._skew_reduction < 1.0:
            prediction.size_multiplier = 1.0 + (prediction.size_multiplier - 1.0) * self._skew_reduction
            prediction.spread_adjustment *= self._skew_reduction
        
        return prediction
    
    def update_with_outcome(self, predicted: str, actual: str) -> None:
        """Update predictor with actual outcome and apply risk controls."""
        self.predictor.update_accuracy(predicted, actual)
        
        # Check consecutive misses
        consecutive = self.predictor.get_consecutive_misses()
        
        if consecutive >= self.severe_miss_threshold:
            # Disable for cooldown period
            self._cooldown_until = datetime.now() + pd.Timedelta(minutes=self.cooldown_minutes)
            self._skew_reduction = 1.0  # Reset after cooldown
            
        elif consecutive >= self.consecutive_miss_threshold:
            # Reduce skew by 50%
            self._skew_reduction = 0.5
        else:
            # Gradually restore
            if self._skew_reduction < 1.0:
                self._skew_reduction = min(1.0, self._skew_reduction + 0.1)
        
        # Check overall accuracy
        accuracy = self.predictor.get_accuracy(window=100)
        if accuracy < self.accuracy_min and self.predictor._prediction_count > 100:
            self._is_disabled = True
    
    def reset(self) -> None:
        """Reset all risk controls."""
        self._is_disabled = False
        self._cooldown_until = None
        self._skew_reduction = 1.0


def create_predictor(
    config: Optional[PredictorConfig] = None,
    model_path: Optional[str] = None,
    with_manager: bool = True
) -> Union[LightGBMPredictor, PredictorManager]:
    """
    Factory function to create predictor.
    
    Args:
        config: Predictor configuration
        model_path: Path to load model from
        with_manager: Whether to wrap in risk manager
        
    Returns:
        Predictor or PredictorManager instance
    """
    predictor = LightGBMPredictor(config)
    
    if model_path:
        predictor.load_model(model_path)
    
    if with_manager:
        return PredictorManager(predictor)
    
    return predictor
