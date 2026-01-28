"""
v4.0 Final Strategy: HMM + AGGRESSIVE Volatility + Direction
Optimized for Market Making on GRVT/Binance

Backtest Results:
- PnL: $14,567 (vs baseline $8,679)
- Sharpe: 23.48 (vs baseline 14.55)
- MDD: 0.42% (vs baseline 0.68%)
"""

import pickle
import logging
from dataclasses import dataclass
from typing import Optional, Tuple
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class StrategyConfig:
    """v4.0 Strategy configuration."""
    # Volatility thresholds (percentiles from training)
    vol_high_percentile: float = 70
    vol_low_percentile: float = 30
    
    # AGGRESSIVE adjustments
    high_vol_spread_mult: float = 0.8   # Tighter spread
    high_vol_size_mult: float = 1.3     # Larger size
    low_vol_spread_mult: float = 1.3    # Wider spread
    low_vol_size_mult: float = 0.8      # Smaller size
    
    # Direction model
    direction_threshold: float = 0.52   # Use signal if prob > this
    direction_layer_shift: int = 1      # Layers to shift
    direction_size_mult: float = 1.15   # Size multiplier for direction
    
    # Model paths
    vol_model_path: str = 'data/volatility_model_range_15m.pkl'
    dir_model_path: str = 'data/direction_model_binary.pkl'


class StrategyV4:
    """
    v4.0 Market Making Strategy with ML Enhancement.
    
    Components:
    - HMM regime detection (existing)
    - Volatility prediction (AGGRESSIVE mode)
    - Direction prediction (binary classifier)
    """
    
    def __init__(self, config: StrategyConfig = None):
        self.config = config or StrategyConfig()
        
        self.vol_model = None
        self.dir_model = None
        self.vol_thresholds = {'high': 0.4, 'low': 0.25}
        
        self._load_models()
    
    def _load_models(self):
        """Load ML models."""
        # Volatility model
        try:
            path = Path(self.config.vol_model_path)
            if path.exists():
                with open(path, 'rb') as f:
                    self.vol_model = pickle.load(f)
                logger.info(f"Loaded volatility model: {path}")
        except Exception as e:
            logger.warning(f"Could not load volatility model: {e}")
        
        # Direction model
        try:
            path = Path(self.config.dir_model_path)
            if path.exists():
                with open(path, 'rb') as f:
                    self.dir_model = pickle.load(f)
                logger.info(f"Loaded direction model: {path}")
        except Exception as e:
            logger.warning(f"Could not load direction model: {e}")
    
    def set_volatility_thresholds(self, vol_history: np.ndarray):
        """Set volatility thresholds from historical data."""
        self.vol_thresholds['high'] = np.percentile(vol_history, self.config.vol_high_percentile)
        self.vol_thresholds['low'] = np.percentile(vol_history, self.config.vol_low_percentile)
        logger.info(f"Vol thresholds: low={self.vol_thresholds['low']:.4f}, high={self.vol_thresholds['high']:.4f}")
    
    def predict_volatility(self, features: pd.Series) -> Optional[float]:
        """Predict future volatility."""
        if self.vol_model is None:
            return None
        try:
            feat_arr = features[self.vol_model['feature_names']].values.reshape(1, -1)
            return self.vol_model['model'].predict(feat_arr)[0]
        except:
            return None
    
    def predict_direction(self, features: pd.Series) -> Tuple[Optional[str], float]:
        """Predict direction (UP/DOWN) and probability."""
        if self.dir_model is None:
            return None, 0.5
        try:
            feat_arr = features[self.dir_model['feature_names']].values.reshape(1, -1)
            prob = self.dir_model['model'].predict(feat_arr)[0]
            direction = 'UP' if prob > 0.5 else 'DOWN'
            confidence = max(prob, 1 - prob)
            return direction, confidence
        except:
            return None, 0.5
    
    def get_adjustments(
        self,
        features: pd.Series,
        hmm_regime: str = 'low_vol'
    ) -> dict:
        """
        Get MM parameter adjustments based on ML predictions.
        
        Returns:
            dict with spread_mult, size_mult, bid_layers, ask_layers, bid_size_mult, ask_size_mult
        """
        result = {
            'spread_mult': 1.0,
            'size_mult': 1.0,
            'bid_layers': 0,  # Layer offset
            'ask_layers': 0,
            'bid_size_mult': 1.0,
            'ask_size_mult': 1.0,
            'vol_regime': 'normal',
            'direction': None,
        }
        
        # 1. Volatility adjustments (AGGRESSIVE)
        vol_pred = self.predict_volatility(features)
        if vol_pred is not None:
            if vol_pred > self.vol_thresholds['high']:
                # High volatility -> AGGRESSIVE
                result['spread_mult'] = self.config.high_vol_spread_mult
                result['size_mult'] = self.config.high_vol_size_mult
                result['vol_regime'] = 'high'
            elif vol_pred < self.vol_thresholds['low']:
                # Low volatility -> Conservative
                result['spread_mult'] = self.config.low_vol_spread_mult
                result['size_mult'] = self.config.low_vol_size_mult
                result['vol_regime'] = 'low'
        
        # 2. Direction adjustments
        direction, confidence = self.predict_direction(features)
        if direction and confidence >= self.config.direction_threshold:
            result['direction'] = direction
            shift = self.config.direction_layer_shift
            size_mult = self.config.direction_size_mult
            
            if direction == 'UP':
                result['bid_layers'] = shift
                result['ask_layers'] = -shift
                result['bid_size_mult'] = size_mult
                result['ask_size_mult'] = 1 / size_mult
            else:  # DOWN
                result['bid_layers'] = -shift
                result['ask_layers'] = shift
                result['bid_size_mult'] = 1 / size_mult
                result['ask_size_mult'] = size_mult
        
        return result


# Usage example
if __name__ == '__main__':
    strategy = StrategyV4()
    
    print("v4.0 Strategy Loaded")
    print(f"  Volatility model: {'OK' if strategy.vol_model else 'NOT LOADED'}")
    print(f"  Direction model: {'OK' if strategy.dir_model else 'NOT LOADED'}")
    print(f"  Config: {strategy.config}")
