"""
ML Module for LightGBM Direction Predictor
v4.0 - Modular, Multi-Exchange Compatible Design
"""

from ml.feature_engineering import (
    FeatureEngineer,
    FeatureConfig,
    BaseFeatureProvider,
    BinanceFeatureProvider,
    GRVTFeatureProvider,
    get_feature_engineer,
    create_target,
    EXCHANGE_PROVIDERS,
)

from ml.walk_forward_validator import (
    WalkForwardValidator,
    ExpandingWindowValidator,
    PurgedKFoldValidator,
    create_validator,
    validate_no_leakage,
)

from ml.lightgbm_predictor import (
    LightGBMPredictor,
    PredictorConfig,
    PredictorManager,
    Prediction,
    create_predictor,
)

from ml.drift_detector import (
    DriftDetector,
    DriftConfig,
    AccuracyMonitor,
    PerformanceMonitor,
)

__all__ = [
    # Feature Engineering
    'FeatureEngineer',
    'FeatureConfig',
    'BaseFeatureProvider',
    'BinanceFeatureProvider',
    'GRVTFeatureProvider',
    'get_feature_engineer',
    'create_target',
    'EXCHANGE_PROVIDERS',
    
    # Validation
    'WalkForwardValidator',
    'ExpandingWindowValidator',
    'PurgedKFoldValidator',
    'create_validator',
    'validate_no_leakage',
    
    # Predictor
    'LightGBMPredictor',
    'PredictorConfig',
    'PredictorManager',
    'Prediction',
    'create_predictor',
    
    # Monitoring
    'DriftDetector',
    'DriftConfig',
    'AccuracyMonitor',
    'PerformanceMonitor',
]

__version__ = '4.0.0'
