"""
Improved Direction Models: Binary Classification & Return Regression
v4.1 - Better problem formulation for market making and CTA

Two approaches:
1. Binary Classification: UP vs DOWN (removes NEUTRAL noise)
2. Return Regression: Predict actual return value (continuous signal)
"""

import sys
from pathlib import Path
import pickle
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import lightgbm as lgb
import optuna
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from optuna.samplers import TPESampler

sys.path.insert(0, str(Path(__file__).parent.parent))

from ml.feature_engineering import get_feature_engineer
from ml.walk_forward_validator import WalkForwardValidator

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_binary_target(
    df: pd.DataFrame,
    horizon: int = 15,
    threshold_pct: float = 0.10,
    min_movement: bool = True
) -> pd.Series:
    """
    Create binary target (UP=1, DOWN=0) for significant movements.
    
    Args:
        df: OHLCV DataFrame
        horizon: Periods ahead to predict
        threshold_pct: Minimum movement to include (filters noise)
        min_movement: If True, only include rows with |return| > threshold
        
    Returns:
        Series with binary labels
    """
    future_return = df['close'].shift(-horizon) / df['close'] - 1
    future_return_pct = future_return * 100
    
    # Create labels: 1=UP, 0=DOWN
    labels = (future_return_pct > 0).astype(int)
    
    # Filter out small movements (noise)
    if min_movement:
        significant = abs(future_return_pct) >= threshold_pct
        labels = labels.where(significant, np.nan)
    
    return labels


def create_regression_target(
    df: pd.DataFrame,
    horizon: int = 15,
    scale: float = 100.0
) -> pd.Series:
    """
    Create regression target (future return in percentage).
    
    Args:
        df: OHLCV DataFrame
        horizon: Periods ahead to predict
        scale: Scaling factor (100 = percentage)
        
    Returns:
        Series with return values
    """
    future_return = df['close'].shift(-horizon) / df['close'] - 1
    return future_return * scale  # Return in percentage


class ImprovedDirectionTrainer:
    """
    Trainer for improved direction models.
    
    Supports:
    - Binary Classification (UP vs DOWN)
    - Return Regression (continuous)
    """
    
    def __init__(
        self,
        data_path: str = 'data/btcusdt_1m_1year.csv',
        output_dir: str = 'data',
        exchange: str = 'binance'
    ):
        self.data_path = Path(data_path)
        self.output_dir = Path(output_dir)
        self.exchange = exchange
        
        self.fe = get_feature_engineer(exchange)
        self.df: Optional[pd.DataFrame] = None
        self.features: Optional[pd.DataFrame] = None
    
    def load_data(self) -> pd.DataFrame:
        """Load and prepare data."""
        logger.info(f"Loading data from {self.data_path}...")
        
        df = pd.read_csv(self.data_path)
        
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
        
        logger.info(f"Loaded {len(df):,} rows")
        self.df = df
        return df
    
    def compute_features(self) -> pd.DataFrame:
        """Compute features."""
        if self.df is None:
            self.load_data()
        
        logger.info("Computing features...")
        self.features = self.fe.compute_features(self.df)
        logger.info(f"Computed {self.features.shape[1]} features")
        return self.features
    
    def train_binary_classifier(
        self,
        horizon: int = 15,
        threshold_pct: float = 0.10,
        n_trials: int = 30
    ) -> Dict[str, Any]:
        """
        Train binary classifier (UP vs DOWN).
        
        Args:
            horizon: Prediction horizon in minutes
            threshold_pct: Minimum movement threshold
            n_trials: Optuna trials
            
        Returns:
            Training results
        """
        if self.features is None:
            self.compute_features()
        
        logger.info(f"Training Binary Classifier (horizon={horizon}m, threshold={threshold_pct}%)")
        
        # Create target
        y = create_binary_target(self.df, horizon=horizon, threshold_pct=threshold_pct)
        
        # Filter NaN
        valid_mask = ~(self.features.isna().any(axis=1) | y.isna())
        X = self.features[valid_mask]
        y = y[valid_mask].astype(int)
        
        logger.info(f"Training samples: {len(X):,}")
        logger.info(f"Class balance: UP={y.sum():,} ({y.mean():.1%}), DOWN={len(y)-y.sum():,}")
        
        # Walk-forward validation
        validator = WalkForwardValidator(n_splits=5, train_period_days=60, test_period_days=7)
        
        # Hyperparameter tuning
        def objective(trial):
            params = {
                'objective': 'binary',
                'metric': 'binary_logloss',
                'verbosity': -1,
                'boosting_type': 'gbdt',
                'num_leaves': trial.suggest_int('num_leaves', 20, 100),
                'max_depth': trial.suggest_int('max_depth', 3, 12),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
                'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 50, 200),
                'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 0.9),
                'bagging_fraction': trial.suggest_float('bagging_fraction', 0.6, 0.9),
                'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
            }
            
            scores = []
            for train_idx, test_idx in validator.split(X):
                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
                
                train_data = lgb.Dataset(X_train, label=y_train)
                valid_data = lgb.Dataset(X_test, label=y_test, reference=train_data)
                
                model = lgb.train(
                    params, train_data,
                    valid_sets=[valid_data],
                    callbacks=[lgb.early_stopping(50, verbose=False)]
                )
                
                y_pred = (model.predict(X_test) > 0.5).astype(int)
                scores.append(accuracy_score(y_test, y_pred))
            
            return np.mean(scores)
        
        study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=42))
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        
        logger.info(f"Best CV Accuracy: {study.best_value:.4f}")
        
        # Train final model
        split_idx = int(len(X) * 0.85)
        X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]
        
        final_params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'verbosity': -1,
            **study.best_params
        }
        
        train_data = lgb.Dataset(X_train, label=y_train)
        valid_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        
        model = lgb.train(
            final_params, train_data,
            valid_sets=[valid_data],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)]
        )
        
        # Evaluate
        y_pred = (model.predict(X_val) > 0.5).astype(int)
        y_prob = model.predict(X_val)
        
        accuracy = accuracy_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred)
        
        logger.info(f"Final Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
        logger.info(classification_report(y_val, y_pred, target_names=['DOWN', 'UP']))
        
        # Save model
        model_path = self.output_dir / 'direction_model_binary.pkl'
        model_data = {
            'model': model,
            'type': 'binary',
            'feature_names': X.columns.tolist(),
            'version': datetime.now().strftime('%Y%m%d_%H%M%S'),
            'metadata': {
                'horizon': horizon,
                'threshold_pct': threshold_pct,
                'accuracy': accuracy,
                'f1_score': f1,
                'best_params': study.best_params,
            }
        }
        
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Model saved to {model_path}")
        
        return model_data
    
    def train_return_regressor(
        self,
        horizon: int = 15,
        n_trials: int = 30
    ) -> Dict[str, Any]:
        """
        Train return regression model.
        
        Args:
            horizon: Prediction horizon in minutes
            n_trials: Optuna trials
            
        Returns:
            Training results
        """
        if self.features is None:
            self.compute_features()
        
        logger.info(f"Training Return Regressor (horizon={horizon}m)")
        
        # Create target
        y = create_regression_target(self.df, horizon=horizon)
        
        # Filter NaN
        valid_mask = ~(self.features.isna().any(axis=1) | y.isna())
        X = self.features[valid_mask]
        y = y[valid_mask]
        
        logger.info(f"Training samples: {len(X):,}")
        logger.info(f"Target stats: mean={y.mean():.4f}%, std={y.std():.4f}%")
        
        # Walk-forward validation
        validator = WalkForwardValidator(n_splits=5, train_period_days=60, test_period_days=7)
        
        # Hyperparameter tuning
        def objective(trial):
            params = {
                'objective': 'regression',
                'metric': 'rmse',
                'verbosity': -1,
                'boosting_type': 'gbdt',
                'num_leaves': trial.suggest_int('num_leaves', 20, 100),
                'max_depth': trial.suggest_int('max_depth', 3, 12),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
                'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 50, 200),
                'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 0.9),
                'bagging_fraction': trial.suggest_float('bagging_fraction', 0.6, 0.9),
                'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
            }
            
            scores = []
            for train_idx, test_idx in validator.split(X):
                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
                
                train_data = lgb.Dataset(X_train, label=y_train)
                valid_data = lgb.Dataset(X_test, label=y_test, reference=train_data)
                
                model = lgb.train(
                    params, train_data,
                    valid_sets=[valid_data],
                    callbacks=[lgb.early_stopping(50, verbose=False)]
                )
                
                y_pred = model.predict(X_test)
                
                # Use directional accuracy (sign match) as metric
                direction_accuracy = ((y_pred > 0) == (y_test > 0)).mean()
                scores.append(direction_accuracy)
            
            return np.mean(scores)
        
        study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=42))
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        
        logger.info(f"Best CV Direction Accuracy: {study.best_value:.4f}")
        
        # Train final model
        split_idx = int(len(X) * 0.85)
        X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]
        
        final_params = {
            'objective': 'regression',
            'metric': 'rmse',
            'verbosity': -1,
            **study.best_params
        }
        
        train_data = lgb.Dataset(X_train, label=y_train)
        valid_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        
        model = lgb.train(
            final_params, train_data,
            valid_sets=[valid_data],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)]
        )
        
        # Evaluate
        y_pred = model.predict(X_val)
        
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        mae = mean_absolute_error(y_val, y_pred)
        r2 = r2_score(y_val, y_pred)
        direction_acc = ((y_pred > 0) == (y_val > 0)).mean()
        
        logger.info(f"RMSE: {rmse:.4f}%, MAE: {mae:.4f}%, R2: {r2:.4f}")
        logger.info(f"Direction Accuracy: {direction_acc:.4f}")
        
        # Save model
        model_path = self.output_dir / 'direction_model_regression.pkl'
        model_data = {
            'model': model,
            'type': 'regression',
            'feature_names': X.columns.tolist(),
            'version': datetime.now().strftime('%Y%m%d_%H%M%S'),
            'metadata': {
                'horizon': horizon,
                'rmse': rmse,
                'mae': mae,
                'r2': r2,
                'direction_accuracy': direction_acc,
                'best_params': study.best_params,
            }
        }
        
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Model saved to {model_path}")
        
        return model_data


def main():
    """Train both improved models."""
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='data/btcusdt_1m_1year.csv')
    parser.add_argument('--output', type=str, default='data')
    parser.add_argument('--trials', type=int, default=30)
    parser.add_argument('--horizon', type=int, default=15)
    parser.add_argument('--model', type=str, default='both', choices=['binary', 'regression', 'both'])
    
    args = parser.parse_args()
    
    trainer = ImprovedDirectionTrainer(
        data_path=args.data,
        output_dir=args.output
    )
    
    trainer.load_data()
    trainer.compute_features()
    
    if args.model in ['binary', 'both']:
        logger.info("\n" + "="*60)
        logger.info("Training BINARY CLASSIFIER")
        logger.info("="*60)
        binary_result = trainer.train_binary_classifier(
            horizon=args.horizon,
            n_trials=args.trials
        )
    
    if args.model in ['regression', 'both']:
        logger.info("\n" + "="*60)
        logger.info("Training RETURN REGRESSOR")
        logger.info("="*60)
        regression_result = trainer.train_return_regressor(
            horizon=args.horizon,
            n_trials=args.trials
        )
    
    logger.info("\n✅ Training complete!")


if __name__ == '__main__':
    main()
