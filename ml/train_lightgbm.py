"""
LightGBM Training Script with Optuna Hyperparameter Tuning
v4.0 - Walk-Forward Validation for Time Series

This script trains the LightGBM direction predictor using:
- Walk-forward validation to prevent future data leakage
- Optuna for hyperparameter optimization
- Class balancing for imbalanced labels
"""

import os
import sys
import pickle
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import lightgbm as lgb
import optuna
from sklearn.metrics import accuracy_score, f1_score, classification_report
from optuna.samplers import TPESampler

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ml.feature_engineering import FeatureEngineer, create_target, get_feature_engineer
from ml.walk_forward_validator import WalkForwardValidator, create_validator


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class LightGBMTrainer:
    """
    Trainer for LightGBM direction predictor.
    
    Handles:
    - Data loading and preprocessing
    - Feature engineering
    - Hyperparameter tuning with Optuna
    - Walk-forward validation
    - Model saving with metadata
    """
    
    def __init__(
        self,
        data_path: str = 'data/market_data.db',
        output_path: str = 'data/direction_model_lgb.pkl',
        exchange: str = 'binance',
        symbol: str = 'BTCUSDT'
    ):
        self.data_path = Path(data_path)
        self.output_path = Path(output_path)
        self.exchange = exchange
        self.symbol = symbol
        
        self.feature_engineer = get_feature_engineer(exchange)
        self.validator = create_validator('walk_forward', n_splits=5)
        
        self.X: Optional[pd.DataFrame] = None
        self.y: Optional[pd.Series] = None
        self.best_params: Dict[str, Any] = {}
        self.best_model: Optional[lgb.Booster] = None
        self.training_metadata: Dict[str, Any] = {}
    
    def load_data(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        table_name: str = 'candles_1m'
    ) -> pd.DataFrame:
        """
        Load OHLCV data from database.
        
        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            table_name: Database table name
            
        Returns:
            DataFrame with OHLCV data
        """
        import sqlite3
        
        conn = sqlite3.connect(self.data_path)
        
        query = f"SELECT * FROM {table_name}"
        conditions = []
        
        if start_date:
            conditions.append(f"timestamp >= '{start_date}'")
        if end_date:
            conditions.append(f"timestamp <= '{end_date}'")
        
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        
        query += " ORDER BY timestamp"
        
        df = pd.read_sql(query, conn)
        conn.close()
        
        # Convert timestamp
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
        
        logger.info(f"Loaded {len(df)} rows from {self.data_path}")
        return df
    
    def load_csv_data(self, path: str) -> pd.DataFrame:
        """Load data from CSV file."""
        df = pd.read_csv(path)
        
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
        
        logger.info(f"Loaded {len(df)} rows from {path}")
        return df
    
    def prepare_features(
        self,
        df: pd.DataFrame,
        target_horizon: int = 1,
        target_threshold: float = 0.05
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare features and target for training.
        
        Args:
            df: Raw OHLCV data
            target_horizon: Prediction horizon in periods
            target_threshold: Threshold for UP/DOWN classification
            
        Returns:
            (X, y) tuple
        """
        logger.info("Computing features...")
        
        # Compute features
        X = self.feature_engineer.compute_features(df)
        
        # Create target
        y = create_target(df, horizon=target_horizon, threshold_pct=target_threshold)
        
        # Remove NaN rows
        valid_mask = ~(X.isna().any(axis=1) | y.isna())
        X = X[valid_mask]
        y = y[valid_mask]
        
        logger.info(f"Prepared {len(X)} samples with {X.shape[1]} features")
        logger.info(f"Target distribution: {y.value_counts().to_dict()}")
        
        self.X = X
        self.y = y
        
        return X, y
    
    def get_class_weights(self, y: pd.Series) -> Dict[int, float]:
        """Calculate class weights for imbalanced data."""
        counts = y.value_counts()
        total = len(y)
        n_classes = len(counts)
        
        weights = {}
        for label, count in counts.items():
            weights[int(label)] = total / (n_classes * count)
        
        return weights
    
    def create_objective(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_splits: int = 5
    ):
        """Create Optuna objective function."""
        
        validator = WalkForwardValidator(
            n_splits=n_splits,
            train_period_days=60,
            test_period_days=7,
            gap_days=1
        )
        
        class_weights = self.get_class_weights(y)
        
        def objective(trial: optuna.Trial) -> float:
            params = {
                'objective': 'multiclass',
                'num_class': 3,
                'metric': 'multi_logloss',
                'verbosity': -1,
                'boosting_type': 'gbdt',
                
                # Tunable parameters
                'num_leaves': trial.suggest_int('num_leaves', 20, 100),
                'max_depth': trial.suggest_int('max_depth', 3, 12),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
                'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                
                # Regularization
                'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 50, 200),
                'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 0.9),
                'bagging_fraction': trial.suggest_float('bagging_fraction', 0.6, 0.9),
                'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
                'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True),
                'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True),
            }
            
            scores = []
            
            for train_idx, test_idx in validator.split(X):
                X_train = X.iloc[train_idx]
                X_test = X.iloc[test_idx]
                y_train = y.iloc[train_idx]
                y_test = y.iloc[test_idx]
                
                # Create sample weights
                sample_weights = y_train.map(class_weights)
                
                # Create datasets
                train_data = lgb.Dataset(
                    X_train, label=y_train,
                    weight=sample_weights
                )
                valid_data = lgb.Dataset(
                    X_test, label=y_test,
                    reference=train_data
                )
                
                # Train
                model = lgb.train(
                    params,
                    train_data,
                    valid_sets=[valid_data],
                    callbacks=[
                        lgb.early_stopping(stopping_rounds=50, verbose=False)
                    ]
                )
                
                # Predict
                y_pred = model.predict(X_test).argmax(axis=1)
                score = accuracy_score(y_test, y_pred)
                scores.append(score)
            
            return np.mean(scores)
        
        return objective
    
    def tune_hyperparameters(
        self,
        n_trials: int = 100,
        timeout_hours: float = 2.0,
        n_jobs: int = 1
    ) -> Dict[str, Any]:
        """
        Tune hyperparameters using Optuna.
        
        Args:
            n_trials: Number of trials
            timeout_hours: Maximum time in hours
            n_jobs: Number of parallel jobs
            
        Returns:
            Best parameters
        """
        if self.X is None or self.y is None:
            raise ValueError("Call prepare_features first")
        
        logger.info(f"Starting hyperparameter tuning with {n_trials} trials...")
        
        objective = self.create_objective(self.X, self.y)
        
        study = optuna.create_study(
            direction='maximize',
            sampler=TPESampler(seed=42)
        )
        
        study.optimize(
            objective,
            n_trials=n_trials,
            timeout=timeout_hours * 3600,
            n_jobs=n_jobs,
            show_progress_bar=True
        )
        
        self.best_params = study.best_params
        logger.info(f"Best accuracy: {study.best_value:.4f}")
        logger.info(f"Best params: {self.best_params}")
        
        # Save study results
        self.training_metadata['optuna_study'] = {
            'best_trial': study.best_trial.number,
            'best_value': study.best_value,
            'n_trials': len(study.trials),
        }
        
        return self.best_params
    
    def train_final_model(
        self,
        params: Optional[Dict[str, Any]] = None,
        train_size: float = 0.85
    ) -> lgb.Booster:
        """
        Train final model with best parameters.
        
        Args:
            params: Model parameters (uses best_params if None)
            train_size: Fraction of data for training
            
        Returns:
            Trained model
        """
        if self.X is None or self.y is None:
            raise ValueError("Call prepare_features first")
        
        params = params or self.best_params
        if not params:
            raise ValueError("No parameters provided")
        
        # Use last portion for validation (respecting time series)
        split_idx = int(len(self.X) * train_size)
        
        X_train = self.X.iloc[:split_idx]
        X_val = self.X.iloc[split_idx:]
        y_train = self.y.iloc[:split_idx]
        y_val = self.y.iloc[split_idx:]
        
        logger.info(f"Training on {len(X_train)} samples, validating on {len(X_val)}")
        
        # Class weights
        class_weights = self.get_class_weights(y_train)
        sample_weights = y_train.map(class_weights)
        
        # Full params
        full_params = {
            'objective': 'multiclass',
            'num_class': 3,
            'metric': 'multi_logloss',
            'verbosity': -1,
            'boosting_type': 'gbdt',
            **params
        }
        
        # Create datasets
        train_data = lgb.Dataset(
            X_train, label=y_train,
            weight=sample_weights
        )
        valid_data = lgb.Dataset(
            X_val, label=y_val,
            reference=train_data
        )
        
        # Train
        self.best_model = lgb.train(
            full_params,
            train_data,
            valid_sets=[valid_data],
            callbacks=[
                lgb.early_stopping(stopping_rounds=50),
                lgb.log_evaluation(period=100)
            ]
        )
        
        # Evaluate
        y_pred = self.best_model.predict(X_val).argmax(axis=1)
        
        accuracy = accuracy_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred, average='weighted')
        
        logger.info(f"Final model - Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
        logger.info("\nClassification Report:")
        logger.info(classification_report(y_val, y_pred, target_names=['DOWN', 'NEUTRAL', 'UP']))
        
        # Update metadata
        self.training_metadata.update({
            'accuracy': accuracy,
            'f1_score': f1,
            'train_size': len(X_train),
            'val_size': len(X_val),
            'n_features': X_train.shape[1],
            'feature_names': X_train.columns.tolist(),
        })
        
        return self.best_model
    
    def save_model(self, path: Optional[str] = None) -> str:
        """
        Save model with metadata.
        
        Args:
            path: Output path (uses self.output_path if None)
            
        Returns:
            Path where model was saved
        """
        if self.best_model is None:
            raise ValueError("No model to save")
        
        output_path = Path(path or self.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Prepare model data
        model_data = {
            'model': self.best_model,
            'feature_names': self.training_metadata.get('feature_names', []),
            'version': datetime.now().strftime('%Y%m%d_%H%M%S'),
            'metadata': {
                **self.training_metadata,
                'best_params': self.best_params,
                'exchange': self.exchange,
                'symbol': self.symbol,
            }
        }
        
        with open(output_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Model saved to {output_path}")
        
        # Save metadata as JSON for inspection
        meta_path = output_path.with_suffix('.json')
        with open(meta_path, 'w') as f:
            json.dump({
                'version': model_data['version'],
                'accuracy': self.training_metadata.get('accuracy'),
                'f1_score': self.training_metadata.get('f1_score'),
                'n_features': self.training_metadata.get('n_features'),
                'train_size': self.training_metadata.get('train_size'),
                'best_params': self.best_params,
            }, f, indent=2)
        
        return str(output_path)
    
    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """Get feature importance from trained model."""
        if self.best_model is None:
            raise ValueError("No model trained")
        
        importance = self.best_model.feature_importance(importance_type='gain')
        feature_names = self.training_metadata.get('feature_names', 
                                                    [f'f{i}' for i in range(len(importance))])
        
        df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        return df.head(top_n)


def main():
    """Main training function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train LightGBM direction predictor')
    parser.add_argument('--data', type=str, default='data/market_data.db',
                       help='Path to data file')
    parser.add_argument('--output', type=str, default='data/direction_model_lgb.pkl',
                       help='Output model path')
    parser.add_argument('--trials', type=int, default=100,
                       help='Number of Optuna trials')
    parser.add_argument('--timeout', type=float, default=2.0,
                       help='Timeout in hours')
    parser.add_argument('--exchange', type=str, default='binance',
                       help='Exchange name')
    parser.add_argument('--symbol', type=str, default='BTCUSDT',
                       help='Trading symbol')
    parser.add_argument('--skip-tuning', action='store_true',
                       help='Skip hyperparameter tuning')
    
    args = parser.parse_args()
    
    # Initialize trainer
    trainer = LightGBMTrainer(
        data_path=args.data,
        output_path=args.output,
        exchange=args.exchange,
        symbol=args.symbol
    )
    
    # Load data
    if args.data.endswith('.csv'):
        df = trainer.load_csv_data(args.data)
    else:
        df = trainer.load_data()
    
    # Prepare features
    trainer.prepare_features(df)
    
    # Tune hyperparameters
    if not args.skip_tuning:
        trainer.tune_hyperparameters(
            n_trials=args.trials,
            timeout_hours=args.timeout
        )
    else:
        # Default parameters
        trainer.best_params = {
            'num_leaves': 50,
            'max_depth': 8,
            'learning_rate': 0.05,
            'n_estimators': 300,
            'min_data_in_leaf': 100,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'lambda_l1': 0.1,
            'lambda_l2': 0.1,
        }
    
    # Train final model
    trainer.train_final_model()
    
    # Save model
    trainer.save_model()
    
    # Show feature importance
    logger.info("\nTop 20 Feature Importance:")
    logger.info(trainer.get_feature_importance(20).to_string())
    
    logger.info("\n✅ Training complete!")


if __name__ == '__main__':
    main()
