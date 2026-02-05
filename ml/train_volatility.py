"""
Volatility Prediction Model for Market Making
v4.2 - Spread and Size Optimization

Predicts future volatility to optimize:
- Spread width (wider in high vol, tighter in low vol)
- Order size (smaller in high vol, larger in low vol)
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
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from optuna.samplers import TPESampler

sys.path.insert(0, str(Path(__file__).parent.parent))

from ml.feature_engineering import get_feature_engineer
from ml.walk_forward_validator import WalkForwardValidator

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_volatility_targets(
    df: pd.DataFrame,
    horizons: List[int] = [5, 15, 60]
) -> pd.DataFrame:
    """
    Create multiple volatility targets.
    
    Targets:
    - Realized volatility (std of returns)
    - High-Low range
    - Max drawdown in window
    
    Args:
        df: OHLCV DataFrame
        horizons: List of prediction horizons in minutes
        
    Returns:
        DataFrame with volatility targets
    """
    targets = pd.DataFrame(index=df.index)
    
    for h in horizons:
        # Future realized volatility (annualized)
        future_returns = df['close'].pct_change().shift(-1)
        future_vol = future_returns.rolling(h).std().shift(-h) * np.sqrt(525600) * 100
        targets[f'vol_{h}m'] = future_vol
        
        # Future high-low range (%)
        future_high = df['high'].rolling(h).max().shift(-h)
        future_low = df['low'].rolling(h).min().shift(-h)
        targets[f'range_{h}m'] = (future_high - future_low) / df['close'] * 100
        
        # Future max drawdown (%)
        future_close = df['close'].shift(-h)
        future_max = df['close'].rolling(h).max().shift(-h)
        targets[f'mdd_{h}m'] = (future_max - future_close) / future_max * 100
    
    return targets


class VolatilityPredictor:
    """
    Volatility prediction model for market making.
    
    Predicts future volatility to optimize spread/size.
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
        self.models: Dict[str, Any] = {}
    
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
    
    def train_volatility_model(
        self,
        target_name: str = 'range_15m',
        n_trials: int = 30
    ) -> Dict[str, Any]:
        """
        Train volatility prediction model.
        
        Args:
            target_name: Which volatility target to predict
            n_trials: Optuna trials
            
        Returns:
            Training results
        """
        if self.features is None:
            self.compute_features()
        
        logger.info(f"Training Volatility Model: {target_name}")
        
        # Create targets
        targets = create_volatility_targets(self.df)
        y = targets[target_name]
        
        # Filter NaN
        valid_mask = ~(self.features.isna().any(axis=1) | y.isna())
        X = self.features[valid_mask]
        y = y[valid_mask]
        
        logger.info(f"Training samples: {len(X):,}")
        logger.info(f"Target stats: mean={y.mean():.4f}%, std={y.std():.4f}%, median={y.median():.4f}%")
        
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
                
                # Use R² as metric (higher = better)
                r2 = r2_score(y_test, y_pred)
                scores.append(r2)
            
            return np.mean(scores)
        
        study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=42))
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        
        logger.info(f"Best CV R²: {study.best_value:.4f}")
        
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
        
        # Directional accuracy for vol changes
        y_val_diff = y_val.diff().dropna()
        y_pred_diff = pd.Series(y_pred, index=y_val.index).diff().dropna()
        common_idx = y_val_diff.index.intersection(y_pred_diff.index)
        direction_acc = ((y_pred_diff[common_idx] > 0) == (y_val_diff[common_idx] > 0)).mean()
        
        # Regime accuracy (high/low vol classification)
        vol_median = y_val.median()
        regime_pred = y_pred > vol_median
        regime_actual = y_val > vol_median
        regime_acc = (regime_pred == regime_actual).mean()
        
        logger.info(f"Validation Results:")
        logger.info(f"  RMSE: {rmse:.4f}%")
        logger.info(f"  MAE: {mae:.4f}%")
        logger.info(f"  R²: {r2:.4f}")
        logger.info(f"  Direction Accuracy: {direction_acc:.4f}")
        logger.info(f"  Regime Accuracy: {regime_acc:.4f}")
        
        # Feature importance
        importance = pd.DataFrame({
            'feature': X.columns,
            'importance': model.feature_importance(importance_type='gain')
        }).sort_values('importance', ascending=False)
        
        logger.info(f"\nTop 10 Features:")
        for _, row in importance.head(10).iterrows():
            logger.info(f"  {row['feature']}: {row['importance']:.0f}")
        
        # Save model
        model_path = self.output_dir / f'volatility_model_{target_name}.pkl'
        model_data = {
            'model': model,
            'type': 'volatility',
            'target': target_name,
            'feature_names': X.columns.tolist(),
            'version': datetime.now().strftime('%Y%m%d_%H%M%S'),
            'metadata': {
                'rmse': rmse,
                'mae': mae,
                'r2': r2,
                'direction_accuracy': direction_acc,
                'regime_accuracy': regime_acc,
                'target_mean': y.mean(),
                'target_std': y.std(),
                'best_params': study.best_params,
            },
            'feature_importance': importance.to_dict('records')
        }
        
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        # Also save JSON summary
        json_path = self.output_dir / f'volatility_model_{target_name}.json'
        json_data = {
            'target': target_name,
            'version': model_data['version'],
            'metrics': {
                'rmse': rmse,
                'mae': mae,
                'r2': r2,
                'direction_accuracy': direction_acc,
                'regime_accuracy': regime_acc,
            },
            'top_features': importance.head(10).to_dict('records')
        }
        with open(json_path, 'w') as f:
            json.dump(json_data, f, indent=2)
        
        logger.info(f"Model saved to {model_path}")
        
        self.models[target_name] = model_data
        return model_data
    
    def train_all_targets(self, n_trials: int = 30) -> Dict[str, Dict]:
        """Train models for all volatility targets."""
        results = {}
        
        targets = ['range_5m', 'range_15m', 'range_60m']
        
        for target in targets:
            logger.info(f"\n{'='*60}")
            logger.info(f"Training {target}")
            logger.info(f"{'='*60}")
            
            results[target] = self.train_volatility_model(target, n_trials)
        
        # Summary
        logger.info("\n" + "="*60)
        logger.info("VOLATILITY MODEL SUMMARY")
        logger.info("="*60)
        
        for target, result in results.items():
            meta = result['metadata']
            logger.info(f"\n{target}:")
            logger.info(f"  R²: {meta['r2']:.4f}")
            logger.info(f"  Regime Accuracy: {meta['regime_accuracy']:.4f}")
        
        return results


def main():
    """Train volatility prediction models."""
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='data/btcusdt_1m_1year.csv')
    parser.add_argument('--output', type=str, default='data')
    parser.add_argument('--trials', type=int, default=30)
    parser.add_argument('--target', type=str, default='all', 
                        choices=['range_5m', 'range_15m', 'range_60m', 'all'])
    
    args = parser.parse_args()
    
    predictor = VolatilityPredictor(
        data_path=args.data,
        output_dir=args.output
    )
    
    predictor.load_data()
    predictor.compute_features()
    
    if args.target == 'all':
        predictor.train_all_targets(n_trials=args.trials)
    else:
        predictor.train_volatility_model(target_name=args.target, n_trials=args.trials)
    
    logger.info("\n✅ Volatility model training complete!")


if __name__ == '__main__':
    main()
