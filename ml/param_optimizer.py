"""
REGIME_PARAMS Auto-Optimizer
-----------------------------
Uses Optuna Bayesian optimization to find optimal trading parameters
for each market regime using 1-minute backtest data.

Usage:
    from ml.param_optimizer import RegimeParamOptimizer
    optimizer = RegimeParamOptimizer()
    results = optimizer.optimize_all(n_trials=50)
"""
import asyncio
import json
import logging
import os
import sys
import time
from datetime import datetime
from typing import Dict, Optional

import numpy as np
import optuna
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtest.mock_exchange import MockExchange
from strategies.market_maker import MarketMaker
from core.config import Config

logger = logging.getLogger("ParamOptimizer")

# Suppress Optuna's verbose output
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Default output path
OPTIMIZED_PARAMS_PATH = os.path.join("data", "optimized_regime_params.json")


def convert_1m(df: pd.DataFrame) -> pd.DataFrame:
    """Convert raw 1-minute OHLCV to backtest format."""
    df = df.copy()
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    range_pct = (df['high'] - df['low']) / df['close']
    spread = (range_pct * 0.1).clip(0.0001, 0.005)
    df['best_bid'] = df['close'] * (1 - spread / 2)
    df['best_ask'] = df['close'] * (1 + spread / 2)
    ts_col = 'timestamp' if 'timestamp' in df.columns else 'open_time'
    df['timestamp'] = pd.to_datetime(df[ts_col], unit='ms')
    return df[['timestamp', 'open', 'high', 'low', 'close', 'volume', 'best_bid', 'best_ask']]


class RegimeParamOptimizer:
    """
    Bayesian optimizer for market-making regime parameters.
    
    Optimizes:
      gamma, kappa, skew_factor, price_tolerance, grid_spacing,
      order_size_mult, grid_layers, max_position_mult
    
    For each of the 4 regimes:
      low_vol, trend_up, trend_down, high_vol
    """

    # Search space bounds per parameter
    SEARCH_SPACE = {
        'gamma':             (0.1, 3.0),
        'kappa':             (100, 3000),
        'skew_factor':       (0.001, 0.020),
        'price_tolerance':   (0.002, 0.015),
        'grid_spacing':      (0.0008, 0.005),
        'order_size_mult':   (0.3, 1.5),
        'grid_layers':       (3, 12),       # integer
        'max_position_mult': (0.3, 2.0),
    }

    def __init__(self, data_file: str = "data/btcusdt_1m_1year.csv",
                 candles: int = 10000,
                 initial_balance: float = 10000.0,
                 train_ratio: float = 0.7,
                 output_path: str = OPTIMIZED_PARAMS_PATH,
                 storage: Optional[str] = None):
        self.data_file = data_file
        self.candles = candles
        self.initial_balance = initial_balance
        self.train_ratio = train_ratio
        self.output_path = output_path
        self.storage = storage
        self.df_train = None
        self.df_val = None
        
        # Initialize Regime Detector
        try:
            from ml.hmm_regime_detector import HMMRegimeDetector
            self.regime_detector = HMMRegimeDetector()
            logger.info("Loaded HMM Regime Detector")
        except Exception as e:
            logger.error(f"Failed to load HMM Regime Detector: {e}")
            self.regime_detector = None

        self._load_data()

    def _load_data(self):
        """Load, preprocess, and label data with regimes."""
        if not os.path.exists(self.data_file):
            raise FileNotFoundError(f"Data file not found: {self.data_file}")

        df = pd.read_csv(self.data_file)
        df = df.tail(self.candles).reset_index(drop=True)
        df = convert_1m(df)

        # === Regime Labeling ===
        if self.regime_detector and self.regime_detector.is_fitted:
            logger.info("Labeling data with HMM regimes...")
            # Resample to 1h for HMM
            df_hourly = df.set_index('timestamp').resample('1h').agg({
                'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
            }).dropna()
            
            # Predict regimes for hourly data
            # We need to map these back to 1m data
            # HMM predict expects a DataFrame
            # Since predict returns a single string usually for the *last* candle,
            # we need a way to get the sequence. 
            # HMMRegimeDetector.fit returns labels for the whole sequence.
            # But here we are just loading. We need to use 'predict' on the whole sequence?
            # The current 'predict' method in HMMRegimeDetector takes a DF and returns ONE string (the last one).
            # We need to expose the internal 'predict' sequence capability or simulate it.
            
            # Let's use the internal model directly if possible or loop (slow).
            # Actually, HMMRegimeDetector.model.predict(features) returns the sequence.
            # We should add a method to HMMRegimeDetector designed for this, 
            # or just access the internal model if we're careful.
            # For now, let's implement a helper here to batch predict.
            
            features, _ = self.regime_detector._calculate_features(df_hourly)
            features_scaled = self.regime_detector.scaler.transform(features)
            states = self.regime_detector.model.predict(features_scaled)
            
            # Map states to regime names
            regime_map = self.regime_detector.cluster_to_regime
            hourly_regimes = [regime_map.get(s, 'unknown') for s in states]
            
            # Create a Series with hourly timestamps
            # Note: _calculate_features returns indices (df.index) corresponding to features
            # (it might drop initial rows due to rolling windows)
            # fit/predict aligns with these indices.
            
            regime_series = pd.Series(hourly_regimes, index=df_hourly.index[len(df_hourly)-len(states):])
            
            # Map back to 1m DataFrame
            # We'll reindex the 1m DF to match timestamps
            df = df.set_index('timestamp')
            df['regime'] = regime_series.reindex(df.index, method='ffill')
            
            # Fill initial NaNs (before first hour completion) with first valid
            df['regime'] = df['regime'].bfill().fillna('unknown')
            df = df.reset_index()
            
            logger.info(f"Regime distribution:\n{df['regime'].value_counts()}")
        else:
            logger.warning("Regime detector not available. Using 'unknown' for all.")
            df['regime'] = 'unknown'

        split_idx = int(len(df) * self.train_ratio)
        self.df_train = df.iloc[:split_idx].reset_index(drop=True)
        self.df_val = df.iloc[split_idx:].reset_index(drop=True)

        logger.info(f"Loaded {len(df)} candles: {split_idx} train / {len(df) - split_idx} validation")

    async def _run_backtest(self, df: pd.DataFrame, params: Dict, target_regime: Optional[str] = None) -> Dict:
        """
        Run a single backtest with given parameters.
        If target_regime is set, metrics are calculated ONLY for that regime's periods.
        """
        exchange = MockExchange(df.copy(), initial_balance=self.initial_balance)
        # ... (mock methods setup same as before) ...
        exchange.set_as_metrics = lambda x: None
        exchange.set_market_regime = lambda x: None
        exchange.save_live_status = lambda **kwargs: None
        exchange.fetch_and_save_trades = lambda *args: None
        exchange._get_trade_count = lambda *args: 0
        exchange.get_bot_pnl = lambda symbol, price: {
            'bot_net_qty': exchange.position.get('amount', 0.0),
            'bot_avg_entry': exchange.position.get('entryPrice', 0.0),
            'realized_pnl': 0.0,
            'unrealized_pnl': exchange.position.get('unrealizedPnL', 0.0),
            'total_pnl': exchange.position.get('unrealizedPnL', 0.0),
            'trade_count': len(exchange.trade_history),
            'bot_cost_basis': abs(exchange.position.get('amount', 0.0)) * exchange.position.get('entryPrice', 0.0)
        }

        Config.load("config.yaml")
        strategy = MarketMaker(exchange)

        # Disable all ML/adaptive systems
        strategy.regime_detector = None
        strategy.adaptive_tuner = None
        strategy.ml_strategy = None
        strategy.order_flow_analyzer = None
        strategy.funding_arb = None
        strategy.microstructure = None
        strategy.execution_algo = None
        strategy.notifier = None

        # Fixed base params
        strategy.max_loss_usd = 5000.0
        strategy.order_size_usd = 200
        strategy.spread_pct = 0.0015

        # Apply trial params
        strategy._ml_price_tolerance = params['price_tolerance']
        strategy._ml_grid_spacing = params['grid_spacing']
        strategy._ml_order_size_mult = params['order_size_mult']
        strategy._ml_grid_layers = params['grid_layers']
        strategy._ml_max_position_mult = params['max_position_mult']
        strategy._ml_skew_factor = params['skew_factor']
        strategy._last_gamma = params['gamma']
        strategy._last_kappa = params['kappa']
        strategy.grid_layers = params['grid_layers']

        # Count cancels
        cancel_count = 0
        orig_cancel = exchange.cancel_all_orders
        async def counted_cancel(symbol):
            nonlocal cancel_count
            cancel_count += 1
            return await orig_cancel(symbol)
        exchange.cancel_all_orders = counted_cancel

        equity_history = []
        regime_mask = []  # Boolean mask for target regime
        
        strategy.is_running = True

        try:
            while exchange.next_tick():
                await strategy.cycle()
                row = df.iloc[exchange.current_index]
                mid = (row['best_bid'] + row['best_ask']) / 2
                equity = exchange.balance['USDT'] + (exchange.position['amount'] * mid)
                equity_history.append(equity)
                
                # Check regime match
                if target_regime:
                    # current_index is 0-based index in df
                    is_match = (row['regime'] == target_regime)
                    regime_mask.append(is_match)
                else:
                    regime_mask.append(True)
                    
        except Exception as e:
            # logger.error(f"Backtest error: {e}")
            pass

        if not equity_history or len(equity_history) < 10:
            return {'pnl': -999, 'sharpe': -999, 'fills': 0, 'cancels': 0, 'max_dd': -99}

        eq = pd.Series(equity_history)
        mask = pd.Series(regime_mask)
        
        # Calculate metrics ONLY for matching regime ticks
        # PnL: We simply take the sum of differences in equity for the masked periods?
        # Better: PnL is the equity change accumulated *during* the regime.
        
        if target_regime:
            if not mask.any():
                logger.warning(f"Target regime '{target_regime}' not found in data segment.")
                return {
                    'pnl': 0.0, 
                    'sharpe': 0.0, 
                    'fills': 0, 
                    'cancels': 0, 
                    'max_dd': 0.0,
                    'status': 'no_data'
                }

                
            # Filter equity curve
            # For PnL: sum of (eq[t] - eq[t-1]) where mask[t] is True
            eq_diff = eq.diff().fillna(0)
            pnl = eq_diff[mask].sum()
            
            # For Sharpe: returns[mask]
            # Returns are (eq[t] / eq[t-1]) - 1
            # We calculate returns for the whole series, then filter
            returns = eq.pct_change().fillna(0)
            regime_returns = returns[mask]
            
            if len(regime_returns) < 2:
                 sharpe = 0
            else:
                 sharpe = (regime_returns.mean() / regime_returns.std()) * np.sqrt(252 * 24 * 60) if regime_returns.std() > 0 else 0
            
            # Max DD: Hard to define on discontinuous segments. 
            # We'll use the specific segments' worst drop? 
            # Or just use the global max_dd since bad params might wreck us *between* regimes?
            # Actually, if we hold a position into a bad regime, that's not this regime's fault?
            # For safety, let's use the global MaxDD of the *entire* run to penalize blowing up,
            # but optimize PnL only for the regime.
            
            max_eq = eq.cummax()
            dd = (eq - max_eq) / max_eq
            max_dd = dd.min() * 100
            
            # Fills: We need to know if fills happened in the regime.
            # This is tricky with current MockExchange structure.
            # Approx: total fills * (regime_ticks / total_ticks)? No.
            # Real way: Track fills timing. 
            # For now, let's use global fills * %time as a rough proxy, 
            # OR just return global fills for simplicity (fills usually correlate with activity)
            # Actually, we can't easily filter trade history timestamps without editing MockExchange.
            # Let's stick to PnL and Sharpe being regime-specific.
            fills = len(exchange.trade_history) 
            
        else:
            pnl = eq.iloc[-1] - self.initial_balance
            returns = eq.pct_change().dropna()
            sharpe = (returns.mean() / returns.std()) * np.sqrt(252 * 24 * 60) if returns.std() > 0 else 0
            max_eq = eq.cummax()
            dd = (eq - max_eq) / max_eq
            max_dd = dd.min() * 100
            fills = len(exchange.trade_history)

        return {
            'pnl': round(pnl, 4),
            'sharpe': round(sharpe, 4),
            'fills': fills,
            'cancels': cancel_count,
            'max_dd': round(max_dd, 4)
        }

    def _create_objective(self, regime: str):
        """Create Optuna objective function for a given regime."""

        def objective(trial: optuna.Trial) -> float:
            params = {
                'gamma': trial.suggest_float('gamma', *self.SEARCH_SPACE['gamma']),
                'kappa': trial.suggest_float('kappa', *self.SEARCH_SPACE['kappa']),
                'skew_factor': trial.suggest_float('skew_factor', *self.SEARCH_SPACE['skew_factor']),
                'price_tolerance': trial.suggest_float('price_tolerance', *self.SEARCH_SPACE['price_tolerance']),
                'grid_spacing': trial.suggest_float('grid_spacing', *self.SEARCH_SPACE['grid_spacing']),
                'order_size_mult': trial.suggest_float('order_size_mult', *self.SEARCH_SPACE['order_size_mult']),
                'grid_layers': trial.suggest_int('grid_layers', *self.SEARCH_SPACE['grid_layers']),
                'max_position_mult': trial.suggest_float('max_position_mult', *self.SEARCH_SPACE['max_position_mult']),
            }

            result = asyncio.run(self._run_backtest(self.df_train, params, target_regime=regime))

            # Ensure user attributes are set even for failed trials
            trial.set_user_attr('pnl', result['pnl'])
            trial.set_user_attr('sharpe', result['sharpe'])
            trial.set_user_attr('fills', result['fills'])
            trial.set_user_attr('cancels', result['cancels'])
            trial.set_user_attr('max_dd', result['max_dd'])

            if result['pnl'] <= -999:
                return -1000  # Failed backtest

            # Multi-objective scoring:
            # - PnL (40%): core profitability
            # - Sharpe (30%): risk-adjusted returns
            # - Fill rate (30%): order efficiency
            pnl_score = result['pnl']
            sharpe_score = result['sharpe']
            fill_score = result['fills'] / max(result['cancels'], 1) * 100  # fills per 100 cancels

            # Normalize roughly to same scale
            score = 0.4 * pnl_score + 0.3 * max(sharpe_score, -10) + 0.3 * fill_score

            # Penalty for excessive drawdown
            if result['max_dd'] < -5.0:
                score *= 0.5

            return score

        return objective

    def optimize_regime(self, regime: str, n_trials: int = 50) -> Dict:
        """Optimize parameters for a single regime."""
        logger.info(f"\n{'='*60}")
        logger.info(f"Optimizing: {regime.upper()} ({n_trials} trials)")
        logger.info(f"{'='*60}")

        study_name = f"optimize_{regime}_{os.path.basename(self.data_file).replace('.', '_')}"
        
        study = optuna.create_study(
            study_name=study_name,
            direction='maximize',
            sampler=optuna.samplers.TPESampler(seed=42),
            pruner=optuna.pruners.MedianPruner(),
            storage=self.storage,
            load_if_exists=True
        )

        # Calculate remaining trials if resuming
        completed_trials = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
        remaining_trials = max(0, n_trials - completed_trials)
        
        if remaining_trials == 0:
            logger.info(f"Study '{study_name}' already has {completed_trials} trials. Skipping optimization.")
        else:
            logger.info(f"Resuming study '{study_name}': {completed_trials} complete, running {remaining_trials} more.")
            t0 = time.time()
            study.optimize(self._create_objective(regime), n_trials=remaining_trials, show_progress_bar=True)
            elapsed = time.time() - t0
        
        # Recalculate elapsed for checking connection... actually just use what we have or 0
        elapsed = 0  # Placeholder if we skipped

        best = study.best_trial
        best_params = best.params
        best_params['grid_layers'] = int(best_params['grid_layers'])

        # Run validation (out-of-sample)
        # Note: best.user_attrs might be missing if we just loaded a study without running? 
        # No, because best_trial is from storage.
        
        val_result = asyncio.run(self._run_backtest(self.df_val, best_params, target_regime=regime))

        if val_result.get('status') == 'no_data':
            logger.warning(f"⚠️  Validation skipped for {regime} (not found in validation set)")
            val_pnl_str = "N/A"
            val_sharpe_str = "N/A"
        else:
            val_pnl_str = f"${val_result['pnl']:+.2f}"
            val_sharpe_str = f"{val_result['sharpe']:.2f}"

        logger.info(f"\n🏆 Best for {regime.upper()} (trial #{best.number}):")
        logger.info(f"   Score: {best.value:.4f}")
        logger.info(f"   Train PnL: ${best.user_attrs['pnl']:+.2f} | Sharpe: {best.user_attrs['sharpe']:.2f} | "
                     f"Fills: {best.user_attrs['fills']} | Cancels: {best.user_attrs['cancels']}")
        logger.info(f"   Val   PnL: {val_pnl_str} | Sharpe: {val_sharpe_str} | "
                     f"Fills: {val_result['fills']} | Cancels: {val_result['cancels']}")
        logger.info(f"   Time: {elapsed:.1f}s")
        logger.info(f"   Params:")
        for k, v in sorted(best_params.items()):
            logger.info(f"     {k}: {v}")

        return {
            'params': best_params,
            'train': {
                'pnl': best.user_attrs['pnl'],
                'sharpe': best.user_attrs['sharpe'],
                'fills': best.user_attrs['fills'],
                'cancels': best.user_attrs['cancels'],
                'max_dd': best.user_attrs['max_dd'],
                'score': round(best.value, 4)
            },
            'validation': val_result,
            'n_trials': n_trials,
            'elapsed_seconds': round(elapsed, 1)
        }

    def optimize_all(self, n_trials: int = 50) -> Dict:
        """Optimize all 4 regimes and save results."""
        regimes = ['low_vol', 'trend_up', 'trend_down', 'high_vol']
        results = {}

        total_start = time.time()

        for regime in regimes:
            result = self.optimize_regime(regime, n_trials=n_trials)
            results[regime] = result

        total_elapsed = time.time() - total_start

        # Build output
        output = {
            'optimized_at': datetime.now().isoformat(),
            'data_file': self.data_file,
            'data_range': f"{self.df_train['timestamp'].iloc[0]} to {self.df_val['timestamp'].iloc[-1]}",
            'candles_total': len(self.df_train) + len(self.df_val),
            'candles_train': len(self.df_train),
            'candles_val': len(self.df_val),
            'total_elapsed_seconds': round(total_elapsed, 1),
            'regimes': {}
        }

        for regime, result in results.items():
            output['regimes'][regime] = {
                **result['params'],
                'train_pnl': result['train']['pnl'],
                'train_sharpe': result['train']['sharpe'],
                'train_fills': result['train']['fills'],
                'val_pnl': result['validation']['pnl'],
                'val_sharpe': result['validation']['sharpe'],
                'val_fills': result['validation']['fills'],
                'score': result['train']['score'],
                'n_trials': result['n_trials']
            }

        self.save_params(output)
        self._print_summary(output)
        return output

    def save_params(self, output: Dict):
        """Save optimized parameters to JSON."""
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)

        # Convert numpy/pandas types for JSON serialization
        def convert(obj):
            if isinstance(obj, (np.integer,)): return int(obj)
            if isinstance(obj, (np.floating,)): return float(obj)
            if isinstance(obj, pd.Timestamp): return str(obj)
            return obj

        with open(self.output_path, 'w') as f:
            json.dump(output, f, indent=2, default=convert)

        logger.info(f"\n💾 Saved to {self.output_path}")

    def _print_summary(self, output: Dict):
        """Print a comparative summary of all regimes."""
        logger.info(f"\n{'='*80}")
        logger.info("OPTIMIZATION COMPLETE — SUMMARY")
        logger.info(f"{'='*80}")
        logger.info(f"{'Regime':<12} | {'γ':>5} | {'κ':>6} | {'Skew':>6} | {'Tol%':>6} | {'Spc%':>6} | "
                     f"{'Size':>5} | {'Lay':>4} | {'PosM':>5} | {'T.PnL':>7} | {'V.PnL':>7}")
        logger.info("-" * 80)

        for regime, data in output['regimes'].items():
            logger.info(
                f"{regime:<12} | {data['gamma']:>5.2f} | {data['kappa']:>6.0f} | "
                f"{data['skew_factor']*100:>5.2f}% | {data['price_tolerance']*100:>5.2f}% | "
                f"{data['grid_spacing']*100:>5.3f}% | {data['order_size_mult']:>5.2f} | "
                f"{data['grid_layers']:>4} | {data['max_position_mult']:>5.2f} | "
                f"${data['train_pnl']:>+6.2f} | ${data['val_pnl']:>+6.2f}"
            )

        logger.info(f"\nTotal time: {output['total_elapsed_seconds']:.0f}s")


def load_optimized_params(path: str = OPTIMIZED_PARAMS_PATH) -> Optional[Dict]:
    """
    Load optimized parameters from JSON file.
    Returns regime params dict or None if file doesn't exist.
    """
    if not os.path.exists(path):
        return None

    try:
        with open(path, 'r') as f:
            data = json.load(f)

        regimes = data.get('regimes', {})
        if not regimes:
            return None

        # Extract only the trading params (exclude metrics)
        trading_keys = ['gamma', 'kappa', 'skew_factor', 'price_tolerance',
                        'grid_spacing', 'order_size_mult', 'grid_layers', 'max_position_mult']

        result = {}
        for regime, params in regimes.items():
            result[regime] = {k: params[k] for k in trading_keys if k in params}

        logger.info(f"Loaded optimized params from {path} (optimized at: {data.get('optimized_at', 'unknown')})")
        return result

    except Exception as e:
        logger.warning(f"Failed to load optimized params: {e}")
        return None
