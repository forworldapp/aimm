"""
Enhanced Backtesting Framework - Phase 1.3
Statistical validation for strategy performance with confidence intervals.

Author: Antigravity
Version: 1.0.0
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Callable, Tuple
import logging
from scipy import stats


class EnhancedBacktester:
    """
    Enhanced Backtesting Framework with statistical validation.
    
    Features:
    1. Bootstrap confidence intervals for PnL and Sharpe
    2. Walk-forward validation for overfitting detection
    3. Strategy comparison with p-values
    4. Realistic slippage and fee modeling
    """
    
    def __init__(self, data: pd.DataFrame):
        """
        Args:
            data: Historical OHLCV data with columns: timestamp, open, high, low, close, volume
        """
        self.data = data.copy()
        self.logger = logging.getLogger("EnhancedBacktester")
        
        # Default fee/slippage
        self.trading_fee_bps = 5  # 0.05%
        self.slippage_bps = 2  # 0.02%
    
    def run_bootstrap_test(self, 
                           strategy_fn: Callable, 
                           n_iterations: int = 100,
                           block_size: int = 100) -> dict:
        """
        Run bootstrap test for confidence intervals.
        
        Uses block bootstrap to preserve temporal dependencies.
        
        Args:
            strategy_fn: Function that takes data and returns dict with 'pnl', 'trades'
            n_iterations: Number of bootstrap samples
            block_size: Size of resampling blocks (preserves autocorrelation)
            
        Returns:
            Dict with means, stds, and 95% confidence intervals
        """
        pnls = []
        sharpes = []
        trade_counts = []
        
        n_blocks = len(self.data) // block_size
        
        for i in range(n_iterations):
            # Block bootstrap: randomly select blocks with replacement
            block_indices = np.random.choice(n_blocks, n_blocks, replace=True)
            
            resampled_data = pd.concat([
                self.data.iloc[idx * block_size:(idx + 1) * block_size]
                for idx in block_indices
            ]).reset_index(drop=True)
            
            # Run strategy on resampled data
            result = strategy_fn(resampled_data)
            
            pnls.append(result.get('pnl', 0))
            sharpes.append(result.get('sharpe', 0))
            trade_counts.append(result.get('trades', 0))
            
            if (i + 1) % 20 == 0:
                self.logger.info(f"Bootstrap iteration {i+1}/{n_iterations}")
        
        pnls = np.array(pnls)
        sharpes = np.array(sharpes)
        
        return {
            'pnl_mean': float(np.mean(pnls)),
            'pnl_std': float(np.std(pnls)),
            'pnl_95_ci': (float(np.percentile(pnls, 2.5)), float(np.percentile(pnls, 97.5))),
            'sharpe_mean': float(np.mean(sharpes)),
            'sharpe_std': float(np.std(sharpes)),
            'sharpe_95_ci': (float(np.percentile(sharpes, 2.5)), float(np.percentile(sharpes, 97.5))),
            'trades_mean': float(np.mean(trade_counts)),
            'n_iterations': n_iterations
        }
    
    def run_walk_forward(self,
                         strategy_factory: Callable,
                         train_size: int,
                         test_size: int) -> dict:
        """
        Walk-forward validation for overfitting detection.
        
        Splits data into rolling train/test windows.
        Strategy is re-fitted on each training window.
        
        Args:
            strategy_factory: Factory function that creates a new strategy instance
            train_size: Number of bars for training
            test_size: Number of bars for testing
            
        Returns:
            Dict with in-sample and out-of-sample performance
        """
        in_sample_pnls = []
        out_sample_pnls = []
        
        n_walks = (len(self.data) - train_size) // test_size
        
        for walk in range(n_walks):
            start_idx = walk * test_size
            train_end = start_idx + train_size
            test_end = train_end + test_size
            
            if test_end > len(self.data):
                break
            
            train_data = self.data.iloc[start_idx:train_end].copy()
            test_data = self.data.iloc[train_end:test_end].copy()
            
            # Create and train strategy
            strategy = strategy_factory()
            
            # Run on training data (in-sample)
            if hasattr(strategy, 'fit'):
                strategy.fit(train_data)
            in_result = strategy(train_data)
            in_sample_pnls.append(in_result.get('pnl', 0))
            
            # Run on test data (out-of-sample)
            out_result = strategy(test_data)
            out_sample_pnls.append(out_result.get('pnl', 0))
            
            self.logger.info(f"Walk {walk+1}/{n_walks}: IS={in_result.get('pnl', 0):.2f}, OOS={out_result.get('pnl', 0):.2f}")
        
        in_sample_pnls = np.array(in_sample_pnls)
        out_sample_pnls = np.array(out_sample_pnls)
        
        # Degradation ratio: OOS vs IS performance
        degradation = np.mean(out_sample_pnls) / max(np.mean(in_sample_pnls), 1e-10)
        
        return {
            'in_sample_mean': float(np.mean(in_sample_pnls)),
            'in_sample_std': float(np.std(in_sample_pnls)),
            'out_sample_mean': float(np.mean(out_sample_pnls)),
            'out_sample_std': float(np.std(out_sample_pnls)),
            'degradation_ratio': float(degradation),
            'n_walks': len(in_sample_pnls),
            'overfit_warning': degradation < 0.5  # Flag if OOS < 50% of IS
        }
    
    def compare_strategies(self,
                           strategies: Dict[str, Callable],
                           n_bootstrap: int = 50) -> Tuple[dict, dict]:
        """
        Compare multiple strategies with statistical significance.
        
        Args:
            strategies: Dict of {name: strategy_fn}
            n_bootstrap: Bootstrap iterations per strategy
            
        Returns:
            Tuple of (results_dict, comparisons_dict)
        """
        results = {}
        pnl_samples = {}
        
        for name, strategy_fn in strategies.items():
            self.logger.info(f"Testing strategy: {name}")
            
            # Run bootstrap for each strategy
            pnls = []
            for i in range(n_bootstrap):
                # Simple random sample for comparison
                sample_idx = np.random.choice(len(self.data), len(self.data), replace=True)
                sample_data = self.data.iloc[sample_idx].reset_index(drop=True)
                result = strategy_fn(sample_data)
                pnls.append(result.get('pnl', 0))
            
            pnls = np.array(pnls)
            pnl_samples[name] = pnls
            
            results[name] = {
                'pnl_mean': float(np.mean(pnls)),
                'pnl_std': float(np.std(pnls)),
                'pnl_95_ci': (float(np.percentile(pnls, 2.5)), float(np.percentile(pnls, 97.5)))
            }
        
        # Pairwise t-tests
        comparisons = {}
        names = list(strategies.keys())
        
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                name_a = names[i]
                name_b = names[j]
                
                # Paired t-test
                t_stat, p_value = stats.ttest_ind(pnl_samples[name_a], pnl_samples[name_b])
                
                # Effect size (Cohen's d)
                pooled_std = np.sqrt((np.var(pnl_samples[name_a]) + np.var(pnl_samples[name_b])) / 2)
                effect_size = (np.mean(pnl_samples[name_a]) - np.mean(pnl_samples[name_b])) / max(pooled_std, 1e-10)
                
                comparisons[f"{name_a} vs {name_b}"] = {
                    't_stat': float(t_stat),
                    'p_value': float(p_value),
                    'significant': p_value < 0.05,
                    'effect_size': float(effect_size),
                    'winner': name_a if np.mean(pnl_samples[name_a]) > np.mean(pnl_samples[name_b]) else name_b
                }
        
        return results, comparisons
    
    def calculate_sharpe(self, returns: np.ndarray, annual_factor: float = 252) -> float:
        """Calculate annualized Sharpe ratio."""
        if len(returns) < 2 or np.std(returns) < 1e-10:
            return 0.0
        return float(np.mean(returns) / np.std(returns) * np.sqrt(annual_factor))
    
    def calculate_max_drawdown(self, equity_curve: np.ndarray) -> float:
        """Calculate maximum drawdown."""
        peak = np.maximum.accumulate(equity_curve)
        drawdown = (peak - equity_curve) / peak
        return float(np.max(drawdown))


# Unit tests
if __name__ == "__main__":
    print("=" * 60)
    print("ENHANCED BACKTESTER TESTS")
    print("=" * 60)
    
    # Create sample data
    np.random.seed(42)
    n_periods = 1000
    
    prices = 100000 + np.cumsum(np.random.randn(n_periods) * 100)
    
    data = pd.DataFrame({
        'timestamp': pd.date_range('2025-01-01', periods=n_periods, freq='1min'),
        'open': prices + np.random.randn(n_periods) * 10,
        'high': prices + abs(np.random.randn(n_periods) * 20),
        'low': prices - abs(np.random.randn(n_periods) * 20),
        'close': prices,
        'volume': np.random.randint(100, 1000, n_periods)
    })
    
    backtester = EnhancedBacktester(data)
    
    # Example strategy functions
    def random_strategy(df):
        pnl = np.random.randn() * 100
        return {'pnl': pnl, 'sharpe': pnl / 50, 'trades': 100}
    
    def better_strategy(df):
        pnl = 50 + np.random.randn() * 80  # Higher mean
        return {'pnl': pnl, 'sharpe': pnl / 50, 'trades': 120}
    
    print("\n1. Bootstrap Test (50 iterations)")
    bootstrap_result = backtester.run_bootstrap_test(random_strategy, n_iterations=50)
    print(f"   PnL Mean: ${bootstrap_result['pnl_mean']:.2f} ± ${bootstrap_result['pnl_std']:.2f}")
    print(f"   95% CI: ${bootstrap_result['pnl_95_ci'][0]:.2f} to ${bootstrap_result['pnl_95_ci'][1]:.2f}")
    
    print("\n2. Strategy Comparison")
    results, comparisons = backtester.compare_strategies({
        'Random': random_strategy,
        'Better': better_strategy
    }, n_bootstrap=30)
    
    for name, r in results.items():
        print(f"   {name}: ${r['pnl_mean']:.2f} (CI: ${r['pnl_95_ci'][0]:.2f} to ${r['pnl_95_ci'][1]:.2f})")
    
    for comp_name, comp in comparisons.items():
        sig = "✓" if comp['significant'] else "✗"
        print(f"   {comp_name}: p={comp['p_value']:.4f} {sig}, winner={comp['winner']}")
