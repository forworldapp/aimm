"""
Chunked Optuna Regime Optimizer
-------------------------------
Runs optimization ONE REGIME AT A TIME to avoid timeouts.
Each regime result is saved immediately to a partial JSON file.
After all 4 regimes complete, results are merged and applied.

Usage:
    python backtest/run_optuna_chunked.py low_vol
    python backtest/run_optuna_chunked.py trend_up
    python backtest/run_optuna_chunked.py trend_down
    python backtest/run_optuna_chunked.py high_vol
    python backtest/run_optuna_chunked.py --merge   # merge & apply all
    python backtest/run_optuna_chunked.py --all     # run all 4 + merge sequentially
"""
import argparse
import asyncio
import json
import logging
import os
import sys
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.param_optimizer import RegimeParamOptimizer, OPTIMIZED_PARAMS_PATH

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("ChunkedOptimizer")

PARTIAL_DIR = os.path.join("data", "optuna_partial")
ALL_REGIMES = ['low_vol', 'trend_up', 'trend_down', 'high_vol']


STORAGE_URL = "sqlite:///data/optuna_studies.db"

def optimize_single_regime(regime: str, n_trials: int, candles: int):
    """Optimize one regime and save partial result."""
    os.makedirs(PARTIAL_DIR, exist_ok=True)
    os.makedirs("data", exist_ok=True)
    
    logger.info(f"{'='*60}")
    logger.info(f"CHUNKED OPTIMIZATION: {regime.upper()}")
    logger.info(f"  Trials: {n_trials}, Candles: {candles}")
    logger.info(f"  Storage: {STORAGE_URL}")
    logger.info(f"{'='*60}")
    
    t0 = time.time()
    
    optimizer = RegimeParamOptimizer(
        candles=candles,
        train_ratio=0.7,
        storage=STORAGE_URL
    )
    
    # Check if regime exists in training data
    regime_counts = optimizer.df_train['regime'].value_counts()
    if regime not in regime_counts or regime_counts[regime] < 50:
        logger.warning(f"⚠️  Regime '{regime}' not found (or <50 samples) in loaded data. Skipping.")
        logger.info(f"Available regimes: {regime_counts.to_dict()}")
        return None
        
    result = optimizer.optimize_regime(regime, n_trials=n_trials)
    if not result:
        logger.warning(f"Optimization failed or skipped for {regime}.")
        return None

    elapsed = time.time() - t0
    # If we resumed and did 0 trials, elapsed is small. Use stored elapsed if available? 
    # Hard to track cumulative elapsed without more complex logic. 
    # accepted limitation: resuming resets the "elapsed" timer for that chunk.

    
    # Save partial result
    partial = {
        'regime': regime,
        'optimized_at': datetime.now().isoformat(),
        'n_trials': n_trials,
        'candles': candles,
        'elapsed_seconds': round(elapsed, 1),
        'params': result['params'],
        'train': result['train'],
        'validation': result['validation']
    }
    
    partial_path = os.path.join(PARTIAL_DIR, f"{regime}.json")
    with open(partial_path, 'w') as f:
        json.dump(partial, f, indent=2, default=_convert)
    
    logger.info(f"\n✅ {regime.upper()} complete in {elapsed:.0f}s")
    logger.info(f"   Train PnL: ${result['train']['pnl']:+.2f} | Val PnL: ${result['validation']['pnl']:+.2f}")
    logger.info(f"   Fills: {result['train']['fills']} | F/C: {result['train']['fills']/max(result['train']['cancels'],1):.4f}")
    logger.info(f"   Saved to {partial_path}")
    
    return partial


def merge_results():
    """Merge all partial results into the final optimized_regime_params.json."""
    logger.info(f"\n{'='*60}")
    logger.info("MERGING PARTIAL RESULTS")
    logger.info(f"{'='*60}")
    
    regimes = {}
    metadata = {}
    
    for regime in ALL_REGIMES:
        partial_path = os.path.join(PARTIAL_DIR, f"{regime}.json")
        if not os.path.exists(partial_path):
            logger.warning(f"⚠️  Missing: {partial_path} — skipping {regime}")
            continue
        
        with open(partial_path, 'r') as f:
            data = json.load(f)
        
        regimes[regime] = {
            **data['params'],
            'train_pnl': data['train']['pnl'],
            'train_sharpe': data['train']['sharpe'],
            'train_fills': data['train']['fills'],
            'val_pnl': data['validation']['pnl'],
            'val_sharpe': data['validation']['sharpe'],
            'val_fills': data['validation']['fills'],
            'score': data['train']['score'],
            'n_trials': data['n_trials']
        }
        metadata[regime] = {
            'optimized_at': data['optimized_at'],
            'elapsed': data['elapsed_seconds']
        }
    
    if not regimes:
        logger.error("No partial results found! Run individual regimes first.")
        # Don't fail, just return (maybe we only optimized 1 regime)
        return None
    
    total_elapsed = sum(m['elapsed'] for m in metadata.values())
    
    output = {
        'optimized_at': datetime.now().isoformat(),
        'data_file': 'data/btcusdt_1m_1year.csv',
        'total_elapsed_seconds': round(total_elapsed, 1),
        'regimes': regimes
    }
    
    # Save merged result
    os.makedirs(os.path.dirname(OPTIMIZED_PARAMS_PATH), exist_ok=True)
    with open(OPTIMIZED_PARAMS_PATH, 'w') as f:
        json.dump(output, f, indent=2, default=_convert)
    
    # Print summary
    logger.info(f"\n{'='*80}")
    logger.info("MERGED OPTIMIZATION RESULTS")
    logger.info(f"{'='*80}")
    logger.info(f"{'Regime':<12} | {'Tol%':>6} | {'Spc%':>6} | {'γ':>5} | {'κ':>6} | "
                f"{'T.PnL':>7} | {'V.PnL':>7} | {'Fills':>5} | {'Score':>6}")
    logger.info("-" * 80)
    
    for regime, data in regimes.items():
        logger.info(
            f"{regime:<12} | {data['price_tolerance']*100:>5.2f}% | "
            f"{data['grid_spacing']*100:>5.3f}% | {data['gamma']:>5.2f} | "
            f"{data['kappa']:>6.0f} | ${data['train_pnl']:>+6.2f} | "
            f"${data['val_pnl']:>+6.2f} | {data['train_fills']:>5} | "
            f"{data['score']:>6.2f}"
        )
    
    logger.info(f"\nTotal time: {total_elapsed:.0f}s")
    logger.info(f"💾 Saved to {OPTIMIZED_PARAMS_PATH}")
    logger.info(f"\nFound {len(regimes)}/4 regimes.")
    
    if len(regimes) < 4:
        missing = [r for r in ALL_REGIMES if r not in regimes]
        logger.warning(f"Missing regimes: {missing}")
    
    return output


def _convert(obj):
    """JSON serialization helper."""
    import numpy as np
    import pandas as pd
    if isinstance(obj, (np.integer,)): return int(obj)
    if isinstance(obj, (np.floating,)): return float(obj)
    if isinstance(obj, pd.Timestamp): return str(obj)
    return obj


def main():
    parser = argparse.ArgumentParser(description='Chunked Optuna Regime Optimizer')
    parser.add_argument('regime', nargs='?', help='Regime to optimize (low_vol, trend_up, trend_down, high_vol)')
    parser.add_argument('--merge', action='store_true', help='Merge all partial results')
    parser.add_argument('--all', action='store_true', help='Run all regimes sequentially + merge')
    parser.add_argument('--trials', type=int, default=20, help='Number of Optuna trials per regime (default: 20)')
    parser.add_argument('--candles', type=int, default=35000, help='Number of 1-min candles (default: 35000 ≈ 24 days)')
    args = parser.parse_args()

    
    if args.all:
        logger.info("Running ALL 4 regimes sequentially...")
        for regime in ALL_REGIMES:
            optimize_single_regime(regime, args.trials, args.candles)
        merge_results()
    elif args.merge:
        merge_results()
    elif args.regime:
        if args.regime not in ALL_REGIMES:
            logger.error(f"Unknown regime: {args.regime}. Choose from: {ALL_REGIMES}")
            sys.exit(1)
        optimize_single_regime(args.regime, args.trials, args.candles)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
