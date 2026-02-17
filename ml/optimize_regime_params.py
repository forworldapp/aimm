#!/usr/bin/env python
"""
REGIME_PARAMS Optimization Runner
-----------------------------------
CLI to optimize market-making parameters per regime using 1-minute BTC data.

Usage:
    # Full optimization (4 regimes × 50 trials, ~40 min)
    python ml/optimize_regime_params.py

    # Quick test (5 trials per regime, ~5 min)
    python ml/optimize_regime_params.py --n-trials 5

    # Single regime
    python ml/optimize_regime_params.py --regime trend_down --n-trials 30

    # Custom data size
    python ml/optimize_regime_params.py --candles 20000 --n-trials 100
"""
import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.param_optimizer import RegimeParamOptimizer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("OptimizeParams")


def main():
    parser = argparse.ArgumentParser(description="Optimize REGIME_PARAMS via backtest")
    parser.add_argument('--regime', type=str, default=None,
                        choices=['low_vol', 'trend_up', 'trend_down', 'high_vol'],
                        help="Optimize a single regime (default: all)")
    parser.add_argument('--n-trials', type=int, default=50,
                        help="Number of Optuna trials per regime (default: 50)")
    parser.add_argument('--candles', type=int, default=10000,
                        help="Number of 1-minute candles to use (default: 10000 = ~7 days)")
    parser.add_argument('--data', type=str, default="data/btcusdt_1m_1year.csv",
                        help="Path to 1-minute OHLCV data CSV")
    parser.add_argument('--output', type=str, default="data/optimized_regime_params.json",
                        help="Output JSON path")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("REGIME_PARAMS OPTIMIZER")
    logger.info("=" * 60)
    logger.info(f"Data: {args.data}")
    logger.info(f"Candles: {args.candles}")
    logger.info(f"Trials: {args.n_trials}")
    logger.info(f"Regime: {args.regime or 'ALL'}")
    logger.info(f"Output: {args.output}")
    logger.info("=" * 60)

    optimizer = RegimeParamOptimizer(
        data_file=args.data,
        candles=args.candles,
        output_path=args.output
    )

    if args.regime:
        # Optimize single regime
        result = optimizer.optimize_regime(args.regime, n_trials=args.n_trials)
        
        # Load existing params or create new
        import json
        existing = {}
        if os.path.exists(args.output):
            with open(args.output, 'r') as f:
                existing = json.load(f)
        
        if 'regimes' not in existing:
            existing['regimes'] = {}
        
        existing['regimes'][args.regime] = {
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
        existing['optimized_at'] = __import__('datetime').datetime.now().isoformat()
        
        optimizer.save_params(existing)
        logger.info(f"\n✅ {args.regime} optimized and saved!")
    else:
        # Optimize all regimes
        results = optimizer.optimize_all(n_trials=args.n_trials)
        logger.info(f"\n✅ All regimes optimized! Results saved to {args.output}")

    logger.info("\nTo apply: restart the bot — it will auto-load optimized params.")


if __name__ == "__main__":
    main()
