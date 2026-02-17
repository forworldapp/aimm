
"""
Chunked Multi-Period Sweep
--------------------------
Runs the parameter sweep on one period at a time to prevent long-running process interruptions.
Saves partial results to data/sweep_partials/ then merges them.

Periods:
1. Jan 2025 (Calm)
2. May 2025 (Trending)
3. Oct 2025 (Volatile)
4. Dec 2025 (Range)
"""
import asyncio
import pandas as pd
import numpy as np
import logging
import os
import sys
import time
import json
import argparse
from itertools import product
from datetime import datetime

# Adjust path to find modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtest.mock_exchange import MockExchange
from strategies.market_maker import MarketMaker
from core.config import Config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("SweepChunked")

# ── 1. Data Conversion ───────────────────────────────────────────
def convert_1m(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Simple bid/ask estimation
    range_pct = (df['high'] - df['low']) / df['close']
    spread = range_pct * 0.1
    spread = spread.clip(0.0001, 0.005)
    df['best_bid'] = df['close'] * (1 - spread/2)
    df['best_ask'] = df['close'] * (1 + spread/2)
    
    ts_col = 'timestamp' if 'timestamp' in df.columns else 'open_time'
    df['timestamp'] = pd.to_datetime(df[ts_col], unit='ms')
    return df[['timestamp', 'open', 'high', 'low', 'close', 'volume', 'best_bid', 'best_ask']]


# ── 2. The Backtest Logic (Same as before) ──────────────────────
async def run_single(df, tolerance, spacing, initial_balance=10000.0):
    exchange = MockExchange(df.copy(), initial_balance=initial_balance)
    
    # Disable heavy logging/extras for speed
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

    # Disable sub-components
    strategy.regime_detector = None
    strategy.adaptive_tuner = None
    strategy.ml_strategy = None
    strategy.order_flow_analyzer = None
    strategy.funding_arb = None
    strategy.microstructure = None
    strategy.execution_algo = None
    strategy.notifier = None

    strategy.max_loss_usd = 5000.0
    strategy.grid_layers = 7
    strategy.order_size_usd = 200
    strategy.spread_pct = 0.0015

    # Set parameters being tested
    strategy._ml_price_tolerance = tolerance
    strategy._ml_grid_spacing = spacing
    strategy._ml_order_size_mult = 0.8  # Fixed
    strategy._ml_grid_layers = 7       # Fixed
    strategy._ml_max_position_mult = 1.0 # Fixed
    strategy._ml_skew_factor = 0.008   # Fixed

    equity_history = []
    cancel_count = 0

    orig_cancel = exchange.cancel_all_orders
    async def counted_cancel(symbol):
        nonlocal cancel_count
        cancel_count += 1
        return await orig_cancel(symbol)
    exchange.cancel_all_orders = counted_cancel

    strategy.is_running = True

    try:
        while exchange.next_tick():
            await strategy.cycle()
            row = df.iloc[exchange.current_index]
            mid = (row['best_bid'] + row['best_ask']) / 2
            equity = exchange.balance['USDT'] + (exchange.position['amount'] * mid)
            equity_history.append(equity)
    except Exception:
        pass

    if not equity_history:
        return None

    eq = pd.Series(equity_history)
    final_equity = eq.iloc[-1]
    pnl = final_equity - initial_balance
    pnl_pct = (pnl / initial_balance) * 100

    returns = eq.pct_change().dropna()
    if returns.std() > 0:
        sharpe = (returns.mean() / returns.std()) * np.sqrt(252 * 24 * 60)
    else:
        sharpe = 0
        
    max_eq = eq.cummax()
    drawdown = (eq - max_eq) / max_eq
    max_dd = drawdown.min() * 100
    fills = len(exchange.trade_history)

    return {
        'tolerance': tolerance,
        'spacing': spacing,
        'pnl': round(pnl, 2),
        'pnl_pct': round(pnl_pct, 2),
        'sharpe': round(sharpe, 2),
        'max_dd': round(max_dd, 2),
        'fills': fills,
        'cancel_cycles': cancel_count,
        'fills_per_cancel': round(fills / max(cancel_count, 1), 4),
        'final_equity': round(final_equity, 2)
    }


# ── 3. Chunk Execution Logic ─────────────────────────────────────
async def run_period_chunk(period_name, df, tolerances, spacings):
    combos = list(product(tolerances, spacings))
    logger.info(f"Starting {period_name} ({len(df)} candles, {len(combos)} combos)")
    
    results = []
    
    # Ensure directory exists for partials
    os.makedirs("data/sweep_partials", exist_ok=True)
    out_file = f"data/sweep_partials/results_{period_name}.json"
    
    # Check if partial results already exist to resume? 
    # For now, just overwrite or maybe append if we were fancy, but overwrite is simpler for chunking per period.
    
    for i, (tol, spc) in enumerate(combos):
        t0 = time.time()
        res = await run_single(df, tol, spc)
        elapsed = time.time() - t0
        
        if res:
            res['period'] = period_name
            results.append(res)
            logger.info(
                f"[{i+1}/{len(combos)}] {period_name} | "
                f"tol={tol*100:.2f}% spc={spc*100:.2f}% | "
                f"PnL=${res['pnl']:+6.2f} | Fills={res['fills']:3d} | "
                f"Sharpe={res['sharpe']:4.2f} | {elapsed:.1f}s"
            )
            
    # Save partial
    with open(out_file, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved partial results to {out_file}")
    return results


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--period", type=str, help="Specific period to run (Jan2025_Calm, May2025_Trending, etc.)")
    parser.add_argument("--merge", action="store_true", help="Merge all partial results")
    args = parser.parse_args()

    # Define all periods logic
    data_file = "data/btcusdt_1m_1year.csv"
    if not os.path.exists(data_file):
        logger.error(f"Data file not found: {data_file}")
        return

    # REGIME DEFINITIONS
    # We will load data only if needed to save memory if running single chunk?
    # Actually loading 1 year 1m data is fine, it's about 500k rows, ~50MB RAM.
    
    logger.info("Loading data...")
    try:
        raw = pd.read_csv(data_file)
        raw['dt'] = pd.to_datetime(raw['timestamp'], unit='ms')
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return

    periods_map = {
        "Jan2025_Calm": (raw['dt'] >= '2025-01-01') & (raw['dt'] < '2025-02-01'),
        "May2025_Trending": (raw['dt'] >= '2025-05-01') & (raw['dt'] < '2025-06-01'),
        "Oct2025_Volatile": (raw['dt'] >= '2025-10-01') & (raw['dt'] < '2025-11-01'),
        "Dec2025_Range":    (raw['dt'] >= '2025-12-01') & (raw['dt'] < '2026-01-01'),
    }

    # PARAMETER GRID
    tolerances = [0.005, 0.008, 0.010, 0.0125, 0.015]
    spacings   = [0.0015, 0.0020, 0.0030]

    if args.period:
        if args.period not in periods_map:
            logger.error(f"Period {args.period} not found. Available: {list(periods_map.keys())}")
            return
            
        mask = periods_map[args.period]
        period_df = raw[mask].reset_index(drop=True)
        if len(period_df) < 100:
            logger.error(f"Period {args.period} data empty!")
            return
            
        period_df = convert_1m(period_df)
        await run_period_chunk(args.period, period_df, tolerances, spacings)

    elif args.merge:
        logger.info("Merging partial results...")
        all_res = []
        partial_dir = "data/sweep_partials"
        if not os.path.exists(partial_dir):
            logger.error("No partials directory found")
            return
            
        for f in os.listdir(partial_dir):
            if f.endswith(".json"):
                try:
                    with open(os.path.join(partial_dir, f), 'r') as jf:
                        all_res.extend(json.load(jf))
                except Exception as e:
                    logger.error(f"Error reading {f}: {e}")
        
        if not all_res:
            logger.info("No results to merge.")
            return
            
        df = pd.DataFrame(all_res)
        df.to_csv("data/sweep_multi_period_results.csv", index=False)
        logger.info(f"Merged {len(df)} results to data/sweep_multi_period_results.csv")
        
        # Determine best
        if 'pnl' in df.columns:
            cross = df.groupby(['tolerance', 'spacing']).agg({
                'pnl': 'mean', 'sharpe': 'mean', 'fills': 'mean', 'max_dd': 'mean'
            }).reset_index().sort_values('pnl', ascending=False)
            
            best = cross.iloc[0]
            logger.info(f"\n🏆 BEST PARAMETERS (avg across periods):")
            logger.info(f"   tol={best['tolerance']*100:.2f}% spc={best['spacing']*100:.2f}%")
            logger.info(f"   Mean PnL=${best['pnl']:.2f} | Mean Sharpe={best['sharpe']:.2f}")

    else:
        # Run ALL sequentially
        for pname in periods_map.keys():
            logger.info(f"\nStarting automatic sequential run for: {pname}")
            mask = periods_map[pname]
            period_df = raw[mask].reset_index(drop=True)
            period_df = convert_1m(period_df)
            await run_period_chunk(pname, period_df, tolerances, spacings)
            
        logger.info("All chunks done.")


if __name__ == "__main__":
    asyncio.run(main())
