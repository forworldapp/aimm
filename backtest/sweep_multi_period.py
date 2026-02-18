"""
Multi-Period Tolerance/Spacing Sweep — Oct 2025 (volatile) + Jan-Feb 2026 (sideways)
=====================================================================================
Runs identical parameter grid on two distinct market regimes to find
robust parameters that work across conditions.
"""
import asyncio
import pandas as pd
import numpy as np
import logging
import os
import sys
import time
import json
from itertools import product
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtest.mock_exchange import MockExchange
from strategies.market_maker import MarketMaker
from core.config import Config

logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("MultiPeriod")
logger.setLevel(logging.INFO)

# ── Data conversion ──────────────────────────────────────────────
def convert_1m(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    range_pct = (df['high'] - df['low']) / df['close']
    spread = range_pct * 0.1
    spread = spread.clip(0.0001, 0.005)
    df['best_bid'] = df['close'] * (1 - spread/2)
    df['best_ask'] = df['close'] * (1 + spread/2)
    ts_col = 'timestamp' if 'timestamp' in df.columns else 'open_time'
    df['timestamp'] = pd.to_datetime(df[ts_col], unit='ms')
    return df[['timestamp', 'open', 'high', 'low', 'close', 'volume', 'best_bid', 'best_ask']]


# ── Single backtest run ──────────────────────────────────────────
async def run_single(df, tolerance, spacing, initial_balance=10000.0):
    exchange = MockExchange(df.copy(), initial_balance=initial_balance)
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

    strategy._ml_price_tolerance = tolerance
    strategy._ml_grid_spacing = spacing
    strategy._ml_order_size_mult = 0.8
    strategy._ml_grid_layers = 7
    strategy._ml_max_position_mult = 1.0
    strategy._ml_skew_factor = 0.008

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
    sharpe = (returns.mean() / returns.std()) * np.sqrt(252 * 24 * 60) if returns.std() > 0 else 0
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


# ── Period sweep ─────────────────────────────────────────────────
async def sweep_period(period_name, df, tolerances, spacings):
    """Run all parameter combos on one period, return list of result dicts."""
    combos = list(product(tolerances, spacings))
    logger.info(f"\n{'='*90}")
    logger.info(f"  PERIOD: {period_name}  |  {len(df):,} candles  |  {df['timestamp'].iloc[0].date()} → {df['timestamp'].iloc[-1].date()}")
    logger.info(f"  Price: ${df['low'].min():,.0f} – ${df['high'].max():,.0f}  |  {len(combos)} combos")
    logger.info(f"{'='*90}")

    results = []
    for i, (tol, spc) in enumerate(combos):
        t0 = time.time()
        result = await run_single(df, tol, spc)
        elapsed = time.time() - t0

        if result:
            result['period'] = period_name
            results.append(result)
            logger.info(
                f"  [{i+1:2d}/{len(combos)}] tol={tol*100:.2f}% spc={spc*100:.2f}% | "
                f"PnL=${result['pnl']:+8.2f} | Fills={result['fills']:4d} | "
                f"Cancels={result['cancel_cycles']:5d} | Sharpe={result['sharpe']:6.2f} | {elapsed:.0f}s"
            )

    results.sort(key=lambda r: r['pnl'], reverse=True)
    return results


# ── Main ─────────────────────────────────────────────────────────
async def main():
    data_file = "data/btcusdt_1m_1year.csv"
    if not os.path.exists(data_file):
        logger.error(f"Data file not found: {data_file}")
        return

    logger.info("Loading 1-minute data...")
    raw = pd.read_csv(data_file)
    raw['dt'] = pd.to_datetime(raw['timestamp'], unit='ms')
    logger.info(f"Total: {len(raw):,} rows  ({raw['dt'].min()} → {raw['dt'].max()})")

    # ── Define periods ───────────────────────────────────────────
    periods = {
        "Oct2025_Volatile": (raw['dt'] >= '2025-10-01') & (raw['dt'] < '2025-11-01'),
        "Jan-Feb2026_Sideways": (raw['dt'] >= '2026-01-01') & (raw['dt'] <= raw['dt'].max()),
    }

    # ── Parameter grid (focused on sweet spot from prior sweep) ──
    tolerances = [0.005, 0.008, 0.010, 0.0125, 0.015]
    spacings = [0.0015, 0.0020, 0.0030]

    all_results = []

    for period_name, mask in periods.items():
        period_df = raw[mask].reset_index(drop=True)
        if len(period_df) < 100:
            logger.warning(f"Skipping {period_name}: only {len(period_df)} rows")
            continue

        period_df = convert_1m(period_df)
        results = await sweep_period(period_name, period_df, tolerances, spacings)
        all_results.extend(results)

    if not all_results:
        logger.error("No results!")
        return

    # ── Summary tables ───────────────────────────────────────────
    results_df = pd.DataFrame(all_results)

    logger.info(f"\n{'='*110}")
    logger.info("MULTI-PERIOD COMPARISON — TOP 5 PER PERIOD")
    logger.info(f"{'='*110}")
    logger.info(f"{'Period':<25} {'Rank':>4} | {'Tol%':>6} | {'Spc%':>6} | {'PnL$':>10} | {'Sharpe':>7} | {'MaxDD%':>7} | {'Fills':>6} | {'Cancels':>8}")
    logger.info("-" * 110)

    for period in results_df['period'].unique():
        pdf = results_df[results_df['period'] == period].sort_values('pnl', ascending=False)
        for i, (_, r) in enumerate(pdf.head(5).iterrows()):
            logger.info(
                f"{period:<25} {i+1:>4} | {r['tolerance']*100:>5.2f}% | {r['spacing']*100:>5.2f}% | "
                f"${r['pnl']:>+9.2f} | {r['sharpe']:>7.2f} | {r['max_dd']:>6.2f}% | "
                f"{r['fills']:>6} | {r['cancel_cycles']:>8}"
            )
        logger.info("-" * 110)

    # ── Cross-period ranking: average PnL per combo ──────────────
    logger.info(f"\n{'='*110}")
    logger.info("CROSS-PERIOD RANKING — Average Metrics Across Both Periods")
    logger.info(f"{'='*110}")

    cross = results_df.groupby(['tolerance', 'spacing']).agg({
        'pnl': 'mean',
        'sharpe': 'mean',
        'fills': 'mean',
        'cancel_cycles': 'mean',
        'max_dd': 'mean',
        'fills_per_cancel': 'mean'
    }).reset_index()
    cross = cross.sort_values('pnl', ascending=False)

    logger.info(f"{'Rank':>4} | {'Tol%':>6} | {'Spc%':>6} | {'AvgPnL$':>10} | {'AvgSharpe':>10} | {'AvgMaxDD%':>10} | {'AvgFills':>9} | {'AvgCancel':>10} | {'AvgF/C':>8}")
    logger.info("-" * 110)
    for i, (_, r) in enumerate(cross.iterrows()):
        marker = " ◀ BEST" if i == 0 else ""
        logger.info(
            f"{i+1:>4} | {r['tolerance']*100:>5.2f}% | {r['spacing']*100:>5.2f}% | "
            f"${r['pnl']:>+9.2f} | {r['sharpe']:>10.2f} | {r['max_dd']:>9.2f}% | "
            f"{r['fills']:>9.0f} | {r['cancel_cycles']:>10.0f} | {r['fills_per_cancel']:>8.4f}{marker}"
        )

    # ── Save results ─────────────────────────────────────────────
    out_csv = "data/sweep_multi_period_results.csv"
    results_df.to_csv(out_csv, index=False)
    logger.info(f"\nDetailed results → {out_csv}")

    # Save cross-period summary JSON
    best_row = cross.iloc[0]
    summary = {
        "run_at": datetime.now().isoformat(),
        "periods": {
            name: {
                "candles": int(mask.sum()),
                "days": round(mask.sum()/1440, 1)
            }
            for name, mask in periods.items()
        },
        "best_cross_period": {
            "tolerance": float(best_row['tolerance']),
            "spacing": float(best_row['spacing']),
            "avg_pnl": round(float(best_row['pnl']), 2),
            "avg_sharpe": round(float(best_row['sharpe']), 2),
            "avg_fills": int(best_row['fills']),
            "avg_max_dd": round(float(best_row['max_dd']), 2),
        },
        "per_period_best": {}
    }

    for period in results_df['period'].unique():
        pdf = results_df[results_df['period'] == period].sort_values('pnl', ascending=False)
        best = pdf.iloc[0]
        summary["per_period_best"][period] = {
            "tolerance": float(best['tolerance']),
            "spacing": float(best['spacing']),
            "pnl": float(best['pnl']),
            "sharpe": float(best['sharpe']),
            "fills": int(best['fills']),
        }

    summary_file = "data/sweep_multi_period_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Summary → {summary_file}")

    # ── Final recommendation ─────────────────────────────────────
    logger.info(f"\n{'='*60}")
    best = cross.iloc[0]
    logger.info(f"🏆 RECOMMENDED PARAMETERS (cross-period best):")
    logger.info(f"   price_tolerance = {best['tolerance']*100:.2f}%")
    logger.info(f"   grid_spacing    = {best['spacing']*100:.2f}%")
    logger.info(f"   avg PnL = ${best['pnl']:+.2f}  |  avg Sharpe = {best['sharpe']:.2f}")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    asyncio.run(main())
