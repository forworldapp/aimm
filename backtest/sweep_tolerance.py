"""
Parameter Sweep: price_tolerance & grid_spacing optimization
-------------------------------------------------------------
Tests multiple combinations and reports:
  - Total PnL, Sharpe, Max Drawdown
  - Number of fills (trades)
  - Order replacement count (cancel+re-place cycles)

Goal: Find the sweet spot where orders stay long enough to fill
      but still adapt to large price moves.
"""
import asyncio
import pandas as pd
import numpy as np
import logging
import os
import sys
import time
from itertools import product

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtest.mock_exchange import MockExchange
from strategies.market_maker import MarketMaker
from core.config import Config

# Suppress noisy logs during sweep
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Sweep")
logger.setLevel(logging.INFO)

def convert_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    range_pct = (df['high'] - df['low']) / df['close']
    spread = range_pct * 0.1
    spread = spread.clip(0.0001, 0.005)
    df['best_bid'] = df['close'] * (1 - spread/2)
    df['best_ask'] = df['close'] * (1 + spread/2)
    df['timestamp'] = pd.to_datetime(df['open_time'], unit='ms')
    return df[['timestamp', 'open', 'high', 'low', 'close', 'volume', 'best_bid', 'best_ask']]


async def run_single(df, tolerance, spacing, initial_balance=10000.0):
    """Run one backtest with given tolerance and spacing."""
    exchange = MockExchange(df.copy(), initial_balance=initial_balance)
    exchange.set_as_metrics = lambda x: None
    exchange.set_market_regime = lambda x: None
    exchange.save_live_status = lambda **kwargs: None
    exchange.fetch_and_save_trades = lambda *args: None
    exchange._get_trade_count = lambda *args: 0
    
    # Stub get_bot_pnl to return zero-impact values
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
    
    # Disable ML and adaptive systems
    strategy.regime_detector = None
    strategy.adaptive_tuner = None
    strategy.ml_strategy = None
    strategy.order_flow_analyzer = None
    strategy.funding_arb = None
    strategy.microstructure = None
    strategy.execution_algo = None
    strategy.notifier = None
    
    # Fixed params
    strategy.max_loss_usd = 5000.0
    strategy.grid_layers = 7
    strategy.order_size_usd = 200
    strategy.spread_pct = 0.0015
    
    # THE PARAMETERS WE'RE TESTING
    strategy._ml_price_tolerance = tolerance
    strategy._ml_grid_spacing = spacing
    strategy._ml_order_size_mult = 0.8
    strategy._ml_grid_layers = 7
    strategy._ml_max_position_mult = 1.0
    strategy._ml_skew_factor = 0.008
    
    equity_history = []
    cancel_count = 0
    
    # Monkey-patch cancel_all_orders to count replacements
    orig_cancel = exchange.cancel_all_orders
    async def counted_cancel(symbol):
        nonlocal cancel_count
        cancel_count += 1
        return await orig_cancel(symbol)
    exchange.cancel_all_orders = counted_cancel
    
    strategy.is_running = True
    
    try:
        step = 0
        while exchange.next_tick():
            await strategy.cycle()
            step += 1
            
            row = df.iloc[exchange.current_index]
            mid = (row['best_bid'] + row['best_ask']) / 2
            equity = exchange.balance['USDT'] + (exchange.position['amount'] * mid)
            equity_history.append(equity)
            
    except Exception as e:
        pass  # Silently handle errors for sweep
    
    # Calculate metrics
    if not equity_history:
        return None
    
    eq = pd.Series(equity_history)
    final_equity = eq.iloc[-1]
    pnl = final_equity - initial_balance
    pnl_pct = (pnl / initial_balance) * 100
    
    returns = eq.pct_change().dropna()
    sharpe = (returns.mean() / returns.std()) * np.sqrt(252 * 24) if returns.std() > 0 else 0
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


async def main():
    data_file = "data/btc_hourly_1000.csv"
    if not os.path.exists(data_file):
        logger.error(f"Data file not found: {data_file}")
        return
    
    logger.info("Loading data...")
    df = pd.read_csv(data_file)
    df = convert_ohlcv(df)
    logger.info(f"Loaded {len(df)} candles")
    
    # Parameter grid
    tolerances = [0.001, 0.002, 0.003, 0.005, 0.008, 0.01]     # 0.1% to 1.0%
    spacings = [0.001, 0.0015, 0.002, 0.003, 0.004, 0.005]     # 0.1% to 0.5%
    
    combos = list(product(tolerances, spacings))
    logger.info(f"Running {len(combos)} parameter combinations...")
    
    results = []
    for i, (tol, spc) in enumerate(combos):
        t0 = time.time()
        result = await run_single(df, tol, spc)
        elapsed = time.time() - t0
        
        if result:
            results.append(result)
            logger.info(
                f"[{i+1}/{len(combos)}] tol={tol:.4f} spc={spc:.4f} | "
                f"PnL=${result['pnl']:+.2f} ({result['pnl_pct']:+.1f}%) | "
                f"Fills={result['fills']:3d} | Cancels={result['cancel_cycles']:4d} | "
                f"Sharpe={result['sharpe']:.2f} | DD={result['max_dd']:.1f}% | "
                f"{elapsed:.1f}s"
            )
    
    if not results:
        logger.error("No results!")
        return
    
    # Sort by PnL
    results.sort(key=lambda r: r['pnl'], reverse=True)
    
    # Print results table
    logger.info("\n" + "=" * 100)
    logger.info("RESULTS RANKED BY P&L")
    logger.info("=" * 100)
    logger.info(f"{'Rank':>4} | {'Tol%':>6} | {'Spc%':>6} | {'PnL$':>10} | {'PnL%':>7} | {'Sharpe':>7} | {'MaxDD%':>7} | {'Fills':>6} | {'Cancels':>8} | {'Fill/Cancel':>11}")
    logger.info("-" * 100)
    
    for i, r in enumerate(results):
        logger.info(
            f"{i+1:>4} | {r['tolerance']*100:>5.2f}% | {r['spacing']*100:>5.2f}% | "
            f"${r['pnl']:>+9.2f} | {r['pnl_pct']:>+6.2f}% | {r['sharpe']:>7.2f} | "
            f"{r['max_dd']:>6.2f}% | {r['fills']:>6} | {r['cancel_cycles']:>8} | "
            f"{r['fills_per_cancel']:>11.4f}"
        )
    
    # Best by different metrics
    logger.info("\n" + "=" * 60)
    best_pnl = max(results, key=lambda r: r['pnl'])
    best_sharpe = max(results, key=lambda r: r['sharpe'])
    best_fills = max(results, key=lambda r: r['fills'])
    least_cancels = min(results, key=lambda r: r['cancel_cycles'])
    
    logger.info(f"🏆 Best PnL:     tol={best_pnl['tolerance']:.4f} spc={best_pnl['spacing']:.4f} → ${best_pnl['pnl']:+.2f}")
    logger.info(f"📈 Best Sharpe:  tol={best_sharpe['tolerance']:.4f} spc={best_sharpe['spacing']:.4f} → {best_sharpe['sharpe']:.2f}")
    logger.info(f"🔥 Most Fills:   tol={best_fills['tolerance']:.4f} spc={best_fills['spacing']:.4f} → {best_fills['fills']} fills")
    logger.info(f"⚡ Least Cancel: tol={least_cancels['tolerance']:.4f} spc={least_cancels['spacing']:.4f} → {least_cancels['cancel_cycles']} cancels")
    
    # Save CSV
    results_df = pd.DataFrame(results)
    out_file = "data/sweep_tolerance_results.csv"
    results_df.to_csv(out_file, index=False)
    logger.info(f"\nResults saved to {out_file}")


if __name__ == "__main__":
    asyncio.run(main())
