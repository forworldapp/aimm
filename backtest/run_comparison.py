"""
GMM vs HMM Backtest Comparison Script
- Runs 1-month backtest with GMM
- Runs 1-month backtest with HMM
- Outputs comparison table
"""

import sys
import os
import pandas as pd
import asyncio
import logging
from collections import deque

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.config import Config
from backtest.mock_exchange import MockExchange
from strategies.market_maker import MarketMaker

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/backtest_comparison.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("BacktestComparison")


async def run_backtest(model_type: str, duration_minutes: int = 43200) -> dict:
    """
    Run backtest with specified model type.
    Returns: dict with pnl, trades, max_drawdown
    """
    # Load Data
    data_file = "data/btcusdt_1m_1year.csv"
    if not os.path.exists(data_file):
        # Try falling back to hourly data
        data_file = "data/btc_hourly_1year.csv"
        if not os.path.exists(data_file):
            logger.error("No data file found!")
            return {}
    
    logger.info(f"Loading data from {data_file}...")
    df = pd.read_csv(data_file)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df['best_bid'] = df['close'] * (1 - 0.00005)
    df['best_ask'] = df['close'] * (1 + 0.00005)
    
    # Setup
    Config.set("strategy", "regime_model_type", model_type)
    exchange = MockExchange(df, initial_balance=10000.0)
    strategy = MarketMaker(exchange)
    
    # Suppress logs
    logging.getLogger("MarketMaker").setLevel(logging.WARNING)
    
    # Override params
    strategy.max_loss_usd = 100000.0  # No circuit breaker during comparison
    strategy.max_drawdown_pct = 100.0
    strategy.grid_layers = 5
    strategy.order_size_usd = 100.0
    strategy.is_active = True
    
    # Mock prediction
    def mock_predict_live_proba(symbol="BTCUSDT"):
        if strategy.candles.empty or len(strategy.candles) < 50:
            return {}
        return strategy.regime_detector.predict_proba(strategy.candles)
    
    if strategy.regime_detector:
        strategy.regime_detector.predict_live_proba = mock_predict_live_proba
    
    # Run
    steps = min(len(df), duration_minutes)
    equity_history = []
    
    logger.info(f"Running {model_type.upper()} backtest for {steps} minutes...")
    
    start_idx = 50
    exchange.current_index = start_idx
    
    for i in range(start_idx):
        mid = (df.iloc[i]['best_bid'] + df.iloc[i]['best_ask']) / 2
        strategy.price_history.append(mid)
    
    for i in range(start_idx, steps - 1):
        if not exchange.next_tick():
            break
        
        current_idx = exchange.current_index
        window_start = max(0, current_idx - 60)
        recent_data = df.iloc[window_start:current_idx+1].copy()
        strategy.candles = recent_data[['open', 'high', 'low', 'close', 'volume']]
        
        row = df.iloc[current_idx]
        mid = (row['best_bid'] + row['best_ask']) / 2
        strategy.price_history.append(mid)
        
        await strategy.cycle()
        
        if not strategy.is_active:
            break
        
        pos = exchange.position
        equity = exchange.balance['USDT'] + (pos['amount'] * mid)
        equity_history.append(equity)
        
        if i % 10000 == 0:
            logger.info(f"[{model_type.upper()}] Step {i}/{steps}: Equity=${equity:.2f}")
    
    # Calculate metrics
    if not equity_history:
        return {"model": model_type, "pnl": 0, "pnl_pct": 0, "trades": 0, "max_drawdown": 0}
    
    final_equity = equity_history[-1]
    pnl = final_equity - 10000.0
    pnl_pct = (pnl / 10000.0) * 100
    trades = len(exchange.trade_history)
    
    # Max drawdown
    peak = 10000.0
    max_dd = 0.0
    for eq in equity_history:
        if eq > peak:
            peak = eq
        dd = (peak - eq) / peak
        if dd > max_dd:
            max_dd = dd
    
    return {
        "model": model_type,
        "pnl": pnl,
        "pnl_pct": pnl_pct,
        "trades": trades,
        "max_drawdown": max_dd * 100
    }


async def main():
    print("=" * 50)
    print("GMM vs HMM BACKTEST COMPARISON")
    print("=" * 50)
    
    # Run GMM
    gmm_results = await run_backtest("gmm", duration_minutes=43200)
    
    # Run HMM
    hmm_results = await run_backtest("hmm", duration_minutes=43200)
    
    # Results table
    print("\n" + "=" * 50)
    print("COMPARISON RESULTS (1 Month)")
    print("=" * 50)
    print(f"{'Metric':<20} {'GMM':>12} {'HMM':>12}")
    print("-" * 50)
    print(f"{'PnL ($)':<20} {gmm_results.get('pnl', 0):>12.2f} {hmm_results.get('pnl', 0):>12.2f}")
    print(f"{'PnL (%)':<20} {gmm_results.get('pnl_pct', 0):>12.2f} {hmm_results.get('pnl_pct', 0):>12.2f}")
    print(f"{'Total Trades':<20} {gmm_results.get('trades', 0):>12} {hmm_results.get('trades', 0):>12}")
    print(f"{'Max Drawdown (%)':<20} {gmm_results.get('max_drawdown', 0):>12.2f} {hmm_results.get('max_drawdown', 0):>12.2f}")
    print("=" * 50)
    
    # Determine winner
    if hmm_results.get('pnl', 0) > gmm_results.get('pnl', 0):
        print("🏆 WINNER: HMM (Higher PnL)")
    elif gmm_results.get('pnl', 0) > hmm_results.get('pnl', 0):
        print("🏆 WINNER: GMM (Higher PnL)")
    else:
        print("🤝 TIE: Both models performed equally")


if __name__ == "__main__":
    asyncio.run(main())
