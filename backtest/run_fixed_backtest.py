"""
Fixed Parameter Backtest Runner - For comparison with ML version
Uses fixed parameters as specified by user (no ML adjustment)
"""
import asyncio
import pandas as pd
import numpy as np
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtest.mock_exchange import MockExchange
from strategies.market_maker import MarketMaker
from core.config import Config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("FixedBacktest")

def convert_ohlcv_to_orderbook(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    range_pct = (df['high'] - df['low']) / df['close']
    spread = range_pct * 0.1
    spread = spread.clip(0.0001, 0.005)
    df['best_bid'] = df['close'] * (1 - spread/2)
    df['best_ask'] = df['close'] * (1 + spread/2)
    df['timestamp'] = pd.to_datetime(df['open_time'], unit='ms')
    return df[['timestamp', 'open', 'high', 'low', 'close', 'volume', 'best_bid', 'best_ask']]

async def run_fixed_backtest(data_file: str, initial_balance: float = 10000.0):
    if not os.path.exists(data_file):
        logger.error(f"Data file not found: {data_file}")
        return
    
    logger.info(f"Loading data from {data_file}...")
    df = pd.read_csv(data_file)
    df = convert_ohlcv_to_orderbook(df)
    logger.info(f"Converted {len(df)} rows")
    
    exchange = MockExchange(df, initial_balance=initial_balance)
    exchange.set_as_metrics = lambda x: None
    exchange.set_market_regime = lambda x: None
    
    Config.load("config.yaml")
    strategy = MarketMaker(exchange)
    
    # OVERRIDE: Force fixed parameters (disable ML)
    strategy.regime_detector = None  # Disable ML regime detection
    strategy.adaptive_tuner = None   # Disable adaptive tuning
    
    # Fixed parameters as specified
    strategy.max_loss_usd = 2000.0       # Same as ML test
    strategy.max_drawdown_pct = 0.30
    strategy.grid_layers = 5             # Fixed 5 layers
    strategy.order_size_usd = 200        # Fixed $200
    strategy.spread_pct = 0.0025         # Fixed 0.25% spread
    
    # Force fixed ML params (won't be updated by ML)
    strategy._ml_grid_spacing = 0.0025   # Fixed 0.25% grid
    strategy._ml_order_size_mult = 1.0   # Fixed 100%
    strategy._ml_grid_layers = 5         # Fixed 5 layers
    strategy._ml_max_position_mult = 1.0 # Fixed 100%
    strategy._ml_skew_factor = 0.003     # Fixed 0.3%
    strategy._ml_price_tolerance = 0.001 # Fixed 0.1%
    
    equity_history = []
    
    logger.info("=" * 60)
    logger.info("Starting FIXED PARAMETER Backtest")
    logger.info(f"Spread: 0.25% | Layers: 5 | Order: $200")
    logger.info("=" * 60)
    
    strategy.is_running = True
    
    try:
        step = 0
        while exchange.next_tick():
            await strategy.cycle()
            step += 1
            
            pos = exchange.position
            row = df.iloc[exchange.current_index]
            mid_price = (row['best_bid'] + row['best_ask']) / 2
            equity = exchange.balance['USDT'] + (pos['amount'] * mid_price)
            equity_history.append({
                'step': step,
                'timestamp': row['timestamp'],
                'equity': equity,
                'position': pos['amount'],
                'price': mid_price
            })
            
            if step % 100 == 0:
                logger.info(f"Step {step}/{len(df)}: Equity=${equity:,.2f} | Pos={pos['amount']:.4f}")
                
    except Exception as e:
        logger.error(f"Backtest error: {e}")
        import traceback
        traceback.print_exc()
    
    logger.info("=" * 60)
    logger.info("FIXED PARAMETER BACKTEST COMPLETE")
    logger.info("=" * 60)
    
    last_row = df.iloc[min(exchange.current_index, len(df)-1)]
    last_price = (last_row['best_bid'] + last_row['best_ask']) / 2
    final_equity = exchange.balance['USDT'] + (exchange.position['amount'] * last_price)
    pnl = final_equity - initial_balance
    pnl_pct = (pnl / initial_balance) * 100
    
    equity_series = pd.Series([e['equity'] for e in equity_history])
    returns = equity_series.pct_change().dropna()
    sharpe = (returns.mean() / returns.std()) * np.sqrt(252 * 24) if returns.std() > 0 else 0
    max_equity = equity_series.cummax()
    drawdown = (equity_series - max_equity) / max_equity
    max_drawdown = drawdown.min() * 100
    
    logger.info(f"Final Equity: ${final_equity:,.2f}")
    logger.info(f"Total PnL: ${pnl:,.2f} ({pnl_pct:+.2f}%)")
    logger.info(f"Total Trades: {len(exchange.trade_history)}")
    logger.info(f"Sharpe Ratio: {sharpe:.2f}")
    logger.info(f"Max Drawdown: {max_drawdown:.2f}%")
    
    results_df = pd.DataFrame(equity_history)
    results_df.to_csv("data/backtest_fixed_results.csv", index=False)
    logger.info(f"Results saved to data/backtest_fixed_results.csv")

if __name__ == "__main__":
    asyncio.run(run_fixed_backtest("data/btc_hourly_1000.csv"))
