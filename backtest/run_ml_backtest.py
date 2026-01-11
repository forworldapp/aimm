"""
ML Market Maker Backtest Runner
- Converts OHLCV data to bid/ask format
- Runs strategy with ML regime detection
- Reports PnL and performance metrics
"""
import asyncio
import pandas as pd
import numpy as np
import logging
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtest.mock_exchange import MockExchange
from strategies.market_maker import MarketMaker
from ml.regime_detector import RegimeDetector
from core.config import Config

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("MLBacktest")

def convert_ohlcv_to_orderbook(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert OHLCV data to simulated orderbook format.
    - best_bid = close - (high-low)*0.01 (1% of range below close)
    - best_ask = close + (high-low)*0.01 (1% of range above close)
    - This simulates a typical spread
    """
    df = df.copy()
    
    # Calculate spread based on volatility
    range_pct = (df['high'] - df['low']) / df['close']
    spread = range_pct * 0.1  # 10% of range as spread
    spread = spread.clip(0.0001, 0.005)  # Min 0.01%, Max 0.5%
    
    df['best_bid'] = df['close'] * (1 - spread/2)
    df['best_ask'] = df['close'] * (1 + spread/2)
    df['timestamp'] = pd.to_datetime(df['open_time'], unit='ms')
    
    return df[['timestamp', 'open', 'high', 'low', 'close', 'volume', 'best_bid', 'best_ask']]

async def run_ml_backtest(data_file: str, initial_balance: float = 10000.0):
    """
    Run backtest with ML-based market maker strategy.
    """
    if not os.path.exists(data_file):
        logger.error(f"Data file not found: {data_file}")
        return
    
    logger.info(f"Loading data from {data_file}...")
    df = pd.read_csv(data_file)
    
    # Convert OHLCV to orderbook format
    df = convert_ohlcv_to_orderbook(df)
    logger.info(f"Converted {len(df)} rows to orderbook format")
    
    # Initialize Mock Exchange
    exchange = MockExchange(df, initial_balance=initial_balance)
    
    # Add set_as_metrics method to MockExchange (for compatibility)
    exchange.set_as_metrics = lambda x: None
    exchange.set_market_regime = lambda x: None
    
    # Initialize Strategy
    Config.load("config.yaml")
    strategy = MarketMaker(exchange)
    
    # Override loss limit DIRECTLY on strategy for backtest
    strategy.max_loss_usd = 2000.0
    strategy.max_drawdown_pct = 0.30  # 30%
    
    # Track metrics
    equity_history = []
    regime_history = []
    
    logger.info("=" * 60)
    logger.info("Starting ML Market Maker Backtest")
    logger.info(f"Initial Balance: ${initial_balance:,.2f}")
    logger.info(f"Data Points: {len(df)}")
    logger.info("=" * 60)
    
    strategy.is_running = True
    
    try:
        step = 0
        while exchange.next_tick():
            await strategy.cycle()
            step += 1
            
            # Track equity every step
            pos = exchange.position
            row = df.iloc[exchange.current_index]
            mid_price = (row['best_bid'] + row['best_ask']) / 2
            equity = exchange.balance['USDT'] + (pos['amount'] * mid_price)
            equity_history.append({
                'step': step,
                'timestamp': row['timestamp'],
                'equity': equity,
                'position': pos['amount'],
                'price': mid_price,
                'regime': getattr(strategy, 'current_ml_regime', 'unknown')
            })
            
            # Log every 100 steps
            if step % 100 == 0:
                regime = getattr(strategy, 'current_ml_regime', 'unknown')
                logger.info(f"Step {step}/{len(df)}: Equity=${equity:,.2f} | Pos={pos['amount']:.4f} | Regime={regime}")
                
    except KeyboardInterrupt:
        logger.info("Backtest interrupted by user.")
    except Exception as e:
        logger.error(f"Backtest error: {e}")
        import traceback
        traceback.print_exc()
    
    # Final Report
    logger.info("=" * 60)
    logger.info("BACKTEST COMPLETE")
    logger.info("=" * 60)
    
    # Calculate Final Metrics
    last_row = df.iloc[min(exchange.current_index, len(df)-1)]
    last_price = (last_row['best_bid'] + last_row['best_ask']) / 2
    final_equity = exchange.balance['USDT'] + (exchange.position['amount'] * last_price)
    pnl = final_equity - initial_balance
    pnl_pct = (pnl / initial_balance) * 100
    
    # Calculate Sharpe & Drawdown
    equity_series = pd.Series([e['equity'] for e in equity_history])
    returns = equity_series.pct_change().dropna()
    
    sharpe = (returns.mean() / returns.std()) * np.sqrt(252 * 24) if returns.std() > 0 else 0
    max_equity = equity_series.cummax()
    drawdown = (equity_series - max_equity) / max_equity
    max_drawdown = drawdown.min() * 100
    
    # Regime distribution
    regime_df = pd.DataFrame(equity_history)
    regime_counts = regime_df['regime'].value_counts()
    
    logger.info(f"Final Equity: ${final_equity:,.2f}")
    logger.info(f"Total PnL: ${pnl:,.2f} ({pnl_pct:+.2f}%)")
    logger.info(f"Total Trades: {len(exchange.trade_history)}")
    logger.info(f"Sharpe Ratio: {sharpe:.2f}")
    logger.info(f"Max Drawdown: {max_drawdown:.2f}%")
    logger.info("-" * 40)
    logger.info("Regime Distribution:")
    for regime, count in regime_counts.items():
        logger.info(f"  {regime}: {count} ({count/len(equity_history)*100:.1f}%)")
    
    # Save results
    results_df = pd.DataFrame(equity_history)
    results_df.to_csv("data/backtest_results.csv", index=False)
    logger.info(f"Results saved to data/backtest_results.csv")
    
    return {
        'final_equity': final_equity,
        'pnl': pnl,
        'pnl_pct': pnl_pct,
        'trades': len(exchange.trade_history),
        'sharpe': sharpe,
        'max_drawdown': max_drawdown
    }

if __name__ == "__main__":
    data_file = "data/btc_hourly_1000.csv"
    
    if len(sys.argv) > 1:
        data_file = sys.argv[1]
    
    asyncio.run(run_ml_backtest(data_file))
