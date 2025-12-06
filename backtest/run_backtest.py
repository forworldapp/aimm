import asyncio
import pandas as pd
import logging
import os
from backtest.mock_exchange import MockExchange
from strategies.market_maker import MarketMaker
from core.config import Config

# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("BacktestRunner")

async def run_backtest(data_file: str):
    if not os.path.exists(data_file):
        logger.error(f"Data file not found: {data_file}")
        return

    logger.info(f"Loading data from {data_file}...")
    df = pd.read_csv(data_file)
    
    # Initialize Mock Exchange
    exchange = MockExchange(df, initial_balance=10000.0)
    
    # Initialize Strategy
    # We need to override Config for backtest params if needed
    Config.load("config.yaml") 
    strategy = MarketMaker(exchange)
    
    logger.info("Starting Backtest...")
    
    # Run Loop
    # Instead of strategy.run() which has a while loop with sleep,
    # We manually cycle the strategy and tick the exchange.
    
    strategy.is_running = True
    
    while exchange.next_tick():
        await strategy.cycle()
        
        # Optional: Print status every 100 ticks
        if exchange.current_index % 100 == 0:
            pos = exchange.position
            bal = exchange.balance
            equity = bal['USDT'] + (pos['amount'] * df.iloc[exchange.current_index]['best_bid'])
            logger.info(f"Step {exchange.current_index}: Equity=${equity:.2f} | Pos={pos['amount']}")

    # Final Report
    logger.info("Backtest Complete.")
    final_equity = exchange.balance['USDT'] + (exchange.position['amount'] * df.iloc[-1]['best_bid'])
    logger.info(f"Final Equity: ${final_equity:.2f}")
    logger.info(f"Total Trades: {len(exchange.trade_history)}")

if __name__ == "__main__":
    # Example usage: python -m backtest.run_backtest data/ticker_data_12345.csv
    import sys
    if len(sys.argv) > 1:
        asyncio.run(run_backtest(sys.argv[1]))
    else:
        print("Usage: python -m backtest.run_backtest <data_csv_path>")
