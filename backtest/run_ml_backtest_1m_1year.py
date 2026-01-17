import sys
import os
import pandas as pd
import asyncio
import logging
from collections import deque

# Adjust path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.config import Config
from backtest.mock_exchange import MockExchange
from strategies.market_maker import MarketMaker

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/backtest_1m_1year.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("MLBacktest_1m_1Year")

async def run_1m_backtest():
    # 1. Load Data
    data_file = "data/btcusdt_1m_1year.csv"
    if not os.path.exists(data_file):
        logger.error(f"Data file {data_file} not found!")
        return

    logger.info(f"Loading 1m data from {data_file}...")
    df = pd.read_csv(data_file)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    
    # Calculate best_bid/ask for MockExchange
    # In 1m candles, spread is tighter. Assume 0.01% spread around close
    df['best_bid'] = df['close'] * (1 - 0.00005)
    df['best_ask'] = df['close'] * (1 + 0.00005)
    
    logger.info(f"Loaded {len(df)} candles.")

    # 2. Setup Components
    # Inject Config for 1m Model
    Config.set("strategy", "regime_model_path", "data/regime_model_1m.pkl") # Point to 1m model
    Config.set("strategy", "ml_regime_enabled", True)
    
    exchange = MockExchange(df, initial_balance=10000.0)
    strategy = MarketMaker(exchange)
    
    # Suppress MarketMaker logs for speed
    logging.getLogger("MarketMaker").setLevel(logging.WARNING)
    
    # Override Strategy Params (10k Capital, -50% Limit)
    strategy.max_loss_usd = 5000.0   # 50% of $10,000
    strategy.max_drawdown_pct = 0.50 # 50% Drawdown Limit
    strategy.max_position_usd = 5000.0 # Max pos 50% of equity
    strategy.grid_layers = 5 
    strategy.order_size_usd = 100.0

    # Mock Prediction Logic
    def mock_predict_live_proba(symbol="BTCUSDT"):
        if strategy.candles.empty or len(strategy.candles) < 50:
            return {}
        return strategy.regime_detector.predict_proba(strategy.candles)

    # Patch strategy with mock
    strategy.regime_detector.predict_live_proba = mock_predict_live_proba
    
    # Enable strategy for backtest
    strategy.is_active = True

    # 3. Running Backtest
    # FULL DURATION
    steps = len(df)
    equity_history = []
    
    print(f"Starting 1m Backtest for {steps} minutes (1 Year)...")
    
    start_idx = 50 # Warmup
    exchange.current_index = start_idx
    
    # Fill initial history
    for i in range(start_idx):
        mid = (df.iloc[i]['best_bid'] + df.iloc[i]['best_ask']) / 2
        strategy.price_history.append(mid)

    for i in range(start_idx, steps - 1):
        if not exchange.next_tick():
            break
            
        current_idx = exchange.current_index
        
        # Inject recent candles for ML
        window_start = max(0, current_idx - 60)
        recent_data = df.iloc[window_start:current_idx+1].copy()
        
        strategy.candles = recent_data[['open', 'high', 'low', 'close', 'volume']]
        
        # Inject Price
        row = df.iloc[current_idx]
        mid = (row['best_bid'] + row['best_ask']) / 2
        strategy.price_history.append(mid)
        
        # Execute Strategy Cycle
        await strategy.cycle()
        
        # CRITICAL: Stop if circuit breaker triggered
        if not strategy.is_active:
            print(f"!! STOPPING BACKTEST: Strategy stopped at step {i} !!")
            break
        
        # Track Equity
        pos = exchange.position
        equity = exchange.balance['USDT'] + (pos['amount'] * mid)
        equity_history.append(equity)
        
        if i % 10000 == 0:
            print(f"Step {i}/{steps}: Equity=${equity:.2f} | PnL={(equity-10000):.2f}")

    # 4. Results
    final_equity = equity_history[-1]
    pnl = final_equity - 10000.0
    pnl_pct = (pnl / 10000.0) * 100
    trades = len(exchange.trade_history)
    
    print("\n" + "="*30)
    print("1-YEAR BACKTEST RESULTS")
    print("="*30)
    print(f"Final Equity: ${final_equity:.2f}")
    print(f"Total PnL: ${pnl:.2f} ({pnl_pct:.2f}%)")
    print(f"Total Trades: {trades}")
    print("="*30)

if __name__ == "__main__":
    asyncio.run(run_1m_backtest())
