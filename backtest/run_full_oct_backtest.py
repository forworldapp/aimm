import sys
import os
import pandas as pd
import asyncio
import logging
import datetime
from collections import deque

# Adjust path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.config import Config
from backtest.mock_exchange import MockExchange
from strategies.market_maker import MarketMaker

# Setup Logging to Console
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s', # Simple format for clear output
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("FullOctSim")

async def run_full_oct_simulation():
    print("\n" + "="*50)
    print("📅 FULL OCTOBER 2025 BACKTEST (v4.0 STRATEGY)")
    print("="*50)
    
    # 1. Load Data
    data_file = "data/btcusdt_1m_oct2025.csv"
    if not os.path.exists(data_file):
        print(f"❌ Data file not found: {data_file}")
        return

    df = pd.read_csv(data_file)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    
    # NO FILTERING - USE FULL DATASET
    df_sim = df
    
    if df_sim.empty:
        print("❌ Data file is empty.")
        return
        
    print(f"📅 Simulating Period: {df_sim['timestamp'].iloc[0]} to {df_sim['timestamp'].iloc[-1]}")
    print(f"📊 Total Candles: {len(df_sim)}")
    
    # Calculate best_bid/ask for MockExchange
    df_sim['best_bid'] = df_sim['close'] * (1 - 0.00005)
    df_sim['best_ask'] = df_sim['close'] * (1 + 0.00005)

    # 2. Setup Bot
    # Inject Config
    Config.set("strategy", "ml_regime_enabled", True)
    Config.set("strategy", "max_position_usd", 5000.0)
    
    exchange = MockExchange(df_sim, initial_balance=10000.0)
    strategy = MarketMaker(exchange)
    strategy.is_active = True
    
    # 3. Monitor Variables
    max_drawdown = 0.0
    min_equity = 10000.0
    risk_events = []
    
    print("\n🚀 Starting Simulation...\n")
    
    steps = len(df_sim)
    
    for i in range(steps - 1):
        if not exchange.next_tick():
            break
            
        current_idx = exchange.current_index
        
        # Feed Data to Strategy
        window_start = max(0, current_idx - 60)
        recent_data = df_sim.iloc[window_start:current_idx+1]
        strategy.candles = recent_data[['open', 'high', 'low', 'close', 'volume']]
        
        row = df_sim.iloc[current_idx]
        mid = (row['best_bid'] + row['best_ask']) / 2
        strategy.price_history.append(mid)
        strategy._update_history(mid)
        
        # --- Run Cycle ---
        await strategy.cycle()
        
        # --- Risk Monitoring ---
        pos_usd = abs(exchange.position['amount']) * mid
        equity = exchange.balance['USDT'] + (exchange.position['amount'] * mid)
        
        if equity < min_equity:
            min_equity = equity
            dd = (10000 - min_equity) / 10000 * 100
            max_drawdown = max(max_drawdown, dd)
        
        # Check Risk State
        ml_max_pos_mult = getattr(strategy, '_ml_max_position_mult', 1.0)
        eff_max_pos = 5000.0 * ml_max_pos_mult
        
        # Log significant events (less verbose for full month)
        if ml_max_pos_mult < 1.0:
            if not risk_events or risk_events[-1]['type'] != 'limit_reduced':
                # Only log specifically if not already in that state to avoid spam
                if not risk_events or (row['timestamp'] - risk_events[-1]['time']).total_seconds() > 3600 * 4: # Log every 4 hours max
                     pass # Too noisy for full month, rely on daily report
                risk_events.append({'type': 'limit_reduced', 'time': row['timestamp']})

        # Daily Progress Report
        if i % 1440 == 0 and i > 0:
            pnl = equity - 10000
            print(f"[{row['timestamp']}] Daily: Equity=${equity:.0f} (PnL {pnl:+.0f}), DD={max_drawdown:.2f}%")

    # Final Report
    final_equity = exchange.balance['USDT'] + (exchange.position['amount'] * mid)
    total_pnl = final_equity - 10000
    pnl_pct = (total_pnl / 10000) * 100
    
    print("\n" + "="*50)
    print("🏁 FULL MONTH SIMULATION COMPLETE")
    print(f"Final Equity: ${final_equity:.2f}")
    print(f"Total PnL: ${total_pnl:.2f} ({pnl_pct:.2f}%)")
    print(f"Max Drawdown: {max_drawdown:.2f}%")
    print(f"Total Trades: {len(exchange.trade_history)}")
    print("="*50)

if __name__ == "__main__":
    asyncio.run(run_full_oct_simulation())
