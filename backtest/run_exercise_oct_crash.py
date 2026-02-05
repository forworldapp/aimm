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
logger = logging.getLogger("CrashSim")

async def run_crash_simulation():
    print("\n" + "="*50)
    print("🔻 OCT 2025 CRASH SIMULATION (RISK MANAGEMENT TEST)")
    print("="*50)
    
    # 1. Load Data
    data_file = "data/btcusdt_1m_oct2025.csv"
    if not os.path.exists(data_file):
        print(f"❌ Data file not found: {data_file}")
        return

    df = pd.read_csv(data_file)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    
    # Filter for Crash Period (Oct 9 - Oct 12)
    start_date = "2025-10-09"
    end_date = "2025-10-12"
    mask = (df['timestamp'] >= start_date) & (df['timestamp'] < end_date)
    df_sim = df.loc[mask].reset_index(drop=True)
    
    if df_sim.empty:
        print("❌ No data found for specified date range.")
        return
        
    print(f"📅 Simulating Period: {df_sim['timestamp'].iloc[0]} to {df_sim['timestamp'].iloc[-1]}")
    print(f"📊 Total Candles: {len(df_sim)}")
    
    # Calculate best_bid/ask for MockExchange
    # Simulate slightly wider spread in volatile conditions if volume spikes?
    # For now keep simple 0.01%
    df_sim['best_bid'] = df_sim['close'] * (1 - 0.00005)
    df_sim['best_ask'] = df_sim['close'] * (1 + 0.00005)

    # 2. Setup Bot
    # Inject Config
    Config.set("strategy", "ml_regime_enabled", True)
    Config.set("strategy", "max_position_usd", 5000.0) # Baseline
    
    exchange = MockExchange(df_sim, initial_balance=10000.0)
    strategy = MarketMaker(exchange)
    strategy.is_active = True
    
    # Enable V4 or HMM (HMM is robust)
    # Ensure HMM model is loaded pointing to correct path or mocks
    # Assuming standard HMM model works or we rely on built-in logic
    
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
        # Rolling window for features
        window_start = max(0, current_idx - 60)
        recent_data = df_sim.iloc[window_start:current_idx+1]
        strategy.candles = recent_data[['open', 'high', 'low', 'close', 'volume']]
        
        row = df_sim.iloc[current_idx]
        mid = (row['best_bid'] + row['best_ask']) / 2
        strategy.price_history.append(mid)
        strategy._update_history(mid) # Helper to maintain internal history
        
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
        
        # Log significant events
        # 1. High Volatility Trigger (Limit Reduced)
        if ml_max_pos_mult < 1.0:
            if not risk_events or risk_events[-1]['type'] != 'limit_reduced':
                print(f"[{row['timestamp']}] 🛡️ RISK ALERT: High Volatility Detected!")
                print(f"   ↳ Max Position Reduced: $5000 ➔ ${eff_max_pos:.0f} (x{ml_max_pos_mult:.2f})")
                risk_events.append({'type': 'limit_reduced', 'time': row['timestamp']})
                
        # 2. Position Limit Hit
        if pos_usd >= eff_max_pos * 0.95 and pos_usd > 100:
             if not risk_events or (row['timestamp'] - risk_events[-1]['time']).total_seconds() > 3600:
                 print(f"[{row['timestamp']}] ✋ RISK ACTION: Position Limit Hit (${pos_usd:.0f} / ${eff_max_pos:.0f})")
                 print("   ↳ Buying/Selling Paused to prevent further risk")
                 risk_events.append({'type': 'limit_hit', 'time': row['timestamp']})

        # 3. Crash detection (Price drop > 1% in 15m)
        if i > 15:
            price_15m_ago = df_sim.iloc[current_idx-15]['close']
            drop = (mid - price_15m_ago) / price_15m_ago
            if drop < -0.01:
                 if not risk_events or (row['timestamp'] - risk_events[-1]['time']).total_seconds() > 900:
                     print(f"[{row['timestamp']}] 📉 MARKETS DROPPING: {drop*100:.2f}% drop in 15m. Price: ${mid:.2f}")
                     risk_events.append({'type': 'crash', 'time': row['timestamp']})

        if i % 1440 == 0 and i > 0: # Every day
            print(f"[{row['timestamp']}] Daily Report: Equity=${equity:.0f}, Drawdown={max_drawdown:.2f}%")

    print("\n" + "="*50)
    print("🏁 SIMULATION COMPLETE")
    print(f"Final Equity: ${equity:.2f}")
    print(f"Max Drawdown: {max_drawdown:.2f}%")
    print("="*50)

if __name__ == "__main__":
    asyncio.run(run_crash_simulation())
