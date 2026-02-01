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

# Silence all library logs
logging.getLogger("MarketMaker").setLevel(logging.CRITICAL + 1)
logging.getLogger("MockExchange").setLevel(logging.CRITICAL + 1)
logging.basicConfig(level=logging.WARNING)

async def run_scenario_chunked(name, max_loss, df, chunk_days=7):
    print(f"\n🔹 Scenario: {name} (Max Loss: ${max_loss})")
    
    # Setup Exchange & Strategy
    exchange = MockExchange(df, initial_balance=10000.0)
    strategy = MarketMaker(exchange)
    
    # CRITICAL: Monkey patch _load_params to prevent overwriting our overrides
    def no_op_load():
        pass
    strategy._load_params = no_op_load
    
    # Apply Overrides
    strategy.max_loss_usd = max_loss
    strategy.max_position_usd = 5000.0
    strategy.initial_equity = 10000.0 # explicit init
    strategy.is_active = True
    
    # Force V4 components
    Config.set("strategy", "ml_regime_enabled", True)
    
    # Tracking
    min_equity = 10000.0
    max_drawdown = 0.0
    breaker_triggered_count = 0
    total_steps = len(df)
    
    # Chunking
    chunk_size = 1440 * chunk_days # days in minutes
    
    print(f"   Processing {total_steps} candles in {chunk_days}-day chunks...")
    
    current_step = 0
    while current_step < total_steps:
        # Define chunk range
        end_step = min(current_step + chunk_size, total_steps)
        
        # Run chunk
        for i in range(current_step, end_step):
            if not exchange.next_tick():
                break
                
            current_idx = exchange.current_index
            
            # Inject Data
            window_start = max(0, current_idx - 60)
            strategy.candles = df.iloc[window_start:current_idx+1][['open', 'high', 'low', 'close', 'volume']]
            
            mid = (df.iloc[current_idx]['best_bid'] + df.iloc[current_idx]['best_ask']) / 2
            strategy.price_history.append(mid)
            if len(strategy.price_history) > strategy.history_max_len:
                strategy.price_history.pop(0)
            
            # Execute Cycle (only if active)
            if strategy.is_active:
                await strategy.cycle()
            else:
                # If breaker triggered, strategy stops. 
                # For "No Breaker" (max_loss=1e9), this should never happen.
                # For others, we count it.
                if breaker_triggered_count == 0:
                    print(f"   ⚠️ Breaker Triggered at step {i}")
                breaker_triggered_count += 1
            
            # Track Equity
            equity = exchange.get_equity()
            if equity < min_equity:
                min_equity = equity
                dd = (10000 - min_equity) / 10000 * 100
                max_drawdown = max(max_drawdown, dd)
                
        # End of chunk report
        progress = (end_step / total_steps) * 100
        equity = exchange.get_equity()
        print(f"   ► {progress:.0f}% done | Equity: ${equity:.2f} | MDD: {max_drawdown:.2f}%")
        
        current_step = end_step

    final_equity = exchange.get_equity()
    pnl = final_equity - 10000
    pnl_pct = (pnl / 10000) * 100
    
    return {
        "Scenario": name,
        "Limit": f"${max_loss}",
        "Equity": f"${final_equity:.0f}",
        "PnL": f"{pnl_pct:+.2f}%",
        "MDD": f"{max_drawdown:.2f}%",
        "Stopped": "YES" if breaker_triggered_count > 0 else "NO"
    }

async def run_all_robust():
    print("\n" + "="*60)
    print("🚦 ROBUST SCENARIO TEST: OCT 2025 (Monthly)")
    print("="*60)
    
    chunk_days = 7 # Process weekly to keep user updated
    
    # Load Data
    data_file = "data/btcusdt_1m_oct2025.csv"
    if not os.path.exists(data_file):
        print(f"❌ Data file not found: {data_file}")
        return

    df = pd.read_csv(data_file)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df['best_bid'] = df['close'] * (1 - 0.00005)
    df['best_ask'] = df['close'] * (1 + 0.00005)
    
    print(f"Loaded {len(df)} candles.")
    
    results = []
    
    # 1. Baseline ($350)
    res1 = await run_scenario_chunked("Baseline", 350.0, df, chunk_days)
    results.append(res1)
    
    # 2. Expanded ($1000)
    res2 = await run_scenario_chunked("Expanded", 1000.0, df, chunk_days)
    results.append(res2)
    
    # 3. No Breaker (Infinite)
    res3 = await run_scenario_chunked("No Breaker", 1000000.0, df, chunk_days)
    results.append(res3)
    
    # Report
    print("\n" + "="*70)
    print("📊 FINAL COMPARISON REPORT")
    print("="*70)
    print(f"{'Scenario':<12} | {'Limit':<10} | {'Equity':<10} | {'PnL':<10} | {'MDD':<8} | {'Stopped'}")
    print("-" * 70)
    for r in results:
        print(f"{r['Scenario']:<12} | {r['Limit']:<10} | {r['Equity']:<10} | {r['PnL']:<10} | {r['MDD']:<8} | {r['Stopped']}")
    print("-" * 70)

if __name__ == "__main__":
    asyncio.run(run_all_robust())
