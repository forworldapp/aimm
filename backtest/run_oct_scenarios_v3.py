import sys
import os
import pandas as pd
import asyncio
import logging
import gc
import math

# Adjust path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.config import Config
from backtest.mock_exchange import MockExchange
from strategies.market_maker import MarketMaker

# Configure logging to show only critical errors from libraries
logging.getLogger("MarketMaker").setLevel(logging.CRITICAL)
logging.getLogger("MockExchange").setLevel(logging.CRITICAL)
logging.basicConfig(level=logging.WARNING)

async def run_scenario_with_progress(name, max_loss, df):
    print(f"\n🔹 Scenario: {name} (Max Loss LIMIT: ${max_loss})")
    print(f"   Initializing...")
    
    # Setup
    exchange = MockExchange(df, initial_balance=10000.0)
    strategy = MarketMaker(exchange)
    
    # Patch load_params to allow overrides
    strategy._load_params = lambda: None
    
    # Overrides
    strategy.max_loss_usd = max_loss
    strategy.max_position_usd = 5000.0
    strategy.initial_equity = 10000.0
    strategy.is_active = True
    
    # Enable V4
    Config.set("strategy", "ml_regime_enabled", True)
    
    # State tracking
    min_equity = 10000.0
    max_drawdown = 0.0
    breaker_triggered_count = 0
    total_steps = len(df)
    
    # Loop with granular progress
    progress_interval = 2000 # Print every ~1.5 days
    
    for i in range(total_steps):
        if not exchange.next_tick():
            break
            
        current_idx = exchange.current_index
        row = df.iloc[current_idx]
        
        # Inject Data
        window_start = max(0, current_idx - 60)
        # Slicing DataFrame is fast enough for 60 rows
        strategy.candles = df.iloc[window_start:current_idx+1][['open', 'high', 'low', 'close', 'volume']]
        
        mid = (row['best_bid'] + row['best_ask']) / 2
        strategy.price_history.append(mid)
        # Custom optimize: keep history small manually if needed
        if len(strategy.price_history) > 250:
             strategy.price_history = strategy.price_history[-200:]
        
        # Cycle
        if strategy.is_active:
            await strategy.cycle()
        else:
            if breaker_triggered_count == 0:
                 print(f"   ⚠️ Circuit Breaker Triggered at step {i} (Time: {row['timestamp']})")
            breaker_triggered_count += 1
            
        # Metrics
        equity = exchange.get_equity()
        if equity < min_equity:
            min_equity = equity
            dd = (10000 - min_equity) / 10000 * 100
            max_drawdown = max(max_drawdown, dd)
            
        # Progress Log
        if i % progress_interval == 0 and i > 0:
            pct = (i / total_steps) * 100
            print(f"   ... {pct:.1f}% ({i}/{total_steps}) | Eq: ${equity:.0f} | MDD: {max_drawdown:.2f}%")
            sys.stdout.flush() # Force output

    final_equity = exchange.get_equity()
    pnl = final_equity - 10000
    pnl_pct = (pnl / 10000) * 100
    
    print(f"   ► Completed: Eq=${final_equity:.0f} ({pnl_pct:+.2f}%) | MDD={max_drawdown:.2f}%")
    
    # Cleanup to free memory
    del exchange
    del strategy
    gc.collect()
    
    return {
        "Scenario": name,
        "Limit": f"${max_loss}",
        "Equity": f"${final_equity:.0f}",
        "PnL": f"{pnl_pct:+.2f}%",
        "MDD": f"{max_drawdown:.2f}%",
        "Stopped": "YES" if breaker_triggered_count > 0 else "NO"
    }

async def run_all_robust_v3():
    print("\n" + "="*60)
    print("🚦 STABLE SCENARIO BACKTEST: OCT 2025")
    print("="*60)
    sys.stdout.flush()
    
    # Load Data
    data_file = "data/btcusdt_1m_oct2025.csv"
    if not os.path.exists(data_file):
        print(f"❌ Data file not found: {data_file}")
        return

    print("Status: Loading Data...")
    df = pd.read_csv(data_file)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df['best_bid'] = df['close'] * (1 - 0.00005)
    df['best_ask'] = df['close'] * (1 + 0.00005)
    
    print(f"Status: Loaded {len(df)} candles. Starting Scenarios...\n")
    sys.stdout.flush()
    
    results = []
    
    # 1. Baseline ($350)
    res1 = await run_scenario_with_progress("Baseline", 350.0, df)
    results.append(res1)
    
    # 2. Expanded ($1000)
    res2 = await run_scenario_with_progress("Expanded", 1000.0, df)
    results.append(res2)
    
    # 3. No Breaker (Infinite)
    res3 = await run_scenario_with_progress("No Breaker", 1000000.0, df)
    results.append(res3)
    
    # Final Report
    print("\n" + "="*70)
    print("📊 FINAL COMPARISON REPORT")
    print("="*70)
    print(f"{'Scenario':<12} | {'Limit':<10} | {'Equity':<10} | {'PnL':<10} | {'MDD':<8} | {'Stopped'}")
    print("-" * 70)
    for r in results:
        print(f"{r['Scenario']:<12} | {r['Limit']:<10} | {r['Equity']:<10} | {r['PnL']:<10} | {r['MDD']:<8} | {r['Stopped']}")
    print("-" * 70)

if __name__ == "__main__":
    asyncio.run(run_all_robust_v3())
