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

# Setup Logging (Quiet for scenarios, only important info)
logging.basicConfig(level=logging.WARNING)

async def run_scenario(name, max_loss, df):
    print(f"\n🔹 Running Scenario: {name} (Max Loss: ${max_loss})")
    
    # Setup
    Config.set("strategy", "ml_regime_enabled", True)
    # Config.set("risk", "max_loss_usd", max_loss) # This might get overwritten
    
    exchange = MockExchange(df, initial_balance=10000.0)
    strategy = MarketMaker(exchange)
    
    # Override Strategy Params explicitly
    strategy.max_loss_usd = max_loss
    strategy.max_position_usd = 5000.0
    strategy.is_active = True
    
    # Run
    steps = len(df)
    min_equity = 10000.0
    max_drawdown = 0.0
    breaker_triggered = False
    
    for i in range(steps - 1):
        if not exchange.next_tick():
            break
            
        current_idx = exchange.current_index
        
        # Inject Data
        window_start = max(0, current_idx - 60)
        recent_data = df.iloc[window_start:current_idx+1]
        strategy.candles = recent_data[['open', 'high', 'low', 'close', 'volume']]
        
        row = df.iloc[current_idx]
        mid = (row['best_bid'] + row['best_ask']) / 2
        strategy.price_history.append(mid)
        strategy._update_history(mid)
        
        # Cycle
        await strategy.cycle()
        
        # Tracking
        equity = exchange.get_equity()
        if equity < min_equity:
            min_equity = equity
            dd = (10000 - min_equity) / 10000 * 100
            max_drawdown = max(max_drawdown, dd)
            
        # Check Circuit Breaker Status via Logs or State
        # If strategy became inactive and we didn't finish
        if not strategy.is_active and not breaker_triggered and i < steps - 10:
            print(f"   ⚠️ Circuit Breaker Triggered at Step {i} ({row['timestamp']})")
            breaker_triggered = True
            # To simulate "stopping", we keep iterating but strategy does nothing
            # Or we can break? Realistically, bot stops. But for backtest comparison we want to see final equity if it stopped.
            # If stopped, equity remains purely currency? Or positions closed? 
            # MM code cancels all orders. Position remains open? 
            # circuit breaker code: await self.exchange.cancel_all_orders... self.is_active = False
            # It does NOT close position automatically in current code implementation (unless stop_close command used).
            # But the user might manually close. For simulation, let's assume position is held or we just track equity as is.
            # If strategy is inactive, cycle returns immediately. Position value fluctuates with market.
            
    final_equity = exchange.get_equity()
    pnl = final_equity - 10000
    pnl_pct = (pnl / 10000) * 100
    
    return {
        "Scenario": name,
        "Max Loss Limit": f"${max_loss}",
        "Final Equity": f"${final_equity:.2f}",
        "PnL": f"${pnl:.2f} ({pnl_pct:.2f}%)",
        "Max Drawdown": f"{max_drawdown:.2f}%",
        "Breaker Triggered": "YES" if breaker_triggered else "NO"
    }

async def run_all_scenarios():
    print("\n" + "="*60)
    print("🚦 CIRCUIT BREAKER SCENARIO TEST (OCT 2025)")
    print("="*60)
    
    # Load Data Once
    data_file = "data/btcusdt_1m_oct2025.csv"
    if not os.path.exists(data_file):
        print(f"❌ Data file not found: {data_file}")
        return

    df = pd.read_csv(data_file)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    
    # Prep columns
    df['best_bid'] = df['close'] * (1 - 0.00005)
    df['best_ask'] = df['close'] * (1 + 0.00005)
    
    print(f"Loaded {len(df)} candles.")
    
    results = []
    
    # Scenario 1: Baseline ($350)
    res1 = await run_scenario("Baseline", 350.0, df)
    results.append(res1)
    
    # Scenario 2: Expanded ($1000)
    res2 = await run_scenario("Expanded", 1000.0, df)
    results.append(res2)
    
    # Scenario 3: Disabled (Infinite)
    res3 = await run_scenario("No Breaker", 100000.0, df)
    results.append(res3)
    
    # Print Comparison Table
    print("\n" + "="*60)
    print("📊 COMPARISON RESULTS")
    print("="*60)
    
    print(f"{'Scenario':<12} | {'Max Loss':<10} | {'Final Equity':<15} | {'PnL':<15} | {'MDD':<8} | {'Triggered'}")
    print("-" * 80)
    
    for r in results:
        print(f"{r['Scenario']:<12} | {r['Max Loss Limit']:<10} | {r['Final Equity']:<15} | {r['PnL']:<15} | {r['Max Drawdown']:<8} | {r['Breaker Triggered']}")
    print("-" * 80)

if __name__ == "__main__":
    asyncio.run(run_all_scenarios())
