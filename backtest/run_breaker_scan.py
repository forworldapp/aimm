import sys
import os
import pandas as pd
import asyncio
import logging
import gc

# Adjust path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.config import Config
from backtest.mock_exchange import MockExchange
from strategies.market_maker import MarketMaker

logging.getLogger("MarketMaker").setLevel(logging.CRITICAL)
logging.getLogger("MockExchange").setLevel(logging.CRITICAL)
logging.basicConfig(level=logging.WARNING)

async def run_scenario_fast(max_loss, df):
    # Setup
    exchange = MockExchange(df, initial_balance=10000.0)
    strategy = MarketMaker(exchange)
    strategy._load_params = lambda: None
    strategy.max_loss_usd = max_loss
    strategy.max_position_usd = 5000.0
    strategy.initial_equity = 10000.0
    strategy.is_active = True
    Config.set("strategy", "ml_regime_enabled", True)
    
    triggered_step = -1
    triggered_loss = 0.0
    
    # Fast loop
    for i in range(len(df)):
        if not exchange.next_tick(): break
        
        # Inject Data (Minimal)
        window_start = max(0, exchange.current_index - 60)
        strategy.candles = df.iloc[window_start:exchange.current_index+1][['open', 'high', 'low', 'close', 'volume']]
        strategy.price_history.append((df.iloc[exchange.current_index]['best_bid'] + df.iloc[exchange.current_index]['best_ask']) / 2)
        if len(strategy.price_history) > 200: strategy.price_history.pop(0)

        # Cycle
        if strategy.is_active:
            await strategy.cycle()
        else:
            # Stopped
            triggered_step = i
            triggered_loss = 10000.0 - exchange.get_equity()
            break # Stop simulation early if triggered (for speed)
            
    final_equity = exchange.get_equity()
    
    del exchange
    del strategy
    gc.collect()
    
    return {
        "Threshold": max_loss,
        "Triggered": triggered_step != -1,
        "Step": triggered_step,
        "FinalEquity": final_equity,
        "SavedLoss": triggered_loss
    }

async def scan_thresholds():
    print("\n" + "="*60)
    print("🔎 CIRCUIT BREAKER SENSITIVITY SCAN (Oct 2025 Crash)")
    print("="*60)
    
    data_file = "data/btcusdt_1m_oct2025.csv"
    if not os.path.exists(data_file):
        print("Data missing.")
        return
        
    df = pd.read_csv(data_file)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df['best_bid'] = df['close'] * (1 - 0.00005)
    df['best_ask'] = df['close'] * (1 + 0.00005)
    
    # We only need the crash period (first 3 days are enough based on previous runs)
    # Step 166 was the trigger point for $350.
    df_short = df.iloc[:2000] 
    print(f"Loaded {len(df_short)} candles (Zoomed in on crash area).")
    
    thresholds = [150, 200, 250, 300, 350, 400]
    results = []
    
    for th in thresholds:
        print(f"Testing ${th}...", end="\r")
        res = await run_scenario_fast(float(th), df_short)
        results.append(res)
        
    print("\n" + "-"*70)
    print(f"{'Limit':<8} | {'Triggered?':<10} | {'Step':<8} | {'Realized Loss (Approx)':<22} | {'Equity Reached'}")
    print("-" * 70)
    
    for r in results:
        loss_display = f"${r['SavedLoss']:.2f}" if r['Triggered'] else "-"
        print(f"${r['Threshold']:<7} | {str(r['Triggered']):<10} | {r['Step']:<8} | {loss_display:<22} | ${r['FinalEquity']:.2f}")
    print("-" * 70)

if __name__ == "__main__":
    asyncio.run(scan_thresholds())
