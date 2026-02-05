import sys
import os
import pandas as pd
import asyncio
import logging
import datetime

# Adjust path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.config import Config
from backtest.mock_exchange import MockExchange
from strategies.market_maker import MarketMaker

logging.getLogger("MarketMaker").setLevel(logging.CRITICAL)
logging.getLogger("MockExchange").setLevel(logging.CRITICAL)
logging.basicConfig(level=logging.WARNING)

async def run_drawdown_analysis():
    print("\n" + "="*60)
    print("📉 DRAWDOWN FREQUENCY ANALYSIS (False Positive Test)")
    print("="*60)
    
    data_file = "data/btcusdt_1m_oct2025.csv"
    if not os.path.exists(data_file):
        print(f"Data file not found: {data_file}")
        return

    print("Loading 1-month data...")
    df = pd.read_csv(data_file)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df['best_bid'] = df['close'] * (1 - 0.00005)
    df['best_ask'] = df['close'] * (1 + 0.00005)
    
    # Setup Strategy (No Breaker)
    exchange = MockExchange(df, initial_balance=10000.0)
    strategy = MarketMaker(exchange)
    strategy._load_params = lambda: None
    strategy.max_loss_usd = 1000000.0 # Unlimited
    strategy.max_position_usd = 5000.0
    strategy.initial_equity = 10000.0
    strategy.is_active = True
    Config.set("strategy", "ml_regime_enabled", True)
    
    # Tracking
    equity_curve = []
    peak_equity = 10000.0
    
    # Run Simulation
    print(f"Simulating {len(df)} candles...")
    total_steps = len(df)
    
    for i in range(total_steps):
        if not exchange.next_tick(): break
        
        # Inject Data (Optimized window)
        window_start = max(0, i - 60)
        strategy.candles = df.iloc[window_start:i+1][['open', 'high', 'low', 'close', 'volume']]
        strategy.price_history.append((df.iloc[i]['best_bid'] + df.iloc[i]['best_ask']) / 2)
        if len(strategy.price_history) > 200: strategy.price_history.pop(0)

        # Cycle
        await strategy.cycle()
        
        # Track Equity
        curr_eq = exchange.get_equity()
        peak_equity = max(peak_equity, curr_eq)
        drawdown = peak_equity - curr_eq
        
        equity_curve.append({
            'step': i,
            'time': df.iloc[i]['timestamp'],
            'equity': curr_eq,
            'peak': peak_equity,
            'drawdown': drawdown
        })
        
        if i % 5000 == 0:
            print(f"Processing... {i}/{total_steps} ({(i/total_steps)*100:.0f}%)", end="\r")

    print(f"\nSimulation Complete. Analyzing Drawdowns...")
    
    df_eq = pd.DataFrame(equity_curve)
    
    def analyze_threshold(threshold):
        # Identify trigger points (crossing threshold)
        triggers = []
        is_triggered = False
        
        for idx, row in df_eq.iterrows():
            if not is_triggered and row['drawdown'] > threshold:
                # Trigger Event!
                is_triggered = True
                
                # Check next 24h (1440 steps) for recovery
                start_step = idx
                start_time = row['time']
                start_equity = row['equity']
                
                # Look ahead
                future = df_eq.iloc[int(idx):int(idx)+1440]
                recovered = False
                max_future_dd = row['drawdown']
                
                for _, f_row in future.iterrows():
                    max_future_dd = max(max_future_dd, f_row['drawdown'])
                    if f_row['equity'] >= row['peak']: # Recovered to Peak
                        recovered = True
                        break
                
                triggers.append({
                    'time': start_time,
                    'drawdown_at_trigger': row['drawdown'],
                    'max_future_dd': max_future_dd,
                    'recovered_24h': "YES" if recovered else "NO"
                })
            
            # Reset logic: we only count "fresh" triggers. 
            # In real life, bot stops. Here we define "reset" as recovering to peak? 
            # Or just count distinct DD events.
            # Simplified: Reset flag if DD drops below threshold (simulating manual restart?)
            # Better: Reset only if equity hits new peak (full recovery).
            if row['equity'] >= row['peak']:
                is_triggered = False
                
        return triggers

    triggers_200 = analyze_threshold(200)
    triggers_350 = analyze_threshold(350)
    
    print(f"\n📊 COMPARISON: $200 vs $350 Thresholds (False Positive Check)")
    print("-" * 70)
    
    print(f"\n🔹 Setting: $200 Limit")
    print(f"Total Triggers: {len(triggers_200)}")
    for t in triggers_200:
        label = "❌ FALSE ALARM" if t['recovered_24h'] == "YES" else "✅ TRUE SAVE"
        print(f"   {t['time']} | DD: ${t['drawdown_at_trigger']:.0f} -> Max: ${t['max_future_dd']:.0f} | {label} (Recov: {t['recovered_24h']})")

    print(f"\n🔹 Setting: $350 Limit")
    print(f"Total Triggers: {len(triggers_350)}")
    for t in triggers_350:
        label = "❌ FALSE ALARM" if t['recovered_24h'] == "YES" else "✅ TRUE SAVE"
        print(f"   {t['time']} | DD: ${t['drawdown_at_trigger']:.0f} -> Max: ${t['max_future_dd']:.0f} | {label} (Recov: {t['recovered_24h']})")
    
    print("-" * 70)
    if len(triggers_200) > len(triggers_350):
        diff = len(triggers_200) - len(triggers_350)
        print(f"⚠️ RISK: $200 allows {diff} more stops than $350.")
        if diff > 3:
            print("   -> Frequent stops might annoy you (False Positives).")
        else:
            print("   -> Stops are rare enough. $200 is acceptable.")
    else:
        print("✅ $200 is just as stable as $350 (No extra false alarms).")

if __name__ == "__main__":
    asyncio.run(run_drawdown_analysis())
