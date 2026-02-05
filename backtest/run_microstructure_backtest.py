"""
Microstructure Signals Backtest - v5.2 Validation
Compares: Baseline vs Microstructure-Enabled Market Making
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from ml.microstructure import MicrostructureIntegrator

# Configuration
DATA_FILE = "data/btcusdt_1m_1year.csv"
CHUNK_DAYS = 7
MAX_CHUNKS = 52  # Full year
INITIAL_BALANCE = 10000
ORDER_SIZE_USD = 200
SPREAD_PCT = 0.002


class MicrostructureBacktester:
    def __init__(self, use_microstructure=False):
        self.use_microstructure = use_microstructure
        self.pnl = 0.0
        self.trades = []
        self.adverse_selections = 0
        self.total_fills = 0
        self.pending_buy = None
        self.pending_sell = None
        
        if use_microstructure:
            self.ms = MicrostructureIntegrator({
                'vpin': {'bucket_size_usd': 10000, 'n_buckets': 50, 'threshold': 0.7},
                'trade_arrival': {'baseline_window_seconds': 3600, 'elevated_threshold': 2.0},
                'defensive_risk_score': 1.0,
                'cautious_risk_score': 0.5
            })
        else:
            self.ms = None
    
    def run_step(self, row, prev_row):
        mid_price = row['close']
        
        # Simulate trade for microstructure
        if self.ms and prev_row is not None:
            price_change = row['close'] - prev_row['close']
            side = 'buy' if price_change > 0 else 'sell'
            simulated_trade = {
                'price': mid_price,
                'size': row['volume'] / 10 if 'volume' in row else 0.1,
                'side': side
            }
            self.ms.update_trade(simulated_trade)
        
        # Get multipliers
        spread_mult = 1.0
        size_mult = 1.0
        
        if self.ms:
            analysis = self.ms.analyze()
            spread_mult = analysis['spread_mult']
            size_mult = analysis['size_mult']
        
        # Calculate prices
        effective_spread = SPREAD_PCT * spread_mult
        bid_price = mid_price * (1 - effective_spread / 2)
        ask_price = mid_price * (1 + effective_spread / 2)
        
        # Simulate fills
        bid_filled = row['low'] <= bid_price
        ask_filled = row['high'] >= ask_price
        
        # Check adverse selection
        if self.pending_buy:
            self.total_fills += 1
            if mid_price < self.pending_buy['price']:
                self.adverse_selections += 1
            self.pending_buy = None
            
        if self.pending_sell:
            self.total_fills += 1
            if mid_price > self.pending_sell['price']:
                self.adverse_selections += 1
            self.pending_sell = None
        
        # Record fills
        if bid_filled:
            self.pending_buy = {'price': bid_price}
        if ask_filled:
            self.pending_sell = {'price': ask_price}
        
        # PnL from round trips
        if bid_filled and ask_filled:
            order_size = ORDER_SIZE_USD * size_mult / mid_price
            profit = order_size * (ask_price - bid_price)
            self.pnl += profit
            self.trades.append(profit)
    
    def get_results(self):
        adverse_rate = self.adverse_selections / max(1, self.total_fills)
        return {
            'pnl': self.pnl,
            'trades': len(self.trades),
            'adverse_rate': adverse_rate,
            'total_fills': self.total_fills,
            'adverse_selections': self.adverse_selections
        }


def main():
    print("=" * 60)
    print("Microstructure Signals Backtest - v5.2 Validation")
    print("=" * 60)
    
    df = pd.read_csv(DATA_FILE)
    df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
    
    total_rows = len(df)
    rows_per_chunk = CHUNK_DAYS * 24 * 60
    
    print(f"Total rows: {total_rows:,}")
    print(f"Chunk size: {rows_per_chunk:,} rows ({CHUNK_DAYS} days)")
    print(f"Max chunks: {MAX_CHUNKS}")
    
    baseline_total = {'pnl': 0, 'trades': 0, 'adverse_selections': 0, 'total_fills': 0}
    ms_total = {'pnl': 0, 'trades': 0, 'adverse_selections': 0, 'total_fills': 0}
    
    chunk_count = 0
    for start_idx in range(0, total_rows, rows_per_chunk):
        if chunk_count >= MAX_CHUNKS:
            break
        
        end_idx = min(start_idx + rows_per_chunk, total_rows)
        df_chunk = df.iloc[start_idx:end_idx].copy()
        
        chunk_count += 1
        if chunk_count % 10 == 0:
            print(f"Processing chunk {chunk_count}/{MAX_CHUNKS}...")
        
        # Run baseline
        bt_base = MicrostructureBacktester(use_microstructure=False)
        prev = None
        for _, row in df_chunk.iterrows():
            bt_base.run_step(row, prev)
            prev = row
        res_base = bt_base.get_results()
        
        # Run with Microstructure
        bt_ms = MicrostructureBacktester(use_microstructure=True)
        prev = None
        for _, row in df_chunk.iterrows():
            bt_ms.run_step(row, prev)
            prev = row
        res_ms = bt_ms.get_results()
        
        # Accumulate
        for key in baseline_total:
            baseline_total[key] += res_base[key]
            ms_total[key] += res_ms[key]
    
    # Calculate adverse rates
    baseline_total['adverse_rate'] = baseline_total['adverse_selections'] / max(1, baseline_total['total_fills'])
    ms_total['adverse_rate'] = ms_total['adverse_selections'] / max(1, ms_total['total_fills'])
    
    # Results
    print("\n" + "=" * 60)
    print("FINAL RESULTS (1 Year)")
    print("=" * 60)
    
    print(f"\n{'Metric':<25} {'Baseline':>12} {'Microstr':>12} {'Diff':>10}")
    print("-" * 60)
    print(f"{'PnL':.<25} ${baseline_total['pnl']:>11.2f} ${ms_total['pnl']:>11.2f} {ms_total['pnl']-baseline_total['pnl']:>+10.2f}")
    print(f"{'Trades':.<25} {baseline_total['trades']:>12,} {ms_total['trades']:>12,}")
    print(f"{'Adverse Rate':.<25} {baseline_total['adverse_rate']:>11.1%} {ms_total['adverse_rate']:>11.1%} {(ms_total['adverse_rate']-baseline_total['adverse_rate'])*100:>+9.1f}%")
    
    # Verdict
    print("\n" + "=" * 60)
    pnl_improvement = ms_total['pnl'] - baseline_total['pnl']
    adverse_improvement = baseline_total['adverse_rate'] - ms_total['adverse_rate']
    
    if pnl_improvement > 0 and adverse_improvement > 0:
        print(f"✅ MICROSTRUCTURE SIGNALS: +${pnl_improvement:.2f}, Adverse -{adverse_improvement*100:.1f}%")
        print("   RECOMMENDATION: KEEP")
    elif pnl_improvement > 0:
        print(f"⚠️ MICROSTRUCTURE: MIXED (PnL +${pnl_improvement:.2f}, Adverse +{abs(adverse_improvement)*100:.1f}%)")
        print("   RECOMMENDATION: REVIEW")
    elif adverse_improvement > 0:
        print(f"⚠️ MICROSTRUCTURE: MIXED (PnL ${pnl_improvement:.2f}, Adverse -{adverse_improvement*100:.1f}%)")
        print("   RECOMMENDATION: REVIEW")
    else:
        print(f"❌ MICROSTRUCTURE: ${pnl_improvement:.2f}")
        print("   RECOMMENDATION: ROLLBACK")
    print("=" * 60)


if __name__ == "__main__":
    main()
