"""
Funding Rate Arbitrage Backtest - v5.1 Validation
Compares: Baseline vs Funding-Integrated Market Making
Uses chunked processing with simulated funding rates
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from core.funding_monitor import FundingRateMonitor, FundingIntegratedMM

# Configuration
DATA_FILE = "data/btcusdt_1m_1year.csv"
CHUNK_DAYS = 7
MAX_CHUNKS = 52  # Full year
INITIAL_BALANCE = 10000
ORDER_SIZE_USD = 200
SPREAD_PCT = 0.002

# Simulated funding rates (realistic range: -0.1% to +0.3%)
# Based on historical BTC perpetual funding
def simulate_funding_rate(hour_of_day: int, volatility: float = 0.01) -> float:
    """
    Simulate realistic funding rate
    - Tends to be positive in bull markets (longs pay shorts)
    - Higher during high volatility
    """
    base_rate = 0.0001  # 0.01% base
    volatility_factor = volatility * 10  # Higher vol = higher funding
    noise = np.random.normal(0, 0.00005)
    
    return base_rate + volatility_factor * 0.0001 + noise


class FundingBacktester:
    def __init__(self, use_funding=False):
        self.use_funding = use_funding
        self.pnl = 0.0
        self.trades = []
        self.funding_income = 0.0  # Track funding separately
        self.position = 0.0
        self.position_entry_price = 0.0
        
        if use_funding:
            self.funding_monitor = FundingRateMonitor()
            self.funding_integrator = FundingIntegratedMM()
        else:
            self.funding_monitor = None
            self.funding_integrator = None
    
    def run_step(self, row, prev_row, hour_of_day, candle_idx):
        mid_price = row['close']
        
        # Simulate funding rate (changes every 8 hours)
        if candle_idx % (8 * 60) == 0:  # Every 8 hours
            volatility = abs(row['close'] - row['open']) / row['open']
            funding_rate = simulate_funding_rate(hour_of_day, volatility)
            
            if self.funding_monitor:
                self.funding_monitor.update(funding_rate)
            
            # Apply funding payment if we have position
            if abs(self.position) > 0.001:
                # Funding payment = position * price * rate
                # Positive rate: longs pay shorts
                funding_payment = -self.position * mid_price * funding_rate
                self.funding_income += funding_payment
        
        # Get bias from funding if enabled
        bid_mult = 1.0
        ask_mult = 1.0
        
        if self.funding_monitor and self.funding_integrator:
            # Calculate simulated hours to funding based on candle position
            minutes_in_8h = 8 * 60
            minutes_since_funding = candle_idx % minutes_in_8h
            simulated_hours = (minutes_in_8h - minutes_since_funding) / 60
            
            # Override the monitor's hours calculation
            analysis = self.funding_monitor.analyze_opportunity()
            analysis['hours_to_funding'] = simulated_hours
            
            adj = self.funding_integrator.get_adjustment(analysis)
            bid_mult = adj['bid_size_mult']
            ask_mult = adj['ask_size_mult']
            
            # Only freeze in last 30 min before funding
            if simulated_hours < 0.5:
                return  # Skip this candle
        
        # Calculate order prices
        bid_price = mid_price * (1 - SPREAD_PCT / 2)
        ask_price = mid_price * (1 + SPREAD_PCT / 2)
        
        # Simulate fills
        bid_filled = row['low'] <= bid_price
        ask_filled = row['high'] >= ask_price
        
        # Apply funding bias to order sizes
        bid_size = ORDER_SIZE_USD * bid_mult / mid_price
        ask_size = ORDER_SIZE_USD * ask_mult / mid_price
        
        # Track position and PnL
        if bid_filled and ask_filled:
            # Round trip
            profit = min(bid_size, ask_size) * (ask_price - bid_price)
            self.pnl += profit
            self.trades.append(profit)
        elif bid_filled:
            # Add to long position
            self.position += bid_size
            self.position_entry_price = bid_price
        elif ask_filled and self.position > 0:
            # Close long position
            size_to_close = min(ask_size, self.position)
            profit = size_to_close * (ask_price - self.position_entry_price)
            self.pnl += profit
            self.position -= size_to_close
    
    def get_results(self):
        return {
            'pnl': self.pnl,
            'funding_income': self.funding_income,
            'total_pnl': self.pnl + self.funding_income,
            'trades': len(self.trades),
            'final_position': self.position
        }


def main():
    print("=" * 60)
    print("Funding Rate Arbitrage Backtest - v5.1 Validation")
    print("=" * 60)
    
    df = pd.read_csv(DATA_FILE)
    df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
    
    total_rows = len(df)
    rows_per_chunk = CHUNK_DAYS * 24 * 60
    
    print(f"Total rows: {total_rows:,}")
    print(f"Chunk size: {rows_per_chunk:,} rows ({CHUNK_DAYS} days)")
    print(f"Max chunks: {MAX_CHUNKS}")
    
    baseline_total = {'pnl': 0, 'funding_income': 0, 'total_pnl': 0, 'trades': 0}
    funding_total = {'pnl': 0, 'funding_income': 0, 'total_pnl': 0, 'trades': 0}
    
    chunk_count = 0
    for start_idx in range(0, total_rows, rows_per_chunk):
        if chunk_count >= MAX_CHUNKS:
            break
        
        end_idx = min(start_idx + rows_per_chunk, total_rows)
        df_chunk = df.iloc[start_idx:end_idx].copy()
        
        chunk_count += 1
        print(f"\n--- Chunk {chunk_count}/{MAX_CHUNKS} ---")
        
        # Run baseline
        bt_base = FundingBacktester(use_funding=False)
        prev = None
        for i, (_, row) in enumerate(df_chunk.iterrows()):
            hour = row['date'].hour
            bt_base.run_step(row, prev, hour, i)
            prev = row
        res_base = bt_base.get_results()
        print(f"  Baseline: PnL=${res_base['pnl']:.2f}, Funding=${res_base['funding_income']:.2f}")
        
        # Run with Funding
        bt_fund = FundingBacktester(use_funding=True)
        prev = None
        for i, (_, row) in enumerate(df_chunk.iterrows()):
            hour = row['date'].hour
            bt_fund.run_step(row, prev, hour, i)
            prev = row
        res_fund = bt_fund.get_results()
        print(f"  Funding:  PnL=${res_fund['pnl']:.2f}, Funding=${res_fund['funding_income']:.2f}")
        
        # Accumulate
        for key in baseline_total:
            baseline_total[key] += res_base[key]
            funding_total[key] += res_fund[key]
    
    # Final results
    print("\n" + "=" * 60)
    print("FINAL RESULTS (8 weeks)")
    print("=" * 60)
    
    print(f"\n{'Metric':<25} {'Baseline':>12} {'Funding':>12} {'Diff':>10}")
    print("-" * 60)
    print(f"{'Trading PnL':.<25} ${baseline_total['pnl']:>11.2f} ${funding_total['pnl']:>11.2f} {funding_total['pnl']-baseline_total['pnl']:>+10.2f}")
    print(f"{'Funding Income':.<25} ${baseline_total['funding_income']:>11.2f} ${funding_total['funding_income']:>11.2f} {funding_total['funding_income']-baseline_total['funding_income']:>+10.2f}")
    print(f"{'TOTAL PnL':.<25} ${baseline_total['total_pnl']:>11.2f} ${funding_total['total_pnl']:>11.2f} {funding_total['total_pnl']-baseline_total['total_pnl']:>+10.2f}")
    print(f"{'Trades':.<25} {baseline_total['trades']:>12,} {funding_total['trades']:>12,}")
    
    # Verdict
    print("\n" + "=" * 60)
    improvement = funding_total['total_pnl'] - baseline_total['total_pnl']
    pct_improvement = (improvement / baseline_total['total_pnl'] * 100) if baseline_total['total_pnl'] > 0 else 0
    
    if improvement > 0:
        print(f"✅ FUNDING RATE ARBITRAGE: +${improvement:.2f} ({pct_improvement:+.1f}%)")
        print("   RECOMMENDATION: KEEP")
    else:
        print(f"❌ FUNDING RATE ARBITRAGE: ${improvement:.2f} ({pct_improvement:+.1f}%)")
        print("   RECOMMENDATION: ROLLBACK")
    print("=" * 60)


if __name__ == "__main__":
    main()
