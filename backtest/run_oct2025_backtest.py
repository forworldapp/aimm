"""
Order Flow Analysis Backtest - October 2025 Crash Period
Tests effectiveness during high volatility
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from ml.order_flow_analyzer import OrderFlowAnalyzer

# Configuration
DATA_FILE = "data/btcusdt_1m_1year.csv"
INITIAL_BALANCE = 10000
ORDER_SIZE_USD = 200
SPREAD_PCT = 0.002

class SimpleBacktester:
    def __init__(self, use_order_flow=False):
        self.use_order_flow = use_order_flow
        self.pnl = 0.0
        self.trades = []
        self.adverse_selections = 0
        self.total_fills = 0
        self.pending_buy = None
        self.pending_sell = None
        
        if use_order_flow:
            self.of_analyzer = OrderFlowAnalyzer()
        else:
            self.of_analyzer = None
    
    def simulate_orderbook(self, row, prev_row=None):
        mid = row['close']
        spread = mid * SPREAD_PCT
        bids = [[mid - spread * (i+1), np.random.uniform(0.5, 2.0)] for i in range(5)]
        asks = [[mid + spread * (i+1), np.random.uniform(0.5, 2.0)] for i in range(5)]
        
        if prev_row is not None:
            price_change = row['close'] - prev_row['close']
            if price_change > 0:
                for i in range(len(bids)):
                    bids[i][1] *= 1.5
            elif price_change < 0:
                for i in range(len(asks)):
                    asks[i][1] *= 1.5
        
        return {'bids': bids, 'asks': asks, 'mid': mid}
    
    def run_step(self, row, prev_row=None):
        mid_price = row['close']
        orderbook = self.simulate_orderbook(row, prev_row)
        
        spread_mult = 1.0
        bid_size_mult = 1.0
        ask_size_mult = 1.0
        
        if self.of_analyzer:
            adj = self.of_analyzer.get_adjustment_factors(orderbook, None)
            spread_mult = adj['spread_mult']
            bid_size_mult = adj['bid_size_mult']
            ask_size_mult = adj['ask_size_mult']
        
        effective_spread = SPREAD_PCT * spread_mult
        bid_price = mid_price * (1 - effective_spread / 2)
        ask_price = mid_price * (1 + effective_spread / 2)
        
        bid_filled = row['low'] <= bid_price
        ask_filled = row['high'] >= ask_price
        
        # Check previous fills for adverse selection
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
        
        if bid_filled:
            self.pending_buy = {'price': bid_price}
        if ask_filled:
            self.pending_sell = {'price': ask_price}
        
        if bid_filled and ask_filled:
            order_size = ORDER_SIZE_USD * min(bid_size_mult, ask_size_mult) / mid_price
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
    print("Order Flow Backtest - OCTOBER 2025 (Crash Period)")
    print("=" * 60)
    
    df = pd.read_csv(DATA_FILE)
    df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
    
    # Filter October 2025
    oct_df = df[(df['date'].dt.month == 10) & (df['date'].dt.year == 2025)].copy()
    print(f"\nOctober 2025 data: {len(oct_df):,} rows")
    print(f"Date range: {oct_df['date'].min()} to {oct_df['date'].max()}")
    
    # Run baseline
    print("\n--- Running Baseline (no Order Flow) ---")
    bt_base = SimpleBacktester(use_order_flow=False)
    prev = None
    for _, row in oct_df.iterrows():
        bt_base.run_step(row, prev)
        prev = row
    res_base = bt_base.get_results()
    print(f"PnL: ${res_base['pnl']:.2f} | Adverse: {res_base['adverse_rate']:.1%}")
    
    # Run with Order Flow
    print("\n--- Running WITH Order Flow ---")
    bt_of = SimpleBacktester(use_order_flow=True)
    prev = None
    for _, row in oct_df.iterrows():
        bt_of.run_step(row, prev)
        prev = row
    res_of = bt_of.get_results()
    print(f"PnL: ${res_of['pnl']:.2f} | Adverse: {res_of['adverse_rate']:.1%}")
    
    # Results
    print("\n" + "=" * 60)
    print("OCTOBER 2025 RESULTS")
    print("=" * 60)
    print(f"\n{'Metric':<25} {'Baseline':>12} {'Order Flow':>12} {'Diff':>10}")
    print("-" * 60)
    print(f"{'PnL':.<25} ${res_base['pnl']:>11.2f} ${res_of['pnl']:>11.2f} {res_of['pnl']-res_base['pnl']:>+10.2f}")
    print(f"{'Trades':.<25} {res_base['trades']:>12,} {res_of['trades']:>12,}")
    print(f"{'Adverse Rate':.<25} {res_base['adverse_rate']:>11.1%} {res_of['adverse_rate']:>11.1%} {(res_of['adverse_rate']-res_base['adverse_rate'])*100:>+9.1f}%")
    print("=" * 60)

if __name__ == "__main__":
    main()
