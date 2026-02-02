"""
Order Flow Analysis Backtest - v5.0 Validation
Compares: Baseline (no OF) vs Order Flow Enabled
Uses chunked processing to avoid long runtimes.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from ml.order_flow_analyzer import OrderFlowAnalyzer

# Configuration
DATA_FILE = "data/btcusdt_1m_1year.csv"
CHUNK_DAYS = 7  # Process 1 week at a time
MAX_CHUNKS = 8  # ~2 months for broader coverage
INITIAL_BALANCE = 10000
ORDER_SIZE_USD = 200
SPREAD_PCT = 0.002  # 0.2% base spread

class SimpleBacktester:
    """Simplified backtest engine for Order Flow comparison"""
    
    def __init__(self, use_order_flow=False):
        self.use_order_flow = use_order_flow
        self.balance = INITIAL_BALANCE
        self.position = 0.0
        self.entry_price = 0.0
        self.pnl = 0.0
        self.trades = []
        self.adverse_selections = 0
        self.total_fills = 0
        
        if use_order_flow:
            self.of_analyzer = OrderFlowAnalyzer()
        else:
            self.of_analyzer = None
    
    def simulate_orderbook(self, row, prev_row=None):
        """Create synthetic orderbook from candle data"""
        mid = row['close']
        spread = mid * SPREAD_PCT
        
        # Simulate order book with some depth
        bids = [[mid - spread * (i+1), np.random.uniform(0.5, 2.0)] for i in range(5)]
        asks = [[mid + spread * (i+1), np.random.uniform(0.5, 2.0)] for i in range(5)]
        
        # Add imbalance based on price movement
        if prev_row is not None:
            price_change = row['close'] - prev_row['close']
            if price_change > 0:
                # Price went up - simulate more buy pressure
                for i in range(len(bids)):
                    bids[i][1] *= 1.5
            elif price_change < 0:
                # Price went down - simulate more sell pressure
                for i in range(len(asks)):
                    asks[i][1] *= 1.5
        
        return {'bids': bids, 'asks': asks, 'mid': mid}
    
    def simulate_trade(self, row, prev_row=None):
        """Simulate trade for toxicity calculation"""
        if prev_row is None:
            return None
        
        price_change = row['close'] - prev_row['close']
        side = 'buy' if price_change > 0 else 'sell'
        
        return {
            'price': row['close'],
            'side': side,
            'size': np.random.uniform(0.01, 0.1)
        }
    
    def run_step(self, row, prev_row=None):
        """Run one simulation step"""
        mid_price = row['close']
        
        # Build synthetic orderbook
        orderbook = self.simulate_orderbook(row, prev_row)
        trade = self.simulate_trade(row, prev_row)
        
        # Get spread/size multipliers
        spread_mult = 1.0
        bid_size_mult = 1.0
        ask_size_mult = 1.0
        
        if self.of_analyzer:
            adj = self.of_analyzer.get_adjustment_factors(orderbook, trade)
            spread_mult = adj['spread_mult']
            bid_size_mult = adj['bid_size_mult']
            ask_size_mult = adj['ask_size_mult']
        
        # Calculate order prices
        effective_spread = SPREAD_PCT * spread_mult
        bid_price = mid_price * (1 - effective_spread / 2)
        ask_price = mid_price * (1 + effective_spread / 2)
        
        # Simulate fills based on high/low
        bid_filled = row['low'] <= bid_price
        ask_filled = row['high'] >= ask_price
        
        # Track adverse selection: check PREVIOUS fills against current price
        # If we bought last candle and current mid is BELOW our buy price = adverse
        if hasattr(self, 'pending_buy') and self.pending_buy:
            self.total_fills += 1
            if mid_price < self.pending_buy['price']:
                self.adverse_selections += 1
            self.pending_buy = None
            
        if hasattr(self, 'pending_sell') and self.pending_sell:
            self.total_fills += 1
            if mid_price > self.pending_sell['price']:
                self.adverse_selections += 1
            self.pending_sell = None
        
        # Record this candle's fills for NEXT candle's adverse check
        if bid_filled:
            self.pending_buy = {'price': bid_price, 'size': ORDER_SIZE_USD * bid_size_mult / mid_price}
        if ask_filled:
            self.pending_sell = {'price': ask_price, 'size': ORDER_SIZE_USD * ask_size_mult / mid_price}
        
        # Simplified PnL: spread capture when both sides fill
        if bid_filled and ask_filled:
            # Round trip profit
            order_size = ORDER_SIZE_USD * min(bid_size_mult, ask_size_mult) / mid_price
            profit = order_size * (ask_price - bid_price)
            self.pnl += profit
            self.trades.append({
                'time': row.get('timestamp', 0),
                'profit': profit,
                'spread_mult': spread_mult
            })
        elif bid_filled:
            # Long position (potential inventory risk)
            order_size = ORDER_SIZE_USD * bid_size_mult / mid_price
            self.position += order_size
            self.entry_price = bid_price
        elif ask_filled:
            # Short position or close long
            if self.position > 0:
                # Close long
                profit = self.position * (ask_price - self.entry_price)
                self.pnl += profit
                self.position = 0
    
    def get_results(self):
        """Return backtest results"""
        adverse_rate = self.adverse_selections / max(1, self.total_fills)
        return {
            'final_pnl': self.pnl,
            'num_trades': len(self.trades),
            'adverse_rate': adverse_rate,
            'total_fills': self.total_fills,
            'adverse_selections': self.adverse_selections
        }


def run_chunk(df_chunk, use_order_flow):
    """Run backtest on a single chunk"""
    bt = SimpleBacktester(use_order_flow=use_order_flow)
    
    prev_row = None
    for idx, row in df_chunk.iterrows():
        bt.run_step(row, prev_row)
        prev_row = row
    
    return bt.get_results()


def main():
    print("=" * 60)
    print("Order Flow Analysis Backtest - v5.0 Validation")
    print("=" * 60)
    
    # Load data
    print(f"\nLoading data from {DATA_FILE}...")
    df = pd.read_csv(DATA_FILE)
    
    # Ensure timestamp column
    if 'timestamp' not in df.columns and 'open_time' in df.columns:
        df['timestamp'] = df['open_time']
    
    total_rows = len(df)
    rows_per_chunk = CHUNK_DAYS * 24 * 60  # minutes per chunk
    
    print(f"Total rows: {total_rows:,}")
    print(f"Chunk size: {rows_per_chunk:,} rows ({CHUNK_DAYS} days)")
    print(f"Max chunks: {MAX_CHUNKS}")
    
    # Results accumulators
    baseline_results = {'final_pnl': 0, 'num_trades': 0, 'adverse_rate': 0, 'total_fills': 0, 'adverse_selections': 0}
    of_results = {'final_pnl': 0, 'num_trades': 0, 'adverse_rate': 0, 'total_fills': 0, 'adverse_selections': 0}
    
    chunk_count = 0
    for start_idx in range(0, total_rows, rows_per_chunk):
        if chunk_count >= MAX_CHUNKS:
            break
        
        end_idx = min(start_idx + rows_per_chunk, total_rows)
        df_chunk = df.iloc[start_idx:end_idx].copy()
        
        chunk_count += 1
        print(f"\n--- Chunk {chunk_count}/{MAX_CHUNKS} (rows {start_idx:,}-{end_idx:,}) ---")
        
        # Run baseline
        print("  Running Baseline (no Order Flow)...", end=" ")
        res_base = run_chunk(df_chunk, use_order_flow=False)
        print(f"PnL: ${res_base['final_pnl']:.2f}, Adverse: {res_base['adverse_rate']:.1%}")
        
        # Run with Order Flow
        print("  Running WITH Order Flow...", end=" ")
        res_of = run_chunk(df_chunk, use_order_flow=True)
        print(f"PnL: ${res_of['final_pnl']:.2f}, Adverse: {res_of['adverse_rate']:.1%}")
        
        # Accumulate
        for key in baseline_results:
            baseline_results[key] += res_base[key]
            of_results[key] += res_of[key]
    
    # Final results
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    
    # Calculate average adverse rate
    baseline_results['adverse_rate'] = baseline_results['adverse_selections'] / max(1, baseline_results['total_fills'])
    of_results['adverse_rate'] = of_results['adverse_selections'] / max(1, of_results['total_fills'])
    
    print(f"\n{'Metric':<25} {'Baseline':>15} {'Order Flow':>15} {'Diff':>12}")
    print("-" * 70)
    print(f"{'Total PnL':.<25} ${baseline_results['final_pnl']:>14.2f} ${of_results['final_pnl']:>14.2f} {of_results['final_pnl'] - baseline_results['final_pnl']:>+12.2f}")
    print(f"{'Num Trades':.<25} {baseline_results['num_trades']:>15,} {of_results['num_trades']:>15,}")
    print(f"{'Total Fills':.<25} {baseline_results['total_fills']:>15,} {of_results['total_fills']:>15,}")
    print(f"{'Adverse Selections':.<25} {baseline_results['adverse_selections']:>15,} {of_results['adverse_selections']:>15,}")
    print(f"{'Adverse Rate':.<25} {baseline_results['adverse_rate']:>14.1%} {of_results['adverse_rate']:>14.1%} {(of_results['adverse_rate'] - baseline_results['adverse_rate'])*100:>+11.1f}%")
    
    # Verdict
    print("\n" + "=" * 60)
    pnl_improvement = of_results['final_pnl'] - baseline_results['final_pnl']
    adverse_improvement = baseline_results['adverse_rate'] - of_results['adverse_rate']
    
    if pnl_improvement > 0 and adverse_improvement > 0:
        print("✅ ORDER FLOW ANALYSIS: SUPERIOR")
        print(f"   PnL Improvement: +${pnl_improvement:.2f}")
        print(f"   Adverse Selection Reduction: {adverse_improvement*100:.1f}%")
    elif pnl_improvement > 0:
        print("⚠️ ORDER FLOW ANALYSIS: MIXED (PnL better, Adverse worse)")
    elif adverse_improvement > 0:
        print("⚠️ ORDER FLOW ANALYSIS: MIXED (Adverse better, PnL worse)")
    else:
        print("❌ ORDER FLOW ANALYSIS: UNDERPERFORMED")
    
    print("=" * 60)


if __name__ == "__main__":
    main()
