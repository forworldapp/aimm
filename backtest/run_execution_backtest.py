"""
Execution Algorithm Backtest - v5.4 Validation
Compares: Immediate execution vs TWAP slicing
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from core.execution_algo import TWAPExecutor

# Configuration
DATA_FILE = "data/btcusdt_1m_1year.csv"
CHUNK_DAYS = 7
MAX_CHUNKS = 52
ORDER_SIZE_USD = 2000  # Larger orders to see impact


class ExecutionBacktester:
    def __init__(self, use_twap=False, slice_duration_minutes=10):
        self.use_twap = use_twap
        self.slice_duration = slice_duration_minutes
        self.total_slippage = 0.0
        self.executions = []
        
        if use_twap:
            self.executor = TWAPExecutor({'slice_interval_seconds': 60})
        else:
            self.executor = None
    
    def simulate_execution(self, df: pd.DataFrame, start_idx: int, qty: float, side: str) -> float:
        """Simulate order execution and return realized price"""
        if not self.use_twap:
            # Immediate execution - take full impact
            row = df.iloc[start_idx]
            mid = row['close']
            impact = 0.0005 * (qty / 0.1)  # 5bps per 0.1 BTC
            impact = min(impact, 0.005)  # Cap at 50 bps
            
            if side == 'buy':
                realized = mid * (1 + impact)
            else:
                realized = mid * (1 - impact)
            
            return realized
        else:
            # TWAP - slice over time
            num_slices = min(10, self.slice_duration)
            prices = []
            
            for i in range(num_slices):
                idx = min(start_idx + i, len(df) - 1)
                row = df.iloc[idx]
                mid = row['close']
                
                # Smaller impact per slice
                slice_qty = qty / num_slices
                impact = 0.0005 * (slice_qty / 0.1)
                impact = min(impact, 0.001)  # Cap at 10 bps per slice
                
                if side == 'buy':
                    prices.append(mid * (1 + impact))
                else:
                    prices.append(mid * (1 - impact))
            
            return sum(prices) / len(prices) if prices else 0
    
    def run_chunk(self, df: pd.DataFrame) -> Dict:
        """Run chunk and measure execution quality"""
        total_slippage = 0.0
        num_executions = 0
        
        # Simulate random large orders throughout chunk
        np.random.seed(42)
        order_indices = np.random.choice(len(df) - 20, size=min(10, len(df) // 100), replace=False)
        
        for idx in order_indices:
            side = 'buy' if np.random.random() > 0.5 else 'sell'
            qty = ORDER_SIZE_USD / df.iloc[idx]['close']
            
            mid_price = df.iloc[idx]['close']
            realized_price = self.simulate_execution(df, idx, qty, side)
            
            # Calculate slippage
            if side == 'buy':
                slippage_pct = (realized_price - mid_price) / mid_price
            else:
                slippage_pct = (mid_price - realized_price) / mid_price
            
            total_slippage += slippage_pct * ORDER_SIZE_USD
            num_executions += 1
        
        return {
            'slippage_usd': total_slippage,
            'executions': num_executions
        }


def main():
    print("=" * 60)
    print("Execution Algorithm Backtest - v5.4 Validation")
    print("=" * 60)
    
    df = pd.read_csv(DATA_FILE)
    total_rows = len(df)
    rows_per_chunk = CHUNK_DAYS * 24 * 60
    
    print(f"Total rows: {total_rows:,}")
    print(f"Order size: ${ORDER_SIZE_USD}")
    
    immediate_total = {'slippage_usd': 0, 'executions': 0}
    twap_total = {'slippage_usd': 0, 'executions': 0}
    
    chunk_count = 0
    for start_idx in range(0, total_rows, rows_per_chunk):
        if chunk_count >= MAX_CHUNKS:
            break
        
        end_idx = min(start_idx + rows_per_chunk, total_rows)
        df_chunk = df.iloc[start_idx:end_idx].reset_index(drop=True)
        
        chunk_count += 1
        if chunk_count % 10 == 0:
            print(f"Processing chunk {chunk_count}/{MAX_CHUNKS}...")
        
        # Immediate execution
        bt_imm = ExecutionBacktester(use_twap=False)
        res_imm = bt_imm.run_chunk(df_chunk)
        
        # TWAP execution
        bt_twap = ExecutionBacktester(use_twap=True, slice_duration_minutes=10)
        res_twap = bt_twap.run_chunk(df_chunk)
        
        for key in immediate_total:
            immediate_total[key] += res_imm[key]
            twap_total[key] += res_twap[key]
    
    # Results
    print("\n" + "=" * 60)
    print("FINAL RESULTS (1 Year)")
    print("=" * 60)
    
    print(f"\n{'Metric':<25} {'Immediate':>12} {'TWAP':>12} {'Savings':>10}")
    print("-" * 60)
    print(f"{'Total Slippage':.<25} ${immediate_total['slippage_usd']:>11.2f} ${twap_total['slippage_usd']:>11.2f} ${immediate_total['slippage_usd']-twap_total['slippage_usd']:>+9.2f}")
    print(f"{'Executions':.<25} {immediate_total['executions']:>12,} {twap_total['executions']:>12,}")
    
    avg_imm = immediate_total['slippage_usd'] / max(1, immediate_total['executions'])
    avg_twap = twap_total['slippage_usd'] / max(1, twap_total['executions'])
    print(f"{'Avg Slippage/Order':.<25} ${avg_imm:>11.2f} ${avg_twap:>11.2f}")
    
    # Verdict
    print("\n" + "=" * 60)
    savings = immediate_total['slippage_usd'] - twap_total['slippage_usd']
    
    if savings > 0:
        print(f"✅ TWAP EXECUTION: Saves ${savings:.2f} in slippage")
        print("   RECOMMENDATION: KEEP")
    else:
        print(f"❌ TWAP EXECUTION: No improvement (${savings:.2f})")
        print("   RECOMMENDATION: REVIEW")
    print("=" * 60)


if __name__ == "__main__":
    main()
