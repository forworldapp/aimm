"""
Cross-Asset Hedging Backtest - v5.3 Validation
Simulates BTC/ETH correlation-based hedging
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from core.cross_asset_hedger import CrossAssetHedgeIntegrator

# Configuration
DATA_FILE = "data/btcusdt_1m_1year.csv"
CHUNK_DAYS = 7
MAX_CHUNKS = 52
INITIAL_BALANCE = 10000
ORDER_SIZE_USD = 200
SPREAD_PCT = 0.002


class HedgeBacktester:
    def __init__(self, use_hedge=False):
        self.use_hedge = use_hedge
        self.pnl = 0.0
        self.eth_pnl = 0.0
        self.hedge_pnl = 0.0
        self.trades = 0
        self.inventory = 0.0  # ETH inventory
        self.hedge_pos = 0.0  # BTC hedge position
        
        if use_hedge:
            self.hedger = CrossAssetHedgeIntegrator({
                'enabled': True,
                'correlation': {'window_minutes': 1440, 'min_correlation': 0.7},
                'hedge': {'ratio_beta': 0.8, 'min_position_usd': 100, 'rebalance_threshold': 0.1}
            })
        else:
            self.hedger = None
    
    def run_step(self, row, prev_row):
        # Simulate ETH price as BTC * 0.06 with some noise
        btc_price = row['close']
        eth_price = btc_price * 0.06 * (1 + np.random.normal(0, 0.001))
        
        if prev_row is not None:
            prev_btc = prev_row['close']
            prev_eth = prev_btc * 0.06 * (1 + np.random.normal(0, 0.001))
            
            # Mark-to-market PnL
            if self.inventory != 0:
                eth_return = (eth_price - prev_eth) / prev_eth
                self.eth_pnl += self.inventory * prev_eth * eth_return
            
            if self.hedge_pos != 0:
                btc_return = (btc_price - prev_btc) / prev_btc
                self.hedge_pnl += self.hedge_pos * prev_btc * btc_return
        
        # Simulate market making fills
        mid = btc_price
        bid = mid * (1 - SPREAD_PCT / 2)
        ask = mid * (1 + SPREAD_PCT / 2)
        
        bid_filled = row['low'] <= bid
        ask_filled = row['high'] >= ask
        
        if bid_filled and ask_filled:
            # Round trip profit
            qty = ORDER_SIZE_USD / mid
            profit = qty * (ask - bid)
            self.eth_pnl += profit
            self.trades += 1
        elif bid_filled:
            self.inventory += ORDER_SIZE_USD / eth_price
        elif ask_filled and self.inventory > 0:
            sell_size = min(self.inventory, ORDER_SIZE_USD / eth_price)
            self.inventory -= sell_size
        
        # Update hedge if enabled
        if self.hedger:
            result = self.hedger.update_and_analyze(eth_price, btc_price, self.inventory)
            
            if result['action'] == 'rebalance':
                # Execute hedge
                old_pos = self.hedge_pos
                self.hedge_pos = result['hedge_btc']
                # Simulated execution cost
                execution_cost = abs(self.hedge_pos - old_pos) * btc_price * 0.0004  # 4 bps
                self.hedge_pnl -= execution_cost
    
    def get_results(self):
        return {
            'eth_pnl': self.eth_pnl,
            'hedge_pnl': self.hedge_pnl,
            'total_pnl': self.eth_pnl + self.hedge_pnl,
            'trades': self.trades
        }


def main():
    print("=" * 60)
    print("Cross-Asset Hedging Backtest - v5.3 Validation")
    print("=" * 60)
    
    np.random.seed(42)  # Reproducibility
    
    df = pd.read_csv(DATA_FILE)
    total_rows = len(df)
    rows_per_chunk = CHUNK_DAYS * 24 * 60
    
    print(f"Total rows: {total_rows:,}")
    print(f"Max chunks: {MAX_CHUNKS}")
    
    baseline_total = {'eth_pnl': 0, 'hedge_pnl': 0, 'total_pnl': 0, 'trades': 0}
    hedge_total = {'eth_pnl': 0, 'hedge_pnl': 0, 'total_pnl': 0, 'trades': 0}
    
    chunk_count = 0
    for start_idx in range(0, total_rows, rows_per_chunk):
        if chunk_count >= MAX_CHUNKS:
            break
        
        end_idx = min(start_idx + rows_per_chunk, total_rows)
        df_chunk = df.iloc[start_idx:end_idx].copy()
        
        chunk_count += 1
        if chunk_count % 10 == 0:
            print(f"Processing chunk {chunk_count}/{MAX_CHUNKS}...")
        
        # Baseline
        bt_base = HedgeBacktester(use_hedge=False)
        prev = None
        for _, row in df_chunk.iterrows():
            bt_base.run_step(row, prev)
            prev = row
        res_base = bt_base.get_results()
        
        # With Hedge
        bt_hedge = HedgeBacktester(use_hedge=True)
        prev = None
        for _, row in df_chunk.iterrows():
            bt_hedge.run_step(row, prev)
            prev = row
        res_hedge = bt_hedge.get_results()
        
        for key in baseline_total:
            baseline_total[key] += res_base[key]
            hedge_total[key] += res_hedge[key]
    
    # Results
    print("\n" + "=" * 60)
    print("FINAL RESULTS (1 Year)")
    print("=" * 60)
    
    print(f"\n{'Metric':<25} {'Baseline':>12} {'Hedged':>12} {'Diff':>10}")
    print("-" * 60)
    print(f"{'ETH Trading PnL':.<25} ${baseline_total['eth_pnl']:>11.2f} ${hedge_total['eth_pnl']:>11.2f}")
    print(f"{'Hedge PnL':.<25} ${baseline_total['hedge_pnl']:>11.2f} ${hedge_total['hedge_pnl']:>11.2f}")
    print(f"{'TOTAL PnL':.<25} ${baseline_total['total_pnl']:>11.2f} ${hedge_total['total_pnl']:>11.2f} {hedge_total['total_pnl']-baseline_total['total_pnl']:>+10.2f}")
    print(f"{'Trades':.<25} {baseline_total['trades']:>12,} {hedge_total['trades']:>12,}")
    
    # Verdict
    print("\n" + "=" * 60)
    improvement = hedge_total['total_pnl'] - baseline_total['total_pnl']
    
    if improvement > 0:
        print(f"✅ CROSS-ASSET HEDGE: +${improvement:.2f}")
        print("   RECOMMENDATION: KEEP")
    else:
        print(f"❌ CROSS-ASSET HEDGE: ${improvement:.2f}")
        print("   RECOMMENDATION: ROLLBACK")
    print("=" * 60)


if __name__ == "__main__":
    main()
