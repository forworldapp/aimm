"""
v3.8.0 vs v3.10.0 Explicit Comparison Backtest
- Simulates both strategies on the same data
- v3.8.0: Baseline aggressive (no volatility adaptation)
- v3.10.0: Hybrid mode (conservative in low vol, aggressive in high vol)
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import requests
import time
import gc
import warnings
warnings.filterwarnings('ignore')

def fetch_month_1m_data(symbol: str, year: int, month: int) -> pd.DataFrame:
    """Fetch 1 month of 1-minute data"""
    start_date = datetime(year, month, 1)
    if month == 12:
        end_date = datetime(year + 1, 1, 1)
    else:
        end_date = datetime(year, month + 1, 1)
    
    start_time = int(start_date.timestamp() * 1000)
    end_time = int(end_date.timestamp() * 1000)
    
    all_klines = []
    current_start = start_time
    
    while current_start < end_time:
        url = "https://api.binance.com/api/v3/klines"
        params = {
            "symbol": symbol,
            "interval": "1m",
            "startTime": current_start,
            "endTime": end_time,
            "limit": 1000
        }
        
        try:
            response = requests.get(url, params=params, timeout=30)
            data = response.json()
            if not data or isinstance(data, dict):
                break
            all_klines.extend(data)
            current_start = data[-1][0] + 60000
        except:
            time.sleep(1)
            continue
    
    if not all_klines:
        return pd.DataFrame()
    
    df = pd.DataFrame(all_klines, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_volume', 'trades', 'taker_buy_base',
        'taker_buy_quote', 'ignore'
    ])
    
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = df[col].astype(float)
    
    return df

def calculate_volatility(df: pd.DataFrame, window: int = 60) -> pd.DataFrame:
    """Calculate rolling volatility (sigma)"""
    df = df.copy()
    df['returns'] = df['close'].pct_change()
    df['sigma'] = df['returns'].rolling(window, min_periods=1).std()
    
    # Classify volatility regime
    low_thresh = 0.0005   # v3.10.0 threshold
    med_thresh = 0.0008
    
    df['vol_regime'] = np.where(df['sigma'] < low_thresh, 'LOW',
                       np.where(df['sigma'] < med_thresh, 'MEDIUM', 'HIGH'))
    return df

def simulate_v38_strategy(df: pd.DataFrame, spread_pct: float = 0.001,
                          order_usd: float = 100) -> dict:
    """
    v3.8.0 Baseline Strategy
    - Always aggressive
    - No quote skipping
    - Fixed spread multiplier (1.0)
    """
    n = len(df)
    mid = df['close'].values
    high = df['high'].values
    low = df['low'].values
    
    prev_mid = np.roll(mid, 1)
    prev_mid[0] = mid[0]
    
    # Standard spread (no multiplier)
    bid_level = prev_mid * (1 - spread_pct/2)
    ask_level = prev_mid * (1 + spread_pct/2)
    
    # Fill detection
    bid_fill = (low <= bid_level).astype(float)
    ask_fill = (high >= ask_level).astype(float)
    
    # No quote skipping - all quotes active
    # PnL calculation
    spread_capture = spread_pct * order_usd * 0.5
    spread_pnl = (bid_fill + ask_fill) * spread_capture
    
    # Inventory tracking
    net_position = np.cumsum(bid_fill - ask_fill) * (order_usd / mid)
    inventory_pnl = np.zeros(n)
    inventory_pnl[1:] = net_position[:-1] * np.diff(mid)
    
    total_pnl = spread_pnl.sum() + inventory_pnl.sum()
    
    return {
        'total_pnl': total_pnl,
        'spread_pnl': spread_pnl.sum(),
        'inventory_pnl': inventory_pnl.sum(),
        'fills': bid_fill.sum() + ask_fill.sum()
    }

def simulate_v310_hybrid(df: pd.DataFrame, spread_pct: float = 0.001,
                         order_usd: float = 100) -> dict:
    """
    v3.10.0 Hybrid Strategy
    - Low volatility: wider spread (1.2x), 70% quote skip
    - Medium/High volatility: aggressive (1.0x spread, no skip)
    """
    n = len(df)
    mid = df['close'].values
    high = df['high'].values
    low = df['low'].values
    vol_regime = df['vol_regime'].values
    
    prev_mid = np.roll(mid, 1)
    prev_mid[0] = mid[0]
    
    # Generate random values for quote skipping
    np.random.seed(42)  # For reproducibility
    skip_random = np.random.random(n)
    
    # Initialize arrays
    spread_pnl = np.zeros(n)
    bid_fill = np.zeros(n)
    ask_fill = np.zeros(n)
    
    for i in range(1, n):
        regime = vol_regime[i]
        
        if regime == 'LOW':
            # Low vol mode: wider spread, high skip probability
            spread_mult = 1.2
            skip_prob = 0.7
        else:
            # Medium/High vol mode: aggressive
            spread_mult = 1.0
            skip_prob = 0.0
        
        # Check if we skip this quote
        if skip_random[i] < skip_prob:
            continue  # Skip - no quote, no fill
        
        # Calculate spread levels
        effective_spread = spread_pct * spread_mult
        bid_lvl = prev_mid[i] * (1 - effective_spread/2)
        ask_lvl = prev_mid[i] * (1 + effective_spread/2)
        
        # Check fills
        if low[i] <= bid_lvl:
            bid_fill[i] = 1
            spread_pnl[i] += effective_spread * order_usd * 0.5
        
        if high[i] >= ask_lvl:
            ask_fill[i] = 1
            spread_pnl[i] += effective_spread * order_usd * 0.5
    
    # Inventory PnL
    net_position = np.cumsum(bid_fill - ask_fill) * (order_usd / mid)
    inventory_pnl = np.zeros(n)
    inventory_pnl[1:] = net_position[:-1] * np.diff(mid)
    
    total_pnl = spread_pnl.sum() + inventory_pnl.sum()
    
    return {
        'total_pnl': total_pnl,
        'spread_pnl': spread_pnl.sum(),
        'inventory_pnl': inventory_pnl.sum(),
        'fills': bid_fill.sum() + ask_fill.sum()
    }

def run_comparison(months_back: int = 12):
    """Run head-to-head comparison"""
    print("=" * 70)
    print("  📊 v3.8.0 vs v3.10.0 HEAD-TO-HEAD COMPARISON")
    print("  12-Month 1-Minute Backtest")
    print("=" * 70)
    
    results_v38 = []
    results_v310 = []
    
    # Calculate months
    now = datetime.now()
    months = []
    for i in range(months_back, 0, -1):
        target = now - timedelta(days=30*i)
        months.append((target.year, target.month))
    
    for idx, (year, month) in enumerate(months):
        month_name = f"{year}-{month:02d}"
        print(f"\n[{idx+1}/{len(months)}] {month_name}...")
        
        # Fetch data
        df = fetch_month_1m_data("BTCUSDT", year, month)
        if df.empty:
            print(f"  ⚠️ No data")
            continue
        
        # Calculate volatility
        df = calculate_volatility(df)
        
        # Low vol percentage
        low_vol_pct = (df['vol_regime'] == 'LOW').mean() * 100
        
        print(f"  📥 {len(df):,} candles | Low Vol: {low_vol_pct:.1f}%")
        
        # Run both strategies
        v38_result = simulate_v38_strategy(df)
        v310_result = simulate_v310_hybrid(df)
        
        v38_result['month'] = month_name
        v38_result['low_vol_pct'] = low_vol_pct
        v310_result['month'] = month_name
        v310_result['low_vol_pct'] = low_vol_pct
        
        results_v38.append(v38_result)
        results_v310.append(v310_result)
        
        diff = v38_result['total_pnl'] - v310_result['total_pnl']
        winner = "v3.8.0" if diff > 0 else "v3.10.0"
        
        print(f"  v3.8.0: ${v38_result['total_pnl']:.2f} | v3.10.0: ${v310_result['total_pnl']:.2f} | Winner: {winner}")
        
        del df
        gc.collect()
    
    return results_v38, results_v310

def plot_comparison(results_v38: list, results_v310: list, output_path: str):
    """Generate comparison visualization"""
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.2, 1, 1], hspace=0.35, wspace=0.25)
    
    months = [r['month'] for r in results_v38]
    pnl_v38 = [r['total_pnl'] for r in results_v38]
    pnl_v310 = [r['total_pnl'] for r in results_v310]
    
    colors = {'v38': '#2196F3', 'v310': '#FF9800', 'win': '#00C853', 'lose': '#FF1744'}
    
    # 1. Monthly comparison bars (top left)
    ax1 = fig.add_subplot(gs[0, 0])
    x = np.arange(len(months))
    width = 0.35
    ax1.bar(x - width/2, pnl_v38, width, label='v3.8.0 (Baseline)', color=colors['v38'], alpha=0.8)
    ax1.bar(x + width/2, pnl_v310, width, label='v3.10.0 (Hybrid)', color=colors['v310'], alpha=0.8)
    ax1.set_xticks(x)
    ax1.set_xticklabels(months, rotation=45, ha='right', fontsize=8)
    ax1.set_ylabel('Monthly PnL ($)', fontsize=10)
    ax1.set_title('Monthly PnL Comparison', fontsize=11, fontweight='bold')
    ax1.legend(loc='upper left')
    ax1.axhline(y=0, color='gray', linestyle='--')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # 2. Cumulative comparison (top right)
    ax2 = fig.add_subplot(gs[0, 1])
    cum_v38 = np.cumsum(pnl_v38)
    cum_v310 = np.cumsum(pnl_v310)
    ax2.plot(months, cum_v38, 'o-', color=colors['v38'], linewidth=2, markersize=6, label='v3.8.0')
    ax2.plot(months, cum_v310, 's-', color=colors['v310'], linewidth=2, markersize=6, label='v3.10.0')
    ax2.fill_between(months, cum_v38, cum_v310, alpha=0.2, 
                     color=colors['v38'] if cum_v38[-1] > cum_v310[-1] else colors['v310'])
    ax2.set_ylabel('Cumulative PnL ($)', fontsize=10)
    ax2.set_title('Cumulative PnL Over Time', fontsize=11, fontweight='bold')
    ax2.legend(loc='upper left')
    ax2.set_xticklabels(months, rotation=45, ha='right', fontsize=8)
    ax2.grid(True, alpha=0.3)
    
    # 3. Win/Loss per month (middle left)
    ax3 = fig.add_subplot(gs[1, 0])
    diff = [v38 - v310 for v38, v310 in zip(pnl_v38, pnl_v310)]
    bar_colors = [colors['win'] if d > 0 else colors['lose'] for d in diff]
    ax3.bar(months, diff, color=bar_colors, alpha=0.7, edgecolor='black')
    ax3.axhline(y=0, color='gray', linestyle='--')
    ax3.set_ylabel('v3.8.0 Advantage ($)', fontsize=10)
    ax3.set_title('v3.8.0 vs v3.10.0 Difference per Month', fontsize=11, fontweight='bold')
    ax3.set_xticklabels(months, rotation=45, ha='right', fontsize=8)
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Count wins
    v38_wins = sum(1 for d in diff if d > 0)
    v310_wins = len(diff) - v38_wins
    ax3.text(0.02, 0.98, f"v3.8.0 wins: {v38_wins} months", transform=ax3.transAxes, 
             fontsize=9, va='top', color=colors['v38'], fontweight='bold')
    ax3.text(0.02, 0.90, f"v3.10.0 wins: {v310_wins} months", transform=ax3.transAxes,
             fontsize=9, va='top', color=colors['v310'], fontweight='bold')
    
    # 4. Total comparison pie (middle right)
    ax4 = fig.add_subplot(gs[1, 1])
    total_v38 = sum(pnl_v38)
    total_v310 = sum(pnl_v310)
    
    # Bar comparison
    strategies = ['v3.8.0\n(Baseline)', 'v3.10.0\n(Hybrid)']
    totals = [total_v38, total_v310]
    bar_colors = [colors['v38'], colors['v310']]
    bars = ax4.bar(strategies, totals, color=bar_colors, alpha=0.8, edgecolor='black', width=0.5)
    
    for bar, total in zip(bars, totals):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                f'${total:.0f}', ha='center', fontsize=12, fontweight='bold')
    
    ax4.axhline(y=0, color='gray', linestyle='--')
    ax4.set_ylabel('Total 12-Month PnL ($)', fontsize=10)
    ax4.set_title('Total PnL Comparison', fontsize=11, fontweight='bold')
    
    # 5. Summary table (bottom)
    ax5 = fig.add_subplot(gs[2, :])
    ax5.axis('off')
    
    # Calculate stats
    fills_v38 = sum(r['fills'] for r in results_v38)
    fills_v310 = sum(r['fills'] for r in results_v310)
    avg_low_vol = np.mean([r['low_vol_pct'] for r in results_v38])
    
    table_data = [
        ['Metric', 'v3.8.0 (Baseline)', 'v3.10.0 (Hybrid)', 'Winner'],
        ['Total PnL', f'${total_v38:.0f}', f'${total_v310:.0f}', 
         'v3.8.0 ✓' if total_v38 > total_v310 else 'v3.10.0 ✓'],
        ['Monthly Wins', f'{v38_wins}/12', f'{v310_wins}/12',
         'v3.8.0 ✓' if v38_wins > v310_wins else 'v3.10.0 ✓'],
        ['Total Fills', f'{fills_v38:,.0f}', f'{fills_v310:,.0f}',
         'v3.8.0 ✓' if fills_v38 > fills_v310 else 'v3.10.0 ✓'],
        ['Avg Low Vol %', f'{avg_low_vol:.1f}%', f'{avg_low_vol:.1f}%', '-'],
    ]
    
    table = ax5.table(
        cellText=table_data[1:],
        colLabels=table_data[0],
        loc='center',
        cellLoc='center',
        colColours=['#E3F2FD', colors['v38'], colors['v310'], '#E8F5E9']
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.3, 2.0)
    
    # Highlight winner row
    winner_str = 'v3.8.0' if total_v38 > total_v310 else 'v3.10.0'
    
    plt.suptitle(f'v3.8.0 vs v3.10.0 - 12 Month Comparison | Winner: {winner_str}', 
                 fontsize=14, fontweight='bold', y=0.98)
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"📊 Chart saved to: {output_path}")

def main():
    results_v38, results_v310 = run_comparison(months_back=12)
    
    if not results_v38:
        print("❌ No results")
        return
    
    # Summary
    total_v38 = sum(r['total_pnl'] for r in results_v38)
    total_v310 = sum(r['total_pnl'] for r in results_v310)
    
    print("\n" + "=" * 70)
    print("  📊 FINAL COMPARISON RESULTS")
    print("=" * 70)
    
    print(f"\n  v3.8.0 (Baseline):  ${total_v38:>10,.2f}")
    print(f"  v3.10.0 (Hybrid):   ${total_v310:>10,.2f}")
    print(f"  Difference:         ${total_v38 - total_v310:>10,.2f}")
    
    winner = "v3.8.0" if total_v38 > total_v310 else "v3.10.0"
    print(f"\n  🏆 WINNER: {winner}")
    
    if winner == "v3.8.0":
        print("\n  ✅ RECOMMENDATION: Use v3.8.0 baseline for production")
        print("     - Set volatility_adaptation.enabled: false")
    else:
        print("\n  ✅ RECOMMENDATION: Use v3.10.0 hybrid for production")
        print("     - Set volatility_adaptation.enabled: true")
        print("     - Set volatility_adaptation.mode: hybrid")
    
    print("=" * 70)
    
    # Generate chart
    output_path = r"C:\Users\camel\.gemini\antigravity\brain\ed4efe75-6a4b-4be3-971f-923ed59be9c8\v38_vs_v310_comparison.png"
    plot_comparison(results_v38, results_v310, output_path)
    
    return results_v38, results_v310

if __name__ == "__main__":
    results_v38, results_v310 = main()
