"""
Chunked 1-Minute Backtest for Market Making
- Processes data month by month to avoid memory issues
- Full 1-year coverage with minute-level accuracy
- Aggregates results for comprehensive analysis
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
    """Fetch 1 month of 1-minute data (~43,200 candles)"""
    # Calculate start/end for the month
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
            current_start = data[-1][0] + 60000  # +1 minute
            
        except Exception as e:
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

def classify_regime_vectorized(df: pd.DataFrame, window: int = 60) -> pd.DataFrame:
    """Classify market regime using vectorized operations"""
    df = df.copy()
    df['returns'] = df['close'].pct_change()
    df['volatility'] = df['returns'].rolling(window, min_periods=1).std()
    df['trend'] = df['close'].pct_change(window)
    
    trend_thresh = 0.01  # 1% movement in 1 hour
    df['regime'] = np.where(df['trend'] > trend_thresh, 'UP',
                   np.where(df['trend'] < -trend_thresh, 'DOWN', 'SIDEWAYS'))
    return df

def simulate_mm_minute(df: pd.DataFrame, spread_pct: float = 0.001,
                       order_usd: float = 100) -> dict:
    """
    Vectorized minute-level market making simulation
    Returns summary stats instead of full dataframe to save memory
    """
    n = len(df)
    if n < 100:
        return None
    
    # Calculate signals
    mid = df['close'].values
    high = df['high'].values
    low = df['low'].values
    
    prev_mid = np.roll(mid, 1)
    prev_mid[0] = mid[0]
    
    bid_level = prev_mid * (1 - spread_pct/2)
    ask_level = prev_mid * (1 + spread_pct/2)
    
    # Fill detection
    bid_fill = (low <= bid_level).astype(float)
    ask_fill = (high >= ask_level).astype(float)
    
    # PnL calculation
    spread_capture = spread_pct * order_usd
    
    # Simple model: spread captured when filled, minus adverse selection
    pnl_per_fill = spread_capture * 0.5  # Half spread per side
    
    total_bid_fills = bid_fill.sum()
    total_ask_fills = ask_fill.sum()
    
    # Inventory risk from price movement
    returns = np.diff(mid) / mid[:-1]
    returns = np.append(returns, 0)
    
    # Net position changes
    net_position = np.cumsum(bid_fill - ask_fill) * (order_usd / mid)
    inventory_pnl = net_position[:-1] * np.diff(mid)
    inventory_pnl = np.append(inventory_pnl, 0)
    
    # Total PnL
    spread_pnl = (bid_fill + ask_fill) * pnl_per_fill
    total_pnl = spread_pnl.sum() + inventory_pnl.sum()
    
    # Stats by regime
    regime_stats = {}
    for regime in ['UP', 'DOWN', 'SIDEWAYS']:
        mask = df['regime'].values == regime
        if mask.sum() > 0:
            regime_spread_pnl = spread_pnl[mask].sum()
            regime_inv_pnl = inventory_pnl[mask].sum()
            regime_stats[regime] = {
                'minutes': mask.sum(),
                'spread_pnl': regime_spread_pnl,
                'inventory_pnl': regime_inv_pnl,
                'total_pnl': regime_spread_pnl + regime_inv_pnl,
                'fills': (bid_fill[mask] + ask_fill[mask]).sum()
            }
    
    return {
        'total_minutes': n,
        'total_pnl': total_pnl,
        'spread_pnl': spread_pnl.sum(),
        'inventory_pnl': inventory_pnl.sum(),
        'total_fills': total_bid_fills + total_ask_fills,
        'regime_stats': regime_stats,
        'hourly_pnl': total_pnl / (n / 60)
    }

def run_chunked_backtest(months_back: int = 12):
    """Run backtest month by month to avoid memory issues"""
    print("=" * 70)
    print("  📊 CHUNKED 1-MINUTE BACKTEST (Memory-Safe)")
    print("  Processing month by month...")
    print("=" * 70)
    
    results = []
    all_regime_stats = {'UP': [], 'DOWN': [], 'SIDEWAYS': []}
    
    # Calculate months to process
    now = datetime.now()
    months = []
    for i in range(months_back, 0, -1):
        target = now - timedelta(days=30*i)
        months.append((target.year, target.month))
    
    for idx, (year, month) in enumerate(months):
        month_name = f"{year}-{month:02d}"
        print(f"\n[{idx+1}/{len(months)}] Processing {month_name}...")
        
        # Fetch data
        df = fetch_month_1m_data("BTCUSDT", year, month)
        if df.empty:
            print(f"  ⚠️ No data for {month_name}")
            continue
        
        print(f"  📥 Fetched {len(df):,} candles")
        
        # Classify regimes
        df = classify_regime_vectorized(df)
        
        # Run simulation
        stats = simulate_mm_minute(df, spread_pct=0.001, order_usd=100)
        if stats is None:
            continue
        
        stats['month'] = month_name
        stats['year'] = year
        stats['month_num'] = month
        
        # Price info
        stats['price_start'] = df['close'].iloc[0]
        stats['price_end'] = df['close'].iloc[-1]
        stats['price_change_pct'] = (stats['price_end'] / stats['price_start'] - 1) * 100
        
        results.append(stats)
        
        # Aggregate regime stats
        for regime in ['UP', 'DOWN', 'SIDEWAYS']:
            if regime in stats['regime_stats']:
                all_regime_stats[regime].append(stats['regime_stats'][regime])
        
        print(f"  ✅ PnL: ${stats['total_pnl']:.2f} | Fills: {stats['total_fills']:.0f}")
        
        # Memory cleanup
        del df
        gc.collect()
    
    return results, all_regime_stats

def plot_monthly_results(results: list, regime_totals: dict, output_path: str):
    """Generate comprehensive monthly visualization"""
    if not results:
        print("❌ No results to plot")
        return
    
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.2, 1, 1], hspace=0.35, wspace=0.25)
    
    colors = {
        'UP': '#00C853',
        'DOWN': '#FF1744',
        'SIDEWAYS': '#FFC107',
        'profit': '#00C853',
        'loss': '#FF1744'
    }
    
    months = [r['month'] for r in results]
    pnls = [r['total_pnl'] for r in results]
    
    # 1. Monthly PnL bars (top left)
    ax1 = fig.add_subplot(gs[0, 0])
    bar_colors = [colors['profit'] if p >= 0 else colors['loss'] for p in pnls]
    bars = ax1.bar(range(len(months)), pnls, color=bar_colors, alpha=0.7, edgecolor='black')
    ax1.axhline(y=0, color='gray', linestyle='--')
    ax1.set_xticks(range(len(months)))
    ax1.set_xticklabels(months, rotation=45, ha='right', fontsize=8)
    ax1.set_ylabel('Monthly PnL ($)', fontsize=10)
    ax1.set_title('Monthly PnL (1-Minute Simulation)', fontsize=11, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # 2. Cumulative PnL (top right)
    ax2 = fig.add_subplot(gs[0, 1])
    cum_pnl = np.cumsum(pnls)
    ax2.fill_between(range(len(months)), cum_pnl, 0, 
                     where=(cum_pnl >= 0), color=colors['profit'], alpha=0.4)
    ax2.fill_between(range(len(months)), cum_pnl, 0,
                     where=(cum_pnl < 0), color=colors['loss'], alpha=0.4)
    ax2.plot(range(len(months)), cum_pnl, 'k-', linewidth=2)
    ax2.axhline(y=0, color='gray', linestyle='--')
    ax2.set_xticks(range(len(months)))
    ax2.set_xticklabels(months, rotation=45, ha='right', fontsize=8)
    ax2.set_ylabel('Cumulative PnL ($)', fontsize=10)
    ax2.set_title('Cumulative PnL Over 12 Months', fontsize=11, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # 3. PnL by Regime (middle left)
    ax3 = fig.add_subplot(gs[1, 0])
    regimes = ['SIDEWAYS', 'UP', 'DOWN']
    regime_pnls = [regime_totals[r] for r in regimes]
    regime_colors = [colors[r] for r in regimes]
    bars = ax3.bar(regimes, regime_pnls, color=regime_colors, alpha=0.7, edgecolor='black')
    ax3.axhline(y=0, color='gray', linestyle='--')
    ax3.set_ylabel('Total PnL ($)', fontsize=10)
    ax3.set_title('PnL by Market Regime (1-Year Total)', fontsize=11, fontweight='bold')
    for bar, pnl in zip(bars, regime_pnls):
        ypos = bar.get_height() + (20 if pnl >= 0 else -30)
        ax3.text(bar.get_x() + bar.get_width()/2, ypos,
                f'${pnl:.0f}', ha='center', fontsize=10, fontweight='bold')
    
    # 4. Win Rate / Stats (middle right)
    ax4 = fig.add_subplot(gs[1, 1])
    profitable_months = sum(1 for p in pnls if p > 0)
    losing_months = len(pnls) - profitable_months
    ax4.pie([profitable_months, losing_months], 
            labels=[f'Profitable\n({profitable_months} months)', f'Losing\n({losing_months} months)'],
            colors=[colors['profit'], colors['loss']],
            autopct='%1.1f%%', startangle=90, explode=[0.02, 0.02])
    ax4.set_title('Monthly Win Rate', fontsize=11, fontweight='bold')
    
    # 5. Summary Table (bottom)
    ax5 = fig.add_subplot(gs[2, :])
    ax5.axis('off')
    
    # Prepare table data
    table_data = []
    for r in results:
        status = '✓' if r['total_pnl'] >= 0 else '✗'
        table_data.append([
            r['month'],
            f"${r['total_pnl']:.0f}",
            f"{r['total_fills']:.0f}",
            f"${r['hourly_pnl']:.2f}",
            f"{r['price_change_pct']:+.1f}%",
            status
        ])
    
    # Add total row
    total_pnl = sum(pnls)
    total_fills = sum(r['total_fills'] for r in results)
    avg_hourly = np.mean([r['hourly_pnl'] for r in results])
    table_data.append([
        'TOTAL',
        f"${total_pnl:.0f}",
        f"{total_fills:.0f}",
        f"${avg_hourly:.2f}",
        '-',
        '✓' if total_pnl >= 0 else '✗'
    ])
    
    table = ax5.table(
        cellText=table_data,
        colLabels=['Month', 'PnL', 'Fills', 'Hourly Avg', 'BTC Move', 'Status'],
        loc='center',
        cellLoc='center',
        colColours=['#E3F2FD']*6
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)
    
    # Highlight total row
    for j in range(6):
        table[(len(table_data), j)].set_facecolor('#BBDEFB')
        table[(len(table_data), j)].set_text_props(fontweight='bold')
    
    plt.suptitle('1-Year Market Making Backtest (1-Minute Data)', 
                 fontsize=14, fontweight='bold', y=0.98)
    
    # Key insight
    sideways_pnl = regime_totals.get('SIDEWAYS', 0)
    insight = f"Sideways Market PnL: ${sideways_pnl:.0f} | Total: ${total_pnl:.0f}"
    fig.text(0.5, 0.01, insight, ha='center', fontsize=11,
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"📊 Chart saved to: {output_path}")

def main():
    # Run chunked backtest for 12 months
    results, all_regime_stats = run_chunked_backtest(months_back=12)
    
    if not results:
        print("❌ No results obtained")
        return
    
    # Aggregate regime totals
    regime_totals = {}
    for regime in ['UP', 'DOWN', 'SIDEWAYS']:
        if all_regime_stats[regime]:
            regime_totals[regime] = sum(s['total_pnl'] for s in all_regime_stats[regime])
        else:
            regime_totals[regime] = 0
    
    # Print summary
    print("\n" + "=" * 70)
    print("  📊 FINAL RESULTS SUMMARY")
    print("=" * 70)
    
    total_pnl = sum(r['total_pnl'] for r in results)
    total_fills = sum(r['total_fills'] for r in results)
    profitable = sum(1 for r in results if r['total_pnl'] > 0)
    
    print(f"\n  Total PnL: ${total_pnl:.2f}")
    print(f"  Total Fills: {total_fills:,.0f}")
    print(f"  Profitable Months: {profitable}/{len(results)}")
    
    print("\n  By Market Regime:")
    for regime, pnl in regime_totals.items():
        status = '✅' if pnl >= 0 else '⚠️'
        print(f"    {regime}: ${pnl:.2f} {status}")
    
    print("=" * 70)
    
    # Generate chart
    output_path = r"C:\Users\camel\.gemini\antigravity\brain\ed4efe75-6a4b-4be3-971f-923ed59be9c8\minute_backtest_12m.png"
    plot_monthly_results(results, regime_totals, output_path)
    
    print("\n🎉 Backtest completed successfully!")
    
    return results, regime_totals

if __name__ == "__main__":
    results, totals = main()
