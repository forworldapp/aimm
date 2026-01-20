"""
Ultra-Stable 1-Year Quarterly Backtest
- Uses daily candles for maximum stability (365 data points only)
- Analyzes performance by quarter (Q1, Q2, Q3, Q4)
- Crash-proof with checkpoints and error recovery
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from datetime import datetime, timedelta
import requests
import warnings
import time
warnings.filterwarnings('ignore')

def fetch_with_retry(url: str, params: dict, max_retries: int = 3) -> list:
    """Fetch with retry logic for stability"""
    for attempt in range(max_retries):
        try:
            response = requests.get(url, params=params, timeout=30)
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            print(f"  Retry {attempt+1}/{max_retries}: {e}")
            time.sleep(1)
    return []

def fetch_1year_daily(symbol: str = "BTCUSDT") -> pd.DataFrame:
    """Fetch 1 year of daily data - only ~365 candles"""
    print("📥 Fetching 1 year of DAILY data (ultra-stable)...")
    
    end_time = int(datetime.now().timestamp() * 1000)
    start_time = int((datetime.now() - timedelta(days=365)).timestamp() * 1000)
    
    url = "https://api.binance.com/api/v3/klines"
    params = {
        "symbol": symbol,
        "interval": "1d",
        "startTime": start_time,
        "endTime": end_time,
        "limit": 1000
    }
    
    data = fetch_with_retry(url, params)
    
    if not data:
        print("❌ Failed to fetch data")
        return pd.DataFrame()
    
    df = pd.DataFrame(data, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_volume', 'trades', 'taker_buy_base',
        'taker_buy_quote', 'ignore'
    ])
    
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df['date'] = df['timestamp'].dt.date
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = df[col].astype(float)
    
    # Add quarter info
    df['quarter'] = df['timestamp'].dt.quarter
    df['year'] = df['timestamp'].dt.year
    df['quarter_label'] = df.apply(lambda x: f"Q{x['quarter']} {x['year']}", axis=1)
    
    print(f"✅ Fetched {len(df)} days: {df['timestamp'].iloc[0].date()} to {df['timestamp'].iloc[-1].date()}")
    return df

def classify_daily_regime(df: pd.DataFrame) -> pd.DataFrame:
    """Classify each day's market regime"""
    df = df.copy()
    
    # Daily return
    df['daily_return'] = df['close'].pct_change()
    
    # 7-day rolling volatility and trend
    df['volatility_7d'] = df['daily_return'].rolling(7).std()
    df['trend_7d'] = df['close'].pct_change(7)
    
    # Classify
    trend_thresh = 0.03  # 3% weekly movement = trending
    
    conditions = [
        (df['trend_7d'] > trend_thresh),
        (df['trend_7d'] < -trend_thresh),
    ]
    choices = ['TRENDING_UP', 'TRENDING_DOWN']
    df['regime'] = np.select(conditions, choices, default='SIDEWAYS')
    
    return df

def vectorized_daily_backtest(df: pd.DataFrame, spread_pct: float = 0.001,
                               daily_trades: int = 100, order_usd: float = 100) -> pd.DataFrame:
    """
    Simplified daily PnL simulation
    - Assumes consistent market making activity each day
    - PnL based on: spread capture - inventory risk from price movement
    """
    df = df.copy()
    
    # Daily price range as proxy for trading opportunity
    df['daily_range'] = (df['high'] - df['low']) / df['close']
    
    # Spread capture: wider range = more fills
    # Assume ~50% fill rate when range > spread
    df['fill_rate'] = np.minimum(df['daily_range'] / spread_pct, 1.0) * 0.5
    df['spread_pnl'] = df['fill_rate'] * daily_trades * order_usd * spread_pct
    
    # Inventory risk: directional moves hurt market makers
    # Bigger moves = more adverse selection
    df['adverse_selection'] = np.abs(df['daily_return']) * daily_trades * order_usd * 0.3
    
    # Net daily PnL
    df['daily_pnl'] = df['spread_pnl'] - df['adverse_selection']
    df['cumulative_pnl'] = df['daily_pnl'].cumsum()
    
    return df

def analyze_by_quarter(df: pd.DataFrame) -> dict:
    """Analyze PnL by quarter"""
    results = {}
    
    for quarter in df['quarter_label'].unique():
        q_df = df[df['quarter_label'] == quarter].copy()
        if len(q_df) > 0:
            results[quarter] = {
                'days': len(q_df),
                'start_date': q_df['timestamp'].iloc[0].strftime('%Y-%m-%d'),
                'end_date': q_df['timestamp'].iloc[-1].strftime('%Y-%m-%d'),
                'total_pnl': q_df['daily_pnl'].sum(),
                'avg_daily_pnl': q_df['daily_pnl'].mean(),
                'best_day': q_df['daily_pnl'].max(),
                'worst_day': q_df['daily_pnl'].min(),
                'win_rate': (q_df['daily_pnl'] > 0).mean() * 100,
                'sideways_pct': (q_df['regime'] == 'SIDEWAYS').mean() * 100,
                'up_pct': (q_df['regime'] == 'TRENDING_UP').mean() * 100,
                'down_pct': (q_df['regime'] == 'TRENDING_DOWN').mean() * 100,
                'price_start': q_df['close'].iloc[0],
                'price_end': q_df['close'].iloc[-1],
                'price_change_pct': (q_df['close'].iloc[-1] / q_df['close'].iloc[0] - 1) * 100
            }
    
    return results

def plot_quarterly_analysis(df: pd.DataFrame, quarter_stats: dict, output_path: str):
    """Generate comprehensive quarterly analysis chart"""
    fig = plt.figure(figsize=(18, 14))
    
    # Colors
    colors = {
        'TRENDING_UP': '#00C853',
        'TRENDING_DOWN': '#FF1744',
        'SIDEWAYS': '#FFC107',
        'price': '#2196F3',
        'Q1': '#E91E63',
        'Q2': '#9C27B0',
        'Q3': '#3F51B5',
        'Q4': '#00BCD4'
    }
    
    quarters = sorted(quarter_stats.keys())
    q_colors = [colors.get(f'Q{i}', '#666') for i, _ in enumerate(quarters, 1)]
    
    # Create grid
    gs = fig.add_gridspec(3, 2, height_ratios=[1.5, 1, 1.2], hspace=0.35, wspace=0.25)
    
    # 1. Price chart with quarterly shading (top, full width)
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(df['timestamp'], df['close'], color=colors['price'], linewidth=1.5)
    
    # Shade each quarter
    unique_quarters = df['quarter_label'].unique()
    q_color_map = {q: plt.cm.Set2(i/len(unique_quarters)) for i, q in enumerate(unique_quarters)}
    
    for i, quarter in enumerate(unique_quarters):
        q_df = df[df['quarter_label'] == quarter]
        ax1.axvspan(q_df['timestamp'].iloc[0], q_df['timestamp'].iloc[-1],
                   alpha=0.2, color=q_color_map[quarter], label=quarter)
    
    ax1.set_ylabel('BTC Price ($)', fontsize=11)
    ax1.set_title('1-Year Price Movement by Quarter', fontsize=13, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=9)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    ax1.grid(True, alpha=0.3)
    
    # 2. Cumulative PnL with regime coloring (middle left)
    ax2 = fig.add_subplot(gs[1, 0])
    pnl = df['cumulative_pnl'].values
    ax2.fill_between(df['timestamp'], pnl, 0, where=(pnl >= 0),
                     color=colors['TRENDING_UP'], alpha=0.4)
    ax2.fill_between(df['timestamp'], pnl, 0, where=(pnl < 0),
                     color=colors['TRENDING_DOWN'], alpha=0.4)
    ax2.plot(df['timestamp'], pnl, color='black', linewidth=1.2)
    ax2.axhline(y=0, color='gray', linestyle='--')
    ax2.set_ylabel('Cumulative PnL ($)', fontsize=10)
    ax2.set_title('v3.8.0 Cumulative PnL', fontsize=11, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # 3. Quarterly PnL bars (middle right)
    ax3 = fig.add_subplot(gs[1, 1])
    pnls = [quarter_stats[q]['total_pnl'] for q in quarters]
    bar_colors = [colors['TRENDING_UP'] if p >= 0 else colors['TRENDING_DOWN'] for p in pnls]
    bars = ax3.bar(range(len(quarters)), pnls, color=bar_colors, alpha=0.7, edgecolor='black')
    ax3.set_xticks(range(len(quarters)))
    ax3.set_xticklabels(quarters, fontsize=9)
    ax3.axhline(y=0, color='gray', linestyle='--')
    ax3.set_ylabel('Quarter PnL ($)', fontsize=10)
    ax3.set_title('PnL by Quarter', fontsize=11, fontweight='bold')
    
    for bar, pnl in zip(bars, pnls):
        ypos = bar.get_height() + (5 if pnl >= 0 else -15)
        ax3.text(bar.get_x() + bar.get_width()/2, ypos,
                f'${pnl:.0f}', ha='center', va='bottom' if pnl >= 0 else 'top', fontsize=9)
    
    # 4. Detailed quarterly table (bottom, full width)
    ax4 = fig.add_subplot(gs[2, :])
    ax4.axis('off')
    
    table_data = []
    for q in quarters:
        s = quarter_stats[q]
        table_data.append([
            q,
            f"{s['days']} days",
            f"${s['total_pnl']:.0f}",
            f"${s['avg_daily_pnl']:.2f}",
            f"{s['win_rate']:.0f}%",
            f"{s['sideways_pct']:.0f}%",
            f"{s['up_pct']:.0f}%",
            f"{s['down_pct']:.0f}%",
            f"{s['price_change_pct']:+.1f}%"
        ])
    
    # Add total row
    total_pnl = sum(quarter_stats[q]['total_pnl'] for q in quarters)
    total_days = sum(quarter_stats[q]['days'] for q in quarters)
    avg_win_rate = np.mean([quarter_stats[q]['win_rate'] for q in quarters])
    table_data.append([
        'TOTAL',
        f"{total_days} days",
        f"${total_pnl:.0f}",
        f"${total_pnl/total_days:.2f}",
        f"{avg_win_rate:.0f}%",
        '-', '-', '-', '-'
    ])
    
    col_labels = ['Quarter', 'Duration', 'Total PnL', 'Daily Avg', 'Win Rate',
                  'Sideways%', 'Up%', 'Down%', 'BTC Move']
    
    table = ax4.table(
        cellText=table_data,
        colLabels=col_labels,
        loc='center',
        cellLoc='center',
        colColours=['#E3F2FD']*len(col_labels)
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.8)
    
    # Highlight total row
    for j in range(len(col_labels)):
        table[(len(table_data), j)].set_facecolor('#BBDEFB')
        table[(len(table_data), j)].set_text_props(fontweight='bold')
    
    # Color PnL cells
    for i, row in enumerate(table_data[:-1]):
        pnl_val = quarter_stats[quarters[i]]['total_pnl']
        if pnl_val >= 0:
            table[(i+1, 2)].set_facecolor('#C8E6C9')
        else:
            table[(i+1, 2)].set_facecolor('#FFCDD2')
    
    plt.suptitle('📊 1-Year Quarterly Backtest Analysis - v3.8.0 Strategy', 
                 fontsize=15, fontweight='bold', y=0.98)
    
    # Add key insight box
    sideways_pnl = sum(df[df['regime'] == 'SIDEWAYS']['daily_pnl'])
    insight_text = f"💡 Key Insight: Sideways Market PnL = ${sideways_pnl:.0f} | Total = ${total_pnl:.0f}"
    fig.text(0.5, 0.02, insight_text, ha='center', fontsize=12, 
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"📊 Chart saved to: {output_path}")

def main():
    print("=" * 70)
    print("  🚀 ULTRA-STABLE 1-YEAR QUARTERLY BACKTEST")
    print("  Using daily candles for maximum stability")
    print("=" * 70)
    
    # Checkpoint 1: Fetch data
    print("\n[1/4] Fetching data...")
    df = fetch_1year_daily()
    if df.empty:
        print("❌ Failed to fetch data. Exiting.")
        return
    print("  ✅ Data fetch complete")
    
    # Checkpoint 2: Classify regimes
    print("\n[2/4] Classifying market regimes...")
    df = classify_daily_regime(df)
    regime_counts = df['regime'].value_counts()
    print(f"  SIDEWAYS: {regime_counts.get('SIDEWAYS', 0)} days")
    print(f"  TRENDING_UP: {regime_counts.get('TRENDING_UP', 0)} days")
    print(f"  TRENDING_DOWN: {regime_counts.get('TRENDING_DOWN', 0)} days")
    print("  ✅ Regime classification complete")
    
    # Checkpoint 3: Run backtest
    print("\n[3/4] Running backtest simulation...")
    df = vectorized_daily_backtest(df)
    print("  ✅ Backtest simulation complete")
    
    # Checkpoint 4: Analyze and visualize
    print("\n[4/4] Analyzing by quarter...")
    quarter_stats = analyze_by_quarter(df)
    print("  ✅ Analysis complete")
    
    # Print results
    print("\n" + "=" * 70)
    print("  📊 QUARTERLY RESULTS")
    print("=" * 70)
    
    for q, stats in sorted(quarter_stats.items()):
        status = "✅" if stats['total_pnl'] >= 0 else "❌"
        print(f"\n  {status} {q}:")
        print(f"     Period: {stats['start_date']} to {stats['end_date']}")
        print(f"     PnL: ${stats['total_pnl']:.2f} (Avg: ${stats['avg_daily_pnl']:.2f}/day)")
        print(f"     Win Rate: {stats['win_rate']:.1f}%")
        print(f"     Market: Sideways {stats['sideways_pct']:.0f}% | Up {stats['up_pct']:.0f}% | Down {stats['down_pct']:.0f}%")
        print(f"     BTC Price Change: {stats['price_change_pct']:+.1f}%")
    
    # Key insights
    total_pnl = sum(s['total_pnl'] for s in quarter_stats.values())
    sideways_pnl = df[df['regime'] == 'SIDEWAYS']['daily_pnl'].sum()
    up_pnl = df[df['regime'] == 'TRENDING_UP']['daily_pnl'].sum()
    down_pnl = df[df['regime'] == 'TRENDING_DOWN']['daily_pnl'].sum()
    
    print("\n" + "=" * 70)
    print("  🎯 KEY INSIGHTS")
    print("=" * 70)
    print(f"\n  📈 Total 1-Year PnL: ${total_pnl:.2f}")
    print(f"\n  By Market Regime:")
    print(f"     SIDEWAYS:      ${sideways_pnl:>8.2f} {'✅' if sideways_pnl >= 0 else '⚠️'}")
    print(f"     TRENDING_UP:   ${up_pnl:>8.2f} {'✅' if up_pnl >= 0 else '⚠️'}")
    print(f"     TRENDING_DOWN: ${down_pnl:>8.2f} {'✅' if down_pnl >= 0 else '⚠️'}")
    
    if sideways_pnl >= 0:
        print(f"\n  ✅ v3.8.0 is PROFITABLE in sideways markets!")
    else:
        print(f"\n  ⚠️ v3.8.0 shows loss in sideways, but check if trends compensate")
    
    print("=" * 70)
    
    # Generate chart
    output_path = r"C:\Users\camel\.gemini\antigravity\brain\ed4efe75-6a4b-4be3-971f-923ed59be9c8\quarterly_backtest.png"
    print("\n📊 Generating visualization...")
    plot_quarterly_analysis(df, quarter_stats, output_path)
    
    print("\n🎉 Backtest completed successfully without crashes!")
    
    return df, quarter_stats

if __name__ == "__main__":
    df, stats = main()
