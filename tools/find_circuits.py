import pandas as pd

df = pd.read_csv('data/trade_history_BTC_USDT_Perp.csv')
df['realized_pnl'] = pd.to_numeric(df['realized_pnl'], errors='coerce').fillna(0)
df['dt'] = pd.to_datetime(df['timestamp'], unit='s')

# Find large loss trades (circuit breaker liquidations)
big_losses = df[df.realized_pnl < -20].sort_values('realized_pnl')
print("=== Large Loss Trades (> $20) ===")
for _, r in big_losses.iterrows():
    dt_str = r['dt'].strftime("%Y-%m-%d %H:%M")
    print(f"  {dt_str} | PnL: ${r['realized_pnl']:.2f} | {r['note']}")

print(f"\nTotal large loss events: {len(big_losses)}")

# Check for gaps in trading (= bot stopped = circuit breaker)
df_sorted = df.sort_values('timestamp')
df_sorted['gap_hours'] = df_sorted['timestamp'].diff() / 3600
gaps = df_sorted[df_sorted['gap_hours'] > 6]
print(f"\n=== Trading Gaps > 6 hours (bot stopped) ===")
for _, r in gaps.iterrows():
    dt_str = r['dt'].strftime("%Y-%m-%d %H:%M")
    print(f"  {dt_str} | Gap: {r['gap_hours']:.1f} hours")
print(f"\nTotal gaps: {len(gaps)}")
