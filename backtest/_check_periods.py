"""Quick check: data coverage for Oct 2025 and Jan 2026."""
import pandas as pd

df = pd.read_csv("data/btcusdt_1m_1year.csv")
df["dt"] = pd.to_datetime(df["timestamp"], unit="ms")

print(f"Full data: {df['dt'].min()} → {df['dt'].max()}  ({len(df):,} rows)")
print()

# October 2025
oct = df[(df["dt"] >= "2025-10-01") & (df["dt"] < "2025-11-01")]
if len(oct) > 0:
    print(f"=== Oct 2025 (급변동장) ===")
    print(f"  Rows: {len(oct):,}  ({len(oct)/1440:.1f} days)")
    print(f"  Period: {oct['dt'].min()} → {oct['dt'].max()}")
    print(f"  Price: ${oct['low'].min():,.0f} → ${oct['high'].max():,.0f}")
    pct_move = (oct['close'].iloc[-1] - oct['close'].iloc[0]) / oct['close'].iloc[0] * 100
    daily_vol = oct['close'].pct_change().std() * (1440**0.5) * 100
    print(f"  Net move: {pct_move:+.1f}%,  Daily vol: {daily_vol:.1f}%")
else:
    print("Oct 2025: NO DATA")

print()

# January 2026
jan = df[(df["dt"] >= "2026-01-01") & (df["dt"] < "2026-02-01")]
if len(jan) > 0:
    print(f"=== Jan 2026 (횡보장) ===")
    print(f"  Rows: {len(jan):,}  ({len(jan)/1440:.1f} days)")
    print(f"  Period: {jan['dt'].min()} → {jan['dt'].max()}")
    print(f"  Price: ${jan['low'].min():,.0f} → ${jan['high'].max():,.0f}")
    pct_move = (jan['close'].iloc[-1] - jan['close'].iloc[0]) / jan['close'].iloc[0] * 100
    daily_vol = jan['close'].pct_change().std() * (1440**0.5) * 100
    print(f"  Net move: {pct_move:+.1f}%,  Daily vol: {daily_vol:.1f}%")
else:
    print("Jan 2026: NO DATA")
