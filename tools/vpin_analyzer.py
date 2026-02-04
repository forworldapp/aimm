"""
VPIN Distribution Analyzer
Analyze historical VPIN values to determine optimal thresholds
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import pandas as pd
import numpy as np
from collections import deque
from typing import List, Dict
import json


class VPINAnalyzer:
    """Analyze VPIN distribution from historical trade data"""
    
    def __init__(self, bucket_size_usd: float = 10000, n_buckets: int = 50):
        self.bucket_size_usd = bucket_size_usd
        self.n_buckets = n_buckets
        self.vpin_values: List[float] = []
        
    def calculate_vpin_series(self, trades: pd.DataFrame) -> List[float]:
        """
        Calculate VPIN values from trade data
        
        Expected columns: price, size, side (buy/sell)
        """
        buckets = deque(maxlen=self.n_buckets)
        current_bucket_volume = 0.0
        current_buy_volume = 0.0
        vpin_series = []
        
        for _, trade in trades.iterrows():
            trade_volume_usd = trade['price'] * trade['size']
            current_bucket_volume += trade_volume_usd
            
            if trade.get('side', 'buy') == 'buy':
                current_buy_volume += trade_volume_usd
            
            # Bucket complete
            if current_bucket_volume >= self.bucket_size_usd:
                sell_volume = current_bucket_volume - current_buy_volume
                imbalance = abs(current_buy_volume - sell_volume) / current_bucket_volume
                buckets.append(imbalance)
                
                # Calculate VPIN
                if len(buckets) > 0:
                    vpin = sum(buckets) / len(buckets)
                    vpin_series.append(vpin)
                
                # Reset
                current_bucket_volume = 0.0
                current_buy_volume = 0.0
        
        self.vpin_values = vpin_series
        return vpin_series
    
    def analyze_distribution(self) -> Dict:
        """Analyze VPIN distribution and recommend thresholds"""
        if not self.vpin_values:
            return {"error": "No VPIN values calculated"}
        
        arr = np.array(self.vpin_values)
        
        percentiles = {
            'p50': np.percentile(arr, 50),
            'p75': np.percentile(arr, 75),
            'p90': np.percentile(arr, 90),
            'p95': np.percentile(arr, 95),
            'p99': np.percentile(arr, 99),
        }
        
        stats = {
            'count': len(arr),
            'mean': float(np.mean(arr)),
            'std': float(np.std(arr)),
            'min': float(np.min(arr)),
            'max': float(np.max(arr)),
        }
        
        # Threshold recommendations
        recommendations = {
            'conservative': round(percentiles['p90'], 3),  # 상위 10%만 필터
            'moderate': round(percentiles['p75'], 3),      # 상위 25% 필터
            'aggressive': round(percentiles['p50'], 3),    # 상위 50% 필터
        }
        
        # Count alerts at different thresholds
        threshold_analysis = {}
        for thresh in [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]:
            alerts = sum(1 for v in arr if v > thresh)
            threshold_analysis[str(thresh)] = {
                'alerts': alerts,
                'alert_rate': round(alerts / len(arr) * 100, 2),
            }
        
        return {
            'stats': stats,
            'percentiles': percentiles,
            'recommendations': recommendations,
            'threshold_analysis': threshold_analysis,
        }
    
    def print_report(self, analysis: Dict):
        """Print formatted analysis report"""
        print("=" * 60)
        print("VPIN Distribution Analysis Report")
        print("=" * 60)
        
        stats = analysis['stats']
        print(f"\n📊 Statistics (n={stats['count']:,})")
        print(f"   Mean: {stats['mean']:.4f}")
        print(f"   Std:  {stats['std']:.4f}")
        print(f"   Min:  {stats['min']:.4f}")
        print(f"   Max:  {stats['max']:.4f}")
        
        pct = analysis['percentiles']
        print(f"\n📈 Percentiles")
        print(f"   P50: {pct['p50']:.4f}")
        print(f"   P75: {pct['p75']:.4f}")
        print(f"   P90: {pct['p90']:.4f}")
        print(f"   P95: {pct['p95']:.4f}")
        print(f"   P99: {pct['p99']:.4f}")
        
        rec = analysis['recommendations']
        print(f"\n🎯 Recommended Thresholds")
        print(f"   Conservative (P90): {rec['conservative']}")
        print(f"   Moderate (P75):     {rec['moderate']}")
        print(f"   Aggressive (P50):   {rec['aggressive']}")
        
        thresh = analysis['threshold_analysis']
        print(f"\n⚡ Alert Rates by Threshold")
        for t, data in thresh.items():
            print(f"   {t}: {data['alerts']:,} alerts ({data['alert_rate']:.1f}%)")
        
        print("\n" + "=" * 60)


def generate_synthetic_trades(n_trades: int = 10000, base_price: float = 50000) -> pd.DataFrame:
    """Generate synthetic trade data for testing"""
    np.random.seed(42)
    
    prices = base_price + np.cumsum(np.random.randn(n_trades) * 10)
    sizes = np.abs(np.random.randn(n_trades) * 0.01) + 0.001
    sides = np.random.choice(['buy', 'sell'], n_trades, p=[0.52, 0.48])  # Slight buy bias
    
    return pd.DataFrame({
        'price': prices,
        'size': sizes,
        'side': sides
    })


def main():
    parser = argparse.ArgumentParser(description='VPIN Distribution Analyzer')
    parser.add_argument('--data', type=str, help='Path to trades CSV file')
    parser.add_argument('--threshold', type=float, default=0.5, help='Test threshold')
    parser.add_argument('--bucket-size', type=float, default=10000, help='Bucket size USD')
    parser.add_argument('--n-buckets', type=int, default=50, help='Number of buckets')
    parser.add_argument('--synthetic', action='store_true', help='Use synthetic data')
    parser.add_argument('--output', type=str, help='Output JSON file')
    
    args = parser.parse_args()
    
    # Load data
    if args.synthetic or not args.data:
        print("📦 Using synthetic trade data (10,000 trades)")
        trades = generate_synthetic_trades()
    else:
        print(f"📂 Loading trades from {args.data}")
        trades = pd.read_csv(args.data)
    
    # Analyze
    analyzer = VPINAnalyzer(
        bucket_size_usd=args.bucket_size,
        n_buckets=args.n_buckets
    )
    
    print(f"🔄 Calculating VPIN series...")
    analyzer.calculate_vpin_series(trades)
    
    print(f"📊 Analyzing distribution...")
    analysis = analyzer.analyze_distribution()
    
    # Print report
    analyzer.print_report(analysis)
    
    # Save output
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(analysis, f, indent=2)
        print(f"\n💾 Saved to {args.output}")
    
    # Test specific threshold
    if args.threshold:
        thresh_data = analysis['threshold_analysis'].get(str(args.threshold), {})
        print(f"\n🎯 At threshold={args.threshold}:")
        print(f"   Alerts: {thresh_data.get('alerts', 'N/A')}")
        print(f"   Rate: {thresh_data.get('alert_rate', 'N/A')}%")


if __name__ == "__main__":
    main()
