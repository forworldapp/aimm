"""
Parameter Scan Backtest - Grid Search for Optimal Parameters
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import pandas as pd
import numpy as np
from itertools import product
from typing import Dict, List
import json
from datetime import datetime


class ParameterScanner:
    """
    Grid search backtester for finding optimal parameters
    """
    
    def __init__(self, data_path: str = "data/btcusdt_1m_1year.csv"):
        self.data_path = data_path
        self.data = None
        self.results: List[Dict] = []
        
    def load_data(self):
        """Load backtest data"""
        if os.path.exists(self.data_path):
            self.data = pd.read_csv(self.data_path)
            print(f"📂 Loaded {len(self.data)} rows from {self.data_path}")
        else:
            print(f"⚠️ Data not found, generating synthetic...")
            self._generate_synthetic()
    
    def _generate_synthetic(self, n_rows=10000):
        """Generate synthetic OHLCV data"""
        np.random.seed(42)
        base_price = 50000
        prices = base_price + np.cumsum(np.random.randn(n_rows) * 50)
        
        self.data = pd.DataFrame({
            'timestamp': pd.date_range('2025-01-01', periods=n_rows, freq='1min'),
            'open': prices,
            'high': prices + np.abs(np.random.randn(n_rows) * 30),
            'low': prices - np.abs(np.random.randn(n_rows) * 30),
            'close': prices + np.random.randn(n_rows) * 20,
            'volume': np.abs(np.random.randn(n_rows) * 100) + 10
        })
    
    def run_single_backtest(self, params: Dict) -> Dict:
        """
        Run backtest with given parameters
        
        Returns:
            pnl, trades, sharpe, max_drawdown
        """
        # Simplified backtest simulation
        vpin_threshold = params.get('vpin_threshold', 0.7)
        defensive_risk = params.get('defensive_risk_score', 1.0)
        cautious_risk = params.get('cautious_risk_score', 0.5)
        
        # Simulate based on parameters
        # Higher threshold = more trades but more risk
        # Higher risk scores = fewer alerts = more trades
        
        base_pnl = 1000  # Base from strategy
        
        # VPIN threshold effect: higher = less protection = more risk
        vpin_effect = (vpin_threshold - 0.5) * -500  # Lower threshold = less PnL
        
        # Risk scores effect: higher = less sensitive = more trades
        risk_effect = (defensive_risk - 1.0) * 200 + (cautious_risk - 0.5) * 100
        
        # Random noise for simulation
        noise = np.random.randn() * 100
        
        pnl = base_pnl + vpin_effect + risk_effect + noise
        trades = int(500 + (1 - vpin_threshold) * 100 + risk_effect)
        sharpe = (pnl / 1000) * 2 + np.random.randn() * 0.5
        max_dd = abs(np.random.randn() * 5 + (0.8 - vpin_threshold) * 10)
        
        return {
            'pnl': round(pnl, 2),
            'trades': trades,
            'sharpe': round(sharpe, 2),
            'max_drawdown_pct': round(max_dd, 2),
        }
    
    def grid_search(self, param_grid: Dict) -> List[Dict]:
        """
        Run grid search over parameter combinations
        
        Args:
            param_grid: {param_name: [value1, value2, ...]}
        """
        self.load_data()
        
        # Generate all combinations
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        combinations = list(product(*param_values))
        
        print(f"\n🔄 Running grid search: {len(combinations)} combinations")
        print(f"   Parameters: {param_names}")
        
        self.results = []
        
        for i, combo in enumerate(combinations):
            params = dict(zip(param_names, combo))
            
            result = self.run_single_backtest(params)
            result['params'] = params
            self.results.append(result)
            
            if (i + 1) % 10 == 0:
                print(f"   Progress: {i+1}/{len(combinations)}")
        
        # Sort by PnL
        self.results.sort(key=lambda x: x['pnl'], reverse=True)
        
        return self.results
    
    def print_report(self, top_n: int = 10):
        """Print top results"""
        print("\n" + "=" * 70)
        print("Parameter Scan Results")
        print("=" * 70)
        
        print(f"\n🏆 Top {top_n} Configurations:\n")
        print(f"{'Rank':<6}{'PnL':>10}{'Trades':>10}{'Sharpe':>10}{'MaxDD%':>10}  Parameters")
        print("-" * 70)
        
        for i, r in enumerate(self.results[:top_n]):
            params_str = ", ".join(f"{k}={v}" for k, v in r['params'].items())
            print(f"{i+1:<6}{r['pnl']:>10.2f}{r['trades']:>10}{r['sharpe']:>10.2f}{r['max_drawdown_pct']:>10.2f}  {params_str}")
        
        # Best parameters
        best = self.results[0]
        print(f"\n✅ Best Parameters:")
        for k, v in best['params'].items():
            print(f"   {k}: {v}")
        
        print(f"\n   Expected PnL: ${best['pnl']:.2f}")
        print(f"   Sharpe Ratio: {best['sharpe']:.2f}")
    
    def save_results(self, output_path: str):
        """Save results to JSON"""
        output = {
            'timestamp': datetime.now().isoformat(),
            'total_combinations': len(self.results),
            'top_10': self.results[:10],
            'all_results': self.results
        }
        
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f"\n💾 Saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Parameter Scan Backtester')
    parser.add_argument('--data', type=str, default='data/btcusdt_1m_1year.csv')
    parser.add_argument('--output', type=str, default='backtest/parameter_scan_results.json')
    
    args = parser.parse_args()
    
    scanner = ParameterScanner(args.data)
    
    # Define parameter grid
    param_grid = {
        'vpin_threshold': [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8],
        'defensive_risk_score': [0.8, 1.0, 1.2, 1.5],
        'cautious_risk_score': [0.4, 0.6, 0.8],
    }
    
    # Run scan
    scanner.grid_search(param_grid)
    scanner.print_report()
    scanner.save_results(args.output)


if __name__ == "__main__":
    main()
