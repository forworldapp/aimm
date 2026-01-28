"""
Backtest: Volatility Model and Combined (Direction + Volatility)
v4.2 - Compare HMM-only vs +Volatility vs +Direction+Volatility
"""

import sys
from pathlib import Path
import pickle
import json
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from ml.feature_engineering import get_feature_engineer


@dataclass
class BacktestConfig:
    """Backtest configuration."""
    initial_capital: float = 10000.0
    order_size_usd: float = 200.0
    maker_fee_bps: float = 2.0
    
    # Grid MM settings
    base_grid_layers: int = 5
    base_grid_spacing_pct: float = 0.15
    
    # Risk
    max_position_usd: float = 2000.0
    
    # Volatility thresholds (percentiles)
    vol_high_pct: float = 70  # Above this = high vol
    vol_low_pct: float = 30   # Below this = low vol
    
    # Direction threshold
    direction_threshold: float = 0.52  # Use direction signal if prob > this
    
    chunk_size: int = 50000


@dataclass
class BacktestResult:
    """Backtest results."""
    model_name: str
    total_pnl: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    equity_curve: List[float] = field(default_factory=list)


class CombinedStrategyBacktest:
    """
    Backtests comparing:
    1. HMM-only (baseline)
    2. HMM + Volatility model
    3. HMM + Volatility + Direction (combined)
    """
    
    def __init__(self, config: BacktestConfig = None):
        self.config = config or BacktestConfig()
        self.fe = get_feature_engineer('binance')
        
        # Models
        self.vol_model = None
        self.dir_model = None
        self.vol_thresholds = {}
        
        # State
        self.position = 0.0
        self.cash = self.config.initial_capital
        self.equity_history = []
    
    def load_models(
        self,
        vol_model_path: str = 'data/volatility_model_range_15m.pkl',
        dir_model_path: str = 'data/direction_model_binary.pkl'
    ):
        """Load volatility and direction models."""
        # Volatility model
        try:
            with open(vol_model_path, 'rb') as f:
                self.vol_model = pickle.load(f)
            print(f"Loaded volatility model: {vol_model_path}")
        except Exception as e:
            print(f"Could not load volatility model: {e}")
        
        # Direction model
        try:
            with open(dir_model_path, 'rb') as f:
                self.dir_model = pickle.load(f)
            print(f"Loaded direction model: {dir_model_path}")
        except Exception as e:
            print(f"Could not load direction model: {e}")
    
    def reset(self):
        """Reset state."""
        self.position = 0.0
        self.cash = self.config.initial_capital
        self.equity_history = []
    
    def predict_volatility(self, features: pd.Series) -> Optional[float]:
        """Predict volatility."""
        if self.vol_model is None:
            return None
        try:
            model = self.vol_model['model']
            feat_arr = features[self.vol_model['feature_names']].values.reshape(1, -1)
            return model.predict(feat_arr)[0]
        except:
            return None
    
    def predict_direction(self, features: pd.Series) -> Tuple[Optional[str], float]:
        """Predict direction (UP/DOWN) and probability."""
        if self.dir_model is None:
            return None, 0.5
        try:
            model = self.dir_model['model']
            feat_arr = features[self.dir_model['feature_names']].values.reshape(1, -1)
            prob = model.predict(feat_arr)[0]
            direction = 'UP' if prob > 0.5 else 'DOWN'
            confidence = max(prob, 1 - prob)
            return direction, confidence
        except:
            return None, 0.5
    
    def simulate_step(
        self,
        row: pd.Series,
        features: pd.Series,
        use_vol: bool = False,
        use_dir: bool = False,
        vol_pred: Optional[float] = None,
        dir_pred: Optional[str] = None,
        dir_conf: float = 0.5
    ) -> float:
        """
        Simulate one step of grid MM.
        
        Returns: PnL for this step
        """
        price = row['close']
        high = row['high']
        low = row['low']
        
        # Base parameters
        layers = self.config.base_grid_layers
        spacing = self.config.base_grid_spacing_pct / 100
        base_size = self.config.order_size_usd / price
        
        # Volatility adjustments
        spread_mult = 1.0
        size_mult = 1.0
        
        if use_vol and vol_pred is not None:
            # AGGRESSIVE: High vol -> tighter spread, larger size
            if vol_pred > self.vol_thresholds.get('high', 0.2):
                spread_mult = 0.8   # Tighter spread (aggressive)
                size_mult = 1.3     # Larger size
            elif vol_pred < self.vol_thresholds.get('low', 0.1):
                spread_mult = 1.3   # Wider spread (conservative)
                size_mult = 0.8     # Smaller size
        
        # Direction adjustments
        bid_layers = layers
        ask_layers = layers
        bid_size_mult = 1.0
        ask_size_mult = 1.0
        
        if use_dir and dir_pred and dir_conf >= self.config.direction_threshold:
            if dir_pred == 'UP':
                bid_layers = min(7, layers + 1)
                ask_layers = max(3, layers - 1)
                bid_size_mult = 1.15
                ask_size_mult = 0.85
            elif dir_pred == 'DOWN':
                ask_layers = min(7, layers + 1)
                bid_layers = max(3, layers - 1)
                ask_size_mult = 1.15
                bid_size_mult = 0.85
        
        # Apply spread multiplier
        adjusted_spacing = spacing * spread_mult
        
        # Simulate fills
        pnl = 0.0
        
        for i in range(bid_layers):
            bid_price = price * (1 - adjusted_spacing * (i + 1))
            if low <= bid_price:
                size = base_size * size_mult * bid_size_mult
                new_pos_value = (self.position + size) * price
                if abs(new_pos_value) <= self.config.max_position_usd:
                    fee = size * bid_price * self.config.maker_fee_bps / 10000
                    self.position += size
                    self.cash -= (size * bid_price + fee)
                    pnl -= fee
        
        for i in range(ask_layers):
            ask_price = price * (1 + adjusted_spacing * (i + 1))
            if high >= ask_price:
                size = base_size * size_mult * ask_size_mult
                new_pos_value = (self.position - size) * price
                if abs(new_pos_value) <= self.config.max_position_usd:
                    fee = size * ask_price * self.config.maker_fee_bps / 10000
                    self.position -= size
                    self.cash += (size * ask_price - fee)
                    pnl -= fee
        
        # Mark-to-market
        equity = self.cash + self.position * price
        if self.equity_history:
            pnl += equity - self.equity_history[-1]
        
        self.equity_history.append(equity)
        return pnl
    
    def run_backtest(
        self,
        df: pd.DataFrame,
        features_df: pd.DataFrame,
        model_name: str,
        use_vol: bool = False,
        use_dir: bool = False
    ) -> BacktestResult:
        """Run backtest for a specific configuration."""
        self.reset()
        
        print(f"\nRunning: {model_name}")
        print(f"  Vol model: {use_vol}, Dir model: {use_dir}")
        
        # Calculate volatility thresholds
        if use_vol and self.vol_model:
            # Predict on first 10% of data to set thresholds
            sample_idx = min(50000, len(df) // 10)
            sample_preds = []
            for i in range(100, sample_idx, 100):
                pred = self.predict_volatility(features_df.iloc[i])
                if pred is not None:
                    sample_preds.append(pred)
            
            if sample_preds:
                self.vol_thresholds['high'] = np.percentile(sample_preds, self.config.vol_high_pct)
                self.vol_thresholds['low'] = np.percentile(sample_preds, self.config.vol_low_pct)
                print(f"  Vol thresholds: low={self.vol_thresholds['low']:.4f}, high={self.vol_thresholds['high']:.4f}")
        
        pnl_history = []
        chunk_size = self.config.chunk_size
        n_chunks = (len(df) - 100) // chunk_size + 1
        
        for chunk_idx in range(n_chunks):
            start_idx = 100 + chunk_idx * chunk_size
            end_idx = min(start_idx + chunk_size, len(df))
            
            if start_idx >= len(df):
                break
            
            chunk_pnl = 0.0
            
            for i in range(start_idx, end_idx):
                feat = features_df.iloc[i] if i < len(features_df) else None
                
                # Predictions
                vol_pred = self.predict_volatility(feat) if use_vol and feat is not None else None
                dir_pred, dir_conf = self.predict_direction(feat) if use_dir and feat is not None else (None, 0.5)
                
                step_pnl = self.simulate_step(
                    df.iloc[i],
                    feat,
                    use_vol=use_vol,
                    use_dir=use_dir,
                    vol_pred=vol_pred,
                    dir_pred=dir_pred,
                    dir_conf=dir_conf
                )
                pnl_history.append(step_pnl)
                chunk_pnl += step_pnl
            
            progress = (chunk_idx + 1) / n_chunks * 100
            equity = self.equity_history[-1] if self.equity_history else self.config.initial_capital
            print(f"  Chunk {chunk_idx + 1}/{n_chunks} ({progress:.0f}%): Equity ${equity:.2f}")
        
        # Calculate metrics
        return self._calculate_metrics(model_name, pnl_history)
    
    def _calculate_metrics(self, model_name: str, pnl_history: List[float]) -> BacktestResult:
        """Calculate performance metrics."""
        pnl_series = pd.Series(pnl_history)
        equity = pd.Series(self.equity_history)
        
        total_pnl = equity.iloc[-1] - self.config.initial_capital if len(equity) > 0 else 0
        
        # Win rate (daily windows)
        window_size = 1440
        n_windows = len(pnl_series) // window_size
        if n_windows > 0:
            daily_pnl = [pnl_series[i*window_size:(i+1)*window_size].sum() for i in range(n_windows)]
            wins = sum(1 for p in daily_pnl if p > 0)
            win_rate = wins / len(daily_pnl)
        else:
            win_rate = 0.5
        
        # Sharpe
        returns = pnl_series / self.config.initial_capital
        sharpe = returns.mean() / returns.std() * np.sqrt(525600) if returns.std() > 0 else 0
        
        # Max drawdown
        peak = equity.expanding().max()
        drawdown = (equity - peak) / peak
        max_dd = abs(drawdown.min()) if len(drawdown) > 0 else 0
        
        # Profit factor
        gross_profit = pnl_series[pnl_series > 0].sum()
        gross_loss = abs(pnl_series[pnl_series < 0].sum())
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        print(f"  Results: PnL=${total_pnl:.2f}, Sharpe={sharpe:.2f}, MDD={max_dd:.1%}")
        
        return BacktestResult(
            model_name=model_name,
            total_pnl=total_pnl,
            sharpe_ratio=sharpe,
            max_drawdown=max_dd,
            win_rate=win_rate,
            profit_factor=profit_factor,
            equity_curve=self.equity_history.copy()
        )


def main():
    """Run comparison backtest."""
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='data/btcusdt_1m_1year.csv')
    
    args = parser.parse_args()
    
    print("="*70)
    print("BACKTEST: HMM vs +Volatility vs +Direction+Volatility")
    print("="*70)
    
    # Load data
    print(f"\nLoading {args.data}...")
    df = pd.read_csv(args.data)
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
    print(f"Loaded {len(df):,} rows")
    
    # Compute features
    print("Computing features...")
    fe = get_feature_engineer('binance')
    features_df = fe.compute_features(df)
    print(f"Computed {features_df.shape[1]} features")
    
    # Initialize backtest
    bt = CombinedStrategyBacktest()
    bt.load_models()
    
    results = {}
    
    # 1. Baseline (HMM-only)
    results['baseline'] = bt.run_backtest(
        df, features_df, 
        model_name="HMM-only (baseline)",
        use_vol=False, use_dir=False
    )
    
    # 2. HMM + Volatility
    results['vol_only'] = bt.run_backtest(
        df, features_df,
        model_name="HMM + Volatility",
        use_vol=True, use_dir=False
    )
    
    # 3. HMM + Volatility + Direction (combined)
    results['combined'] = bt.run_backtest(
        df, features_df,
        model_name="HMM + Vol + Direction",
        use_vol=True, use_dir=True
    )
    
    # Summary
    print("\n" + "="*70)
    print("COMPARISON SUMMARY")
    print("="*70)
    
    baseline = results['baseline']
    
    print(f"\n{'Model':<25} {'PnL':>12} {'Sharpe':>10} {'MDD':>10} {'vs Base':>12}")
    print("-"*70)
    
    for name, result in results.items():
        pnl_diff = result.total_pnl - baseline.total_pnl
        diff_str = f"${pnl_diff:+.0f}" if name != 'baseline' else "-"
        print(f"{result.model_name:<25} ${result.total_pnl:>11.2f} {result.sharpe_ratio:>10.2f} {result.max_drawdown:>9.1%} {diff_str:>12}")
    
    # Save results
    output = {
        'timestamp': datetime.now().isoformat(),
        'results': {
            name: {
                'total_pnl': r.total_pnl,
                'sharpe_ratio': r.sharpe_ratio,
                'max_drawdown': r.max_drawdown,
                'win_rate': r.win_rate,
                'profit_factor': r.profit_factor,
            }
            for name, r in results.items()
        }
    }
    
    with open('backtest/combined_model_results.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\nResults saved to backtest/combined_model_results.json")


if __name__ == '__main__':
    main()
