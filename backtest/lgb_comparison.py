"""
Backtest Comparison: HMM-only vs HMM+LightGBM
v4.0 - Chunked processing for long backtests

Compares strategy performance with and without LightGBM direction predictor.
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

from ml.feature_engineering import get_feature_engineer, create_target
from ml.lightgbm_predictor import LightGBMPredictor


@dataclass
class BacktestConfig:
    """Backtest configuration."""
    # Data
    data_path: str = 'data/btcusdt_1m_1year.csv'
    
    # Trading
    initial_capital: float = 10000.0
    order_size_usd: float = 200.0
    maker_fee_bps: float = 2.0
    taker_fee_bps: float = 5.0
    slippage_bps: float = 1.0
    
    # Grid MM settings
    grid_layers: int = 5
    grid_spacing_pct: float = 0.15
    
    # Risk
    max_position_usd: float = 2000.0
    
    # LightGBM settings
    lgb_confidence_threshold: float = 0.55
    lgb_size_multiplier_max: float = 1.5
    
    # Chunked processing
    chunk_size: int = 50000  # Process 50k rows at a time
    
    # Output
    output_dir: str = 'backtest'


@dataclass
class TradeRecord:
    """Record of a single trade."""
    timestamp: datetime
    side: str  # 'buy' or 'sell'
    price: float
    size: float
    fee: float
    pnl: float = 0.0
    model: str = 'baseline'  # 'baseline' or 'lgb'


@dataclass 
class BacktestResult:
    """Results from a backtest run."""
    model_name: str
    total_pnl: float
    total_trades: int
    win_rate: float
    sharpe_ratio: float
    max_drawdown: float
    profit_factor: float
    avg_trade_pnl: float
    
    # Time series
    equity_curve: List[float] = field(default_factory=list)
    daily_returns: List[float] = field(default_factory=list)


class StrategySimulator:
    """
    Simulates grid market making strategy.
    
    Compares:
    - Baseline: HMM-only parameter adjustment
    - Enhanced: HMM + LightGBM direction bias
    """
    
    def __init__(self, config: BacktestConfig):
        self.config = config
        self.fe = get_feature_engineer('binance')
        self.lgb_predictor: Optional[LightGBMPredictor] = None
        
        # State
        self.position = 0.0
        self.cash = config.initial_capital
        self.equity_history: List[float] = []
        self.trades: List[TradeRecord] = []
    
    def load_lgb_model(self, path: str = 'data/direction_model_lgb.pkl') -> bool:
        """Load LightGBM model."""
        self.lgb_predictor = LightGBMPredictor()
        return self.lgb_predictor.load_model(path)
    
    def reset(self):
        """Reset simulator state."""
        self.position = 0.0
        self.cash = self.config.initial_capital
        self.equity_history = []
        self.trades = []
    
    def simulate_grid_mm(
        self,
        row: pd.Series,
        features: pd.Series,
        use_lgb: bool = False,
        hmm_regime: str = 'low_vol'
    ) -> float:
        """
        Simulate one step of grid market making.
        
        Returns:
            PnL for this step
        """
        price = row['close']
        high = row['high']
        low = row['low']
        
        # Base grid parameters (from HMM regime)
        base_layers = self.config.grid_layers
        base_size = self.config.order_size_usd / price
        spacing = self.config.grid_spacing_pct / 100
        
        # LightGBM enhancement
        bid_layers = base_layers
        ask_layers = base_layers
        bid_size_mult = 1.0
        ask_size_mult = 1.0
        
        if use_lgb and self.lgb_predictor:
            try:
                pred = self.lgb_predictor.predict(features, hmm_regime)
                
                # Use probability differential instead of argmax
                # This gives continuous skew even when NEUTRAL is highest
                up_prob = pred.probabilities['UP']
                down_prob = pred.probabilities['DOWN']
                prob_diff = up_prob - down_prob  # Range: -1 to +1
                
                # Apply skew if there's meaningful directional signal
                # Even 5% difference should create some skew
                skew_threshold = 0.02  # 2% probability difference
                
                if abs(prob_diff) >= skew_threshold:
                    # Scale factor: prob_diff of 0.1 -> 20% skew
                    skew_factor = prob_diff * 2.0  # Range: -2 to +2
                    
                    if prob_diff > 0:  # Bullish
                        # More bid layers, larger bid size
                        layer_shift = min(2, int(skew_factor * 2))
                        bid_layers = min(7, base_layers + layer_shift)
                        ask_layers = max(3, base_layers - layer_shift)
                        bid_size_mult = min(1.0 + skew_factor * 0.5, self.config.lgb_size_multiplier_max)
                        ask_size_mult = max(0.7, 1.0 - skew_factor * 0.3)
                    else:  # Bearish
                        # More ask layers, larger ask size
                        layer_shift = min(2, int(-skew_factor * 2))
                        ask_layers = min(7, base_layers + layer_shift)
                        bid_layers = max(3, base_layers - layer_shift)
                        ask_size_mult = min(1.0 - skew_factor * 0.5, self.config.lgb_size_multiplier_max)
                        bid_size_mult = max(0.7, 1.0 + skew_factor * 0.3)
            except:
                pass  # Fallback to baseline
        
        # Calculate grid levels
        mid_price = price
        pnl = 0.0
        
        # Simulate grid fills based on high/low
        for i in range(bid_layers):
            bid_price = mid_price * (1 - spacing * (i + 1))
            if low <= bid_price:
                # Bid filled
                size = base_size * bid_size_mult
                
                # Check position limit
                new_pos_value = (self.position + size) * price
                if abs(new_pos_value) <= self.config.max_position_usd:
                    fee = size * bid_price * self.config.maker_fee_bps / 10000
                    self.position += size
                    self.cash -= (size * bid_price + fee)
                    pnl -= fee
        
        for i in range(ask_layers):
            ask_price = mid_price * (1 + spacing * (i + 1))
            if high >= ask_price:
                # Ask filled
                size = base_size * ask_size_mult
                
                # Check position limit
                new_pos_value = (self.position - size) * price
                if abs(new_pos_value) <= self.config.max_position_usd:
                    fee = size * ask_price * self.config.maker_fee_bps / 10000
                    self.position -= size
                    self.cash += (size * ask_price - fee)
                    pnl -= fee
        
        # Mark-to-market PnL
        equity = self.cash + self.position * price
        if self.equity_history:
            pnl += equity - self.equity_history[-1]
        
        self.equity_history.append(equity)
        
        return pnl
    
    def run_backtest(
        self,
        df: pd.DataFrame,
        use_lgb: bool = False,
        progress_callback=None
    ) -> BacktestResult:
        """
        Run full backtest on data.
        
        Args:
            df: OHLCV DataFrame
            use_lgb: Whether to use LightGBM enhancement
            progress_callback: Optional callback for progress updates
        """
        self.reset()
        
        model_name = 'HMM+LightGBM' if use_lgb else 'HMM-only'
        print(f"\n{'='*60}")
        print(f"Running backtest: {model_name}")
        print(f"Data: {len(df):,} rows")
        print(f"{'='*60}")
        
        # Compute features
        print("Computing features...")
        features_df = self.fe.compute_features(df)
        
        # Simple HMM regime simulation (based on volatility)
        volatility = df['close'].pct_change().rolling(60).std() * 100
        returns = df['close'].pct_change().rolling(20).sum() * 100
        
        # Track PnL
        pnl_history = []
        chunk_size = self.config.chunk_size
        n_chunks = (len(df) - 60) // chunk_size + 1
        
        print(f"Processing in {n_chunks} chunks...")
        
        for chunk_idx in range(n_chunks):
            start_idx = 60 + chunk_idx * chunk_size  # Skip warmup
            end_idx = min(start_idx + chunk_size, len(df))
            
            if start_idx >= len(df):
                break
            
            chunk_pnl = 0.0
            
            for i in range(start_idx, end_idx):
                # Determine HMM regime (simplified)
                vol = volatility.iloc[i] if not pd.isna(volatility.iloc[i]) else 0.5
                ret = returns.iloc[i] if not pd.isna(returns.iloc[i]) else 0
                
                if vol > 1.5:
                    regime = 'high_vol'
                elif ret > 2:
                    regime = 'trend_up'
                elif ret < -2:
                    regime = 'trend_down'
                else:
                    regime = 'low_vol'
                
                # Get features for this row
                if i < len(features_df):
                    feat = features_df.iloc[i]
                else:
                    feat = None
                
                # Simulate strategy
                step_pnl = self.simulate_grid_mm(
                    df.iloc[i],
                    feat,
                    use_lgb=use_lgb,
                    hmm_regime=regime
                )
                pnl_history.append(step_pnl)
                chunk_pnl += step_pnl
            
            # Progress update
            progress = (chunk_idx + 1) / n_chunks * 100
            equity = self.equity_history[-1] if self.equity_history else self.config.initial_capital
            print(f"  Chunk {chunk_idx + 1}/{n_chunks} ({progress:.0f}%): PnL ${chunk_pnl:.2f}, Equity ${equity:.2f}")
            
            if progress_callback:
                progress_callback(chunk_idx + 1, n_chunks, chunk_pnl)
        
        # Calculate metrics
        result = self._calculate_metrics(model_name, pnl_history)
        
        print(f"\n{model_name} Results:")
        print(f"  Total PnL: ${result.total_pnl:.2f}")
        print(f"  Sharpe Ratio: {result.sharpe_ratio:.2f}")
        print(f"  Max Drawdown: {result.max_drawdown:.1%}")
        print(f"  Win Rate: {result.win_rate:.1%}")
        
        return result
    
    def _calculate_metrics(
        self,
        model_name: str,
        pnl_history: List[float]
    ) -> BacktestResult:
        """Calculate performance metrics."""
        pnl_series = pd.Series(pnl_history)
        equity = pd.Series(self.equity_history)
        
        # Total PnL
        total_pnl = equity.iloc[-1] - self.config.initial_capital if len(equity) > 0 else 0
        
        # Win rate (on daily-equivalent windows - 1440 minutes per day)
        window_size = 1440
        n_windows = len(pnl_series) // window_size
        if n_windows > 0:
            daily_pnl = [pnl_series[i*window_size:(i+1)*window_size].sum() for i in range(n_windows)]
            wins = sum(1 for p in daily_pnl if p > 0)
            win_rate = wins / len(daily_pnl)
        else:
            win_rate = 0.5
        
        # Sharpe ratio (annualized, assuming 1-min data)
        returns = pnl_series / self.config.initial_capital
        sharpe = returns.mean() / returns.std() * np.sqrt(525600) if returns.std() > 0 else 0
        
        # Max drawdown
        peak = equity.expanding().max()
        drawdown = (equity - peak) / peak
        max_dd = drawdown.min() if len(drawdown) > 0 else 0
        
        # Profit factor
        gross_profit = pnl_series[pnl_series > 0].sum()
        gross_loss = abs(pnl_series[pnl_series < 0].sum())
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        return BacktestResult(
            model_name=model_name,
            total_pnl=total_pnl,
            total_trades=len(self.trades),
            win_rate=win_rate,
            sharpe_ratio=sharpe,
            max_drawdown=abs(max_dd),
            profit_factor=profit_factor,
            avg_trade_pnl=total_pnl / max(len(pnl_history), 1),
            equity_curve=self.equity_history,
            daily_returns=pnl_history
        )


def run_comparison_backtest(
    data_path: str = 'data/btcusdt_1m_1year.csv',
    output_dir: str = 'backtest'
) -> Dict[str, BacktestResult]:
    """
    Run comparison backtest between HMM-only and HMM+LightGBM.
    
    Returns:
        Dict with results for each model
    """
    print("="*70)
    print("BACKTEST COMPARISON: HMM-only vs HMM+LightGBM")
    print("="*70)
    
    # Load data
    print(f"\nLoading data from {data_path}...")
    df = pd.read_csv(data_path)
    
    # Convert timestamp
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
    
    print(f"Loaded {len(df):,} rows")
    print(f"Period: {df.index[0]} to {df.index[-1]}")
    
    # Config
    config = BacktestConfig(
        data_path=data_path,
        chunk_size=50000
    )
    
    # Initialize simulator
    sim = StrategySimulator(config)
    
    # Load LightGBM model
    if not sim.load_lgb_model():
        print("Warning: Could not load LightGBM model")
    
    results = {}
    
    # Run baseline (HMM-only)
    results['hmm_only'] = sim.run_backtest(df, use_lgb=False)
    
    # Run enhanced (HMM+LightGBM)
    results['hmm_lgb'] = sim.run_backtest(df, use_lgb=True)
    
    # Comparison summary
    print("\n" + "="*70)
    print("COMPARISON SUMMARY")
    print("="*70)
    
    baseline = results['hmm_only']
    enhanced = results['hmm_lgb']
    
    print(f"\n{'Metric':<25} {'HMM-only':>15} {'HMM+LightGBM':>15} {'Improvement':>15}")
    print("-"*70)
    
    pnl_diff = enhanced.total_pnl - baseline.total_pnl
    print(f"{'Total PnL':<25} ${baseline.total_pnl:>14.2f} ${enhanced.total_pnl:>14.2f} ${pnl_diff:>14.2f}")
    
    sharpe_diff = enhanced.sharpe_ratio - baseline.sharpe_ratio
    print(f"{'Sharpe Ratio':<25} {baseline.sharpe_ratio:>15.2f} {enhanced.sharpe_ratio:>15.2f} {sharpe_diff:>+15.2f}")
    
    print(f"{'Max Drawdown':<25} {baseline.max_drawdown:>14.1%} {enhanced.max_drawdown:>14.1%}")
    print(f"{'Win Rate':<25} {baseline.win_rate:>14.1%} {enhanced.win_rate:>14.1%}")
    print(f"{'Profit Factor':<25} {baseline.profit_factor:>15.2f} {enhanced.profit_factor:>15.2f}")
    
    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    results_summary = {
        'timestamp': datetime.now().isoformat(),
        'data_path': data_path,
        'data_rows': len(df),
        'hmm_only': {
            'total_pnl': baseline.total_pnl,
            'sharpe_ratio': baseline.sharpe_ratio,
            'max_drawdown': baseline.max_drawdown,
            'win_rate': baseline.win_rate,
            'profit_factor': baseline.profit_factor,
        },
        'hmm_lgb': {
            'total_pnl': enhanced.total_pnl,
            'sharpe_ratio': enhanced.sharpe_ratio,
            'max_drawdown': enhanced.max_drawdown,
            'win_rate': enhanced.win_rate,
            'profit_factor': enhanced.profit_factor,
        },
        'improvement': {
            'pnl_diff': pnl_diff,
            'sharpe_diff': sharpe_diff,
        }
    }
    
    with open(output_path / 'lgb_comparison_results.json', 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    print(f"\nResults saved to {output_path / 'lgb_comparison_results.json'}")
    
    # Save equity curves
    equity_df = pd.DataFrame({
        'hmm_only': baseline.equity_curve,
        'hmm_lgb': enhanced.equity_curve
    })
    equity_df.to_csv(output_path / 'lgb_equity_curves.csv', index=False)
    
    return results


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Backtest comparison')
    parser.add_argument('--data', type=str, default='data/btcusdt_1m_1year.csv')
    parser.add_argument('--output', type=str, default='backtest')
    
    args = parser.parse_args()
    
    run_comparison_backtest(args.data, args.output)


if __name__ == '__main__':
    main()
