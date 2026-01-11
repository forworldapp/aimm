"""
Adaptive Parameter Tuner (Phase 3)
- Adjusts γ (gamma) and κ (kappa) based on recent trade performance
- Uses online learning: immediate feedback after each trade
- No historical data required - learns in real-time
"""

import json
import os
import time
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
import logging

@dataclass
class AdaptiveParams:
    """Current adaptive parameters and performance metrics."""
    gamma: float = 1.0
    kappa: float = 1000.0
    recent_pnl: float = 0.0
    win_rate: float = 0.5
    avg_spread: float = 0.002
    fill_rate: float = 0.5
    adjustment_count: int = 0
    last_updated: float = 0.0


class AdaptiveParameterTuner:
    """
    Online adaptive tuning of A&S parameters based on recent performance.
    
    Strategy:
    - Track recent trades (last N trades)
    - If profitable: explore tighter spreads (higher kappa)
    - If losing: widen spreads for more margin (lower kappa)
    - Adjust gamma based on inventory risk exposure
    """
    
    # Tuning bounds
    GAMMA_MIN = 0.2
    GAMMA_MAX = 2.0
    KAPPA_MIN = 100
    KAPPA_MAX = 3000
    
    # Learning rates
    GAMMA_LR = 0.05  # 5% adjustment per step
    KAPPA_LR = 0.1   # 10% adjustment per step
    
    def __init__(self, state_file: str = "data/adaptive_params.json"):
        self.logger = logging.getLogger("AdaptiveTuner")
        self.state_file = state_file
        
        # Recent trade tracking
        self.recent_trades: List[Dict] = []
        self.max_trades = 20  # Rolling window
        
        # Current parameters
        self.params = AdaptiveParams()
        
        # Load previous state if exists
        self._load_state()
    
    def _load_state(self):
        """Load saved state from disk."""
        try:
            if os.path.exists(self.state_file):
                with open(self.state_file, 'r') as f:
                    data = json.load(f)
                    self.params = AdaptiveParams(**data.get('params', {}))
                    self.recent_trades = data.get('recent_trades', [])
                    self.logger.info(f"Loaded adaptive state: γ={self.params.gamma:.2f}, κ={self.params.kappa:.0f}")
        except Exception as e:
            self.logger.warning(f"Failed to load adaptive state: {e}")
    
    def _save_state(self):
        """Save current state to disk."""
        try:
            data = {
                'params': asdict(self.params),
                'recent_trades': self.recent_trades[-self.max_trades:]
            }
            os.makedirs(os.path.dirname(self.state_file), exist_ok=True)
            with open(self.state_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            self.logger.warning(f"Failed to save adaptive state: {e}")
    
    def record_trade(self, trade: Dict):
        """
        Record a trade and update parameters based on performance.
        
        trade: {
            'profit': float,      # PnL of this trade
            'spread': float,      # Spread at time of trade
            'side': str,          # 'buy' or 'sell'
            'filled': bool,       # Was order filled?
            'position': float     # Position after trade
        }
        """
        self.recent_trades.append({
            **trade,
            'timestamp': time.time()
        })
        
        # Keep only recent trades
        if len(self.recent_trades) > self.max_trades:
            self.recent_trades = self.recent_trades[-self.max_trades:]
        
        # Update performance metrics
        self._update_metrics()
        
        # Adjust parameters
        self._adjust_parameters()
        
        # Save state
        self._save_state()
    
    def _update_metrics(self):
        """Calculate performance metrics from recent trades."""
        if not self.recent_trades:
            return
        
        # Total PnL
        profits = [t.get('profit', 0) for t in self.recent_trades]
        self.params.recent_pnl = sum(profits)
        
        # Win rate
        wins = sum(1 for p in profits if p > 0)
        self.params.win_rate = wins / len(profits) if profits else 0.5
        
        # Average spread
        spreads = [t.get('spread', 0.002) for t in self.recent_trades if t.get('spread')]
        self.params.avg_spread = sum(spreads) / len(spreads) if spreads else 0.002
        
        # Fill rate
        filled = sum(1 for t in self.recent_trades if t.get('filled', True))
        self.params.fill_rate = filled / len(self.recent_trades) if self.recent_trades else 0.5
    
    def _adjust_parameters(self):
        """Adjust γ and κ based on recent performance."""
        if len(self.recent_trades) < 5:
            return  # Need minimum trades for adjustment
        
        old_gamma = self.params.gamma
        old_kappa = self.params.kappa
        
        # === Kappa Adjustment (Spread Control) ===
        # High PnL + Good fill rate → Tighter spread (increase kappa)
        # Low PnL or Low fill rate → Wider spread (decrease kappa)
        
        if self.params.recent_pnl > 0 and self.params.fill_rate > 0.3:
            # Profitable and filling: Try tighter spreads
            self.params.kappa *= (1 + self.KAPPA_LR)
        elif self.params.recent_pnl < 0:
            # Losing: Widen spreads for more margin
            self.params.kappa *= (1 - self.KAPPA_LR)
        elif self.params.fill_rate < 0.2:
            # Not filling: Tighten spreads
            self.params.kappa *= (1 + self.KAPPA_LR * 0.5)
        
        # === Gamma Adjustment (Risk Control) ===
        # High win rate → More aggressive (lower gamma)
        # Low win rate → More conservative (higher gamma)
        
        if self.params.win_rate > 0.6:
            # Winning streak: Slightly more aggressive
            self.params.gamma *= (1 - self.GAMMA_LR)
        elif self.params.win_rate < 0.4:
            # Losing streak: More conservative
            self.params.gamma *= (1 + self.GAMMA_LR)
        
        # Clamp to bounds
        self.params.gamma = max(self.GAMMA_MIN, min(self.GAMMA_MAX, self.params.gamma))
        self.params.kappa = max(self.KAPPA_MIN, min(self.KAPPA_MAX, self.params.kappa))
        
        # Round for stability
        self.params.gamma = round(self.params.gamma, 2)
        self.params.kappa = round(self.params.kappa, 0)
        
        self.params.adjustment_count += 1
        self.params.last_updated = time.time()
        
        if old_gamma != self.params.gamma or old_kappa != self.params.kappa:
            self.logger.info(
                f"Adaptive adjustment #{self.params.adjustment_count}: "
                f"γ {old_gamma:.2f}→{self.params.gamma:.2f}, "
                f"κ {old_kappa:.0f}→{self.params.kappa:.0f} "
                f"(PnL: ${self.params.recent_pnl:.2f}, WR: {self.params.win_rate:.0%})"
            )
    
    def get_params(self) -> Dict[str, float]:
        """Get current adaptive parameters."""
        return {
            'gamma': self.params.gamma,
            'kappa': self.params.kappa,
            'recent_pnl': self.params.recent_pnl,
            'win_rate': self.params.win_rate,
            'fill_rate': self.params.fill_rate,
            'adjustment_count': self.params.adjustment_count
        }
    
    def get_display_metrics(self) -> Dict:
        """Get metrics for dashboard display."""
        return {
            'adaptive_gamma': self.params.gamma,
            'adaptive_kappa': self.params.kappa,
            'recent_pnl': round(self.params.recent_pnl, 2),
            'win_rate': round(self.params.win_rate * 100, 1),
            'fill_rate': round(self.params.fill_rate * 100, 1),
            'adjustments': self.params.adjustment_count,
            'trade_count': len(self.recent_trades)
        }


# Test
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    tuner = AdaptiveParameterTuner()
    
    # Simulate some trades
    test_trades = [
        {'profit': 0.5, 'spread': 0.002, 'filled': True, 'side': 'buy'},
        {'profit': -0.3, 'spread': 0.002, 'filled': True, 'side': 'sell'},
        {'profit': 0.8, 'spread': 0.002, 'filled': True, 'side': 'buy'},
        {'profit': 0.2, 'spread': 0.002, 'filled': True, 'side': 'sell'},
        {'profit': -0.1, 'spread': 0.002, 'filled': True, 'side': 'buy'},
    ]
    
    for trade in test_trades:
        tuner.record_trade(trade)
    
    print("\n=== Adaptive Parameters ===")
    print(tuner.get_params())
    print(tuner.get_display_metrics())
