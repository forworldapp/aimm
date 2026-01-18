"""
Integrated Market Maker - Phase 5.1
Combines all ML components into unified trading system.

Author: Antigravity
Version: 1.0.0
"""

import numpy as np
import time
import logging
from typing import Dict, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class SystemMetrics:
    """Real-time system metrics for dashboard."""
    
    # Performance
    total_pnl: float = 0.0
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    sharpe_ratio: float = 0.0
    
    # Risk
    current_inventory: float = 0.0
    max_inventory: float = 0.0
    inventory_ratio: float = 0.0
    drawdown: float = 0.0
    max_drawdown: float = 0.0
    
    # Regime
    current_regime: str = "unknown"
    regime_confidence: float = 0.0
    regime_probs: Dict[str, float] = field(default_factory=dict)
    
    # Adverse Selection
    as_probability: float = 0.0
    as_spread_adjust: int = 0
    as_trades_detected: int = 0
    
    # Orders
    current_spread_bps: float = 0.0
    bid_size: float = 0.0
    ask_size: float = 0.0
    fill_rate: float = 0.0
    total_trades: int = 0
    
    # Funding & Liquidation
    funding_prediction: float = 0.0
    inventory_bias: float = 0.0
    liquidation_alert: bool = False
    liquidation_severity: float = 0.0
    
    # Bandit
    bandit_selected_spread: float = 0.0
    bandit_exploration_rate: float = 0.0
    
    # Model Status
    regime_model_fitted: bool = False
    as_model_fitted: bool = False
    fill_model_fitted: bool = False
    online_learning_active: bool = False
    drift_detected: bool = False
    
    # System
    last_update: str = ""
    uptime_seconds: float = 0.0
    cycle_count: int = 0


class IntegratedMarketMaker:
    """
    Integrated Market Maker with all ML components.
    
    Components:
    - DynamicOrderSizer (Phase 1.1)
    - AdverseSelectionDetector (Phase 1.2)
    - RegimeDetector / HMMRegimeDetector (Phase 2)
    - FillProbabilityModel (Phase 3.1)
    - FundingPredictor / LiquidationDetector (Phase 3.2/3.3)
    - ContextualBanditSpread (Phase 4.1)
    - OnlineModelUpdater (Phase 4.2)
    """
    
    def __init__(self, base_strategy):
        """
        Args:
            base_strategy: MarketMaker instance with exchange connection
        """
        self.strategy = base_strategy
        self.logger = logging.getLogger("IntegratedMM")
        
        # Metrics
        self.metrics = SystemMetrics()
        self.start_time = time.time()
        
        # Import components
        self._init_components()
    
    def _init_components(self):
        """Initialize all ML components."""
        
        # Dynamic Order Sizer
        try:
            from ml.dynamic_order_sizer import DynamicOrderSizer
            self.order_sizer = DynamicOrderSizer(
                base_size=getattr(self.strategy, 'order_size_usd', 200),
                max_inventory=getattr(self.strategy, 'max_position_usd', 5000)
            )
            self.logger.info("✓ DynamicOrderSizer loaded")
        except Exception as e:
            self.order_sizer = None
            self.logger.warning(f"✗ DynamicOrderSizer: {e}")
        
        # Adverse Selection Detector
        try:
            from ml.adverse_selection import AdverseSelectionDetector
            self.as_detector = AdverseSelectionDetector()
            self.logger.info("✓ AdverseSelectionDetector loaded")
        except Exception as e:
            self.as_detector = None
            self.logger.warning(f"✗ AdverseSelectionDetector: {e}")
        
        # Fill Probability Model
        try:
            from ml.fill_probability import FillProbabilityModel
            self.fill_model = FillProbabilityModel()
            self.logger.info("✓ FillProbabilityModel loaded")
        except Exception as e:
            self.fill_model = None
            self.logger.warning(f"✗ FillProbabilityModel: {e}")
        
        # Funding Predictor
        try:
            from ml.funding_liquidation import FundingPredictor, LiquidationDetector
            self.funding_predictor = FundingPredictor()
            self.liquidation_detector = LiquidationDetector()
            self.logger.info("✓ FundingPredictor & LiquidationDetector loaded")
        except Exception as e:
            self.funding_predictor = None
            self.liquidation_detector = None
            self.logger.warning(f"✗ Funding/Liquidation: {e}")
        
        # Contextual Bandit
        try:
            from ml.contextual_bandit import ContextualBanditSpread
            self.bandit = ContextualBanditSpread()
            self.logger.info("✓ ContextualBanditSpread loaded")
        except Exception as e:
            self.bandit = None
            self.logger.warning(f"✗ ContextualBandit: {e}")
        
        # Online Learning
        try:
            from ml.online_learning import OnlineModelUpdater
            # Will be used to wrap other models later
            self.online_updater = None  # Initialize when needed
            self.logger.info("✓ OnlineLearning available")
        except Exception as e:
            self.online_updater = None
    
    def update_metrics(self, market_data: dict):
        """Update all system metrics."""
        
        self.metrics.cycle_count += 1
        self.metrics.uptime_seconds = time.time() - self.start_time
        self.metrics.last_update = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Extract market data
        mid_price = market_data.get('mid_price', 0)
        inventory = market_data.get('inventory', 0)
        volatility = market_data.get('volatility', 0.01)
        
        # Inventory metrics
        self.metrics.current_inventory = inventory
        max_inv = getattr(self.strategy, 'max_position_usd', 5000)
        self.metrics.max_inventory = max_inv
        self.metrics.inventory_ratio = inventory / max_inv if max_inv > 0 else 0
        
        # Regime metrics
        if hasattr(self.strategy, 'regime_detector') and self.strategy.regime_detector:
            self.metrics.regime_model_fitted = self.strategy.regime_detector.is_fitted
            regime_probs = market_data.get('regime_probs', {})
            if regime_probs:
                self.metrics.regime_probs = regime_probs
                best_regime = max(regime_probs, key=regime_probs.get)
                self.metrics.current_regime = best_regime
                self.metrics.regime_confidence = regime_probs.get(best_regime, 0)
        
        # Adverse Selection metrics
        if self.as_detector:
            stats = self.as_detector.get_stats()
            self.metrics.as_model_fitted = self.as_detector.is_fitted
            self.metrics.as_trades_detected = stats.get('adverse_trades', 0)
        
        # Funding metrics
        if self.funding_predictor:
            self.metrics.inventory_bias = self.funding_predictor.get_inventory_bias()
            pred, _ = self.funding_predictor.predict_next_funding()
            self.metrics.funding_prediction = pred
        
        # Liquidation metrics
        if self.liquidation_detector:
            stats = self.liquidation_detector.get_stats()
            self.metrics.liquidation_alert = stats.get('cascade_active', False)
            self.metrics.liquidation_severity = stats.get('severity', 0)
        
        # Bandit metrics
        if self.bandit:
            stats = self.bandit.get_stats()
            self.metrics.bandit_exploration_rate = 1 / (1 + stats.get('total_observations', 0) * 0.01)
        
        # Fill model metrics
        if self.fill_model:
            self.metrics.fill_model_fitted = self.fill_model.is_fitted
    
    def get_quote_adjustment(self, market_data: dict) -> dict:
        """
        Get quote adjustments from all ML components.
        
        Returns:
            Dict with spread_add_bps, size_mult, should_pause
        """
        adjustment = {
            'spread_add_bps': 0,
            'size_mult': 1.0,
            'should_pause': False,
            'inventory_bias': 0
        }
        
        volatility = market_data.get('volatility', 0.01)
        inventory = market_data.get('inventory', 0)
        regime = market_data.get('regime', 0)
        
        # Adverse Selection adjustment
        if self.as_detector and market_data.get('last_trade'):
            _, as_prob = self.as_detector.predict(market_data['last_trade'])
            self.metrics.as_probability = as_prob
            
            as_adj = self.as_detector.get_adjustment(as_prob)
            adjustment['spread_add_bps'] += as_adj.get('spread_add_bps', 0)
            adjustment['size_mult'] *= as_adj.get('size_mult', 1.0)
            if as_adj.get('pause_seconds', 0) > 0:
                adjustment['should_pause'] = True
            
            self.metrics.as_spread_adjust = as_adj.get('spread_add_bps', 0)
        
        # Liquidation adjustment
        if self.liquidation_detector:
            oi_change = market_data.get('oi_change', 0)
            price_change = market_data.get('price_change', 0)
            volume_spike = market_data.get('volume_spike', 1)
            
            is_cascade, severity, direction = self.liquidation_detector.detect(
                oi_change, price_change, volume_spike
            )
            
            if is_cascade:
                liq_adj = self.liquidation_detector.get_adjustment()
                adjustment['spread_add_bps'] += int(severity * 5)
                adjustment['size_mult'] *= liq_adj.get('size_mult', 1.0)
        
        # Funding bias
        if self.funding_predictor:
            adjustment['inventory_bias'] = self.funding_predictor.get_inventory_bias()
        
        # Contextual Bandit spread selection
        if self.bandit:
            context = self.bandit.extract_context(
                volatility=volatility,
                inventory_ratio=self.metrics.inventory_ratio,
                regime=regime,
                book_imbalance=market_data.get('book_imbalance', 0),
                hour_of_day=datetime.now().hour
            )
            spread, arm = self.bandit.select_spread(context)
            self.metrics.bandit_selected_spread = spread
        
        return adjustment
    
    def get_order_sizes(self, market_data: dict) -> Tuple[float, float]:
        """Get dynamic bid/ask sizes."""
        if not self.order_sizer:
            base = getattr(self.strategy, 'order_size_usd', 200)
            return base, base
        
        inventory = market_data.get('inventory', 0)
        volatility = market_data.get('volatility', 0.01)
        book_depth = market_data.get('book_depth', 10)
        
        bid_size, ask_size = self.order_sizer.calculate(inventory, volatility, book_depth)
        
        self.metrics.bid_size = bid_size
        self.metrics.ask_size = ask_size
        
        return bid_size, ask_size
    
    def get_metrics_dict(self) -> dict:
        """Get all metrics as dictionary for dashboard."""
        return {
            # Performance
            'total_pnl': f"${self.metrics.total_pnl:.2f}",
            'realized_pnl': f"${self.metrics.realized_pnl:.2f}",
            'unrealized_pnl': f"${self.metrics.unrealized_pnl:.2f}",
            'sharpe_ratio': f"{self.metrics.sharpe_ratio:.2f}",
            
            # Risk
            'inventory': f"${self.metrics.current_inventory:.0f}",
            'inventory_ratio': f"{self.metrics.inventory_ratio:.1%}",
            'max_drawdown': f"{self.metrics.max_drawdown:.2%}",
            
            # Regime
            'regime': self.metrics.current_regime,
            'regime_confidence': f"{self.metrics.regime_confidence:.1%}",
            
            # Adverse Selection
            'as_prob': f"{self.metrics.as_probability:.1%}",
            'as_spread_adj': f"+{self.metrics.as_spread_adjust}bps",
            
            # Orders
            'spread': f"{self.metrics.current_spread_bps:.1f}bps",
            'bid_size': f"${self.metrics.bid_size:.0f}",
            'ask_size': f"${self.metrics.ask_size:.0f}",
            'fill_rate': f"{self.metrics.fill_rate:.1%}",
            'trades': str(self.metrics.total_trades),
            
            # Funding & Liquidation
            'funding_pred': f"{self.metrics.funding_prediction:.4%}",
            'inv_bias': f"{self.metrics.inventory_bias:+.2f}",
            'liq_alert': "⚠️ YES" if self.metrics.liquidation_alert else "✓ No",
            
            # Bandit
            'bandit_spread': f"{self.metrics.bandit_selected_spread:.0f}bps",
            
            # Model Status
            'regime_fitted': "✓" if self.metrics.regime_model_fitted else "✗",
            'as_fitted': "✓" if self.metrics.as_model_fitted else "✗",
            'fill_fitted': "✓" if self.metrics.fill_model_fitted else "✗",
            'drift': "⚠️" if self.metrics.drift_detected else "✓",
            
            # System
            'uptime': f"{self.metrics.uptime_seconds/60:.1f}m",
            'cycles': str(self.metrics.cycle_count),
            'last_update': self.metrics.last_update
        }
    
    def print_dashboard(self):
        """Print real-time dashboard to console."""
        m = self.get_metrics_dict()
        
        print("\033[2J\033[H")  # Clear screen
        print("=" * 70)
        print("      📊 INTEGRATED MARKET MAKER DASHBOARD (Phase 5)")
        print("=" * 70)
        
        print(f"\n{'─' * 70}")
        print("│ 💰 PERFORMANCE                                                     │")
        print(f"{'─' * 70}")
        print(f"│  PnL: {m['total_pnl']:>12}  │  Sharpe: {m['sharpe_ratio']:>8}  │  Trades: {m['trades']:>8}  │")
        
        print(f"\n{'─' * 70}")
        print("│ ⚖️  RISK                                                           │")
        print(f"{'─' * 70}")
        print(f"│  Inventory: {m['inventory']:>10} ({m['inventory_ratio']:>6})  │  MaxDD: {m['max_drawdown']:>8}  │")
        
        print(f"\n{'─' * 70}")
        print("│ 🎯 REGIME & SIGNALS                                                │")
        print(f"{'─' * 70}")
        print(f"│  Regime: {m['regime']:>12} ({m['regime_confidence']:>5})  │  Liq Alert: {m['liq_alert']:>8}  │")
        
        print(f"\n{'─' * 70}")
        print("│ 📈 ORDERS                                                          │")
        print(f"{'─' * 70}")
        print(f"│  Spread: {m['spread']:>8}  │  Bid: {m['bid_size']:>8}  │  Ask: {m['ask_size']:>8}  │")
        print(f"│  Bandit: {m['bandit_spread']:>8}  │  AS Adj: {m['as_spread_adj']:>6}  │  Fill: {m['fill_rate']:>6}  │")
        
        print(f"\n{'─' * 70}")
        print("│ 🤖 ML MODEL STATUS                                                 │")
        print(f"{'─' * 70}")
        print(f"│  Regime: {m['regime_fitted']:>3}  │  AS: {m['as_fitted']:>3}  │  Fill: {m['fill_fitted']:>3}  │  Drift: {m['drift']:>3}  │")
        
        print(f"\n{'─' * 70}")
        print(f"│ ⏱️  System: {m['uptime']} uptime, {m['cycles']} cycles  │  Last: {m['last_update']}")
        print("=" * 70)


# Demo/Test
if __name__ == "__main__":
    print("=" * 60)
    print("INTEGRATED MARKET MAKER TEST")
    print("=" * 60)
    
    # Create mock strategy
    class MockStrategy:
        order_size_usd = 200
        max_position_usd = 5000
        regime_detector = None
    
    mock = MockStrategy()
    integrated = IntegratedMarketMaker(mock)
    
    # Simulate market data
    market_data = {
        'mid_price': 100000,
        'inventory': 1500,
        'volatility': 0.015,
        'book_depth': 8,
        'regime': 1,
        'book_imbalance': 0.1,
        'oi_change': 0,
        'price_change': 0,
        'volume_spike': 1.2
    }
    
    # Update and get adjustments
    integrated.update_metrics(market_data)
    adj = integrated.get_quote_adjustment(market_data)
    bid, ask = integrated.get_order_sizes(market_data)
    
    print(f"\n✓ Quote Adjustment: {adj}")
    print(f"✓ Sizes: bid=${bid:.0f}, ask=${ask:.0f}")
    
    # Print dashboard
    integrated.print_dashboard()
