"""
GRVT Market Maker Strategy V1.4
-------------------------------
Features:
- Bollinger Bands Mean Reversion (Buy Low/Sell High at Band Edges).
- Dynamic Spread (ATR-based volatility adaptation).
- Grid Spacing Optimization (Prevent simultaneous fills).
- RSI Safety Filter (Overbought/Oversold protection).
- Aggressive Entry Mode (Maximize fill rate on signals).

Author: Antigravity
Version: 1.4.1 (Hotfix)
Last Updated: 2025-12-18
Changelog:
- Fixed Persistence: Properly restoring Paper Exchange state on restart.
- Fixed Inventory Logic: Added missing inventory initialization and sync for exit logic.
- Fixed Import Error: Corrected filter module imports.
"""

import sys
import os
import time
import json
import logging
import asyncio
import statistics
import pandas as pd
from datetime import datetime

# Adjust path for local imports if needed
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.config import Config
from core.risk_manager import RiskManager
from core.paper_exchange import PaperGrvtExchange # Assuming this is needed based on the instruction's import list, though not used in the provided snippet.

# New Filters Import
from strategies.filters import RSIFilter, MAFilter, ADXFilter, ATRFilter, BollingerFilter, ComboFilter, ChopFilter

# ML Regime Detection
try:
    from ml.regime_detector import RegimeDetector
    from ml.hmm_regime_detector import HMMRegimeDetector
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    RegimeDetector = None
    HMMRegimeDetector = None

# Adaptive Parameter Tuning
try:
    from ml.adaptive_tuner import AdaptiveParameterTuner
    ADAPTIVE_AVAILABLE = True
except ImportError:
    ADAPTIVE_AVAILABLE = False
    AdaptiveParameterTuner = None

# Dynamic Order Sizing (Phase 1.1)
try:
    from ml.dynamic_order_sizer import DynamicOrderSizer
    DYNAMIC_SIZER_AVAILABLE = True
except ImportError:
    DYNAMIC_SIZER_AVAILABLE = False
    DynamicOrderSizer = None

# Adverse Selection Detection (Phase 1.2)
try:
    from ml.adverse_selection import AdverseSelectionDetector
    ADVERSE_SELECTION_AVAILABLE = True
except ImportError:
    ADVERSE_SELECTION_AVAILABLE = False
    AdverseSelectionDetector = None

# v4.0 ML Strategy
try:
    from ml.strategy_v4 import StrategyV4
    from ml.feature_engineering import get_feature_engineer
    ML_V4_AVAILABLE = True
except ImportError:
    ML_V4_AVAILABLE = False
    StrategyV4 = None

# v5.0 Order Flow Analysis
try:
    from ml.order_flow_analyzer import OrderFlowAnalyzer
    ORDER_FLOW_AVAILABLE = True
except ImportError:
    ORDER_FLOW_AVAILABLE = False
    OrderFlowAnalyzer = None

# v5.1 Funding Rate Arbitrage
try:
    from core.funding_monitor import FundingRateMonitor, FundingIntegratedMM
    FUNDING_AVAILABLE = True
except ImportError:
    FUNDING_AVAILABLE = False
    FundingRateMonitor = None
    FundingIntegratedMM = None


def round_tick_size(price, tick_size):
    return round(price / tick_size) * tick_size

class MarketMaker:
    """
    Enhanced Market Maker Strategy for GRVT.
    Supports Adaptive Regime Detection using various Technical Filters.
    """

    def __init__(self, exchange):
        self.exchange = exchange
        self.symbol = Config.get("exchange", "symbol", "BTC_USDT_Perp")
        
        self.logger = logging.getLogger("MarketMaker")
        
        # Exchange Info
        self.tick_size = 0.1 
        
        # Trend & Volatility State
        self.price_history = []
        self.history_max_len = 200  # Increased for ML features (needs 60m+)
        
        # Candle Data for Advanced Filters (OHLC)
        self.candles = pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close'])
        self.current_candle = None
        self.last_candle_time = 0
        
        # Command Control
        self.command_file = os.path.join("data", "command.json")
        self.is_active = False 
        self.is_running = True
        
        # Risk State
        self.initial_equity = None
        self.inventory = 0.0 # Position tracking

        # Load Params & Initialize Filter
        self._load_params()
        
        # Initialize v4.0 Strategy
        if ML_V4_AVAILABLE:
            try:
                self.strategy_v4 = StrategyV4()
                # Initialize Feature Engineer (exchange name from config or default to binance for paper)
                exchange_name = 'binance' # Default for now as models are trained on it
                self.fe = get_feature_engineer(exchange_name)
                self.logger.info("✅ v4.0 ML Strategy Initialized")
            except Exception as e:
                self.logger.error(f"❌ Failed to initialize v4.0 Strategy: {e}")
                self.strategy_v4 = None
        else:
            self.strategy_v4 = None
        
        # v5.0 Order Flow
        self.order_flow = None
        if ORDER_FLOW_AVAILABLE and Config.get("order_flow_analysis", "enabled", False):
            self.order_flow = OrderFlowAnalyzer(Config.get("order_flow_analysis", {}))
            self.logger.info("🌊 Order Flow Analysis Enabled")
        
        # v5.1 Funding Rate
        self.funding_monitor = None
        self.funding_integrator = None
        if FUNDING_AVAILABLE and Config.get("funding_rate_arbitrage", "enabled", False):
            funding_config = Config.get("funding_rate_arbitrage", {})
            self.funding_monitor = FundingRateMonitor(funding_config.get('monitoring', {}))
            self.funding_integrator = FundingIntegratedMM(funding_config.get('integration', {}))
            self.logger.info("💰 Funding Rate Arbitrage Enabled")
        
    def _load_params(self):
        """Load strategy parameters from config.yaml"""
        Config.load("config.yaml") # Force reload
        
        self.base_spread = float(Config.get("strategy", "spread_pct", 0.0002))
        self.order_size_usd = float(Config.get("strategy", "order_size_usd", 100.0))
        self.amount = 0.0 # Deprecated, auto-calc
        # self.amount = float(Config.get("strategy", "order_amount", 0.001))
        self.refresh_interval = int(Config.get("strategy", "refresh_interval", 3))
        self.skew_factor = float(Config.get("risk", "inventory_skew_factor", 0.05))
        self.max_position_usd = float(Config.get("risk", "max_position_usd", 500.0))  # Position limit
        self.max_loss_usd = float(Config.get("risk", "max_loss_usd", 50.0))  # Circuit breaker
        self.grid_layers = int(Config.get("strategy", "grid_layers", 3))
        self.entry_anchor_mode = Config.get("strategy", "entry_anchor_mode", False)
        
        # Strategy Selector
        self.trend_strategy = Config.get("strategy", "trend_strategy", "bollinger")
        self.latched_regime = None # Memory for signal latch
        
        self.filter_strategy = self._initialize_filter(self.trend_strategy)
        self.rsi_filter = RSIFilter() # Auxiliary filter
        
        self.logger.info(f"Loaded Params: Layers={self.grid_layers}, MaxPos=${self.max_position_usd}, MaxLoss=${self.max_loss_usd}")
        
        self.risk_manager = RiskManager()
        self.risk_manager.max_drawdown = float(Config.get("risk", "max_drawdown_pct", 0.10))
        
        # ML Regime Detector
        self.ml_regime_enabled = Config.get("strategy", "ml_regime_enabled", False)
        self.regime_detector = None
        if self.ml_regime_enabled and ML_AVAILABLE:
            try:
                # Model selector: 'gmm' or 'hmm'
                model_type = Config.get("strategy", "regime_model_type", "gmm")
                
                if model_type == "hmm" and HMMRegimeDetector:
                    model_path = Config.get("strategy", "regime_model_hmm_path", "data/regime_model_hmm.pkl")
                    self.regime_detector = HMMRegimeDetector(model_path=model_path)
                    self.logger.info(f"HMM Regime Detector loading from {model_path}")
                else:
                    model_path = Config.get("strategy", "regime_model_path", "data/regime_model.pkl")
                    self.regime_detector = RegimeDetector(model_path=model_path)
                    self.logger.info(f"GMM Regime Detector loading from {model_path}")
                
                if self.regime_detector.is_fitted:
                    self.logger.info(f"ML Regime Detector ({model_type.upper()}) loaded successfully")
                else:
                    self.logger.warning("ML Regime Detector not fitted, will use static params")
            except Exception as e:
                self.logger.warning(f"Failed to load ML Regime Detector: {e}")
        
        self.current_ml_regime = "unknown"
        
        # Adaptive Parameter Tuner
        self.adaptive_enabled = Config.get("strategy", "adaptive_tuning_enabled", False)
        self.adaptive_tuner = None
        if self.adaptive_enabled and ADAPTIVE_AVAILABLE:
            try:
                self.adaptive_tuner = AdaptiveParameterTuner()
                self.logger.info("Adaptive Parameter Tuner loaded successfully")
            except Exception as e:
                self.logger.warning(f"Failed to load Adaptive Tuner: {e}")

        # Dynamic Order Sizer (Phase 1.1)
        self.dynamic_sizer = None
        self.dynamic_sizing_enabled = Config.get("strategy", "dynamic_sizing_enabled", True)
        if self.dynamic_sizing_enabled and DYNAMIC_SIZER_AVAILABLE:
            try:
                self.dynamic_sizer = DynamicOrderSizer(
                    base_size=self.order_size_usd,
                    max_inventory=self.max_position_usd
                )
                self.logger.info(f"Dynamic Order Sizer loaded (base=${self.order_size_usd}, max_inv=${self.max_position_usd})")
            except Exception as e:
                self.logger.warning(f"Failed to load Dynamic Order Sizer: {e}")

        # Adverse Selection Detector (Phase 1.2)
        self.as_detector = None
        self.as_detection_enabled = Config.get("strategy", "as_detection_enabled", True)
        if self.as_detection_enabled and ADVERSE_SELECTION_AVAILABLE:
            try:
                self.as_detector = AdverseSelectionDetector()
                self.logger.info("Adverse Selection Detector loaded")
            except Exception as e:
                self.logger.warning(f"Failed to load AS Detector: {e}")
        
        # AS adjustment tracking
        self.as_spread_add_bps = 0
        self.as_size_mult = 1.0
        self.as_pause_until = 0

    def _initialize_filter(self, name):
        name = str(name).lower()
        if name == 'adx': return ADXFilter()
        if name == 'atr': return ATRFilter()
        if name == 'chop': return ChopFilter()
        if name == 'combo': return ComboFilter()
        if name == 'rsi':
             conf = Config.get("strategy", "rsi", {})
             return RSIFilter(conf.get('period', 14), conf.get('overbought', 70), conf.get('oversold', 30))
        if name == 'bollinger':
             conf = Config.get("strategy", "bollinger", {})
             return BollingerFilter(conf.get('period', 20), conf.get('std_dev', 2.0))
             conf = Config.get("strategy", "rsi", {})
             return RSIFilter(conf.get('period', 14), conf.get('overbought', 70), conf.get('oversold', 30))
        if name == 'ma_trend' or name == 'adaptive': return MAFilter() 
        return None # 'off'

    async def check_command(self):
        """Check for external commands from dashboard."""
        if os.path.exists(self.command_file):
            try:
                with open(self.command_file, "r") as f:
                    data = json.load(f)
                
                command = data.get("command")
                if command:
                    self.logger.info(f"Received command: {command}")
                    
                    if command == "start":
                        self.is_active = True
                        self.initial_equity = None # Reset drawdown baseline
                        self.logger.info("Bot STARTED.")
                    elif command == "stop":
                        self.is_active = False
                        await self.exchange.cancel_all_orders(self.symbol)
                        self.logger.info("Bot PAUSED.")
                    elif command == "stop_close":
                        self.is_active = False
                        await self.exchange.cancel_all_orders(self.symbol)
                        await self.exchange.close_position(self.symbol) # Market Close
                        self.logger.info("Bot STOPPED & CLOSED.")
                    elif command == "reload_config":
                        self._load_params()
                        self.logger.info("Configuration Reloaded.")
                    elif command == "shutdown":
                        self.is_active = False
                        self.is_running = False
                        self.logger.info("Shutdown sequence initiated.")

                    elif command == "restart":
                        self.logger.info("Received command: restart")
                        self.is_running = False # Break main loop
                        os.remove(self.command_file)
                        return 'restart'

                    # Clear command file
                    os.remove(self.command_file)
                    
            except Exception as e:
                self.logger.error(f"Error reading command: {e}")

    def _update_history(self, price):
        self.price_history.append(price)
        if len(self.price_history) > self.history_max_len:
            self.price_history.pop(0)

    def _update_candle(self, price, timestamp):
        """Update 1-minute OHLC candles."""
        dt = datetime.fromtimestamp(timestamp)
        current_minute = dt.replace(second=0, microsecond=0)
        
        if self.current_candle is None:
            self.current_candle = {
                'timestamp': current_minute,
                'open': price, 'high': price, 'low': price, 'close': price
            }
        elif self.current_candle['timestamp'] != current_minute:
            new_row = pd.DataFrame([self.current_candle])
            self.candles = pd.concat([self.candles, new_row], ignore_index=True)
            if len(self.candles) > 100:
                self.candles = self.candles.iloc[-100:]
            self.current_candle = {
                'timestamp': current_minute,
                'open': price, 'high': price, 'low': price, 'close': price
            }
        else:
            self.current_candle['high'] = max(self.current_candle['high'], price)
            self.current_candle['low'] = min(self.current_candle['low'], price)
            self.current_candle['close'] = price

    def _detect_market_regime(self):
        """Use the selected Filter Strategy to detect regime and append RSI status."""
        # Allow grid trading even when filter/ML not ready - default to neutral
        if not self.filter_strategy:
            if hasattr(self.exchange, "set_market_regime"):
                self.exchange.set_market_regime('NEUTRAL (Grid)')
            return 'neutral'  # Changed from 'ranging' to 'neutral' to allow trading
            
        if self.current_candle:
            df = pd.concat([self.candles, pd.DataFrame([self.current_candle])], ignore_index=True)
        else:
            df = self.candles
            
        regime = self.filter_strategy.analyze(df)
        
        if self.rsi_filter:
            rsi_val = self.rsi_filter.analyze(df)
            rsi_num = getattr(self.rsi_filter, 'last_rsi', 50.0)
            
            # [Safety Filter] Block signal if RSI does not confirm
            if 'buy_signal' in regime and rsi_val != 'oversold':
                regime = 'neutral' # Blocked by RSI
            elif 'sell_signal' in regime and rsi_val != 'overbought':
                regime = 'neutral' # Blocked by RSI

            if rsi_val != 'neutral' and rsi_val != 'waiting':
                 regime += f" | {rsi_val.upper()} ({rsi_num:.1f})"
            elif rsi_val == 'neutral':
                 regime += f" | RSI: {rsi_num:.1f}"
        
        if self.filter_strategy and 'BB' in self.filter_strategy.name:
            pct_b = getattr(self.filter_strategy, 'last_pct_b', 0.5)
            regime += f" | BB%: {pct_b:.2f}"

        # Shorten display: e.g. "BUY (BB)" instead of "BUY_SIGNAL (BB(20, 2.0))"
        short_name = self.filter_strategy.name.split('(')[0] # "BB"
        status_str = f"{regime.upper()} ({short_name})"
        
        if hasattr(self.exchange, "set_market_regime"):
            self.exchange.set_market_regime(status_str)
            
        return regime

    def _get_trend_skew(self):
        """Calculate skew based on selected filter strategy."""
        
        # 0. Update Candle (Called in cycle, but ensure data exists)
        if not self.filter_strategy:
             if hasattr(self.exchange, "set_market_regime"):
                self.exchange.set_market_regime('OFF (Pure Grid)')
             return 0.0

        # 1. Detect Regime
        regime = self._detect_market_regime()
        
        if regime == 'waiting':
             # Treat waiting as neutral - allow grid trading while ML initializes
             regime = 'neutral'
             
        if regime == 'ranging':
            regime = 'neutral'  # Ranging is effectively neutral for grid MM 
        
        if len(self.price_history) < 20: return 0.0
        short_ma = statistics.mean(self.price_history[-10:])
        long_ma = statistics.mean(self.price_history[-20:])
        
        if long_ma == 0: return 0.0
        diff_pct = (short_ma - long_ma) / long_ma
        
        return max(-0.001, min(0.001, diff_pct * 10))

    def _get_volatility_multiplier(self):
        if len(self.price_history) < 20:
            return 1.0
        stdev = statistics.stdev(self.price_history[-20:])
        mean_price = statistics.mean(self.price_history[-20:])
        vol_pct = stdev / mean_price
        base_vol = 0.0001 
        multiplier = max(1.0, vol_pct / base_vol)
        return min(multiplier, 5.0)

    def _calculate_dynamic_spread(self):
        """
        Avellaneda-Stoikov Optimal Spread Calculation.
        
        Formula: δ = γ × σ² × (T-t) + (2/γ) × ln(1 + γ/κ)
        
        Returns spread as a fraction (e.g., 0.005 = 0.5%)
        """
        import math
        
        as_conf = Config.get("strategy", "avellaneda_stoikov", {})
        if not as_conf.get('enabled', False):
            # Fallback to legacy ATR-based spread
            return self._calculate_legacy_spread()
        
        # A&S Parameters (base from config)
        gamma = float(as_conf.get('gamma', 0.3))  # Risk aversion
        kappa = float(as_conf.get('kappa', 1.5))  # Order book liquidity
        min_spread = float(as_conf.get('min_spread', 0.001))
        max_spread = float(as_conf.get('max_spread', 0.02))
        session_hours = float(as_conf.get('session_length_hours', 24))
        
        
        # ML Regime Override: Use blended params if GMM probability available
        if self.regime_detector and self.regime_detector.is_fitted:
            try:
                # Use live Binance data for probability-based prediction
                probs = self.regime_detector.predict_live_proba(symbol="BTCUSDT")
                
                if probs:
                    # Determine dominant regime for logging/display
                    dominant_regime = max(probs, key=probs.get)
                    self.current_ml_regime = dominant_regime
                    
                    # Initialize blended params
                    blended = {
                        'gamma': 0.0, 'kappa': 0.0, 'skew_factor': 0.0,
                        'price_tolerance': 0.0, 'grid_spacing': 0.0,
                        'order_size_mult': 0.0, 'grid_layers': 0.0,
                        'max_position_mult': 0.0
                    }
                    
                    # Calculate weighted average
                    for regime, prob in probs.items():
                        r_params = self.regime_detector.get_params_for_regime(regime)
                        blended['gamma'] += r_params.get('gamma', 0) * prob
                        blended['kappa'] += r_params.get('kappa', 0) * prob
                        blended['skew_factor'] += r_params.get('skew_factor', 0) * prob
                        blended['price_tolerance'] += r_params.get('price_tolerance', 0) * prob
                        blended['grid_spacing'] += r_params.get('grid_spacing', 0) * prob
                        blended['order_size_mult'] += r_params.get('order_size_mult', 0) * prob
                        blended['grid_layers'] += r_params.get('grid_layers', 0) * prob
                        blended['max_position_mult'] += r_params.get('max_position_mult', 0) * prob
                    
                    # Apply blended params
                    gamma = blended['gamma']
                    kappa = blended['kappa']
                    
                    self._last_gamma = gamma
                    self._last_kappa = kappa
                    self._ml_skew_factor = blended['skew_factor']
                    self._ml_price_tolerance = blended['price_tolerance']
                    self._ml_grid_spacing = blended['grid_spacing']
                    self._ml_order_size_mult = blended['order_size_mult']
                    self._ml_grid_layers = round(blended['grid_layers'])  # Integer for layers
                    self._ml_max_position_mult = blended['max_position_mult']
                    
                    # Format probability string for log
                    prob_str = " ".join([f"{k[:2]}={v:.2f}" for k, v in probs.items() if v > 0.05])
                    
                    self.logger.info(f"🧠 ML: {dominant_regime}({probs[dominant_regime]:.2f}) | {prob_str}")
                    self.logger.info(f"   ↳ Blended: γ={gamma:.2f} κ={kappa:.0f} grid={self._ml_grid_layers}x{self._ml_grid_spacing*100:.2f}%")
                else:
                    self.current_ml_regime = "unknown"
                    
            except Exception as e:
                self.logger.debug(f"ML regime prediction failed: {e}")
                import traceback
                traceback.print_exc()
        
        # Adaptive Tuner Override: Fine-tune based on recent performance
        if self.adaptive_tuner:
            try:
                adaptive_params = self.adaptive_tuner.get_params()
                gamma = adaptive_params['gamma']
                kappa = adaptive_params['kappa']
                self.logger.debug(f"Adaptive Override: γ={gamma}, κ={kappa}")
            except Exception as e:
                self.logger.debug(f"Adaptive tuner failed: {e}")
        
        # Calculate volatility (σ) - annualized returns std
        sigma = self._calculate_volatility_sigma()
        
        # Time factor (T-t): Normalized to session length
        # For 24/7 crypto, we use a rolling window approach
        # t/T approaches 1 as we near "session end" (less aggressive)
        # We keep (T-t) relatively stable at 0.5 for continuous trading
        time_factor = 0.5  # Neutral time factor for crypto markets
        
        # Optimal Spread Formula
        try:
            # Term 1: Risk-adjusted volatility component
            # Scale σ by 100 so micro-volatility has meaningful impact
            # Without scaling: 0.01% σ → σ²=0.000001% → negligible
            # With scaling: 0.01% × 100 = 1% → σ²=0.01% → meaningful
            sigma_scaled = sigma * 100
            term1 = gamma * (sigma_scaled ** 2) * time_factor
            
            # Term 2: Liquidity-adjusted component (base spread)
            term2 = (2 / gamma) * math.log(1 + gamma / kappa)
            
            optimal_spread = term1 + term2
            
            # Clamp to configured bounds
            optimal_spread = max(min_spread, min(max_spread, optimal_spread))
            
            self.logger.debug(f"A&S Spread: σ={sigma:.4f}, γ={gamma}, κ={kappa} → δ={optimal_spread:.4f}")
            
        except Exception as e:
            self.logger.warning(f"A&S spread calc error: {e}, using base spread")
            optimal_spread = self.base_spread
        
        # Phase 7: Hybrid Volatility Strategy v3.10.0
        vol_adapt_conf = Config.get("strategy", "volatility_adaptation", {})
        if vol_adapt_conf.get('enabled', False) and vol_adapt_conf.get('mode') == 'hybrid':
            thresholds = vol_adapt_conf.get('thresholds', {})
            low_thresh = float(thresholds.get('low', 0.0005))
            med_thresh = float(thresholds.get('medium', 0.0008))
            
            # Determine volatility regime based on sigma
            if sigma < low_thresh:
                vol_regime = 'low'
                mode_conf = vol_adapt_conf.get('low_vol_mode', {})
            else:
                vol_regime = 'high' if sigma >= med_thresh else 'medium'
                mode_conf = vol_adapt_conf.get('high_vol_mode', {})
            
            # Get mode-specific parameters
            spread_mult = float(mode_conf.get('spread_multiplier', 1.0))
            
            # Apply multiplier
            adjusted_spread = optimal_spread * spread_mult
            
            # Ensure still within bounds
            adjusted_spread = max(min_spread, min(max_spread, adjusted_spread))
            
            # Store for dashboard/logging and cycle() skip logic
            self._vol_regime = vol_regime
            self._vol_spread_mult = spread_mult
            self._vol_skip_prob = float(mode_conf.get('quote_skip_prob', 0.0))
            
            if vol_regime == 'low':
                self.logger.debug(f"📉 Hybrid LOW: σ={sigma:.4f}, spread×{spread_mult:.1f}, skip={self._vol_skip_prob:.0%}")
            
            return adjusted_spread
        
        return optimal_spread
    
    def _calculate_volatility_sigma(self) -> float:
        """
        Calculate market volatility (σ) for A&S model.
        Uses rolling standard deviation of log returns.
        """
        if len(self.price_history) < 20:
            return 0.001  # Default low volatility
        
        prices = self.price_history[-20:]
        
        # Calculate log returns
        returns = []
        for i in range(1, len(prices)):
            if prices[i-1] > 0:
                log_return = (prices[i] - prices[i-1]) / prices[i-1]
                returns.append(log_return)
        
        if len(returns) < 5:
            return 0.001
        
        # Standard deviation of returns (volatility proxy)
        sigma = statistics.stdev(returns)
        
        # Scale to reasonable range (prevent extreme values)
        sigma = max(0.0001, min(0.1, sigma))
        
        return sigma
    
    def _calculate_reservation_price(self, mid_price: float, inventory_ratio: float) -> float:
        """
        Avellaneda-Stoikov Reservation Price.
        
        Formula: r = s - q × γ × σ² × (T-t)
        
        This shifts the reference price based on inventory position.
        - Long position (q > 0): Reservation price < mid price (encourage selling)
        - Short position (q < 0): Reservation price > mid price (encourage buying)
        """
        as_conf = Config.get("strategy", "avellaneda_stoikov", {})
        if not as_conf.get('enabled', False):
            return mid_price
        
        gamma = float(as_conf.get('gamma', 0.3))
        sigma = self._calculate_volatility_sigma()
        time_factor = 0.5  # Neutral for 24/7 markets
        
        # q = inventory ratio (-1 to +1 range)
        q = inventory_ratio
        
        # Reservation price adjustment
        adjustment = q * gamma * (sigma ** 2) * time_factor * mid_price
        
        reservation_price = mid_price - adjustment
        
        self.logger.debug(f"A&S Reservation: mid={mid_price:.2f}, q={q:.3f} → r={reservation_price:.2f}")
        
        return reservation_price
    
    def _calculate_legacy_spread(self):
        """Calculate spread based on ATR (Volatility) with USD limits."""
        conf = Config.get("strategy", "dynamic_spread", {})
        if not conf.get('enabled', False):
            return self.base_spread
            
        if len(self.candles) < 20:
            return self.base_spread
            
        # Current Price (Approx from close)
        current_price = self.candles['close'].iloc[-1]
        
        # Calculate ATR(14)
        high = self.candles['high']
        low = self.candles['low']
        close = self.candles['close']
        
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(14).mean().iloc[-1]
        
        # ATR Logic
        ref_atr = conf.get('reference_atr', 100.0)
        multiplier = atr / ref_atr
        multiplier = max(0.5, min(multiplier, 3.0)) # 0.5x ~ 3.0x
        
        raw_spread_pct = self.base_spread * multiplier
        raw_spread_usd = current_price * raw_spread_pct
        
        # Enforce User Limits ($30 ~ $200) -> Adjusted to $100 min per observation
        min_usd = 100.0
        max_usd = 200.0
        
        final_usd = max(min_usd, min(max_usd, raw_spread_usd))
        final_pct = final_usd / current_price if current_price > 0 else self.base_spread
        
        return final_pct
    
    async def check_drawdown(self):
        """Check Max Drawdown safety stop."""
        status = await self.exchange.get_account_summary()
        current_equity = status.get('total_equity', 0.0)
        
        if self.initial_equity is None:
            self.initial_equity = current_equity
            self.logger.info(f"Initial Equity Set: {self.initial_equity}")
            return True
            
        # Drawdown check
        dd_pct = (self.initial_equity - current_equity) / self.initial_equity
        if dd_pct > self.risk_manager.max_drawdown:
            self.logger.critical(f"MAX DRAWDOWN REACHED! {dd_pct*100:.2f}% >= {self.risk_manager.max_drawdown*100:.2f}%")
            self.logger.critical("STOPPING BOT & CLOSING POSITIONS.")
            
            # Close all
            await self.exchange.cancel_all_orders(self.symbol)
            await self.exchange.close_position(self.symbol)
            
            self.is_active = False # Stop Bot
            return False
            
        return True

    async def cycle(self):
        # 1. Get Data
        orderbook = await self.exchange.get_orderbook(self.symbol)
        position = await self.exchange.get_position(self.symbol)
        
        if not orderbook or 'bids' not in orderbook:
            return
        
        # Phase 7 v3.10.0: Hybrid Quote Skip
        # Use _vol_skip_prob set by _calculate_dynamic_spread()
        vol_adapt_conf = Config.get("strategy", "volatility_adaptation", {})
        if vol_adapt_conf.get('enabled', False) and vol_adapt_conf.get('mode') == 'hybrid':
            import random
            skip_prob = getattr(self, '_vol_skip_prob', 0.0)
            vol_regime = getattr(self, '_vol_regime', 'high')
            
            if random.random() < skip_prob:
                # Skip this cycle - don't place new quotes
                self.logger.debug(f"📉 Hybrid Skip: {vol_regime} vol, skipping cycle (p={skip_prob:.0%})")
                return

        try:
            bids = orderbook['bids']
            asks = orderbook['asks']
            best_bid = float(bids[0]['price']) if isinstance(bids[0], dict) else float(bids[0][0])
            best_ask = float(asks[0]['price']) if isinstance(asks[0], dict) else float(asks[0][0])
            mid_price = (best_bid + best_ask) / 2
        except:
            return
        
        self._update_history(mid_price)
        self._update_candle(mid_price, time.time())
        current_pos_qty = position.get('amount', 0.0)
        self.inventory = current_pos_qty
        
        # --- Circuit Breaker Check ---
        unrealized_pnl = position.get('unrealizedPnL', 0.0)
        if unrealized_pnl < -self.max_loss_usd:
            self.logger.critical(f"🚨 CIRCUIT BREAKER: Loss ${abs(unrealized_pnl):.2f} exceeds max ${self.max_loss_usd:.2f}")
            self.logger.critical("Cancelling all orders and LIQUIDATING position!")
            
            # 1. Cancel Open Orders (Prevent stacking)
            await self.exchange.cancel_all_orders(self.symbol)
            
            # 2. Liquidate Position (Market Close)
            await self.exchange.close_position(self.symbol)
            
            # 3. Stop Logic
            self.is_active = False
            return
        
        # Log Status
        rsi_status = self.rsi_filter.analyze(self.candles)
        # 1. Detect Regime & Apply Latch
        current_regime = self._detect_market_regime()
        rsi_status = self.rsi_filter.analyze(self.candles) # Update last_rsi
        last_rsi = getattr(self.rsi_filter, 'last_rsi', 50.0)
        
        # Reset Latch if flat
        if current_pos_qty == 0:
            self.latched_regime = None
        
        # Reset Latch based on RSI thresholds (Hysteresis)
        # User Rule: Keep Buy Latch if RSI <= 40 / Keep Sell Latch if RSI >= 60
        if self.latched_regime == 'buy_signal' and last_rsi > 40:
             self.latched_regime = None # Release to Neutral
        elif self.latched_regime == 'sell_signal' and last_rsi < 60:
             self.latched_regime = None # Release to Neutral

        # Update Latch if new signal appears (Overrides Reset)
        if 'buy_signal' in current_regime or 'sell_signal' in current_regime:
            self.latched_regime = current_regime
            
        # Use Latched Regime if Current is Neutral but we have a Latch
        effective_regime = current_regime
        if current_regime == 'neutral' and self.latched_regime:
             effective_regime = self.latched_regime
             # Add visual indicator
             effective_regime += " (Latched)"
             
        # Log Status
        self.logger.info(f"Pos: {current_pos_qty:.4f} | Mid: {mid_price:.2f} | Regime: {effective_regime} | RSI: {last_rsi:.1f} | Equity: {position.get('unrealizedPnL', 0):.2f}")

        # Sync to Paper Exchange
        if hasattr(self.exchange, 'set_market_regime'):
            self.exchange.set_market_regime(effective_regime)

        # --- v4.0 ML Integration ---
        # Reset ML multipliers
        ml_spread_mult = 1.0
        ml_size_mult = 1.0
        ml_bid_offset = 0
        ml_ask_offset = 0
        
        if self.strategy_v4 and len(self.candles) >= 60:
            try:
                # 1. Compute Features
                # We need to ensure we pass a DataFrame with enough history
                # self.candles has ['timestamp', 'open', 'high', 'low', 'close']
                # Feature engineer expects this format plus volume (if available)
                df_candles = self.candles.copy()
                if 'volume' not in df_candles.columns:
                    df_candles['volume'] = 1000.0 # Dummy volume if not available
                    
                # Fix dtypes
                for col in ['open', 'high', 'low', 'close', 'volume']:
                    df_candles[col] = df_candles[col].astype(float)
                df_candles['timestamp'] = pd.to_datetime(df_candles['timestamp'], unit='s')
                df_candles.set_index('timestamp', inplace=True)
                
                features = self.fe.compute_features(df_candles)
                
                if not features.empty:
                    last_row = features.iloc[-1]
                    
                    # 2. Get v4 Adjustments
                    adj = self.strategy_v4.get_adjustments(last_row, effective_regime)
                    
                    ml_spread_mult = adj['spread_mult']
                    ml_size_mult = adj['size_mult']
                    ml_bid_offset = adj['bid_layers']
                    ml_ask_offset = adj['ask_layers']
                    
                    # Store for Dashboard
                    self.last_ml_metrics = {
                        'vol_value': self.strategy_v4.predict_volatility(last_row),
                        'confidence': 0.0,
                        'spread_mult': ml_spread_mult,
                        'size_mult': ml_size_mult,
                        'direction': adj.get('direction'),
                        'vol_regime': adj.get('vol_regime')
                    }
                    if adj.get('direction'):
                        _, conf = self.strategy_v4.predict_direction(last_row)
                        self.last_ml_metrics['confidence'] = conf
                    
                    # Update regime string for dashboard
                    self.current_ml_regime = f"v4:{adj['vol_regime']}"
                    if adj.get('direction'):
                        self.current_ml_regime += f"_{adj['direction']}"
                    
                    # Log significant adjustments
                    if ml_spread_mult != 1.0 or ml_size_mult != 1.0 or adj.get('direction'):
                        self.logger.info(f"🤖 ML v4: {adj['vol_regime'].upper()} | Dir {adj.get('direction')} | Spread x{ml_spread_mult:.2f} | Size x{ml_size_mult:.2f}")

            except Exception as e:
                self.logger.error(f"v4 ML Error: {e}")
                self.last_ml_metrics = {}
        elif self.strategy_v4:
            # Report initialization progress
            self.last_ml_metrics = {
                'vol_value': 0,
                'confidence': 0,
                'spread_mult': 1.0,
                'size_mult': 1.0,
                'direction': None,
                'vol_regime': f'init ({len(self.candles)}/60)'
            }
            # Also update regime string to show init
            self.current_ml_regime = f"v4:init({len(self.candles)}/60)"

        # --- v5.0 Order Flow Integration ---
        of_spread_mult = 1.0
        of_bid_size_mult = 1.0
        of_ask_size_mult = 1.0
        of_metrics = {}

        if self.order_flow:
            try:
                # Calculate OBI/Toxicity
                # orderbook is available from line 681
                of_adj = self.order_flow.get_adjustment_factors(orderbook, trade=None)
                
                of_spread_mult = of_adj['spread_mult']
                of_bid_size_mult = of_adj['bid_size_mult']
                of_ask_size_mult = of_adj['ask_size_mult']
                of_metrics = of_adj['metrics']
                
                if of_spread_mult > 1.0 or of_bid_size_mult < 1.0 or of_ask_size_mult < 1.0:
                    self.logger.info(f"🌊 Order Flow: OBI={of_metrics.get('obi',0):.2f} Tox={of_metrics.get('toxicity',0):.2f} | Spr x{of_spread_mult:.2f} Bid x{of_bid_size_mult:.2f} Ask x{of_ask_size_mult:.2f}")

            except Exception as e:
                self.logger.error(f"Order Flow Error: {e}")

        # --- v5.1 Funding Rate Integration ---
        fr_bid_size_mult = 1.0
        fr_ask_size_mult = 1.0
        fr_freeze_orders = False
        fr_metrics = {}

        if self.funding_monitor and self.funding_integrator:
            try:
                # Get funding analysis
                fr_analysis = self.funding_monitor.analyze_opportunity()
                
                # Get adjustment
                fr_adj = self.funding_integrator.get_adjustment(
                    fr_analysis,
                    current_inventory=current_pos_qty,
                    max_inventory=self.max_position_usd / mid_price if mid_price > 0 else 1.0
                )
                
                fr_bid_size_mult = fr_adj['bid_size_mult']
                fr_ask_size_mult = fr_adj['ask_size_mult']
                fr_freeze_orders = fr_adj['freeze_orders']
                fr_metrics = fr_adj
                
                if fr_analysis['opportunity']:
                    self.logger.info(f"💰 Funding: {fr_analysis['direction'].upper()} bias | Yield {fr_analysis['annual_yield']:.1f}%/yr | {fr_analysis['hours_to_funding']:.1f}h to funding")
                
            except Exception as e:
                self.logger.error(f"Funding Rate Error: {e}")

        # 2. Calculate Parameters
        # Fix Division by Zero: Use calculated qty based on price
        estimated_qty = (self.order_size_usd / mid_price) if mid_price > 0 else self.amount
        if estimated_qty <= 0: estimated_qty = 0.001 # Fallback
        
        # ML-adjusted skew sensitivity: higher skew_factor = more aggressive inventory adjustment
        ml_skew_factor = getattr(self, '_ml_skew_factor', 0.005)  # Default 0.5%
        skew_multiplier = ml_skew_factor / 0.005  # Normalize to base 0.5%
        max_skew_qty = estimated_qty * 20 / skew_multiplier  # Higher skew = lower threshold
        inventory_ratio = max(-1.0, min(1.0, current_pos_qty / max_skew_qty))
        
        # Determine Spread using A&S formula (or legacy)
        final_spread = self._calculate_dynamic_spread()
        
        # Apply ML Spread Multiplier
        # Apply ML Spread Multiplier
        final_spread *= ml_spread_mult
        
        # Apply Order Flow Spread Multiplier (v5.0)
        final_spread *= of_spread_mult
        
        # === Adverse Selection Adjustment (Phase 1.2) ===
        if self.as_detector and self.as_spread_add_bps > 0:
            # Add AS-detected spread increase
            final_spread += self.as_spread_add_bps / 10000  # bps to fraction
        
        # --- A&S Reservation Price ---
        # Replace simple skew with Avellaneda-Stoikov reservation price
        # This provides more sophisticated inventory-based price adjustment
        reservation_price = self._calculate_reservation_price(mid_price, inventory_ratio)
        
        target_bid = round_tick_size(reservation_price * (1 - final_spread / 2), self.tick_size)
        target_ask = round_tick_size(reservation_price * (1 + final_spread / 2), self.tick_size)
        
        # --- Push A&S Metrics to Exchange for Dashboard ---
        if hasattr(self.exchange, 'set_as_metrics'):
            # Get actual gamma/kappa from calculation context
            actual_gamma = gamma if 'gamma' in dir() else 1.0
            actual_kappa = kappa if 'kappa' in dir() else 1000
            
            # Get adaptive tuner metrics if available
            adaptive_metrics = {}
            if self.adaptive_tuner:
                adaptive_metrics = self.adaptive_tuner.get_display_metrics()
            
            self.exchange.set_as_metrics({
                "reservation_price": round(reservation_price, 2),
                "optimal_spread": round(final_spread * 100, 4),
                "volatility_sigma": round(self._calculate_volatility_sigma() * 100, 4),
                "gamma": actual_gamma,
                "kappa": actual_kappa,
                "ml_regime": getattr(self, 'current_ml_regime', 'disabled'),
                "recent_pnl": adaptive_metrics.get('recent_pnl', 0),
                "win_rate": adaptive_metrics.get('win_rate', 0),
                "adjustments": adaptive_metrics.get('adjustments', 0)
            })

        # Dropdown Check (Pass)

        # --- 3. Entry Guard / Profit Protection (Anchor) ---
        entry_price = position.get('entryPrice', 0.0)
        
        if self.entry_anchor_mode and current_pos_qty != 0 and entry_price > 0:
            # Regime-Based Stop Loss Logic
            loss_tolerance = 0.0 # Default: Zero Tolerance (Strictly Profit Only)
            
            # If Signal detected (Trend) OR Latched, loosen stop loss to allow escape
            if 'buy_signal' in effective_regime or 'sell_signal' in effective_regime:
                 loss_tolerance = 0.005 # Allow 0.5% loss to cut bad positions in trend
            
            # Neutral: Strict "Hold" (loss_tolerance ~ 0)
            
            if current_pos_qty > 0: # Long
                 # DCA: Buy if price < Entry
                 target_bid = min(target_bid, entry_price * 0.9995)
                 # Anchor: Limit Sell Price
                 limit_price = entry_price * (1.0 - loss_tolerance) + (entry_price * 0.0005) # Adjust slightly
                 target_ask = max(target_ask, entry_price * (1 - loss_tolerance)) 
                 
            elif current_pos_qty < 0: # Short
                 # DCA: Sell if price > Entry
                 target_ask = max(target_ask, entry_price * 1.0005)
                 # Anchor: Limit Buy Price
                 target_bid = min(target_bid, entry_price * (1 + loss_tolerance))

        # --- 4. Permission Flags ---
        # Always allow both sides for grid trading (Signal Boost strategy)
        # Only RSI extreme conditions block orders
        allow_buy = rsi_status != 'overbought'
        allow_sell = rsi_status != 'oversold'
        
        # --- 4.1 Max Position Limit (ML-adjusted) ---
        # Block further accumulation when position exceeds ML-adjusted max_position_usd
        ml_max_pos_mult = getattr(self, '_ml_max_position_mult', 1.0)
        effective_max_position = self.max_position_usd * ml_max_pos_mult
        position_usd = abs(current_pos_qty) * mid_price
        if position_usd >= effective_max_position:
            if current_pos_qty > 0:
                allow_buy = False  # Long position at limit → block buying
                self.logger.debug(f"Max position reached: ${position_usd:.0f} >= ${effective_max_position:.0f}, blocking BUY")
            elif current_pos_qty < 0:
                allow_sell = False  # Short position at limit → block selling
                self.logger.debug(f"Max position reached: ${position_usd:.0f} >= ${effective_max_position:.0f}, blocking SELL")
        
        # DEBUG: Log Allow Status
        self.logger.info(f"DEBUG: AllowBuy={allow_buy} AllowSell={allow_sell} Pos=${position_usd:.0f}/{(effective_max_position):.0f} (x{ml_max_pos_mult})")
                
        # --- 5. Generate Grid Orders ---
        buy_orders = []
        sell_orders = []
        
        # Grid Spacing: Use ML-adjusted value or fallback to 0.12%
        ml_grid = getattr(self, '_ml_grid_spacing', 0.0012)
        layer_spacing = max(final_spread, ml_grid)
        
        # Order Size Multiplier from ML Regime
        ml_size_mult_legacy = getattr(self, '_ml_order_size_mult', 1.0)
        # Combine legacy ML (if any) with v4 ML
        total_size_mult = ml_size_mult_legacy * ml_size_mult
        
        # Grid Layers: Use ML-adjusted count or fallback to config
        ml_grid_layers = getattr(self, '_ml_grid_layers', self.grid_layers)

        for i in range(ml_grid_layers):
            # Apply Directional Offset to Layers
            # UP trend (dir='UP'): bid_layers +1, ask_layers -1 (handled by offsets)
            # We want to shift the grid: 
            #   If bid_offset > 0: start bids closer (or add more bids?)
            #   StrategyV4 implementation: bid_layers bias means we want MORE probability to buy?
            #   Actually v4 logic: bid_layers = shift. 
            #   Let's interpret shift as: skewing the grid center.
            #   But 'market_maker.py' builds grid from mid_price outwards.
            #   Simplest impl: Adjust spacing or start index?
            
            # v4 Logic Interpretation:
            # bid_layers > 0 means we want to be more aggressive on bids.
            # We can implement this by shifting the start price for bids/asks.
            pass # just a comment placeholder
            
            # Linearly spaced grid with ML offsets
            # If bid_offset = 1 (UP trend), we want bids to be closer/more aggressive.
            #   standard: mid * (1 - spacing * (i+1))
            #   aggressive: mid * (1 - spacing * i)  <- start at mid!
            # Let's simply modulate 'i' index.
            
            eff_i_bid = max(0, i - ml_bid_offset)
            eff_i_ask = max(0, i - ml_ask_offset)
            
            bid_p = round_tick_size(target_bid * (1 - (layer_spacing * (eff_i_bid + 1))), self.tick_size)
            ask_p = round_tick_size(target_ask * (1 + (layer_spacing * (eff_i_ask + 1))), self.tick_size)
            
            # === Dynamic Order Sizing (Phase 1.1) ===
            if self.dynamic_sizer and self.order_size_usd > 0:
                # Get volatility for sizing
                volatility = self._calculate_volatility_sigma() if hasattr(self, '_calculate_volatility_sigma') else 0.01
                # Get order book depth (simplified - use default 10)
                book_depth = 10.0
                
                # Calculate asymmetric bid/ask sizes
                bid_size_usd, ask_size_usd = self.dynamic_sizer.calculate(
                    inventory=current_pos_qty * mid_price,  # Convert to USD
                    volatility=volatility,
                    book_depth=book_depth
                )
                
                # Apply ML size multiplier
                bid_size_usd *= total_size_mult
                ask_size_usd *= total_size_mult
                
                # Convert to quantity
                bid_qty = round(bid_size_usd / mid_price, 3)
                ask_qty = round(ask_size_usd / mid_price, 3)
                
                # Enforce minimums
                bid_qty = max(bid_qty, 0.001)
                ask_qty = max(ask_qty, 0.001)
            else:
                # Fallback: Fixed order size
                if self.order_size_usd > 0:
                    adjusted_size_usd = self.order_size_usd * total_size_mult
                    raw_qty = adjusted_size_usd / mid_price
                    qty = round(raw_qty, 3)
                    qty = max(qty, 0.001)
                else:
                    qty = self.amount
                bid_qty = qty
                ask_qty = qty
            
            # Apply Order Flow Size Multipliers (v5.0)
            bid_qty *= of_bid_size_mult
            ask_qty *= of_ask_size_mult
            
            # Apply Funding Rate Size Multipliers (v5.1)
            bid_qty *= fr_bid_size_mult
            ask_qty *= fr_ask_size_mult
            
            # Re-check minimums
            bid_qty = max(bid_qty, 0.001)
            ask_qty = max(ask_qty, 0.001)
            
            if allow_buy:
                buy_orders.append((bid_p, bid_qty))
            if allow_sell:
                sell_orders.append((ask_p, ask_qty))
        
        # --- 5.1 Signal Boost: Add aggressive orders when signal detected ---
        # This adds EXTRA orders close to market price in signal direction
        if 'buy_signal' in effective_regime and allow_buy:
            # Add 2 extra aggressive buy orders very close to mid price
            for i in range(2):
                aggressive_price = round_tick_size(mid_price * (1 - 0.0005 * (i + 1)), self.tick_size)  # 0.05%, 0.1% below mid
                buy_orders.append((aggressive_price, bid_qty))
            self.logger.info(f"📈 SIGNAL BOOST: +2 aggressive BUY orders near ${mid_price:.0f}")
        elif 'sell_signal' in effective_regime and allow_sell:
            # Add 2 extra aggressive sell orders very close to mid price
            for i in range(2):
                aggressive_price = round_tick_size(mid_price * (1 + 0.0005 * (i + 1)), self.tick_size)  # 0.05%, 0.1% above mid
                sell_orders.append((aggressive_price, ask_qty))
            self.logger.info(f"📉 SIGNAL BOOST: +2 aggressive SELL orders near ${mid_price:.0f}")
            
        # --- 6. Smart Order Management (v1.9.1) ---
        # Only update orders if prices changed significantly (>0.5% tolerance for stable/conservative mode)
        PRICE_TOLERANCE = 0.005  # 0.5%
        
        # Get existing orders (Support both GRVT API format and Paper Exchange format)
        existing_orders = await self.exchange.get_open_orders(self.symbol)
        existing_buys = {}
        existing_sells = {}
        
        for o in existing_orders:
            order_id = o.get('order_id', o.get('id'))
            # Paper Exchange format: direct price/side
            if 'price' in o and 'side' in o:
                price = float(o.get('price', 0))
                if o.get('side') == 'buy':
                    existing_buys[order_id] = price
                else:
                    existing_sells[order_id] = price
            # GRVT API format: legs[0] contains order details
            elif o.get('legs'):
                leg = o['legs'][0]
                price = float(leg.get('limit_price', 0))
                if leg.get('is_buying_asset'):
                    existing_buys[order_id] = price
                else:
                    existing_sells[order_id] = price
        
        new_buy_prices = set(p for p, q in buy_orders)
        new_sell_prices = set(p for p, q in sell_orders)
        
        # Check if orders need update (use ML-adjusted tolerance)
        PRICE_TOLERANCE = getattr(self, '_ml_price_tolerance', 0.001)  # ML-adjusted or 0.1% default
        
        def prices_match(existing_prices, new_prices, tolerance):
            # Strict count check causes constant updates if any order is filled/cancelled externally
            # if len(existing_prices) != len(new_prices): return False 
            
            # Relaxed Check: Ensure all NEW orders are represented in EXISTING orders within tolerance
            # Does not force update if we have EXTRA existing orders (e.g. manual trades)
            # But here we want to maintain EXACT grid
            if len(existing_prices) != len(new_prices):
                 return False
                 
            if not existing_prices or not new_prices:
                return len(existing_prices) == len(new_prices)
            existing_set = set(existing_prices.values())
            for new_p in new_prices:
                matched = any(abs(new_p - old_p) / old_p < tolerance for old_p in existing_set if old_p > 0)
                if not matched:
                    return False
            return True
        
        buys_need_update = not prices_match(existing_buys, new_buy_prices, PRICE_TOLERANCE)
        sells_need_update = not prices_match(existing_sells, new_sell_prices, PRICE_TOLERANCE)
        
        # Debug: Log order comparison
        if buys_need_update or sells_need_update:
            self.logger.info(f"Order update needed: buys={buys_need_update}, sells={sells_need_update}")
            self.logger.info(f"Count diff: Buys {len(existing_buys)} vs {len(new_buy_prices)} | Sells {len(existing_sells)} vs {len(new_sell_prices)}")
            # self.logger.debug(f"New buy prices: {sorted(list(new_buy_prices))}")
        
        # Skip update if no new orders to place (avoid cancel+empty replace loop)
        if not buy_orders and not sell_orders:
            pass  # Keep existing orders
        elif buys_need_update or sells_need_update:
            # Cancel and replace only if needed
            self.logger.info(f"Updating orders: buys_changed={buys_need_update}, sells_changed={sells_need_update}")
            await self.exchange.cancel_all_orders(self.symbol)
            
            for p, q in buy_orders:
                await self.exchange.place_limit_order(self.symbol, 'buy', p, q)
            for p, q in sell_orders:
                await self.exchange.place_limit_order(self.symbol, 'sell', p, q)
        # else: Keep existing orders (no action needed)
        
        # Push ML metrics to dashboard
        if hasattr(self.exchange, 'set_as_metrics'):
            ml_regime = getattr(self, 'current_ml_regime', 'unknown')
            adaptive_metrics = {}
            if self.adaptive_tuner:
                try:
                    adaptive_metrics = self.adaptive_tuner.get_display_metrics()
                except:
                    pass
            
            ml_metrics = getattr(self, 'last_ml_metrics', {})
            
            self.exchange.set_as_metrics({
                "reservation_price": mid_price,
                "optimal_spread": final_spread if 'final_spread' in dir() else 0.002,
                "volatility_sigma": 0.01,
                "gamma": getattr(self, '_last_gamma', 1.0),
                "kappa": getattr(self, '_last_kappa', 1000),
                "ml_regime": ml_regime,
                "recent_pnl": adaptive_metrics.get('recent_pnl', 0),
                "win_rate": adaptive_metrics.get('win_rate', 50),
                "adjustments": adaptive_metrics.get('adjustments', 0),
                # v4 Metrics
                "ml_vol_value": ml_metrics.get('vol_value', 0),
                "ml_confidence": ml_metrics.get('confidence', 0),
                "ml_spread_mult": ml_metrics.get('spread_mult', 1.0),
                "ml_spread_mult": ml_metrics.get('spread_mult', 1.0),
                "ml_size_mult": ml_metrics.get('size_mult', 1.0),
                # v5 Metrics
                "of_obi": of_metrics.get('obi', 0.0),
                "of_toxicity": of_metrics.get('toxicity', 0.0),
                "ml_direction": ml_metrics.get('direction'),
                "ml_vol_regime": ml_metrics.get('vol_regime')
            })
        
        # Save status for dashboard (Live mode)
        if hasattr(self.exchange, 'save_live_status'):
            open_orders = await self.exchange.get_open_orders(self.symbol)
            status = await self.exchange.get_account_summary()
            self.exchange.save_live_status(
                symbol=self.symbol,
                mid_price=mid_price,
                regime=effective_regime,
                position=position,
                open_orders=open_orders,
                equity=status.get('total_equity', 0.0)
            )
            # Also fetch and save trade history for dashboard
            # if hasattr(self.exchange, 'fetch_and_save_trades'):
            #     self.exchange.fetch_and_save_trades(self.symbol)

    async def run(self):
        self.logger.info("Strategy Started")
        self.is_running = True
        self.is_active = True # Force Auto-Start
        while self.is_running:
            try:
                cmd_res = await self.check_command()
                if cmd_res == 'restart':
                    return 'restart' # Signal main to restart

                if not self.is_active:
                    await asyncio.sleep(2)
                    continue
                
                if not await self.check_drawdown():
                    continue

                await self.cycle()
            except Exception as e:
                self.logger.error(f"Error in strategy cycle: {e}")
            
            await asyncio.sleep(self.refresh_interval)
        return 'stop'
