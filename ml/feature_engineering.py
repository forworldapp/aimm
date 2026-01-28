"""
Feature Engineering Pipeline for LightGBM Direction Predictor
v4.0 - Modular, Multi-Exchange Compatible Design

This module provides a modular, exchange-agnostic feature engineering pipeline
that can be easily ported to different exchanges and trading products.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import numpy as np
import pandas as pd


@dataclass
class FeatureConfig:
    """Configuration for feature engineering."""
    # Price features
    return_windows: List[int] = field(default_factory=lambda: [1, 5, 15, 60])
    volatility_windows: List[int] = field(default_factory=lambda: [20, 60])
    
    # Technical indicators
    rsi_periods: List[int] = field(default_factory=lambda: [7, 14])
    bb_period: int = 20
    bb_std: float = 2.0
    macd_params: Tuple[int, int, int] = (12, 26, 9)
    ema_periods: Tuple[int, int] = (9, 21)
    atr_period: int = 14
    adx_period: int = 14
    
    # Microstructure (optional, exchange-dependent)
    use_orderbook: bool = True
    orderbook_levels: int = 5
    
    # Trade flow (optional)
    use_trade_flow: bool = True
    cvd_windows: List[int] = field(default_factory=lambda: [1, 5])
    
    # Derivatives (optional, crypto-specific)
    use_derivatives: bool = True
    
    # Time features
    use_time_features: bool = True
    trading_sessions: Dict[str, Tuple[int, int]] = field(default_factory=lambda: {
        'asia': (1, 9),     # UTC
        'europe': (7, 16),
        'us': (13, 22),
    })
    
    # HMM regime (if available)
    use_regime_features: bool = True


class BaseFeatureProvider(ABC):
    """
    Abstract base class for exchange-specific feature providers.
    
    Subclass this for each exchange/product to handle data format differences.
    """
    
    @abstractmethod
    def get_ohlcv(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract OHLCV columns in standard format."""
        pass
    
    @abstractmethod
    def get_orderbook_imbalance(self, df: pd.DataFrame) -> Optional[pd.Series]:
        """Get orderbook imbalance if available."""
        pass
    
    @abstractmethod
    def get_trade_flow(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Get trade flow metrics (buy/sell ratio, CVD, etc.)."""
        pass
    
    @abstractmethod
    def get_derivatives_data(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Get derivatives data (funding, OI, etc.)."""
        pass


class BinanceFeatureProvider(BaseFeatureProvider):
    """Feature provider for Binance Futures."""
    
    def get_ohlcv(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize Binance OHLCV format."""
        return df[['open', 'high', 'low', 'close', 'volume']].copy()
    
    def get_orderbook_imbalance(self, df: pd.DataFrame) -> Optional[pd.Series]:
        """Calculate orderbook imbalance from Binance data."""
        if 'bid_volume' in df.columns and 'ask_volume' in df.columns:
            total = df['bid_volume'] + df['ask_volume']
            return (df['bid_volume'] - df['ask_volume']) / total.replace(0, np.nan)
        return None
    
    def get_trade_flow(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Get trade flow from Binance aggTrades."""
        result = pd.DataFrame(index=df.index)
        
        if 'taker_buy_volume' in df.columns:
            taker_sell = df['volume'] - df['taker_buy_volume']
            total = df['volume'].replace(0, np.nan)
            result['buy_sell_ratio'] = df['taker_buy_volume'] / taker_sell.replace(0, np.nan)
            result['cvd_raw'] = df['taker_buy_volume'] - taker_sell
            return result
        return None
    
    def get_derivatives_data(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Get Binance futures derivatives data."""
        result = pd.DataFrame(index=df.index)
        
        if 'funding_rate' in df.columns:
            result['funding_rate'] = df['funding_rate']
        if 'open_interest' in df.columns:
            result['open_interest'] = df['open_interest']
            result['oi_change_1h'] = df['open_interest'].pct_change(60) * 100
        if 'long_short_ratio' in df.columns:
            result['long_short_ratio'] = df['long_short_ratio']
            
        return result if len(result.columns) > 0 else None


class GRVTFeatureProvider(BaseFeatureProvider):
    """Feature provider for GRVT Exchange."""
    
    def get_ohlcv(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize GRVT OHLCV format."""
        # GRVT uses same column names
        return df[['open', 'high', 'low', 'close', 'volume']].copy()
    
    def get_orderbook_imbalance(self, df: pd.DataFrame) -> Optional[pd.Series]:
        """Calculate orderbook imbalance from GRVT data."""
        if 'best_bid_size' in df.columns and 'best_ask_size' in df.columns:
            total = df['best_bid_size'] + df['best_ask_size']
            return (df['best_bid_size'] - df['best_ask_size']) / total.replace(0, np.nan)
        return None
    
    def get_trade_flow(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Get trade flow from GRVT trades."""
        # GRVT specific implementation
        return None
    
    def get_derivatives_data(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Get GRVT derivatives data."""
        result = pd.DataFrame(index=df.index)
        
        if 'funding_rate' in df.columns:
            result['funding_rate'] = df['funding_rate']
            
        return result if len(result.columns) > 0 else None


class FeatureEngineer:
    """
    Main feature engineering class.
    
    Computes all features for LightGBM direction predictor.
    Exchange-agnostic through provider pattern.
    """
    
    def __init__(
        self,
        config: Optional[FeatureConfig] = None,
        provider: Optional[BaseFeatureProvider] = None
    ):
        self.config = config or FeatureConfig()
        self.provider = provider or BinanceFeatureProvider()
        self._feature_names: List[str] = []
    
    @property
    def feature_names(self) -> List[str]:
        """Get list of computed feature names."""
        return self._feature_names
    
    def compute_features(
        self,
        df: pd.DataFrame,
        regime_probs: Optional[Dict[str, float]] = None
    ) -> pd.DataFrame:
        """
        Compute all features for the given OHLCV data.
        
        Args:
            df: DataFrame with OHLCV data
            regime_probs: Optional HMM regime probabilities
            
        Returns:
            DataFrame with computed features
        """
        features = pd.DataFrame(index=df.index)
        
        # Get standardized OHLCV
        ohlcv = self.provider.get_ohlcv(df)
        
        # 1. Price features
        price_features = self._compute_price_features(ohlcv)
        features = pd.concat([features, price_features], axis=1)
        
        # 2. Technical indicators
        tech_features = self._compute_technical_features(ohlcv)
        features = pd.concat([features, tech_features], axis=1)
        
        # 3. Microstructure features (optional)
        if self.config.use_orderbook:
            micro_features = self._compute_microstructure_features(df, ohlcv)
            if micro_features is not None:
                features = pd.concat([features, micro_features], axis=1)
        
        # 4. Trade flow features (optional)
        if self.config.use_trade_flow:
            flow_features = self._compute_trade_flow_features(df)
            if flow_features is not None:
                features = pd.concat([features, flow_features], axis=1)
        
        # 5. Derivatives features (optional)
        if self.config.use_derivatives:
            deriv_features = self._compute_derivatives_features(df)
            if deriv_features is not None:
                features = pd.concat([features, deriv_features], axis=1)
        
        # 6. Time features
        if self.config.use_time_features:
            time_features = self._compute_time_features(df)
            features = pd.concat([features, time_features], axis=1)
        
        # 7. Regime features (if provided)
        if self.config.use_regime_features and regime_probs is not None:
            regime_features = self._compute_regime_features(df, regime_probs)
            features = pd.concat([features, regime_features], axis=1)
        
        # Update feature names
        self._feature_names = features.columns.tolist()
        
        return features
    
    def _compute_price_features(self, ohlcv: pd.DataFrame) -> pd.DataFrame:
        """Compute price-based features."""
        features = pd.DataFrame(index=ohlcv.index)
        close = ohlcv['close']
        
        # Returns at different windows
        for window in self.config.return_windows:
            features[f'returns_{window}m'] = close.pct_change(window) * 100
        
        # Volatility at different windows
        for window in self.config.volatility_windows:
            features[f'volatility_{window}'] = close.pct_change().rolling(window).std() * 100
        
        # High-low range
        features['high_low_range'] = (ohlcv['high'] - ohlcv['low']) / close * 100
        
        # Close to VWAP approximation (simplified)
        typical_price = (ohlcv['high'] + ohlcv['low'] + ohlcv['close']) / 3
        vwap = (typical_price * ohlcv['volume']).rolling(20).sum() / ohlcv['volume'].rolling(20).sum()
        features['close_to_vwap'] = (close - vwap) / close * 100
        
        return features
    
    def _compute_technical_features(self, ohlcv: pd.DataFrame) -> pd.DataFrame:
        """Compute technical indicator features."""
        features = pd.DataFrame(index=ohlcv.index)
        close = ohlcv['close']
        high = ohlcv['high']
        low = ohlcv['low']
        
        # RSI
        for period in self.config.rsi_periods:
            features[f'rsi_{period}'] = self._compute_rsi(close, period)
        
        # Bollinger Bands
        bb_mid = close.rolling(self.config.bb_period).mean()
        bb_std = close.rolling(self.config.bb_period).std()
        bb_upper = bb_mid + self.config.bb_std * bb_std
        bb_lower = bb_mid - self.config.bb_std * bb_std
        
        features['bb_pct'] = (close - bb_lower) / (bb_upper - bb_lower)
        features['bb_width'] = (bb_upper - bb_lower) / bb_mid * 100
        
        # MACD
        fast, slow, signal = self.config.macd_params
        ema_fast = close.ewm(span=fast, adjust=False).mean()
        ema_slow = close.ewm(span=slow, adjust=False).mean()
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal, adjust=False).mean()
        
        features['macd'] = macd
        features['macd_signal'] = macd_signal
        features['macd_histogram'] = macd - macd_signal
        
        # EMA Cross
        ema_short, ema_long = self.config.ema_periods
        ema_s = close.ewm(span=ema_short, adjust=False).mean()
        ema_l = close.ewm(span=ema_long, adjust=False).mean()
        features['ema_cross'] = (ema_s > ema_l).astype(float)
        
        # ATR
        tr = pd.concat([
            high - low,
            (high - close.shift()).abs(),
            (low - close.shift()).abs()
        ], axis=1).max(axis=1)
        features['atr'] = tr.rolling(self.config.atr_period).mean()
        
        # ADX (simplified)
        features['adx'] = self._compute_adx(high, low, close, self.config.adx_period)
        
        return features
    
    def _compute_microstructure_features(
        self,
        df: pd.DataFrame,
        ohlcv: pd.DataFrame
    ) -> Optional[pd.DataFrame]:
        """Compute market microstructure features."""
        features = pd.DataFrame(index=df.index)
        
        # Orderbook imbalance
        imbalance = self.provider.get_orderbook_imbalance(df)
        if imbalance is not None:
            features['orderbook_imbalance'] = imbalance
        
        # Spread (if available)
        if 'spread' in df.columns:
            features['spread_bps'] = df['spread'] / ohlcv['close'] * 10000
        elif 'best_bid' in df.columns and 'best_ask' in df.columns:
            features['spread_bps'] = (df['best_ask'] - df['best_bid']) / ohlcv['close'] * 10000
        
        # Mid price velocity
        mid = ohlcv['close']  # Approximation
        features['mid_velocity'] = mid.diff().rolling(5).mean()
        
        return features if len(features.columns) > 0 else None
    
    def _compute_trade_flow_features(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Compute trade flow features."""
        trade_flow = self.provider.get_trade_flow(df)
        if trade_flow is None:
            return None
        
        features = pd.DataFrame(index=df.index)
        
        if 'buy_sell_ratio' in trade_flow.columns:
            features['buy_sell_ratio'] = trade_flow['buy_sell_ratio']
        
        if 'cvd_raw' in trade_flow.columns:
            for window in self.config.cvd_windows:
                features[f'cvd_{window}m'] = trade_flow['cvd_raw'].rolling(window).sum()
        
        # Volume ratio
        if 'volume' in df.columns:
            vol_ma = df['volume'].rolling(20).mean()
            features['volume_ratio'] = df['volume'] / vol_ma
        
        return features if len(features.columns) > 0 else None
    
    def _compute_derivatives_features(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Compute derivatives-specific features."""
        deriv_data = self.provider.get_derivatives_data(df)
        if deriv_data is None:
            return None
        
        features = pd.DataFrame(index=df.index)
        
        if 'funding_rate' in deriv_data.columns:
            features['funding_rate'] = deriv_data['funding_rate']
            features['funding_rate_ma'] = deriv_data['funding_rate'].rolling(8).mean()
        
        if 'oi_change_1h' in deriv_data.columns:
            features['oi_change_1h'] = deriv_data['oi_change_1h']
        
        if 'long_short_ratio' in deriv_data.columns:
            features['long_short_ratio'] = deriv_data['long_short_ratio']
        
        return features if len(features.columns) > 0 else None
    
    def _compute_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute time-based features."""
        features = pd.DataFrame(index=df.index)
        
        # Get datetime index
        if isinstance(df.index, pd.DatetimeIndex):
            dt = df.index
        elif 'timestamp' in df.columns:
            dt = pd.to_datetime(df['timestamp'])
        else:
            return features
        
        # Cyclical encoding of hour
        hour = dt.hour
        features['hour_sin'] = np.sin(2 * np.pi * hour / 24)
        features['hour_cos'] = np.cos(2 * np.pi * hour / 24)
        
        # Day of week
        dow = dt.dayofweek
        features['dow_sin'] = np.sin(2 * np.pi * dow / 7)
        features['dow_cos'] = np.cos(2 * np.pi * dow / 7)
        
        # Trading sessions
        for session, (start, end) in self.config.trading_sessions.items():
            if start < end:
                features[f'is_{session}_session'] = ((hour >= start) & (hour < end)).astype(float)
            else:  # Wraps around midnight
                features[f'is_{session}_session'] = ((hour >= start) | (hour < end)).astype(float)
        
        # Minutes to funding (crypto-specific, every 8h)
        minutes_in_day = hour * 60 + dt.minute
        funding_times = [0, 8*60, 16*60, 24*60]  # 00:00, 08:00, 16:00 UTC
        
        min_to_funding = []
        for m in minutes_in_day:
            next_funding = min(t for t in funding_times if t > m) if any(t > m for t in funding_times) else funding_times[1]
            min_to_funding.append(next_funding - m if next_funding > m else (24*60 - m + funding_times[1]))
        features['minutes_to_funding'] = min_to_funding
        
        return features
    
    def _compute_regime_features(
        self,
        df: pd.DataFrame,
        regime_probs: Dict[str, float]
    ) -> pd.DataFrame:
        """Compute HMM regime-based features."""
        features = pd.DataFrame(index=df.index)
        
        # One-hot regime indicators
        regimes = ['low_vol', 'high_vol', 'trend_up', 'trend_down']
        for regime in regimes:
            features[f'regime_{regime}'] = regime_probs.get(regime, 0.0)
        
        return features
    
    @staticmethod
    def _compute_rsi(series: pd.Series, period: int) -> pd.Series:
        """Compute RSI indicator."""
        delta = series.diff()
        gain = delta.where(delta > 0, 0).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs = gain / loss.replace(0, np.nan)
        return 100 - (100 / (1 + rs))
    
    @staticmethod
    def _compute_adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
        """Compute ADX indicator (simplified)."""
        plus_dm = high.diff()
        minus_dm = -low.diff()
        
        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
        
        tr = pd.concat([
            high - low,
            (high - close.shift()).abs(),
            (low - close.shift()).abs()
        ], axis=1).max(axis=1)
        
        atr = tr.rolling(period).mean()
        plus_di = 100 * (plus_dm.rolling(period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(period).mean() / atr)
        
        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
        adx = dx.rolling(period).mean()
        
        return adx


def create_target(
    df: pd.DataFrame,
    horizon: int = 1,
    threshold_pct: float = 0.05,
    close_col: str = 'close'
) -> pd.Series:
    """
    Create classification target for direction prediction.
    
    Args:
        df: DataFrame with close prices
        horizon: Prediction horizon in periods
        threshold_pct: Threshold for UP/DOWN classification
        close_col: Name of close price column
        
    Returns:
        Series with target labels (0=DOWN, 1=NEUTRAL, 2=UP)
    """
    future_returns = df[close_col].pct_change(horizon).shift(-horizon) * 100
    
    target = pd.cut(
        future_returns,
        bins=[-np.inf, -threshold_pct, threshold_pct, np.inf],
        labels=[0, 1, 2]
    ).astype(float)
    
    return target


# Registry for exchange providers
EXCHANGE_PROVIDERS = {
    'binance': BinanceFeatureProvider,
    'grvt': GRVTFeatureProvider,
}


def get_feature_engineer(
    exchange: str = 'binance',
    config: Optional[FeatureConfig] = None
) -> FeatureEngineer:
    """
    Factory function to get feature engineer for specific exchange.
    
    Args:
        exchange: Exchange name ('binance', 'grvt', etc.)
        config: Optional feature configuration
        
    Returns:
        Configured FeatureEngineer instance
    """
    provider_class = EXCHANGE_PROVIDERS.get(exchange.lower())
    if provider_class is None:
        raise ValueError(f"Unknown exchange: {exchange}. Available: {list(EXCHANGE_PROVIDERS.keys())}")
    
    return FeatureEngineer(config=config, provider=provider_class())
