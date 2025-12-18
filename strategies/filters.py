"""
Strategy Filters Module (V1.4)
------------------------------
Contains technical indicator logic used by the Market Maker strategy.
- RSI: Relative Strength Index
- Bollinger Bands: Volatility bands for mean reversion
- ATR: Average True Range for volatility measurement
"""

import pandas as pd
import numpy as np
from abc import ABC, abstractmethod

class TrendFilter(ABC):
    @abstractmethod
    def analyze(self, df: pd.DataFrame) -> str:
        """
        Analyze candles and return 'ranging' or 'trending'.
        df should have columns: ['open', 'high', 'low', 'close']
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        pass

class MAFilter(TrendFilter):
    def __init__(self, short_win=3, long_win=10, threshold=0.0003):
        self.short_win = short_win
        self.long_win = long_win
        self.threshold = threshold

    @property
    def name(self):
        return "MA Divergence"

    def analyze(self, df: pd.DataFrame) -> str:
        if len(df) < self.long_win:
            return 'waiting'
            
        close = df['close']
        short_ma = close.rolling(window=self.short_win).mean().iloc[-1]
        long_ma = close.rolling(window=self.long_win).mean().iloc[-1]
        
        if long_ma == 0: return 'ranging'
        
        divergence = abs(short_ma - long_ma) / long_ma
        return 'trending' if divergence >= self.threshold else 'ranging'

class ADXFilter(TrendFilter):
    def __init__(self, period=14, threshold=25):
        self.period = period
        self.threshold = threshold

    @property
    def name(self):
        return "ADX Strength"

    def analyze(self, df: pd.DataFrame) -> str:
        if len(df) < self.period * 2:
            return 'waiting'

        high = df['high']
        low = df['low']
        close = df['close']
        prev_close = close.shift(1)
        
        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=self.period).mean()

        up_move = high - high.shift(1)
        down_move = low.shift(1) - low
        
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
        
        plus_dm = pd.Series(plus_dm, index=df.index)
        minus_dm = pd.Series(minus_dm, index=df.index)
        
        smooth_plus = plus_dm.rolling(window=self.period).mean()
        smooth_minus = minus_dm.rolling(window=self.period).mean()
        
        plus_di = 100 * (smooth_plus / atr)
        minus_di = 100 * (smooth_minus / atr)
        
        dx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, 1))
        adx = dx.rolling(window=self.period).mean().iloc[-1]
        
        return 'trending' if adx >= self.threshold else 'ranging'

class ATRFilter(TrendFilter):
    def __init__(self, period=14, threshold_pct=0.005):
        self.period = period
        self.threshold_pct = threshold_pct 

    @property
    def name(self):
        return "ATR Volatility"

    def analyze(self, df: pd.DataFrame) -> str:
        if len(df) < self.period + 1:
            return 'waiting'
            
        high = df['high']
        low = df['low']
        close = df['close']
        prev_close = close.shift(1)
        
        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=self.period).mean().iloc[-1]
        
        current_price = close.iloc[-1]
        atr_pct = atr / current_price
        
        return 'trending' if atr_pct >= self.threshold_pct else 'ranging'

class ChopFilter(TrendFilter):
    def __init__(self, period=14):
        self.period = period

    @property
    def name(self):
        return "Choppiness Index"

    def analyze(self, df: pd.DataFrame) -> str:
        if len(df) < self.period + 1:
            return 'waiting'
            
        high = df['high']
        low = df['low']
        close = df['close']
        
        prev_close = close.shift(1)
        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        sum_tr = tr.rolling(window=self.period).sum()
        max_hi = high.rolling(window=self.period).max()
        min_lo = low.rolling(window=self.period).min()
        
        range_hl = max_hi - min_lo
        range_hl = range_hl.replace(0, 1e-9)
        
        chop = 100 * np.log10(sum_tr / range_hl) / np.log10(self.period)
        current_chop = chop.iloc[-1]
        
        if current_chop > 60:
            return 'ranging'
        elif current_chop < 40:
            return 'trending'
        else:
            return 'ranging' 

class ComboFilter(TrendFilter):
    """
    Combines ADX and ATR.
    Must meet BOTH conditions (Strong Trend + High Volatility).
    """
    def __init__(self):
        # [TODO: v1.4] Restore period to 14 for Production/Live Trading for better accuracy.
        # Currently set to 7 for faster testing/debugging.
        self.adx = ADXFilter(period=7, threshold=25)
        self.atr = ATRFilter(period=7)

    @property
    def name(self):
        return "ADX + ATR Combo"

    def analyze(self, df: pd.DataFrame) -> str:
        r1 = self.adx.analyze(df)
        r2 = self.atr.analyze(df)
        
        if r1 == 'waiting' or r2 == 'waiting':
            return 'waiting'
            
        if r1 == 'trending' and r2 == 'trending':
            return 'trending'
            
        return 'ranging'

class RSIFilter(TrendFilter):
    """
    RSI Filter for Overbought/Oversold detection.
    """
    def __init__(self, period=14, overbought=70, oversold=30):
        self.period = period
        self.overbought = overbought
        self.oversold = oversold

    def analyze(self, df):
        """
        Returns:
        - 'overbought' if RSI > 70
        - 'oversold' if RSI < 30
        - 'neutral' otherwise
        - 'waiting' if insufficient data
        """
        if len(df) < self.period + 1:
            return 'waiting'
        
        # Calculate RSI using pandas
        close_delta = df['close'].diff()
        up = close_delta.clip(lower=0)
        down = -1 * close_delta.clip(upper=0)
        
        ma_up = up.ewm(com=self.period - 1, adjust=True, min_periods=self.period).mean()
        ma_down = down.ewm(com=self.period - 1, adjust=True, min_periods=self.period).mean()
        
        rsi = ma_up / (ma_up + ma_down) * 100
        
        current_rsi = rsi.iloc[-1]
        self.last_rsi = current_rsi # Store for display
        
        if current_rsi > self.overbought:
            return 'overbought'
        elif current_rsi < self.oversold:
            return 'oversold'
        else:
            return 'neutral'
            
    @property
    def name(self):
        return f"RSI({self.period})"
        
class BollingerFilter(TrendFilter):
    """
    Bollinger Bands Filter for Mean Reversion.
    Buy at Lower Band, Sell at Upper Band.
    """
    def __init__(self, period=20, std_dev=2.0):
        self.period = period
        self.std_dev = std_dev
        self.last_pct_b = 0.5 # Track %B for display

    def analyze(self, df):
        """
        Returns:
        - 'buy_signal' if Price <= Lower Band
        - 'sell_signal' if Price >= Upper Band
        - 'neutral' otherwise
        - 'waiting' if insufficient data
        """
        if len(df) < self.period:
            return 'waiting'
            
        close = df['close']
        
        # Calculate Bollinger Bands
        ma = close.rolling(window=self.period).mean()
        std = close.rolling(window=self.period).std()
        
        upper = ma + (std * self.std_dev)
        lower = ma - (std * self.std_dev)
        
        current_close = close.iloc[-1]
        current_upper = upper.iloc[-1]
        current_lower = lower.iloc[-1]
        
        # Calculate %B (Percent Bandwidth) for display
        # %B = (Price - Lower) / (Upper - Lower)
        if current_upper != current_lower:
            self.last_pct_b = (current_close - current_lower) / (current_upper - current_lower)
        
        if current_close <= current_lower:
            return 'buy_signal'
        elif current_close >= current_upper:
            return 'sell_signal'
        else:
            return 'neutral'
            
    @property
    def name(self):
        return f"BB({self.period}, {self.std_dev})"
