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
