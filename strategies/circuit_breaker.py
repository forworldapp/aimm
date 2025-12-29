import time
import pandas as pd
import logging
from core.config import Config

class CircuitBreaker:
    def __init__(self):
        self.logger = logging.getLogger("CircuitBreaker")
        self.enabled = Config.get("risk", "circuit_breaker", {}).get("enabled", False)
        self.window = Config.get("risk", "circuit_breaker", {}).get("window_minutes", 60)
        self.multiplier = Config.get("risk", "circuit_breaker", {}).get("sigma_multiplier", 4.0)
        self.cooldown_mins = Config.get("risk", "circuit_breaker", {}).get("cooldown_minutes", 15)
        
        self.cooldown_until = 0
        self.triggered_at = 0
        
    def check_volatility(self, candles: pd.DataFrame):
        """
        Check if current volatility exceeds ATR threshold.
        Returns: (is_triggered, status_message)
        """
        if not self.enabled:
            return False, "Disabled"
            
        # Check Cooldown
        if time.time() < self.cooldown_until:
            remaining = int(self.cooldown_until - time.time())
            return True, f"Cooldown active ({remaining}s remaining)"
            
        if len(candles) < self.window:
            return False, "Not enough data"
            
        try:
            # Calculate ATR (Average High-Low Range)
            df = candles.copy()
            df['range'] = df['high'] - df['low']
            
            # Baseline: Average range of last N minutes
            # Access by position to be safe with dataframe
            avg_range = df['range'].tail(self.window).mean()
            
            # Current Volatility: Last completed candle OR rolling range
            current_range = df['range'].iloc[-1]
            
            # Safety for divide by zero
            if avg_range == 0: avg_range = 1e-9
            
            ratio = current_range / avg_range
            
            if ratio > self.multiplier:
                self.trigger(ratio, current_range, avg_range)
                return True, f"TRIGGERED! Volatility {ratio:.1f}x (Curr:{current_range:.1f} > Avg:{avg_range:.1f})"
            
            return False, f"Normal (Vol: {ratio:.1f}x)"
            
        except Exception as e:
            self.logger.error(f"Error checking volatility: {e}")
            return False, "Error"

    def trigger(self, ratio, current, avg):
        self.triggered_at = time.time()
        self.cooldown_until = self.triggered_at + (self.cooldown_mins * 60)
        self.logger.critical(f"🛑 CIRCUIT BREAKER TRIGGERED! Ratio: {ratio:.2f}x (Current: {current:.2f}, Avg: {avg:.2f}). Pausing for {self.cooldown_mins} mins.")
