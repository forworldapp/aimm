"""
Microstructure Signals - v5.2
VPIN, Trade Arrival Rate, Volume Clock for detecting informed trading
"""
import logging
from collections import deque
from typing import Dict, List, Optional
import time

class VPIN:
    """
    Volume-Synchronized Probability of Informed Trading
    
    높은 VPIN = 정보 거래자 활동 증가 = 역선택 위험 증가
    """
    
    def __init__(self, config: dict = None):
        self.config = config or {}
        self.logger = logging.getLogger("VPIN")
        
        # Parameters
        self.bucket_size_usd = self.config.get('bucket_size_usd', 10000)
        self.n_buckets = self.config.get('n_buckets', 50)
        self.threshold = self.config.get('threshold', 0.7)
        
        # State
        self.buckets = deque(maxlen=self.n_buckets)
        self.current_bucket_volume = 0.0
        self.current_buy_volume = 0.0
        
    def update(self, trade: Dict) -> Optional[float]:
        """
        Update with new trade and potentially complete a bucket
        
        Args:
            trade: {price, size, side, is_taker}
            
        Returns:
            New VPIN value if bucket completed, else None
        """
        trade_volume_usd = trade['price'] * trade['size']
        
        self.current_bucket_volume += trade_volume_usd
        if trade['side'] == 'buy':
            self.current_buy_volume += trade_volume_usd
        
        # Check if bucket is complete
        if self.current_bucket_volume >= self.bucket_size_usd:
            sell_volume = self.current_bucket_volume - self.current_buy_volume
            imbalance = abs(self.current_buy_volume - sell_volume) / self.current_bucket_volume
            self.buckets.append(imbalance)
            
            # Reset for next bucket
            self.current_bucket_volume = 0.0
            self.current_buy_volume = 0.0
            
            return self.calculate()
        
        return None
    
    def calculate(self) -> float:
        """Calculate current VPIN from recent buckets"""
        if len(self.buckets) == 0:
            return 0.0
        return sum(self.buckets) / len(self.buckets)
    
    def is_elevated(self) -> bool:
        """Check if VPIN is above threshold"""
        return self.calculate() > self.threshold


class TradeArrivalAnalyzer:
    """
    거래 도착률 분석
    
    거래 빈도 급증 = 정보 이벤트 가능성
    """
    
    def __init__(self, config: dict = None):
        self.config = config or {}
        self.logger = logging.getLogger("TradeArrival")
        
        # Parameters
        self.baseline_window_seconds = self.config.get('baseline_window_seconds', 3600)
        self.elevated_threshold = self.config.get('elevated_threshold', 2.0)
        
        # State
        self.trade_times = deque(maxlen=10000)  # Store timestamps
        
    def record_trade(self, timestamp: float = None):
        """Record a trade arrival"""
        if timestamp is None:
            timestamp = time.time()
        self.trade_times.append(timestamp)
    
    def calculate_arrival_rate(self, current_time: float = None) -> Dict:
        """
        Calculate current arrival rate vs baseline
        
        Returns:
            current_rate: trades per second
            baseline_rate: historical average
            ratio: current/baseline
            is_elevated: bool
        """
        if current_time is None:
            current_time = time.time()
        
        # Recent 1 minute
        one_minute_ago = current_time - 60
        recent_trades = sum(1 for t in self.trade_times if t > one_minute_ago)
        current_rate = recent_trades / 60
        
        # Baseline (last hour)
        baseline_start = current_time - self.baseline_window_seconds
        baseline_trades = sum(1 for t in self.trade_times if t > baseline_start)
        baseline_rate = baseline_trades / self.baseline_window_seconds if self.baseline_window_seconds > 0 else 0
        
        ratio = current_rate / baseline_rate if baseline_rate > 0 else 1.0
        
        return {
            'current_rate': current_rate,
            'baseline_rate': baseline_rate,
            'ratio': ratio,
            'is_elevated': ratio > self.elevated_threshold
        }


class VolumeClock:
    """
    볼륨 기반 시간 척도
    
    시간 기반이 아닌 거래량 기반으로 시장 상태 측정
    → 변동성 높을 때 더 빠르게 반응
    """
    
    def __init__(self, config: dict = None):
        self.config = config or {}
        self.logger = logging.getLogger("VolumeClock")
        
        # Parameters
        self.volume_per_tick_usd = self.config.get('volume_per_tick_usd', 50000)
        
        # State
        self.accumulated_volume = 0.0
        self.volume_ticks = 0
        self.last_tick_time = time.time()
        
    def update(self, trade_volume_usd: float) -> bool:
        """
        Update with new trade volume
        
        Returns:
            True if new volume tick completed
        """
        self.accumulated_volume += trade_volume_usd
        
        if self.accumulated_volume >= self.volume_per_tick_usd:
            self.volume_ticks += 1
            self.accumulated_volume -= self.volume_per_tick_usd
            self.last_tick_time = time.time()
            return True
        
        return False
    
    def get_tick_rate(self, window_seconds: float = 300) -> float:
        """Get volume ticks per second over recent window"""
        # Simplified: return recent tick count
        return self.volume_ticks / max(1, window_seconds)


class MicrostructureIntegrator:
    """
    미시구조 신호 통합 및 조정 계산
    """
    
    def __init__(self, config: dict = None):
        self.config = config or {}
        self.logger = logging.getLogger("MicrostructureIntegrator")
        
        # Components
        vpin_config = self.config.get('vpin', {})
        arrival_config = self.config.get('trade_arrival', {})
        volume_config = self.config.get('volume_clock', {})
        
        self.vpin = VPIN(vpin_config)
        self.arrival = TradeArrivalAnalyzer(arrival_config)
        self.volume_clock = VolumeClock(volume_config)
        
        # Integration params
        self.defensive_risk_score = self.config.get('defensive_risk_score', 1.0)
        self.cautious_risk_score = self.config.get('cautious_risk_score', 0.5)
        
    def update_trade(self, trade: Dict, timestamp: float = None):
        """Update all components with new trade"""
        self.vpin.update(trade)
        self.arrival.record_trade(timestamp)
        trade_volume_usd = trade['price'] * trade['size']
        self.volume_clock.update(trade_volume_usd)
    
    def analyze(self) -> Dict:
        """
        Comprehensive microstructure analysis
        
        Returns:
            action: 'normal' | 'cautious' | 'defensive'
            risk_score: float
            spread_multiplier: float
            size_multiplier: float
            metrics: {vpin, arrival_ratio}
        """
        vpin_value = self.vpin.calculate()
        arrival_data = self.arrival.calculate_arrival_rate()
        arrival_ratio = arrival_data['ratio']
        
        # Calculate risk score
        risk_score = 0.0
        
        if vpin_value > self.vpin.threshold:
            risk_score += (vpin_value - self.vpin.threshold) * 2
        
        if arrival_ratio > self.arrival.elevated_threshold:
            risk_score += (arrival_ratio - self.arrival.elevated_threshold) * 0.5
        
        # Determine action and multipliers
        if risk_score > self.defensive_risk_score:
            action = 'defensive'
            spread_mult = 1.5
            size_mult = 0.5
        elif risk_score > self.cautious_risk_score:
            action = 'cautious'
            spread_mult = 1.2
            size_mult = 0.8
        else:
            action = 'normal'
            spread_mult = 1.0
            size_mult = 1.0
        
        return {
            'action': action,
            'risk_score': risk_score,
            'spread_mult': spread_mult,
            'size_mult': size_mult,
            'metrics': {
                'vpin': vpin_value,
                'arrival_ratio': arrival_ratio,
                'arrival_elevated': arrival_data['is_elevated']
            }
        }


# Convenience function
def create_microstructure_analyzer(config: dict = None) -> MicrostructureIntegrator:
    """Create integrated microstructure analyzer"""
    return MicrostructureIntegrator(config or {})
