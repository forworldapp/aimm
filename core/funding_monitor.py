"""
Funding Rate Monitor & Arbitrage - v5.1
Monitors funding rates and adjusts market making bias accordingly.
"""
import asyncio
import logging
from datetime import datetime, timezone
from typing import Dict, Optional
from collections import deque

class FundingRateMonitor:
    """
    펀딩비 모니터링 및 분석
    
    펀딩비 양수 → 숏 포지션 유리 (롱이 숏에게 지불)
    펀딩비 음수 → 롱 포지션 유리 (숏이 롱에게 지불)
    """
    
    def __init__(self, config: dict = None):
        self.config = config or {}
        self.logger = logging.getLogger("FundingRateMonitor")
        
        # Thresholds
        self.threshold_high = self.config.get('threshold_high', 0.0001)  # 0.01%
        self.threshold_low = self.config.get('threshold_low', -0.0001)
        self.min_yield_annual = self.config.get('min_yield_annual', 5.0)  # 5% annual
        
        # History
        self.history = deque(maxlen=168)  # 1 week of 8-hour intervals
        self.last_funding_rate = 0.0
        self.next_funding_time = None
        
    def update(self, funding_rate: float, next_funding_time: datetime = None):
        """Update current funding rate"""
        self.last_funding_rate = funding_rate
        self.next_funding_time = next_funding_time
        self.history.append({
            'rate': funding_rate,
            'time': datetime.now(timezone.utc)
        })
        
    def calculate_annualized_rate(self, funding_rate: float, interval_hours: int = 8) -> float:
        """
        연환산 펀딩비 계산
        
        Returns:
            Annualized rate as percentage (e.g., 10.0 = 10% annual)
        """
        fundings_per_year = 365 * 24 / interval_hours  # ~1095 for 8h
        return funding_rate * fundings_per_year * 100
    
    def get_hours_to_funding(self) -> float:
        """Get hours until next funding"""
        if not self.next_funding_time:
            # Default: assume 8-hour cycle
            now = datetime.now(timezone.utc)
            # Funding at 00:00, 08:00, 16:00 UTC
            hour = now.hour
            if hour < 8:
                next_hour = 8
            elif hour < 16:
                next_hour = 16
            else:
                next_hour = 24  # Next day 00:00
            
            hours_to = next_hour - hour - (now.minute / 60)
            return max(0, hours_to)
        
        now = datetime.now(timezone.utc)
        delta = self.next_funding_time - now
        return max(0, delta.total_seconds() / 3600)
    
    def analyze_opportunity(self) -> Dict:
        """
        펀딩비 기회 분석
        
        Returns:
            opportunity: bool - 기회 존재 여부
            direction: 'long' | 'short' | 'neutral'
            annual_yield: float - 연환산 수익률
            confidence: float - 신뢰도 (0-1)
            hours_to_funding: float
        """
        funding_rate = self.last_funding_rate
        annual_yield = self.calculate_annualized_rate(funding_rate)
        hours_to_funding = self.get_hours_to_funding()
        
        # Determine direction
        if funding_rate > self.threshold_high:
            direction = 'short'  # 숏 유리 (롱이 지불)
            opportunity = abs(annual_yield) >= self.min_yield_annual
        elif funding_rate < self.threshold_low:
            direction = 'long'  # 롱 유리 (숏이 지불)
            opportunity = abs(annual_yield) >= self.min_yield_annual
        else:
            direction = 'neutral'
            opportunity = False
        
        # Calculate confidence based on history consistency
        if len(self.history) >= 3:
            recent = list(self.history)[-3:]
            same_sign = all(h['rate'] * funding_rate > 0 for h in recent)
            confidence = 0.8 if same_sign else 0.5
        else:
            confidence = 0.3
        
        # Adjust confidence by time to funding
        if hours_to_funding < 1:
            confidence *= 1.2  # Boost close to funding
        elif hours_to_funding > 6:
            confidence *= 0.8  # Reduce far from funding
        
        confidence = min(1.0, confidence)
        
        return {
            'opportunity': opportunity,
            'direction': direction,
            'funding_rate': funding_rate,
            'annual_yield': annual_yield,
            'confidence': confidence,
            'hours_to_funding': hours_to_funding
        }


class FundingIntegratedMM:
    """
    펀딩비를 고려한 마켓메이킹 조정
    
    전략:
    1. 펀딩비 양수 → 숏 편향 (ask 공격적, bid 보수적)
    2. 펀딩비 음수 → 롱 편향 (bid 공격적, ask 보수적)
    3. 펀딩 시간 직전 → 유리한 방향 포지션 유지
    """
    
    def __init__(self, config: dict = None):
        self.config = config or {}
        self.logger = logging.getLogger("FundingIntegratedMM")
        
        # Integration weights
        self.funding_weight = self.config.get('funding_weight', 0.3)
        self.max_inventory_skew = self.config.get('max_inventory_skew', 0.5)
        self.freeze_before_minutes = self.config.get('freeze_before_minutes', 30)
        
    def get_adjustment(
        self,
        funding_analysis: Dict,
        current_inventory: float = 0.0,
        max_inventory: float = 1.0
    ) -> Dict:
        """
        펀딩비 기반 조정 계산
        
        Returns:
            bid_size_mult: float - Bid 사이즈 배수
            ask_size_mult: float - Ask 사이즈 배수
            target_inventory_pct: float - 목표 재고 비율 (-1 to 1)
            freeze_orders: bool - 신규 주문 중단 여부
        """
        direction = funding_analysis['direction']
        confidence = funding_analysis['confidence']
        hours_to_funding = funding_analysis['hours_to_funding']
        annual_yield = funding_analysis['annual_yield']
        
        # Default: no adjustment
        bid_mult = 1.0
        ask_mult = 1.0
        target_inventory = 0.0
        freeze = False
        
        # Check freeze condition (30 min before funding)
        if hours_to_funding < (self.freeze_before_minutes / 60):
            freeze = True
            self.logger.info(f"🕐 Funding in {hours_to_funding*60:.0f}min - Freezing new orders")
        
        # Apply funding bias if opportunity exists
        if funding_analysis['opportunity'] and not freeze:
            # Time factor: closer to funding = stronger bias
            time_factor = max(0, 1 - hours_to_funding / 8)
            adjustment_strength = self.funding_weight * confidence * time_factor
            
            if direction == 'short':
                # 숏 유리 → Ask 공격적 (더 많이 팔기), Bid 보수적
                ask_mult = 1.0 + adjustment_strength
                bid_mult = 1.0 - adjustment_strength * 0.5
                target_inventory = -self.max_inventory_skew * time_factor
                
            elif direction == 'long':
                # 롱 유리 → Bid 공격적 (더 많이 사기), Ask 보수적
                bid_mult = 1.0 + adjustment_strength
                ask_mult = 1.0 - adjustment_strength * 0.5
                target_inventory = self.max_inventory_skew * time_factor
        
        # Clamp values
        bid_mult = max(0.5, min(1.5, bid_mult))
        ask_mult = max(0.5, min(1.5, ask_mult))
        target_inventory = max(-1.0, min(1.0, target_inventory))
        
        return {
            'bid_size_mult': bid_mult,
            'ask_size_mult': ask_mult,
            'target_inventory_pct': target_inventory,
            'freeze_orders': freeze,
            'direction': direction,
            'annual_yield': annual_yield,
            'hours_to_funding': hours_to_funding
        }


# Convenience function for integration
def create_funding_analyzer(config: dict = None) -> tuple:
    """Create monitor and integrator pair"""
    config = config or {}
    monitor = FundingRateMonitor(config.get('monitoring', {}))
    integrator = FundingIntegratedMM(config.get('integration', {}))
    return monitor, integrator
