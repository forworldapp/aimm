"""
Execution Algorithms - v5.4
TWAP, VWAP, and Adaptive Execution for minimizing market impact
"""
import logging
import time
from collections import deque
from typing import Dict, List, Optional, Callable
from enum import Enum

class ExecutionStyle(Enum):
    TWAP = "twap"
    VWAP = "vwap"
    ADAPTIVE = "adaptive"


class TWAPExecutor:
    """
    Time-Weighted Average Price Execution
    
    큰 주문을 시간 기반으로 분할 실행
    """
    
    def __init__(self, config: dict = None):
        self.config = config or {}
        self.logger = logging.getLogger("TWAPExecutor")
        
        # Parameters
        self.slice_interval_seconds = self.config.get('slice_interval_seconds', 60)
        self.participation_rate = self.config.get('participation_rate', 0.1)
        
        # State
        self.active_orders = []
        
    def create_schedule(
        self,
        total_qty: float,
        duration_minutes: int,
        side: str
    ) -> List[Dict]:
        """
        Create TWAP execution schedule
        
        Args:
            total_qty: Total quantity to execute
            duration_minutes: Total execution duration
            side: 'buy' or 'sell'
            
        Returns:
            List of scheduled orders with timing
        """
        num_slices = max(1, duration_minutes * 60 // self.slice_interval_seconds)
        qty_per_slice = total_qty / num_slices
        
        schedule = []
        current_time = time.time()
        
        for i in range(int(num_slices)):
            execute_at = current_time + (i * self.slice_interval_seconds)
            schedule.append({
                'slice_id': i,
                'execute_at': execute_at,
                'quantity': qty_per_slice,
                'side': side,
                'status': 'pending'
            })
        
        self.active_orders = schedule
        return schedule
    
    def get_next_slice(self, current_time: float = None) -> Optional[Dict]:
        """Get next slice to execute"""
        if current_time is None:
            current_time = time.time()
        
        for order in self.active_orders:
            if order['status'] == 'pending' and order['execute_at'] <= current_time:
                return order
        
        return None
    
    def mark_executed(self, slice_id: int, executed_price: float):
        """Mark slice as executed"""
        for order in self.active_orders:
            if order['slice_id'] == slice_id:
                order['status'] = 'executed'
                order['executed_price'] = executed_price
                order['executed_at'] = time.time()
                break
    
    def get_execution_summary(self) -> Dict:
        """Get summary of execution"""
        executed = [o for o in self.active_orders if o['status'] == 'executed']
        
        if not executed:
            return {'executed_qty': 0, 'vwap': 0, 'slices_done': 0}
        
        total_qty = sum(o['quantity'] for o in executed)
        vwap = sum(o['quantity'] * o.get('executed_price', 0) for o in executed) / total_qty if total_qty > 0 else 0
        
        return {
            'executed_qty': total_qty,
            'vwap': vwap,
            'slices_done': len(executed),
            'slices_total': len(self.active_orders)
        }


class VWAPExecutor:
    """
    Volume-Weighted Average Price Execution
    
    거래량 프로파일에 맞춰 주문 분배
    """
    
    def __init__(self, config: dict = None):
        self.config = config or {}
        self.logger = logging.getLogger("VWAPExecutor")
        
        # Volume profile (simplified: hour-of-day based)
        self.volume_profile = self.config.get('volume_profile', self._default_profile())
        
        # State
        self.active_orders = []
        self.volume_history = deque(maxlen=1440)  # 1 day of minutes
        
    def _default_profile(self) -> Dict[int, float]:
        """Default hourly volume profile (0-23 UTC)"""
        # Crypto tends to have higher volume in US/EU overlap and Asian open
        return {
            0: 0.04, 1: 0.04, 2: 0.03, 3: 0.03, 4: 0.03, 5: 0.03,
            6: 0.04, 7: 0.04, 8: 0.05, 9: 0.05, 10: 0.05, 11: 0.05,
            12: 0.04, 13: 0.04, 14: 0.05, 15: 0.06, 16: 0.06, 17: 0.05,
            18: 0.04, 19: 0.04, 20: 0.04, 21: 0.04, 22: 0.04, 23: 0.04
        }
    
    def record_volume(self, volume: float, hour: int):
        """Record actual volume for profile adaptation"""
        self.volume_history.append({'hour': hour, 'volume': volume})
    
    def get_target_participation(self, hour: int) -> float:
        """Get target participation rate for current hour"""
        return self.volume_profile.get(hour, 0.04)
    
    def calculate_order_size(
        self,
        remaining_qty: float,
        remaining_time_hours: float,
        current_hour: int,
        current_volume: float
    ) -> float:
        """
        Calculate how much to execute now based on VWAP targeting
        
        Args:
            remaining_qty: Quantity left to execute
            remaining_time_hours: Hours left in execution window
            current_hour: Current hour (0-23)
            current_volume: Current market volume
            
        Returns:
            Quantity to execute this period
        """
        if remaining_time_hours <= 0:
            return remaining_qty
        
        # Get expected volume participation
        target_pct = self.get_target_participation(current_hour)
        
        # Adjust based on actual volume
        participation_rate = min(0.2, target_pct)  # Cap at 20%
        
        size = current_volume * participation_rate
        size = min(size, remaining_qty)
        
        return size


class AdaptiveExecutor:
    """
    Adaptive Execution Strategy
    
    시장 상태에 따라 TWAP/VWAP 전환 및 속도 조절
    """
    
    def __init__(self, config: dict = None):
        self.config = config or {}
        self.logger = logging.getLogger("AdaptiveExecutor")
        
        # Sub-executors
        self.twap = TWAPExecutor(self.config.get('twap', {}))
        self.vwap = VWAPExecutor(self.config.get('vwap', {}))
        
        # Adaptation params
        self.volatility_threshold = self.config.get('volatility_threshold', 0.02)
        self.spread_threshold = self.config.get('spread_threshold', 0.003)
        
        # State
        self.current_style = ExecutionStyle.TWAP
        
    def analyze_market_conditions(self, spread: float, volatility: float) -> ExecutionStyle:
        """Determine best execution style based on market conditions"""
        if volatility > self.volatility_threshold:
            return ExecutionStyle.TWAP  # Steady pace in volatile markets
        elif spread > self.spread_threshold:
            return ExecutionStyle.VWAP  # Follow volume in wide spreads
        else:
            return ExecutionStyle.ADAPTIVE
    
    def calculate_urgency(
        self,
        elapsed_pct: float,
        executed_pct: float
    ) -> float:
        """
        Calculate execution urgency (0-2)
        
        <1: Ahead of schedule
        =1: On schedule
        >1: Behind schedule
        """
        if elapsed_pct == 0:
            return 1.0
        
        urgency = elapsed_pct / executed_pct if executed_pct > 0 else 2.0
        return max(0.5, min(2.0, urgency))
    
    def get_adjusted_size(
        self,
        base_size: float,
        urgency: float,
        spread: float
    ) -> float:
        """Adjust order size based on urgency and market conditions"""
        if urgency > 1.5:
            # Behind schedule - more aggressive
            return base_size * 1.3
        elif urgency < 0.8:
            # Ahead of schedule - can be more passive
            return base_size * 0.7
        else:
            return base_size


class ExecutionIntegrator:
    """
    MarketMaker 통합을 위한 실행 알고리즘 인터페이스
    """
    
    def __init__(self, config: dict = None):
        self.config = config or {}
        self.logger = logging.getLogger("ExecutionIntegrator")
        self.adaptive = AdaptiveExecutor(config)
        self.enabled = self.config.get('enabled', True)
        
        # Active execution tasks
        self.active_tasks = []
    
    def start_execution(
        self,
        qty: float,
        side: str,
        duration_minutes: int = 30,
        style: str = 'adaptive'
    ) -> Dict:
        """Start a new execution task"""
        if not self.enabled:
            return {'status': 'disabled'}
        
        schedule = self.adaptive.twap.create_schedule(qty, duration_minutes, side)
        
        task = {
            'id': len(self.active_tasks),
            'qty': qty,
            'side': side,
            'duration_minutes': duration_minutes,
            'style': style,
            'schedule': schedule,
            'start_time': time.time()
        }
        
        self.active_tasks.append(task)
        
        return {
            'status': 'started',
            'task_id': task['id'],
            'slices': len(schedule)
        }
    
    def get_next_action(self, spread: float = 0.001, volatility: float = 0.01) -> Dict:
        """Get next execution action"""
        if not self.active_tasks:
            return {'action': 'none'}
        
        # Check first active task
        task = self.active_tasks[0]
        next_slice = self.adaptive.twap.get_next_slice()
        
        if next_slice:
            # Adjust based on market conditions
            style = self.adaptive.analyze_market_conditions(spread, volatility)
            urgency = self.adaptive.calculate_urgency(
                elapsed_pct=0.5,  # Simplified
                executed_pct=0.4
            )
            
            adjusted_qty = self.adaptive.get_adjusted_size(
                next_slice['quantity'],
                urgency,
                spread
            )
            
            return {
                'action': 'execute',
                'slice_id': next_slice['slice_id'],
                'side': next_slice['side'],
                'quantity': adjusted_qty,
                'style': style.value,
                'urgency': urgency
            }
        
        return {'action': 'wait'}


# Convenience function
def create_executor(config: dict = None) -> ExecutionIntegrator:
    """Create execution algorithm integrator"""
    return ExecutionIntegrator(config or {})
