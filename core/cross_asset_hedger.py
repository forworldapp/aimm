"""
Cross-Asset Hedging Module - v5.3
BTC/ETH 상관관계를 활용한 방향성 리스크 헤지
"""
import logging
import numpy as np
from collections import deque
from typing import Dict, Tuple, Optional

class CorrelationAnalyzer:
    """
    크로스 에셋 상관관계 분석
    
    ETH 마켓메이킹 시 BTC로 방향성 헤지
    """
    
    def __init__(self, config: dict = None):
        self.config = config or {}
        self.logger = logging.getLogger("CorrelationAnalyzer")
        
        # Parameters
        self.window_minutes = self.config.get('window_minutes', 1440)  # 1일
        self.min_correlation = self.config.get('min_correlation', 0.7)
        
        # Price history
        self.eth_returns = deque(maxlen=self.window_minutes)
        self.btc_returns = deque(maxlen=self.window_minutes)
        self.last_eth_price = None
        self.last_btc_price = None
        
    def update(self, eth_price: float, btc_price: float):
        """Update with new prices"""
        if self.last_eth_price and self.last_btc_price:
            # Calculate log returns
            eth_return = np.log(eth_price / self.last_eth_price)
            btc_return = np.log(btc_price / self.last_btc_price)
            
            self.eth_returns.append(eth_return)
            self.btc_returns.append(btc_return)
        
        self.last_eth_price = eth_price
        self.last_btc_price = btc_price
    
    def calculate_correlation(self) -> float:
        """Calculate rolling correlation between ETH and BTC"""
        if len(self.eth_returns) < 30:  # Minimum samples
            return 0.0
        
        eth_arr = np.array(self.eth_returns)
        btc_arr = np.array(self.btc_returns)
        
        correlation = np.corrcoef(eth_arr, btc_arr)[0, 1]
        return correlation if not np.isnan(correlation) else 0.0
    
    def is_correlated(self) -> bool:
        """Check if correlation is above threshold"""
        return abs(self.calculate_correlation()) >= self.min_correlation


class CrossAssetHedger:
    """
    크로스 에셋 헤지 관리
    
    β = Cov(ETH, BTC) / Var(BTC)
    Hedge Position = -ETH_Exposure * β * hedge_ratio_beta
    """
    
    def __init__(self, config: dict = None):
        self.config = config or {}
        self.logger = logging.getLogger("CrossAssetHedger")
        
        # Correlation analyzer
        corr_config = self.config.get('correlation', {})
        self.correlation_analyzer = CorrelationAnalyzer(corr_config)
        
        # Hedge parameters
        hedge_config = self.config.get('hedge', {})
        self.hedge_ratio_beta = hedge_config.get('ratio_beta', 0.8)  # 80% hedge
        self.min_position_usd = hedge_config.get('min_position_usd', 100)
        self.rebalance_threshold = hedge_config.get('rebalance_threshold', 0.1)  # 10%
        
        # State
        self.current_hedge_position = 0.0  # BTC position for hedging
        self.last_calculated_hedge = 0.0
        
    def update_prices(self, eth_price: float, btc_price: float):
        """Update price data"""
        self.correlation_analyzer.update(eth_price, btc_price)
    
    def calculate_hedge_ratio(self) -> float:
        """
        Calculate beta-based hedge ratio
        
        β = Cov(ETH, BTC) / Var(BTC)
        """
        if len(self.correlation_analyzer.eth_returns) < 30:
            return 0.0
        
        eth_arr = np.array(self.correlation_analyzer.eth_returns)
        btc_arr = np.array(self.correlation_analyzer.btc_returns)
        
        covariance = np.cov(eth_arr, btc_arr)[0, 1]
        btc_variance = np.var(btc_arr)
        
        if btc_variance == 0:
            return 0.0
        
        beta = covariance / btc_variance
        correlation = self.correlation_analyzer.calculate_correlation()
        
        # Reduce hedge ratio if correlation is low
        if abs(correlation) < self.correlation_analyzer.min_correlation:
            beta *= abs(correlation) / self.correlation_analyzer.min_correlation
        
        return beta * self.hedge_ratio_beta
    
    def calculate_hedge_position(
        self,
        eth_inventory: float,
        eth_price: float,
        btc_price: float
    ) -> Dict:
        """
        Calculate required BTC hedge position
        
        Returns:
            hedge_btc: Required BTC position (negative = short)
            hedge_usd: Hedge value in USD
            action: 'none' | 'rebalance'
            correlation: Current correlation
        """
        hedge_ratio = self.calculate_hedge_ratio()
        correlation = self.correlation_analyzer.calculate_correlation()
        
        # ETH exposure in USD
        eth_exposure_usd = eth_inventory * eth_price
        
        # Required BTC hedge (opposite direction)
        btc_hedge_usd = -eth_exposure_usd * hedge_ratio
        btc_hedge_position = btc_hedge_usd / btc_price if btc_price > 0 else 0
        
        # Check if rebalancing needed
        position_diff = abs(btc_hedge_position - self.current_hedge_position)
        current_value = abs(self.current_hedge_position * btc_price)
        
        needs_rebalance = False
        if current_value > 0:
            if position_diff / abs(self.current_hedge_position) > self.rebalance_threshold:
                needs_rebalance = True
        elif abs(btc_hedge_usd) >= self.min_position_usd:
            needs_rebalance = True
        
        # Only hedge if correlation is meaningful
        if not self.correlation_analyzer.is_correlated():
            btc_hedge_position = 0
            needs_rebalance = self.current_hedge_position != 0
        
        self.last_calculated_hedge = btc_hedge_position
        
        return {
            'hedge_btc': btc_hedge_position,
            'hedge_usd': btc_hedge_usd,
            'action': 'rebalance' if needs_rebalance else 'none',
            'correlation': correlation,
            'hedge_ratio': hedge_ratio,
            'eth_exposure_usd': eth_exposure_usd
        }
    
    def update_hedge_position(self, new_position: float):
        """Update current hedge position after execution"""
        self.current_hedge_position = new_position


class CrossAssetHedgeIntegrator:
    """
    MarketMaker 통합을 위한 인터페이스
    """
    
    def __init__(self, config: dict = None):
        self.config = config or {}
        self.logger = logging.getLogger("CrossAssetHedgeIntegrator")
        self.hedger = CrossAssetHedger(config)
        self.enabled = self.config.get('enabled', True)
        
    def update_and_analyze(
        self,
        eth_price: float,
        btc_price: float,
        eth_inventory: float
    ) -> Dict:
        """
        Update prices and get hedge recommendation
        """
        if not self.enabled:
            return {
                'enabled': False,
                'action': 'none',
                'hedge_btc': 0,
                'correlation': 0
            }
        
        self.hedger.update_prices(eth_price, btc_price)
        hedge_info = self.hedger.calculate_hedge_position(eth_inventory, eth_price, btc_price)
        
        return {
            'enabled': True,
            **hedge_info
        }


# Convenience function
def create_hedger(config: dict = None) -> CrossAssetHedgeIntegrator:
    """Create cross-asset hedge integrator"""
    return CrossAssetHedgeIntegrator(config or {})
