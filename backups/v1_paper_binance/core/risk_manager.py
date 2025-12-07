import logging
from typing import Dict
from core.config import Config

class RiskManager:
    """
    Handles risk checks and limits.
    """
    def __init__(self):
        self.logger = logging.getLogger("RiskManager")
        self.max_pos_usd = float(Config.get("risk", "max_position_usd", 1000.0))
        self.max_drawdown = float(Config.get("risk", "max_drawdown_pct", 0.05))

    def check_trade_allowed(self, current_position_usd: float, new_order_usd: float) -> bool:
        """
        Check if a new trade violates position limits.
        """
        projected_exposure = abs(current_position_usd) + new_order_usd
        
        if projected_exposure > self.max_pos_usd:
            self.logger.warning(f"Risk Reject: Position limit reached. ({projected_exposure} > {self.max_pos_usd})")
            return False
            
        return True

    def calculate_skew(self, current_position_qty: float, max_position_qty: float, skew_factor: float) -> float:
        """
        Calculate price skew based on inventory.
        Returns a float to adjust the mid price.
        
        Logic:
        - Long Position -> Return Negative (Lower prices to sell)
        - Short Position -> Return Positive (Raise prices to buy)
        """
        if max_position_qty == 0: return 0.0
        
        # Ratio of current position to max allowed (-1.0 to 1.0)
        ratio = current_position_qty / max_position_qty
        
        # Skew adjustment (e.g., if ratio is 0.5 and factor is 0.001 (0.1%), skew is 0.0005)
        # We return a multiplier for the spread or a direct price offset.
        # Here we return a 'skew intensity' to be applied to the mid price offset.
        
        return -1 * ratio * skew_factor
