from decimal import Decimal

def check_arb_opportunity(yes_ask: float, no_ask: float, threshold: float = 0.98) -> tuple[bool, float]:
    """
    Checks if the sum of YES and NO ask prices is below the threshold.

    Args:
        yes_ask (float): Best ask price for YES token.
        no_ask (float): Best ask price for NO token.
        threshold (float): The combined price threshold to trigger an arb signal.

    Returns:
        tuple[bool, float]: (Is Arb Opportunity, Total Cost)
    """
    if yes_ask is None or no_ask is None:
        return False, 0.0
    
    # Simple float addition for MVP (Decimal preferred for production financial apps)
    total_cost = yes_ask + no_ask
    
    is_arb = total_cost < threshold
    
    return is_arb, total_cost
