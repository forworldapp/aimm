from decimal import Decimal, ROUND_DOWN

def round_step_size(quantity: float, step_size: float) -> float:
    """
    Rounds a quantity to a specific step size (e.g., 0.001).
    """
    if step_size == 0: return quantity
    q = Decimal(str(quantity))
    s = Decimal(str(step_size))
    target = (q // s) * s
    return float(target)

def round_tick_size(price: float, tick_size: float) -> float:
    """
    Rounds a price to a specific tick size (e.g., 0.5).
    """
    if tick_size == 0: return price
    p = Decimal(str(price))
    t = Decimal(str(tick_size))
    # Rounding usually happens to the nearest tick
    target = (p / t).quantize(Decimal('1'), rounding=ROUND_DOWN) * t
    return float(target)
