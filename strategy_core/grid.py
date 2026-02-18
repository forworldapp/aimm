"""
Grid Price Generation — Pure Functions.

Extracted from aimm/strategies/market_maker.py grid placement logic.
"""

import numpy as np
from typing import List, Tuple


def generate_grid_prices(
    reservation_price: float,
    spread: float,
    grid_layers: int,
    grid_spacing: float = 0.002,
    skew: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate bid and ask grid price levels.

    Matches aimm MarketMaker's grid placement in cycle().

    Args:
        reservation_price: A&S reservation price (inventory-adjusted mid)
        spread: Optimal spread (fraction)
        grid_layers: Number of grid levels per side
        grid_spacing: Distance between grid levels (fraction)
        skew: Trend skew offset (fraction, positive = bullish)

    Returns:
        (bid_prices, ask_prices): Arrays of grid price levels
    """
    half_spread = spread / 2.0

    # Base bid/ask from reservation price
    base_bid = reservation_price * (1.0 - half_spread + skew)
    base_ask = reservation_price * (1.0 + half_spread + skew)

    bid_prices = np.array([
        base_bid * (1.0 - i * grid_spacing)
        for i in range(grid_layers)
    ])

    ask_prices = np.array([
        base_ask * (1.0 + i * grid_spacing)
        for i in range(grid_layers)
    ])

    return bid_prices, ask_prices


def generate_grid_sizes(
    base_size_usd: float,
    grid_layers: int,
    order_size_mult: float = 1.0,
    position_qty: float = 0.0,
    max_position_usd: float = 5000.0,
    mid_price: float = 100000.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate order sizes for each grid level.

    Args:
        base_size_usd: Base order size in USD
        grid_layers: Number of grid levels per side
        order_size_mult: ML regime-based size multiplier
        position_qty: Current position quantity (BTC)
        max_position_usd: Maximum position in USD
        mid_price: Current mid price

    Returns:
        (bid_sizes, ask_sizes): Arrays of order sizes in base currency
    """
    size_usd = base_size_usd * order_size_mult
    size_base = size_usd / mid_price if mid_price > 0 else 0.0

    # Position limit check
    current_pos_usd = abs(position_qty * mid_price)

    bid_sizes = np.full(grid_layers, size_base)
    ask_sizes = np.full(grid_layers, size_base)

    # If near max position, reduce sizes on the side that increases position
    if current_pos_usd > max_position_usd * 0.9:
        if position_qty > 0:
            bid_sizes *= 0.1  # Reduce buys when long
        else:
            ask_sizes *= 0.1  # Reduce sells when short

    return bid_sizes, ask_sizes


def vectorized_grid_mids(
    reservation_prices: np.ndarray,
    spreads: np.ndarray,
    skews: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Vectorized best bid/ask calculation for entire price series.

    For fast backtesting: compute all bid/ask levels at once.

    Args:
        reservation_prices: Array of reservation prices
        spreads: Array of spreads
        skews: Array of trend skews

    Returns:
        (best_bids, best_asks): Arrays of best bid/ask prices
    """
    half_spreads = spreads / 2.0
    best_bids = reservation_prices * (1.0 - half_spreads + skews)
    best_asks = reservation_prices * (1.0 + half_spreads + skews)
    return best_bids, best_asks
