"""
Avellaneda-Stoikov Spread & Reservation Price — Pure Functions.

Extracted from aimm/strategies/market_maker.py for shared use.
All functions are stateless and Numba-compatible where possible.

Formulas:
  Optimal Spread:     δ = γ × (σ×100)² × T + (2/γ) × ln(1 + γ/κ)
  Reservation Price:  r = s - q × γ × σ² × T × s

Note: The σ×100 scaling matches aimm's implementation to ensure
micro-volatility (0.01%) has meaningful impact on spread.
"""

import math
import numpy as np
from typing import Dict, Optional


def optimal_spread(
    sigma: float,
    gamma: float,
    kappa: float,
    T: float = 0.5,
    min_spread: float = 0.001,
    max_spread: float = 0.02,
) -> float:
    """
    Avellaneda-Stoikov optimal spread calculation.

    Matches aimm MarketMaker._calculate_dynamic_spread() exactly.

    Args:
        sigma: Volatility (std of 20-period returns), raw scale
        gamma: Risk aversion parameter
        kappa: Order book liquidity parameter
        T: Time factor (0.5 = neutral for 24/7 crypto)
        min_spread: Minimum spread clamp
        max_spread: Maximum spread clamp

    Returns:
        Optimal spread as fraction (e.g., 0.005 = 0.5%)
    """
    if gamma <= 0 or kappa <= 0:
        return min_spread

    # Scale σ by 100 — matches aimm's implementation
    # Without scaling: 0.01% σ → σ²=0.000001% → negligible
    # With scaling: 0.01% × 100 = 1% → σ²=0.01% → meaningful
    sigma_scaled = sigma * 100.0

    # Term 1: Risk-adjusted volatility component
    term1 = gamma * (sigma_scaled ** 2) * T

    # Term 2: Liquidity-adjusted component (base spread)
    term2 = (2.0 / gamma) * math.log(1.0 + gamma / kappa)

    spread = term1 + term2

    # Clamp to configured bounds
    return max(min_spread, min(max_spread, spread))


def reservation_price(
    mid: float,
    q: float,
    gamma: float,
    sigma: float,
    T: float = 0.5,
) -> float:
    """
    Avellaneda-Stoikov reservation price.

    Matches aimm MarketMaker._calculate_reservation_price() exactly.

    Args:
        mid: Current mid price
        q: Inventory ratio (-1 to +1)
        gamma: Risk aversion parameter
        sigma: Volatility (raw scale, NOT ×100)
        T: Time factor

    Returns:
        Reservation price (shifted mid based on inventory)
    """
    adjustment = q * gamma * (sigma ** 2) * T * mid
    return mid - adjustment


def volatility_sigma_from_prices(prices: np.ndarray, window: int = 20) -> float:
    """
    Calculate volatility (σ) from price array.

    Matches aimm MarketMaker._calculate_volatility_sigma() exactly.
    Uses rolling standard deviation of simple returns over `window` periods.

    Args:
        prices: Array of prices (at least `window` elements)
        window: Lookback window (default 20)

    Returns:
        Volatility σ, clamped to [0.0001, 0.1]
    """
    if len(prices) < window:
        return 0.001  # Default low volatility (matches aimm)

    recent = prices[-window:]
    # Simple returns (matching aimm's implementation)
    returns = np.diff(recent) / recent[:-1]

    if len(returns) < 5:
        return 0.001

    sigma = float(np.std(returns, ddof=1))  # ddof=1 matches statistics.stdev
    return max(0.0001, min(0.1, sigma))


def calculate_spread_with_regime(
    sigma: float,
    regime_params: Dict[str, Dict],
    regime_name: str,
    min_spread: float = 0.001,
    max_spread: float = 0.02,
) -> float:
    """
    Calculate spread using regime-specific γ and κ parameters.

    Args:
        sigma: Current volatility
        regime_params: Dict of regime → {gamma, kappa, ...}
        regime_name: Current regime name (e.g., 'low_vol', 'trend_up')
        min_spread: Minimum spread
        max_spread: Maximum spread

    Returns:
        Optimal spread for the given regime
    """
    if regime_name not in regime_params:
        # Fallback: use first available or defaults
        gamma, kappa = 1.0, 1000.0
    else:
        p = regime_params[regime_name]
        gamma = float(p.get('gamma', 1.0))
        kappa = float(p.get('kappa', 1000.0))

    return optimal_spread(sigma, gamma, kappa,
                          min_spread=min_spread, max_spread=max_spread)


def blend_regime_params(
    regime_probs: Dict[str, float],
    regime_params: Dict[str, Dict],
) -> Dict[str, float]:
    """
    Blend regime parameters using probability weights.

    Matches aimm MarketMaker._calculate_dynamic_spread() blending logic.

    Args:
        regime_probs: Dict of regime → probability (should sum to ~1.0)
        regime_params: Dict of regime → {gamma, kappa, skew_factor, ...}

    Returns:
        Blended parameters dict
    """
    param_keys = [
        'gamma', 'kappa', 'skew_factor', 'price_tolerance',
        'grid_spacing', 'order_size_mult', 'grid_layers', 'max_position_mult'
    ]

    blended = {k: 0.0 for k in param_keys}

    for regime, prob in regime_probs.items():
        if regime not in regime_params:
            continue
        r_params = regime_params[regime]
        for key in param_keys:
            blended[key] += r_params.get(key, 0.0) * prob

    return blended


# --- Vectorized versions for fast backtesting ---

def vectorized_spreads(
    sigmas: np.ndarray,
    gamma: float,
    kappa: float,
    T: float = 0.5,
    min_spread: float = 0.001,
    max_spread: float = 0.02,
) -> np.ndarray:
    """
    Calculate optimal spreads for an entire array of volatilities at once.

    This is the vectorized equivalent of calling optimal_spread() in a loop.
    ~100x faster than the loop version for large arrays.

    Args:
        sigmas: Array of volatility values
        gamma, kappa, T, min_spread, max_spread: Same as optimal_spread()

    Returns:
        Array of optimal spreads
    """
    sigma_scaled = sigmas * 100.0
    term1 = gamma * (sigma_scaled ** 2) * T
    term2 = (2.0 / gamma) * math.log(1.0 + gamma / kappa)
    spreads = term1 + term2
    return np.clip(spreads, min_spread, max_spread)


def vectorized_reservation_prices(
    mids: np.ndarray,
    q: np.ndarray,
    gamma: float,
    sigmas: np.ndarray,
    T: float = 0.5,
) -> np.ndarray:
    """
    Calculate reservation prices for arrays of inputs.

    Args:
        mids: Array of mid prices
        q: Array of inventory ratios
        gamma: Risk aversion parameter
        sigmas: Array of volatilities
        T: Time factor

    Returns:
        Array of reservation prices
    """
    adjustments = q * gamma * (sigmas ** 2) * T * mids
    return mids - adjustments


def rolling_volatility(closes: np.ndarray, window: int = 20) -> np.ndarray:
    """
    Pre-compute rolling volatility for entire price series.

    Replaces: calling _calculate_volatility_sigma() at each tick.

    Args:
        closes: Array of close prices (full series)
        window: Lookback window

    Returns:
        Array of σ values (same length as closes, NaN-padded at start)
    """
    n = len(closes)
    sigmas = np.full(n, 0.001)  # Default

    if n < window:
        return sigmas

    # Simple returns
    returns = np.diff(closes) / closes[:-1]  # length n-1

    # Rolling std with ddof=1 (matches statistics.stdev)
    for i in range(window - 1, len(returns)):
        window_returns = returns[i - window + 1: i + 1]
        s = float(np.std(window_returns, ddof=1))
        sigmas[i + 1] = max(0.0001, min(0.1, s))

    return sigmas
