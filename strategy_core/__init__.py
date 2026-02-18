"""
strategy_core — Shared strategy logic between aimm and aimm-lab.

All functions here are pure (no side effects, no I/O, no async).
They can be used by both:
  - aimm's MarketMaker (async, real-time)
  - aimm-lab's FastBacktester (sync, vectorized)
"""

from strategy_core.spread import (
    optimal_spread,
    reservation_price,
    calculate_spread_with_regime,
    volatility_sigma_from_prices,
)
from strategy_core.grid import generate_grid_prices
from strategy_core.risk import check_circuit_breaker, check_max_drawdown
from strategy_core.config import load_config, load_regime_params
