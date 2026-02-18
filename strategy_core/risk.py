"""
Risk Management — Pure Functions.

Extracted from aimm/strategies/market_maker.py cycle() risk checks.
"""

import numpy as np
from typing import Dict, Tuple


def check_circuit_breaker(
    total_pnl: float,
    max_loss_usd: float,
) -> bool:
    """
    Check if bot should stop due to max loss.

    Matches aimm MarketMaker.cycle() circuit breaker check.

    Args:
        total_pnl: Bot's total P&L (realized + unrealized)
        max_loss_usd: Maximum loss threshold

    Returns:
        True if circuit breaker TRIGGERED (should stop)
    """
    return total_pnl < -max_loss_usd


def check_max_drawdown(
    total_pnl: float,
    cost_basis: float,
    max_drawdown_pct: float = 0.15,
) -> bool:
    """
    Check if max drawdown exceeded.

    Matches aimm MarketMaker.cycle() drawdown check.

    Args:
        total_pnl: Bot's total P&L
        cost_basis: Total cost basis of bot trades
        max_drawdown_pct: Maximum drawdown percentage

    Returns:
        True if drawdown EXCEEDED (should stop)
    """
    if cost_basis <= 0 or total_pnl >= 0:
        return False

    dd_pct = abs(total_pnl) / cost_basis
    return dd_pct > max_drawdown_pct


def calculate_inventory_ratio(
    position_qty: float,
    max_position_qty: float,
) -> float:
    """
    Calculate inventory ratio for A&S reservation price.

    Args:
        position_qty: Current position in base currency
        max_position_qty: Maximum position in base currency

    Returns:
        Inventory ratio in [-1, +1]
    """
    if max_position_qty <= 0:
        return 0.0
    ratio = position_qty / max_position_qty
    return max(-1.0, min(1.0, ratio))


def equity_from_state(
    balance_usd: float,
    position_qty: float,
    mid_price: float,
) -> float:
    """Calculate total equity from state components."""
    return balance_usd + position_qty * mid_price


def calculate_pnl_metrics(equity_history: np.ndarray) -> Dict[str, float]:
    """
    Calculate performance metrics from equity curve.

    Args:
        equity_history: Array of equity values over time

    Returns:
        Dict with pnl, sharpe, max_drawdown, win_rate
    """
    if len(equity_history) < 10:
        return {'pnl': -999.0, 'sharpe': -999.0, 'max_dd': -99.0}

    eq = np.array(equity_history, dtype=np.float64)
    pnl = float(eq[-1] - eq[0])

    # Returns
    returns = np.diff(eq) / eq[:-1]
    returns = returns[np.isfinite(returns)]

    if len(returns) < 2 or np.std(returns) == 0:
        sharpe = 0.0
    else:
        # Annualized Sharpe (1-minute data → 252*24*60 periods/year)
        sharpe = float((np.mean(returns) / np.std(returns)) * np.sqrt(252 * 24 * 60))

    # Max Drawdown
    cummax = np.maximum.accumulate(eq)
    drawdowns = (eq - cummax) / cummax
    max_dd = float(np.min(drawdowns))

    return {
        'pnl': pnl,
        'sharpe': sharpe,
        'max_dd': max_dd,
    }
