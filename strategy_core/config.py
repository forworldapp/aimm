"""
Configuration Loading — Pure Functions.

Loads config.yaml and regime parameter JSONs compatible with aimm format.
"""

import json
import os
from typing import Any, Dict, Optional

import yaml


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """
    Load YAML configuration file.

    Compatible with aimm's config.yaml format.

    Args:
        config_path: Path to config.yaml

    Returns:
        Parsed config dict
    """
    with open(config_path, 'r') as f:
        return yaml.safe_load(f) or {}


def get_config(config: Dict, *keys, default=None) -> Any:
    """
    Nested config access (replaces aimm's Config.get()).

    Example: get_config(cfg, "strategy", "avellaneda_stoikov", "gamma")
    """
    current = config
    for key in keys:
        if isinstance(current, dict):
            current = current.get(key, default)
        else:
            return default
    return current


def load_regime_params(
    filepath: str = "data/optimized_regime_params.json",
) -> Dict[str, Dict]:
    """
    Load optimized regime parameters.

    Compatible with aimm's regime_params JSON format.

    Args:
        filepath: Path to regime params JSON

    Returns:
        Dict of regime_name → {gamma, kappa, grid_layers, ...}
    """
    if not os.path.exists(filepath):
        # Return sensible defaults
        return default_regime_params()

    with open(filepath, 'r') as f:
        data = json.load(f)

    # Handle both formats: flat dict or nested {'regimes': {...}} or {'results': {...}}
    if 'regimes' in data:
        params = {}
        for regime, info in data['regimes'].items():
            if isinstance(info, dict):
                # Filter to trading keys only
                trading_keys = ['gamma', 'kappa', 'skew_factor', 'price_tolerance',
                                'grid_spacing', 'order_size_mult', 'grid_layers', 'max_position_mult']
                params[regime] = {k: info[k] for k in trading_keys if k in info}
        return params

    if 'results' in data:
        params = {}
        for regime, info in data['results'].items():
            if isinstance(info, dict) and 'best_params' in info:
                params[regime] = info['best_params']
            elif isinstance(info, dict):
                params[regime] = info
        return params

    return data


def save_regime_params(
    params: Dict[str, Dict],
    filepath: str = "data/optimized_regime_params.json",
) -> None:
    """
    Save regime parameters in aimm-compatible format.

    Args:
        params: Dict of regime_name → param dict
        filepath: Output path
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(params, f, indent=2)


def default_regime_params() -> Dict[str, Dict]:
    """
    Default regime parameters matching aimm's config.yaml baseline.
    """
    return {
        'low_vol': {
            'gamma': 1.0,
            'kappa': 1000.0,
            'skew_factor': 0.005,
            'price_tolerance': 0.001,
            'grid_spacing': 0.002,
            'order_size_mult': 1.0,
            'grid_layers': 7,
            'max_position_mult': 1.0,
        },
        'trend_up': {
            'gamma': 0.8,
            'kappa': 800.0,
            'skew_factor': 0.01,
            'price_tolerance': 0.0015,
            'grid_spacing': 0.0025,
            'order_size_mult': 1.2,
            'grid_layers': 5,
            'max_position_mult': 1.2,
        },
        'trend_down': {
            'gamma': 1.2,
            'kappa': 1200.0,
            'skew_factor': -0.01,
            'price_tolerance': 0.0015,
            'grid_spacing': 0.0025,
            'order_size_mult': 0.8,
            'grid_layers': 5,
            'max_position_mult': 0.8,
        },
        'high_vol': {
            'gamma': 1.5,
            'kappa': 500.0,
            'skew_factor': 0.002,
            'price_tolerance': 0.002,
            'grid_spacing': 0.004,
            'order_size_mult': 0.6,
            'grid_layers': 4,
            'max_position_mult': 0.6,
        },
    }
