"""
Dynamic Order Sizer - Phase 1.1
Risk-based order sizing considering inventory, volatility, and order book depth.

Author: Antigravity
Version: 1.0.0
"""

import numpy as np
from typing import Tuple
import logging

class DynamicOrderSizer:
    """
    Dynamic order size calculation based on:
    1. Inventory position (asymmetric sizing)
    2. Market volatility
    3. Order book depth
    
    Logic:
    - Long inventory → smaller bid, larger ask (reduce position)
    - Short inventory → larger bid, smaller ask
    - High volatility → reduce both sizes
    - Thin order book → reduce both sizes
    """
    
    def __init__(self, 
                 base_size: float = 200.0,
                 max_inventory: float = 5000.0,
                 min_size_ratio: float = 0.2,
                 max_size_ratio: float = 1.5):
        """
        Args:
            base_size: Default order size in USD
            max_inventory: Maximum allowed inventory position
            min_size_ratio: Minimum size as ratio of base (0.2 = 20%)
            max_size_ratio: Maximum size as ratio of base (1.5 = 150%)
        """
        self.base_size = base_size
        self.max_inventory = max_inventory
        self.min_size_ratio = min_size_ratio
        self.max_size_ratio = max_size_ratio
        self.logger = logging.getLogger("DynamicOrderSizer")
    
    def calculate(self, 
                  inventory: float, 
                  volatility: float, 
                  book_depth: float = 10.0) -> Tuple[float, float]:
        """
        Calculate dynamic bid and ask sizes.
        
        Args:
            inventory: Current inventory position in USD (positive = long)
            volatility: Current market volatility (e.g., 0.01 = 1%)
            book_depth: Order book depth near best price
            
        Returns:
            Tuple of (bid_size, ask_size)
        """
        # 1. Inventory ratio (-1 to 1)
        inv_ratio = np.clip(inventory / self.max_inventory, -1.0, 1.0)
        
        # 2. Inventory-based asymmetric multipliers
        # Long (positive inventory) → reduce bid, increase ask
        # Short (negative inventory) → increase bid, reduce ask
        inventory_sensitivity = 0.4
        bid_inv_mult = 1 - inv_ratio * inventory_sensitivity
        ask_inv_mult = 1 + inv_ratio * inventory_sensitivity
        
        # 3. Volatility adjustment (high vol → smaller sizes)
        # vol=0.01 (1%) → mult=0.5, vol=0.001 (0.1%) → mult≈0.91
        vol_mult = 1 / (1 + volatility * 50)
        vol_mult = np.clip(vol_mult, 0.3, 1.0)
        
        # 4. Order book depth adjustment
        # thin book (depth < 5) → reduce size
        depth_mult = np.clip(book_depth / 10.0, 0.3, 1.0)
        
        # 5. Final calculation
        bid_size = self.base_size * bid_inv_mult * vol_mult * depth_mult
        ask_size = self.base_size * ask_inv_mult * vol_mult * depth_mult
        
        # 6. Apply min/max limits
        min_size = self.base_size * self.min_size_ratio
        max_size = self.base_size * self.max_size_ratio
        
        bid_size = np.clip(bid_size, min_size, max_size)
        ask_size = np.clip(ask_size, min_size, max_size)
        
        return float(bid_size), float(ask_size)
    
    def get_sizing_info(self, 
                        inventory: float, 
                        volatility: float, 
                        book_depth: float = 10.0) -> dict:
        """
        Get detailed sizing information for logging/debugging.
        """
        inv_ratio = np.clip(inventory / self.max_inventory, -1.0, 1.0)
        bid_size, ask_size = self.calculate(inventory, volatility, book_depth)
        
        return {
            'inventory': inventory,
            'inventory_ratio': inv_ratio,
            'volatility': volatility,
            'book_depth': book_depth,
            'bid_size': bid_size,
            'ask_size': ask_size,
            'bid_vs_base': bid_size / self.base_size,
            'ask_vs_base': ask_size / self.base_size,
            'asymmetry': ask_size / bid_size if bid_size > 0 else 0
        }


# Unit tests
if __name__ == "__main__":
    sizer = DynamicOrderSizer(base_size=200, max_inventory=5000)
    
    print("=" * 60)
    print("DYNAMIC ORDER SIZER TESTS")
    print("=" * 60)
    
    test_cases = [
        # (inventory, volatility, book_depth, description)
        (0, 0.005, 10, "Neutral inventory, normal vol"),
        (2500, 0.005, 10, "50% Long, normal vol"),
        (-2500, 0.005, 10, "50% Short, normal vol"),
        (0, 0.02, 10, "Neutral, HIGH volatility"),
        (0, 0.005, 3, "Neutral, THIN order book"),
        (4000, 0.015, 5, "80% Long, high vol, thin book"),
    ]
    
    for inv, vol, depth, desc in test_cases:
        bid, ask = sizer.calculate(inv, vol, depth)
        asymm = ask / bid if bid > 0 else 0
        print(f"\n{desc}:")
        print(f"  Inventory: ${inv:,.0f} | Vol: {vol:.1%} | Depth: {depth}")
        print(f"  Bid: ${bid:.2f} | Ask: ${ask:.2f} | Asymmetry: {asymm:.2f}x")
