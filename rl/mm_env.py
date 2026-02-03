"""
Market Making Gymnasium Environment - v6.0
Custom environment for RL-based market making
"""
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Dict, Tuple, Optional, Any
import pandas as pd


class MarketMakingEnv(gym.Env):
    """
    Market Making Environment for Reinforcement Learning
    
    Observation Space:
        - Mid price (normalized)
        - Spread
        - Volatility (20-period)
        - Inventory position
        - Recent returns (5 periods)
        - Bid/Ask imbalance
    
    Action Space:
        - Spread multiplier (0.5x - 2.0x)
        - Size multiplier (0.5x - 1.5x)
        - Skew (-0.5 to +0.5)
    """
    
    metadata = {"render_modes": ["human"]}
    
    def __init__(
        self,
        data: pd.DataFrame = None,
        initial_balance: float = 10000,
        order_size_usd: float = 200,
        base_spread: float = 0.002,
        max_position_usd: float = 2000,
        episode_length: int = 1440,  # 1 day
        render_mode: str = None
    ):
        super().__init__()
        
        self.render_mode = render_mode
        self.data = data
        self.initial_balance = initial_balance
        self.order_size_usd = order_size_usd
        self.base_spread = base_spread
        self.max_position_usd = max_position_usd
        self.episode_length = episode_length
        
        # Observation: [mid_price_norm, spread, volatility, inventory_ratio, 
        #               return_1, return_5, bid_ask_imbalance, pnl_ratio]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32
        )
        
        # Action: [spread_mult, size_mult, skew]
        self.action_space = spaces.Box(
            low=np.array([0.5, 0.5, -0.5]),
            high=np.array([2.0, 1.5, 0.5]),
            dtype=np.float32
        )
        
        # State
        self.current_step = 0
        self.start_idx = 0
        self.balance = initial_balance
        self.inventory = 0.0
        self.pnl = 0.0
        self.trades = []
        
        # Synthetic data if none provided
        if self.data is None:
            self._generate_synthetic_data()
    
    def _generate_synthetic_data(self, length: int = 50000):
        """Generate synthetic price data for training"""
        np.random.seed(42)
        
        prices = [50000]
        for _ in range(length - 1):
            ret = np.random.normal(0, 0.0005)  # ~0.05% per minute
            prices.append(prices[-1] * (1 + ret))
        
        # Add volume and high/low
        self.data = pd.DataFrame({
            'close': prices,
            'high': [p * (1 + abs(np.random.normal(0, 0.001))) for p in prices],
            'low': [p * (1 - abs(np.random.normal(0, 0.001))) for p in prices],
            'volume': np.random.exponential(100, length)
        })
    
    def _get_observation(self) -> np.ndarray:
        """Build observation vector"""
        idx = self.start_idx + self.current_step
        
        if idx >= len(self.data) - 1:
            idx = len(self.data) - 2
        
        row = self.data.iloc[idx]
        mid_price = row['close']
        
        # Normalized mid price (relative to start)
        start_price = self.data.iloc[self.start_idx]['close']
        mid_norm = (mid_price / start_price) - 1
        
        # Current spread (from high-low)
        spread = (row['high'] - row['low']) / mid_price if mid_price > 0 else 0
        
        # Volatility (20-period)
        lookback = max(0, idx - 20)
        prices = self.data.iloc[lookback:idx+1]['close'].values
        if len(prices) > 1:
            returns = np.diff(np.log(prices))
            volatility = np.std(returns) * np.sqrt(1440)  # Annualized
        else:
            volatility = 0.01
        
        # Inventory ratio
        inventory_usd = self.inventory * mid_price
        inventory_ratio = inventory_usd / self.max_position_usd if self.max_position_usd > 0 else 0
        inventory_ratio = np.clip(inventory_ratio, -1, 1)
        
        # Recent returns
        if idx >= 5:
            ret_1 = (mid_price / self.data.iloc[idx-1]['close']) - 1
            ret_5 = (mid_price / self.data.iloc[idx-5]['close']) - 1
        else:
            ret_1, ret_5 = 0, 0
        
        # Bid/Ask imbalance (simulated)
        imbalance = np.random.normal(0, 0.1)
        
        # PnL ratio
        pnl_ratio = self.pnl / self.initial_balance if self.initial_balance > 0 else 0
        
        return np.array([
            mid_norm, spread, volatility, inventory_ratio,
            ret_1, ret_5, imbalance, pnl_ratio
        ], dtype=np.float32)
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one step in the environment
        
        Args:
            action: [spread_mult, size_mult, skew]
            
        Returns:
            observation, reward, terminated, truncated, info
        """
        spread_mult = np.clip(action[0], 0.5, 2.0)
        size_mult = np.clip(action[1], 0.5, 1.5)
        skew = np.clip(action[2], -0.5, 0.5)
        
        idx = self.start_idx + self.current_step
        if idx >= len(self.data) - 1:
            idx = len(self.data) - 2
        
        row = self.data.iloc[idx]
        mid_price = row['close']
        
        # Calculate bid/ask
        effective_spread = self.base_spread * spread_mult
        bid_price = mid_price * (1 - effective_spread / 2 + skew * effective_spread)
        ask_price = mid_price * (1 + effective_spread / 2 + skew * effective_spread)
        
        # Order sizes
        order_size_usd = self.order_size_usd * size_mult
        bid_size = order_size_usd / mid_price
        ask_size = order_size_usd / mid_price
        
        # Simulate fills
        bid_filled = row['low'] <= bid_price
        ask_filled = row['high'] >= ask_price
        
        step_pnl = 0.0
        adverse_selection = 0
        
        if bid_filled and ask_filled:
            # Round trip
            profit = min(bid_size, ask_size) * (ask_price - bid_price)
            step_pnl += profit
            self.trades.append(profit)
        elif bid_filled:
            # Check for adverse selection
            if idx + 1 < len(self.data):
                next_price = self.data.iloc[idx + 1]['close']
                if next_price < bid_price:
                    adverse_selection = 1
            
            # Add to inventory
            inventory_usd = self.inventory * mid_price
            if inventory_usd < self.max_position_usd:
                self.inventory += bid_size
        elif ask_filled and self.inventory > 0:
            # Check for adverse selection
            if idx + 1 < len(self.data):
                next_price = self.data.iloc[idx + 1]['close']
                if next_price > ask_price:
                    adverse_selection = 1
            
            sell_size = min(ask_size, self.inventory)
            self.inventory -= sell_size
        
        # Mark to market inventory
        if idx + 1 < len(self.data) and self.inventory != 0:
            next_price = self.data.iloc[idx + 1]['close']
            mtm_pnl = self.inventory * (next_price - mid_price)
            step_pnl += mtm_pnl
        
        self.pnl += step_pnl
        self.current_step += 1
        
        # Reward design
        # Positive for profit, penalty for adverse selection and large inventory
        inventory_penalty = abs(self.inventory * mid_price / self.max_position_usd) * 0.001
        adverse_penalty = adverse_selection * 0.005
        
        reward = step_pnl / self.initial_balance - inventory_penalty - adverse_penalty
        
        # Episode termination
        terminated = False
        truncated = self.current_step >= self.episode_length
        
        # Check for bankruptcy
        if self.balance + self.pnl <= 0:
            terminated = True
            reward = -1.0
        
        obs = self._get_observation()
        
        info = {
            'pnl': self.pnl,
            'inventory': self.inventory,
            'trades': len(self.trades),
            'adverse_selection': adverse_selection
        }
        
        return obs, reward, terminated, truncated, info
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """Reset the environment"""
        super().reset(seed=seed)
        
        # Random start point
        max_start = max(0, len(self.data) - self.episode_length - 100)
        if seed is not None:
            np.random.seed(seed)
        self.start_idx = np.random.randint(0, max_start) if max_start > 0 else 0
        
        self.current_step = 0
        self.balance = self.initial_balance
        self.inventory = 0.0
        self.pnl = 0.0
        self.trades = []
        
        return self._get_observation(), {}
    
    def render(self):
        """Render environment state"""
        if self.render_mode == "human":
            print(f"Step: {self.current_step} | PnL: ${self.pnl:.2f} | "
                  f"Inventory: {self.inventory:.4f} | Trades: {len(self.trades)}")


# Register the environment
try:
    gym.register(
        id='MarketMaking-v0',
        entry_point='rl.mm_env:MarketMakingEnv',
    )
except Exception:
    pass  # Already registered
