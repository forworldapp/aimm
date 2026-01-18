"""
Contextual Bandit for Spread Optimization - Phase 4.1
Uses Thompson Sampling to find optimal spread for given market context.

Author: Antigravity
Version: 1.0.0
"""

import numpy as np
from typing import List, Tuple, Optional
import logging


class ContextualBanditSpread:
    """
    Contextual Bandit for dynamic spread optimization.
    
    Uses Thompson Sampling with Bayesian Linear Regression.
    
    Context features:
    - volatility: Current market volatility
    - inventory_ratio: Current inventory position (-1 to 1)
    - regime: Market regime (encoded as 0-3)
    - book_imbalance: Order book imbalance
    - hour_of_day: Trading hour (0-23)
    
    Arms:
    - Different spread levels (e.g., 3, 5, 8, 10, 15, 20 bps)
    
    Reward:
    - PnL from fills at that spread level
    """
    
    def __init__(self, 
                 spread_levels: List[float] = None,
                 context_dim: int = 5,
                 prior_variance: float = 1.0):
        """
        Args:
            spread_levels: List of spread levels in bps
            context_dim: Number of context features
            prior_variance: Prior variance for Bayesian updates
        """
        self.spread_levels = spread_levels or [3, 5, 8, 10, 15, 20, 25]
        self.n_arms = len(self.spread_levels)
        self.context_dim = context_dim
        
        # Bayesian Linear Regression parameters for each arm
        # θ ~ N(μ, Σ)
        self.mu = np.zeros((self.n_arms, context_dim))
        self.sigma = np.array([np.eye(context_dim) * prior_variance 
                               for _ in range(self.n_arms)])
        
        # Precision matrix (inverse of covariance)
        self.precision = np.array([np.eye(context_dim) / prior_variance 
                                   for _ in range(self.n_arms)])
        
        # Sufficient statistics for updates
        self.b = np.zeros((self.n_arms, context_dim))
        
        # Observation noise variance
        self.noise_var = 1.0
        
        # History for analysis
        self.history: List[dict] = []
        self.total_reward = 0
        self.arm_counts = np.zeros(self.n_arms, dtype=int)
        
        self.logger = logging.getLogger("ContextualBandit")
    
    def extract_context(self, 
                        volatility: float,
                        inventory_ratio: float,
                        regime: int,
                        book_imbalance: float,
                        hour_of_day: int = 12) -> np.ndarray:
        """
        Extract normalized context vector.
        """
        context = np.array([
            volatility * 100,  # Scale to ~0-2 range
            inventory_ratio,   # Already -1 to 1
            regime / 3,        # Normalize to 0-1
            book_imbalance,    # Already -1 to 1
            (hour_of_day - 12) / 12  # Normalize to -1 to 1
        ])
        return context
    
    def select_spread(self, context: np.ndarray) -> Tuple[float, int]:
        """
        Select spread using Thompson Sampling.
        
        Args:
            context: Context feature vector
            
        Returns:
            Tuple of (selected_spread_bps, arm_index)
        """
        samples = []
        
        for arm in range(self.n_arms):
            # Sample θ from posterior
            try:
                theta = np.random.multivariate_normal(
                    self.mu[arm], 
                    self.sigma[arm]
                )
            except:
                # Fallback if covariance is not positive definite
                theta = self.mu[arm] + np.random.randn(self.context_dim) * 0.1
            
            # Expected reward = context @ θ
            expected_reward = context @ theta
            samples.append(expected_reward)
        
        # Select arm with highest sampled reward
        best_arm = int(np.argmax(samples))
        selected_spread = self.spread_levels[best_arm]
        
        self.arm_counts[best_arm] += 1
        
        return float(selected_spread), best_arm
    
    def update(self, context: np.ndarray, arm: int, reward: float):
        """
        Update posterior with observed reward.
        
        Uses Bayesian update for linear regression:
        Σ_new = (Σ_prev^{-1} + x x^T / σ²)^{-1}
        μ_new = Σ_new @ (Σ_prev^{-1} @ μ_prev + x r / σ²)
        """
        x = context.reshape(-1, 1)
        
        # Update precision matrix
        self.precision[arm] += x @ x.T / self.noise_var
        
        # Update sufficient statistics
        self.b[arm] += context * reward / self.noise_var
        
        # Recompute covariance (inverse of precision)
        try:
            self.sigma[arm] = np.linalg.inv(self.precision[arm])
            self.mu[arm] = self.sigma[arm] @ self.b[arm]
        except:
            # Fallback if matrix is singular
            self.sigma[arm] = np.eye(self.context_dim) * 0.1
        
        # Track history
        self.total_reward += reward
        self.history.append({
            'context': context.tolist(),
            'arm': arm,
            'spread': self.spread_levels[arm],
            'reward': reward
        })
    
    def get_arm_stats(self) -> dict:
        """Get statistics per arm."""
        stats = {}
        for i, spread in enumerate(self.spread_levels):
            arm_history = [h for h in self.history if h['arm'] == i]
            if arm_history:
                rewards = [h['reward'] for h in arm_history]
                stats[f"{spread}bps"] = {
                    'count': len(arm_history),
                    'avg_reward': float(np.mean(rewards)),
                    'total': float(np.sum(rewards))
                }
            else:
                stats[f"{spread}bps"] = {'count': 0, 'avg_reward': 0, 'total': 0}
        return stats
    
    def get_recommended_spread(self, context: np.ndarray) -> float:
        """
        Get recommended spread (exploitation only, no exploration).
        Uses mean of posterior instead of sampling.
        """
        expected_rewards = []
        
        for arm in range(self.n_arms):
            expected_reward = context @ self.mu[arm]
            expected_rewards.append(expected_reward)
        
        best_arm = int(np.argmax(expected_rewards))
        return float(self.spread_levels[best_arm])
    
    def get_stats(self) -> dict:
        """Get overall statistics."""
        return {
            'total_observations': len(self.history),
            'total_reward': float(self.total_reward),
            'arm_counts': self.arm_counts.tolist(),
            'spread_levels': self.spread_levels
        }


# Unit tests
if __name__ == "__main__":
    print("=" * 60)
    print("CONTEXTUAL BANDIT SPREAD TESTS")
    print("=" * 60)
    
    bandit = ContextualBanditSpread()
    
    # Simulate training
    np.random.seed(42)
    
    print("\nSimulating 100 observations...")
    for i in range(100):
        # Random context
        context = bandit.extract_context(
            volatility=np.random.uniform(0.005, 0.02),
            inventory_ratio=np.random.uniform(-0.5, 0.5),
            regime=np.random.randint(0, 4),
            book_imbalance=np.random.uniform(-0.3, 0.3),
            hour_of_day=np.random.randint(0, 24)
        )
        
        # Select spread
        spread, arm = bandit.select_spread(context)
        
        # Simulate reward (tighter spread = higher fill but lower margin)
        # Optimal is around 8-10 bps for most contexts
        optimal = 8
        noise = np.random.randn() * 2
        reward = 10 - 0.5 * abs(spread - optimal) + noise
        
        # Update
        bandit.update(context, arm, reward)
    
    print("\nArm Statistics:")
    for spread, stats in bandit.get_arm_stats().items():
        print(f"  {spread}: {stats['count']} pulls, avg reward: {stats['avg_reward']:.2f}")
    
    print("\nTest Recommendations:")
    test_contexts = [
        (0.005, 0, 1, 0, "Low vol, neutral"),
        (0.02, 0.5, 3, 0.3, "High vol, long inventory"),
        (0.01, -0.3, 0, -0.2, "Medium vol, short"),
    ]
    
    for vol, inv, regime, imb, desc in test_contexts:
        ctx = bandit.extract_context(vol, inv, regime, imb)
        spread, _ = bandit.select_spread(ctx)
        rec = bandit.get_recommended_spread(ctx)
        print(f"  {desc}: Selected={spread:.0f}bps, Recommended={rec:.0f}bps")
