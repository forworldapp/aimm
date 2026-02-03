"""
RL Agent Training - v6.0
CPU-friendly PPO training for market making
"""
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from typing import Optional
import logging

# Stable-Baselines3 imports (will be installed if not present)
try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False
    print("stable-baselines3 not installed. Run: pip install stable-baselines3")

from rl.mm_env import MarketMakingEnv


class TrainingProgressCallback(BaseCallback):
    """Custom callback for logging training progress"""
    
    def __init__(self, check_freq=1000, verbose=1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.best_mean_reward = -np.inf
    
    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            # Log progress
            if self.verbose > 0:
                print(f"Step {self.n_calls}: Training...")
        return True


def create_env(data: pd.DataFrame = None, **kwargs) -> MarketMakingEnv:
    """Create market making environment"""
    return MarketMakingEnv(data=data, **kwargs)


def train_ppo_agent(
    data: pd.DataFrame = None,
    total_timesteps: int = 50000,  # Reduced for CPU
    save_path: str = "models/mm_ppo",
    learning_rate: float = 3e-4,
    batch_size: int = 64,
    n_epochs: int = 10,
    verbose: int = 1
) -> Optional['PPO']:
    """
    Train PPO agent for market making
    
    Args:
        data: Historical price data (or None for synthetic)
        total_timesteps: Training steps
        save_path: Where to save the model
        learning_rate: PPO learning rate
        batch_size: Minibatch size
        n_epochs: PPO epochs per update
        verbose: Logging verbosity
        
    Returns:
        Trained PPO model
    """
    if not SB3_AVAILABLE:
        print("ERROR: stable-baselines3 not installed")
        return None
    
    print("=" * 60)
    print("RL Agent Training - v6.0")
    print("=" * 60)
    print(f"Total timesteps: {total_timesteps:,}")
    print(f"Learning rate: {learning_rate}")
    print(f"Batch size: {batch_size}")
    print()
    
    # Create environment
    env = DummyVecEnv([lambda: create_env(data)])
    
    # Create PPO model (CPU-optimized)
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=learning_rate,
        n_steps=2048,  # Steps per update
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,  # Exploration
        verbose=verbose,
        device="cpu"  # Force CPU
    )
    
    # Train
    print("Starting training...")
    callback = TrainingProgressCallback(check_freq=5000)
    
    model.learn(
        total_timesteps=total_timesteps,
        callback=callback,
        progress_bar=False  # Avoid tqdm/rich dependency
    )
    
    # Save model
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    model.save(save_path)
    print(f"\nModel saved to: {save_path}")
    
    return model


def evaluate_agent(
    model: 'PPO',
    data: pd.DataFrame = None,
    n_episodes: int = 10
) -> dict:
    """
    Evaluate trained agent
    
    Returns:
        Dictionary with evaluation metrics
    """
    env = create_env(data)
    
    total_rewards = []
    total_pnls = []
    total_trades = []
    
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=ep)
        episode_reward = 0
        done = False
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            done = terminated or truncated
        
        total_rewards.append(episode_reward)
        total_pnls.append(info['pnl'])
        total_trades.append(info['trades'])
    
    results = {
        'mean_reward': np.mean(total_rewards),
        'std_reward': np.std(total_rewards),
        'mean_pnl': np.mean(total_pnls),
        'std_pnl': np.std(total_pnls),
        'mean_trades': np.mean(total_trades),
        'episodes': n_episodes
    }
    
    print("\n" + "=" * 60)
    print("Evaluation Results")
    print("=" * 60)
    print(f"Mean Reward: {results['mean_reward']:.4f} ± {results['std_reward']:.4f}")
    print(f"Mean PnL: ${results['mean_pnl']:.2f} ± ${results['std_pnl']:.2f}")
    print(f"Mean Trades: {results['mean_trades']:.0f}")
    
    return results


class RLAgentWrapper:
    """
    Wrapper for using trained RL agent in MarketMaker
    """
    
    def __init__(self, model_path: str = "models/mm_ppo"):
        self.logger = logging.getLogger("RLAgent")
        self.model = None
        self.enabled = False
        
        self._load_model(model_path)
    
    def _load_model(self, model_path: str):
        """Load trained model"""
        if not SB3_AVAILABLE:
            self.logger.warning("stable-baselines3 not available")
            return
        
        try:
            if os.path.exists(model_path + ".zip"):
                self.model = PPO.load(model_path, device="cpu")
                self.enabled = True
                self.logger.info(f"✅ RL Agent loaded from {model_path}")
            else:
                self.logger.warning(f"Model not found at {model_path}")
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
    
    def get_action(
        self,
        mid_price_norm: float,
        spread: float,
        volatility: float,
        inventory_ratio: float,
        ret_1: float,
        ret_5: float,
        imbalance: float,
        pnl_ratio: float
    ) -> dict:
        """
        Get action from RL agent
        
        Returns:
            spread_mult: float
            size_mult: float
            skew: float
        """
        if not self.enabled or self.model is None:
            return {
                'spread_mult': 1.0,
                'size_mult': 1.0,
                'skew': 0.0,
                'rl_active': False
            }
        
        obs = np.array([
            mid_price_norm, spread, volatility, inventory_ratio,
            ret_1, ret_5, imbalance, pnl_ratio
        ], dtype=np.float32)
        
        action, _ = self.model.predict(obs, deterministic=True)
        
        return {
            'spread_mult': float(np.clip(action[0], 0.5, 2.0)),
            'size_mult': float(np.clip(action[1], 0.5, 1.5)),
            'skew': float(np.clip(action[2], -0.5, 0.5)),
            'rl_active': True
        }


if __name__ == "__main__":
    # Quick training test
    if SB3_AVAILABLE:
        model = train_ppo_agent(
            total_timesteps=10000,  # Short test
            verbose=1
        )
        
        if model:
            evaluate_agent(model, n_episodes=5)
    else:
        print("Please install stable-baselines3: pip install stable-baselines3")
