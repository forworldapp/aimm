"""
RL Agent Training Script - Extended 50k steps
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from rl.train_agent import train_ppo_agent, evaluate_agent, SB3_AVAILABLE

if __name__ == "__main__":
    if not SB3_AVAILABLE:
        print("stable-baselines3 not installed")
        exit(1)
    
    # Load real data if available
    data_path = "data/btcusdt_1m_1year.csv"
    if os.path.exists(data_path):
        print(f"Loading real data from {data_path}...")
        data = pd.read_csv(data_path)
        data = data.rename(columns={'timestamp': 'timestamp'})
    else:
        print("Using synthetic data...")
        data = None
    
    # Extended training
    model = train_ppo_agent(
        data=data,
        total_timesteps=50000,  # 50k steps
        save_path="models/mm_ppo_50k",
        learning_rate=3e-4,
        batch_size=64,
        verbose=1
    )
    
    if model:
        # Evaluate
        results = evaluate_agent(model, data=data, n_episodes=10)
        
        print("\n" + "=" * 60)
        if results['mean_pnl'] > 0:
            print("✅ RL AGENT TRAINING: SUCCESSFUL")
        else:
            print("⚠️ RL AGENT: Needs more training")
        print("=" * 60)
