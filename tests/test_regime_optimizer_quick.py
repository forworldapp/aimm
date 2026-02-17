
import sys
import os
import logging

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.param_optimizer import RegimeParamOptimizer

def test_regime_differentiation():
    print("Initializing optimizer...")
    # Increase to 30,000 candles (~20 days) to ensure we have multiple regimes
    opt = RegimeParamOptimizer(candles=30000)
    
    # Check available regimes
    regime_counts = opt.df_train['regime'].value_counts()
    print(f"\nAvailable regimes in training data:\n{regime_counts}")
    
    available_regimes = regime_counts.index.tolist()
    if len(available_regimes) < 2:
        print("Not enough regimes in data sample to test differentiation. Increase candles.")
        return

    # Pick top 2 regimes to compare
    regimes_to_test = available_regimes[:2]
    print(f"\nTesting differentiation between: {regimes_to_test}")
    
    results = {}
    
    for regime in regimes_to_test:
        print(f"\n--- Testing {regime.upper()} ---")
        # Run 1 trial
        try:
            res = opt.optimize_regime(regime, n_trials=1)
            results[regime] = res
        except Exception as e:
            print(f"Optimization failed for {regime}: {e}")
            
    print("\n\n=== RESULTS COMPARISON ===")
    for regime in results:
        print(f"{regime.upper()} Train PnL: ${results[regime]['train']['pnl']:.2f}")
        
    vals = [res['train']['pnl'] for res in results.values()]
    if len(vals) >= 2 and vals[0] != vals[1]:
        print("\nSUCCESS: PnL values differ between regimes!")
    else:
        print("\nFAILURE: PnL values are identical or insufficient data.")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_regime_differentiation()
