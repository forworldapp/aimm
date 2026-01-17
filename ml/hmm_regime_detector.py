"""
HMM Regime Detection Model (Phase 4 - HMM Upgrade)
- Uses Hidden Markov Model for temporal regime classification
- Captures regime persistence (states tend to persist over time)
- Same interface as RegimeDetector for drop-in replacement
"""

import pandas as pd
import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler
import pickle
import os

class HMMRegimeDetector:
    """
    HMM-based market regime detector.
    
    Key advantage over GMM: Captures temporal dependencies between states.
    Markets tend to stay in trends/ranges, HMM models this behavior.
    """
    
    REGIME_NAMES = {
        0: "low_vol",
        1: "trend_up",
        2: "trend_down", 
        3: "high_vol"
    }
    
    # Same parameters as GMM for fair comparison
    REGIME_PARAMS = {
        "low_vol": {
            "gamma": 1.5, 
            "kappa": 2000, 
            "skew_factor": 0.003,
            "price_tolerance": 0.001,
            "grid_spacing": 0.0010,
            "order_size_mult": 1.0,
            "grid_layers": 10,
            "max_position_mult": 1.4,
            "description": "Tight spread, stable - aggressive accumulation"
        },
        "trend_up": {
            "gamma": 0.5, 
            "kappa": 500, 
            "skew_factor": 0.008,
            "price_tolerance": 0.0015,
            "grid_spacing": 0.0015,
            "order_size_mult": 0.8,
            "grid_layers": 7,
            "max_position_mult": 1.0,
            "description": "Wide spread, favor sells - ride the trend"
        },
        "trend_down": {
            "gamma": 0.5, 
            "kappa": 500, 
            "skew_factor": 0.008,
            "price_tolerance": 0.0015,
            "grid_spacing": 0.0015,
            "order_size_mult": 0.8,
            "grid_layers": 7,
            "max_position_mult": 1.0,
            "description": "Wide spread, favor buys - accumulate dips"
        },
        "high_vol": {
            "gamma": 0.3, 
            "kappa": 200, 
            "skew_factor": 0.002,
            "price_tolerance": 0.002,
            "grid_spacing": 0.0020,
            "order_size_mult": 0.7,
            "grid_layers": 5,
            "max_position_mult": 0.6,
            "description": "Wide spread, conservative - survival mode"
        }
    }
    
    def __init__(self, model_path="data/regime_model_hmm.pkl"):
        self.model_path = model_path
        self.model = None
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.cluster_to_regime = {}
        
        # Try to load existing model
        if os.path.exists(model_path):
            self.load_model()
    
    def _calculate_features(self, df: pd.DataFrame) -> np.ndarray:
        """
        Calculate features for regime detection.
        Same features as GMM for fair comparison.
        """
        df = df.copy()
        
        # Ensure numeric types
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Returns (momentum)
        df['returns'] = df['close'].pct_change()
        
        # Volatility (20-period rolling std)
        df['volatility'] = df['returns'].rolling(20).std()
        
        # ATR Ratio
        df['tr'] = np.maximum(
            df['high'] - df['low'],
            np.maximum(
                abs(df['high'] - df['close'].shift(1)),
                abs(df['low'] - df['close'].shift(1))
            )
        )
        df['atr'] = df['tr'].rolling(14).mean()
        df['atr_ratio'] = df['atr'] / df['close']
        
        # Momentum (10-period)
        df['momentum'] = df['close'].pct_change(10)
        
        # Drop NaN rows
        df = df.dropna()
        
        # Select features
        features = df[['returns', 'volatility', 'atr_ratio', 'momentum']].values
        
        return features, df.index
    
    def fit(self, df: pd.DataFrame, n_components=4):
        """
        Train the HMM regime detection model on historical data.
        """
        features, indices = self._calculate_features(df)
        
        if len(features) < 100:
            raise ValueError(f"Not enough data: {len(features)} rows (need 100+)")
        
        # Scale features
        features_scaled = self.scaler.fit_transform(features)
        
        # Hidden Markov Model with Gaussian emissions
        self.model = GaussianHMM(
            n_components=n_components,
            covariance_type='full',
            n_iter=100,
            random_state=42
        )
        
        # Fit on sequence (HMM considers temporal order)
        self.model.fit(features_scaled)
        labels = self.model.predict(features_scaled)
        
        # Analyze clusters to assign regime names
        self._analyze_clusters(features, labels)
        
        self.is_fitted = True
        print(f"HMM Model fitted on {len(features)} samples")
        
        # Print transition matrix
        print("\n=== State Transition Matrix ===")
        trans = self.model.transmat_
        for i in range(n_components):
            regime_name = self.cluster_to_regime.get(i, f"state_{i}")
            probs = [f"{p:.2f}" for p in trans[i]]
            print(f"{regime_name}: {probs}")
        
        return labels
    
    def _analyze_clusters(self, features, labels):
        """
        Analyze state characteristics to map to regime names.
        Same logic as GMM for consistency.
        """
        df_analysis = pd.DataFrame(features, columns=['returns', 'volatility', 'atr_ratio', 'momentum'])
        df_analysis['cluster'] = labels
        
        cluster_stats = df_analysis.groupby('cluster').mean()
        print("\n=== State Analysis ===")
        print(cluster_stats)
        
        vol_order = cluster_stats['volatility'].sort_values()
        mom_order = cluster_stats['momentum'].sort_values()
        
        self.cluster_to_regime = {
            vol_order.index[0]: "low_vol",
            vol_order.index[-1]: "high_vol",
            mom_order.index[-1]: "trend_up",
            mom_order.index[0]: "trend_down"
        }
        
        # Handle overlap
        assigned = set()
        final_mapping = {}
        for cluster, regime in self.cluster_to_regime.items():
            if regime not in assigned:
                final_mapping[cluster] = regime
                assigned.add(regime)
        
        all_regimes = set(self.REGIME_PARAMS.keys())
        for i in range(4):
            if i not in final_mapping:
                remaining = all_regimes - assigned
                if remaining:
                    final_mapping[i] = remaining.pop()
                    assigned.add(final_mapping[i])
        
        self.cluster_to_regime = final_mapping
        print(f"\nState to Regime Mapping: {self.cluster_to_regime}")
    
    def predict(self, df: pd.DataFrame) -> str:
        """
        Predict current market regime from recent price data.
        """
        if not self.is_fitted:
            return "unknown"
        
        features, _ = self._calculate_features(df)
        
        if len(features) == 0:
            return "unknown"
        
        features_scaled = self.scaler.transform(features)
        
        # HMM predicts sequence, take last state
        states = self.model.predict(features_scaled)
        latest_state = states[-1]
        regime = self.cluster_to_regime.get(latest_state, "unknown")
        
        return regime

    def predict_proba(self, df: pd.DataFrame) -> dict:
        """
        Predict regime probabilities using HMM posterior.
        """
        if not self.is_fitted:
            return {}
        
        features, _ = self._calculate_features(df)
        if len(features) == 0:
            return {}
            
        features_scaled = self.scaler.transform(features)
        
        # Get posterior probabilities for each state
        _, posteriors = self.model.score_samples(features_scaled)
        latest_probs = posteriors[-1]
        
        # Map state probs to regime names
        regime_probs = {}
        for state_idx, prob in enumerate(latest_probs):
            regime_name = self.cluster_to_regime.get(state_idx, f"state_{state_idx}")
            regime_probs[regime_name] = regime_probs.get(regime_name, 0.0) + prob
            
        return regime_probs
    
    def fetch_binance_candles(self, symbol="BTCUSDT", interval="1h", limit=100) -> pd.DataFrame:
        """
        Fetch recent candles from Binance API for live prediction.
        """
        import requests
        
        url = "https://api.binance.com/api/v3/klines"
        params = {
            "symbol": symbol,
            "interval": interval,
            "limit": limit
        }
        
        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            df = pd.DataFrame(data, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                'taker_buy_quote', 'ignore'
            ])
            
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col])
            
            return df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        except Exception as e:
            print(f"Failed to fetch Binance data: {e}")
            return pd.DataFrame()
    
    def predict_live(self, symbol="BTCUSDT") -> str:
        """
        Predict current regime using live Binance data.
        """
        if not self.is_fitted:
            return "unknown"
        
        df = self.fetch_binance_candles(symbol=symbol, interval="1h", limit=100)
        
        if df.empty or len(df) < 50:
            return "unknown"
        
        return self.predict(df)

    def predict_live_proba(self, symbol="BTCUSDT") -> dict:
        """
        Fetch recent data and predict regime probabilities for live trading.
        """
        if not self.is_fitted:
            return {}

        try:
            df = self.fetch_binance_candles(symbol=symbol, limit=100)
            if len(df) < 50:
                print(f"Not enough live data: {len(df)}")
                return {}
            return self.predict_proba(df)
        except Exception as e:
            print(f"Live probability prediction error: {e}")
            return {}
    
    def get_params_for_regime(self, regime: str) -> dict:
        """
        Get A&S parameters for the detected regime.
        """
        return self.REGIME_PARAMS.get(regime, self.REGIME_PARAMS["low_vol"])
    
    def save_model(self):
        """Save trained model to disk."""
        if not self.is_fitted:
            raise ValueError("Model not fitted yet")
        
        data = {
            'model': self.model,
            'scaler': self.scaler,
            'cluster_to_regime': self.cluster_to_regime
        }
        
        with open(self.model_path, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"HMM Model saved to {self.model_path}")
    
    def load_model(self):
        """Load trained model from disk."""
        try:
            with open(self.model_path, 'rb') as f:
                data = pickle.load(f)
            
            self.model = data['model']
            self.scaler = data['scaler']
            self.cluster_to_regime = data['cluster_to_regime']
            self.is_fitted = True
            
            print(f"HMM Model loaded from {self.model_path}")
        except Exception as e:
            print(f"Failed to load HMM model: {e}")
            self.is_fitted = False


# Training script
if __name__ == "__main__":
    # Load BTC data
    df = pd.read_csv("data/btc_hourly_1year.csv")
    print(f"Loaded {len(df)} rows of BTC data")
    
    # Initialize and train
    detector = HMMRegimeDetector()
    labels = detector.fit(df)
    
    # Save model
    detector.save_model()
    
    # Test prediction
    regime = detector.predict(df.tail(100))
    params = detector.get_params_for_regime(regime)
    probs = detector.predict_proba(df.tail(100))
    
    print(f"\n=== Current Regime ===")
    print(f"Regime: {regime}")
    print(f"Probabilities: {probs}")
    print(f"Recommended γ: {params['gamma']}, κ: {params['kappa']}")
    print(f"Description: {params['description']}")
    
    # Distribution of regimes
    print(f"\n=== Regime Distribution ===")
    regime_names = [detector.cluster_to_regime.get(l, 'unknown') for l in labels]
    for regime in set(regime_names):
        count = regime_names.count(regime)
        pct = count / len(regime_names) * 100
        print(f"{regime}: {count} ({pct:.1f}%)")
