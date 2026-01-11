"""
ML Regime Detection Model (Phase 2)
- Uses K-Means clustering to classify market regimes
- 4 Regimes: Trending Up, Trending Down, Ranging, High Volatility
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import pickle
import os

class RegimeDetector:
    """
    ML-based market regime detector using K-Means clustering.
    
    Regimes:
    - 0: Low Volatility (Ranging) -> Tight spread
    - 1: Trending Up -> Skew towards selling
    - 2: Trending Down -> Skew towards buying
    - 3: High Volatility -> Wide spread
    """
    
    REGIME_NAMES = {
        0: "low_vol",
        1: "trend_up",
        2: "trend_down", 
        3: "high_vol"
    }
    
    # A&S Parameters for each regime (Extended with full trading params)
    REGIME_PARAMS = {
        "low_vol": {
            "gamma": 1.5, 
            "kappa": 2000, 
            "skew_factor": 0.003,       # Tight skew for stable market
            "price_tolerance": 0.001,    # 0.1% tolerance
            "grid_spacing": 0.0010,      # 0.10% grid (tight)
            "order_size_mult": 1.0,      # Normal order size
            "grid_layers": 10,           # More layers in stable market
            "max_position_mult": 1.4,    # 140% max position ($7,000)
            "description": "Tight spread, stable - aggressive accumulation"
        },
        "trend_up": {
            "gamma": 0.5, 
            "kappa": 500, 
            "skew_factor": 0.008,        # Higher skew to sell more
            "price_tolerance": 0.0015,   # 0.15% tolerance
            "grid_spacing": 0.0015,      # 0.15% grid
            "order_size_mult": 0.8,      # Smaller orders in trend
            "grid_layers": 7,            # Normal layers
            "max_position_mult": 1.0,    # Normal max position ($5,000)
            "description": "Wide spread, favor sells - ride the trend"
        },
        "trend_down": {
            "gamma": 0.5, 
            "kappa": 500, 
            "skew_factor": 0.008,        # Higher skew to buy more
            "price_tolerance": 0.0015,   # 0.15% tolerance  
            "grid_spacing": 0.0015,      # 0.15% grid
            "order_size_mult": 0.8,      # Smaller orders in trend
            "grid_layers": 7,            # Normal layers
            "max_position_mult": 1.0,    # Normal max position ($5,000)
            "description": "Wide spread, favor buys - accumulate dips"
        },
        "high_vol": {
            "gamma": 0.3, 
            "kappa": 200, 
            "skew_factor": 0.002,        # Reduced skew in chaos
            "price_tolerance": 0.002,    # 0.2% tolerance (wider)
            "grid_spacing": 0.0020,      # 0.20% grid (wide safety)
            "order_size_mult": 0.5,      # Half size for risk reduction
            "grid_layers": 5,            # Fewer layers for safety
            "max_position_mult": 0.6,    # 60% max position ($3,000)
            "description": "Very wide spread, conservative - survival mode"
        }
    }
    
    def __init__(self, model_path="data/regime_model.pkl"):
        self.model_path = model_path
        self.model = None
        self.scaler = StandardScaler()
        self.is_fitted = False
        
        # Try to load existing model
        if os.path.exists(model_path):
            self.load_model()
    
    def _calculate_features(self, df: pd.DataFrame) -> np.ndarray:
        """
        Calculate features for regime detection.
        
        Features:
        - returns: Price momentum
        - volatility: Rolling std of returns
        - atr_ratio: ATR relative to price
        - volume_change: Volume momentum
        """
        df = df.copy()
        
        # Ensure numeric types
        for col in ['open', 'high', 'low', 'close', 'volume']:
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
        
        # Volume change
        df['volume_sma'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        
        # Momentum (10-period)
        df['momentum'] = df['close'].pct_change(10)
        
        # Drop NaN rows
        df = df.dropna()
        
        # Select features
        features = df[['returns', 'volatility', 'atr_ratio', 'momentum']].values
        
        return features, df.index
    
    def fit(self, df: pd.DataFrame, n_clusters=4):
        """
        Train the regime detection model on historical data.
        """
        features, indices = self._calculate_features(df)
        
        if len(features) < 100:
            raise ValueError(f"Not enough data: {len(features)} rows (need 100+)")
        
        # Scale features
        features_scaled = self.scaler.fit_transform(features)
        
        # K-Means clustering
        self.model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        self.model.fit(features_scaled)
        
        # Analyze clusters to assign regime names
        self._analyze_clusters(features, self.model.labels_)
        
        self.is_fitted = True
        print(f"Model fitted on {len(features)} samples")
        
        return self.model.labels_
    
    def _analyze_clusters(self, features, labels):
        """
        Analyze cluster characteristics to map to regime names.
        """
        df_analysis = pd.DataFrame(features, columns=['returns', 'volatility', 'atr_ratio', 'momentum'])
        df_analysis['cluster'] = labels
        
        cluster_stats = df_analysis.groupby('cluster').mean()
        print("\n=== Cluster Analysis ===")
        print(cluster_stats)
        
        # Auto-map clusters based on characteristics
        # High volatility: highest volatility
        # Trend up: positive momentum
        # Trend down: negative momentum
        # Low vol: lowest volatility
        
        vol_order = cluster_stats['volatility'].sort_values()
        mom_order = cluster_stats['momentum'].sort_values()
        
        self.cluster_to_regime = {
            vol_order.index[0]: "low_vol",      # Lowest volatility
            vol_order.index[-1]: "high_vol",    # Highest volatility
            mom_order.index[-1]: "trend_up",    # Highest momentum
            mom_order.index[0]: "trend_down"    # Lowest (negative) momentum
        }
        
        # Handle overlap (some clusters may be assigned twice)
        assigned = set()
        final_mapping = {}
        for cluster, regime in self.cluster_to_regime.items():
            if regime not in assigned:
                final_mapping[cluster] = regime
                assigned.add(regime)
        
        # Fill remaining with any unassigned regime
        all_regimes = set(self.REGIME_PARAMS.keys())
        for i in range(4):
            if i not in final_mapping:
                remaining = all_regimes - assigned
                if remaining:
                    final_mapping[i] = remaining.pop()
                    assigned.add(final_mapping[i])
        
        self.cluster_to_regime = final_mapping
        print(f"\nCluster to Regime Mapping: {self.cluster_to_regime}")
    
    def predict(self, df: pd.DataFrame) -> str:
        """
        Predict current market regime from recent price data.
        
        Returns regime name: 'low_vol', 'trend_up', 'trend_down', 'high_vol'
        """
        if not self.is_fitted:
            return "unknown"
        
        features, _ = self._calculate_features(df)
        
        if len(features) == 0:
            return "unknown"
        
        # Use last row (most recent)
        latest_features = features[-1].reshape(1, -1)
        features_scaled = self.scaler.transform(latest_features)
        
        cluster = self.model.predict(features_scaled)[0]
        regime = self.cluster_to_regime.get(cluster, "unknown")
        
        return regime
    
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
            
            # Convert to numeric
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col])
            
            return df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        except Exception as e:
            print(f"Failed to fetch Binance data: {e}")
            return pd.DataFrame()
    
    def predict_live(self, symbol="BTCUSDT") -> str:
        """
        Predict current regime using live Binance data.
        This bypasses the need for internal candle accumulation.
        """
        if not self.is_fitted:
            return "unknown"
        
        # Fetch 100 recent 1-hour candles
        df = self.fetch_binance_candles(symbol=symbol, interval="1h", limit=100)
        
        if df.empty or len(df) < 50:
            return "unknown"
        
        return self.predict(df)
    
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
        
        print(f"Model saved to {self.model_path}")
    
    def load_model(self):
        """Load trained model from disk."""
        try:
            with open(self.model_path, 'rb') as f:
                data = pickle.load(f)
            
            self.model = data['model']
            self.scaler = data['scaler']
            self.cluster_to_regime = data['cluster_to_regime']
            self.is_fitted = True
            
            print(f"Model loaded from {self.model_path}")
        except Exception as e:
            print(f"Failed to load model: {e}")
            self.is_fitted = False


# Training script
if __name__ == "__main__":
    # Load BTC data
    df = pd.read_csv("data/btc_hourly_1000.csv")
    print(f"Loaded {len(df)} rows of BTC data")
    
    # Initialize and train
    detector = RegimeDetector()
    labels = detector.fit(df)
    
    # Save model
    detector.save_model()
    
    # Test prediction
    regime = detector.predict(df.tail(50))
    params = detector.get_params_for_regime(regime)
    
    print(f"\n=== Current Regime ===")
    print(f"Regime: {regime}")
    print(f"Recommended γ: {params['gamma']}, κ: {params['kappa']}")
    print(f"Description: {params['description']}")
    
    # Distribution of regimes
    print(f"\n=== Regime Distribution ===")
    regime_names = [detector.cluster_to_regime.get(l, 'unknown') for l in labels]
    for regime in set(regime_names):
        count = regime_names.count(regime)
        pct = count / len(regime_names) * 100
        print(f"{regime}: {count} ({pct:.1f}%)")
