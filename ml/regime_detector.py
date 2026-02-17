"""
ML Regime Detection Model (Phase 3 - GMM Upgrade)
- Uses Gaussian Mixture Model for probability-based regime classification
- 4 Regimes: Trending Up, Trending Down, Ranging, High Volatility
- Outputs regime probabilities for parameter blending
"""

import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
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
    
    # Optuna-optimized (2026-02-16, 20 trials × 5000 1m candles)
    # Baseline: gamma=2.28, kappa=2961, tol=1.45% → $7.06 PnL, Sharpe 10.80
    # Differentiated per regime via scaling + domain knowledge
    REGIME_PARAMS = {
        "low_vol": {
            "gamma": 2.28, 
            "kappa": 2961, 
            "skew_factor": 0.005,
            "price_tolerance": 0.010,    # 1.0% - tighter ok in calm markets
            "grid_spacing": 0.0015,      # 0.15% - tight grid for scalping
            "order_size_mult": 1.0,
            "grid_layers": 5,
            "max_position_mult": 1.85,
            "description": "Calm market - tighter tol, more layers, full size"
        },
        "trend_up": {
            "gamma": 2.28, 
            "kappa": 2961, 
            "skew_factor": 0.015,
            "price_tolerance": 0.0145,   # 1.45% - Optuna baseline
            "grid_spacing": 0.0020,      # 0.20% - wider for trend capture
            "order_size_mult": 0.85,
            "grid_layers": 3,
            "max_position_mult": 1.2,
            "description": "Uptrend - high skew to sell into strength"
        },
        "trend_down": {
            "gamma": 2.28, 
            "kappa": 2961, 
            "skew_factor": 0.015,
            "price_tolerance": 0.0145,   # 1.45% - Optuna baseline
            "grid_spacing": 0.0020,      # 0.20% - wider for trend capture
            "order_size_mult": 0.85,
            "grid_layers": 3,
            "max_position_mult": 1.2,
            "description": "Downtrend - high skew to buy dips"
        },
        "high_vol": {
            "gamma": 1.5, 
            "kappa": 1500, 
            "skew_factor": 0.005,
            "price_tolerance": 0.015,    # 1.5% - widest for volatile markets
            "grid_spacing": 0.0030,      # 0.30% - wide grid for safety
            "order_size_mult": 0.7,
            "grid_layers": 3,
            "max_position_mult": 0.8,
            "description": "Volatile - widest tol, small size, survival mode"
        }
    }
    
    def __init__(self, model_path="data/regime_model.pkl"):
        self.model_path = model_path
        self.model = None
        self.scaler = StandardScaler()
        self.is_fitted = False
        self._optimized_params = {}  # Loaded from JSON if available
        
        # Load optimized params if available
        self._load_optimized_params()
        
        # Try to load existing model
        if os.path.exists(model_path):
            self.load_model()
    
    def _load_optimized_params(self):
        """Load data-driven optimized params from JSON (overrides hardcoded REGIME_PARAMS)."""
        params_path = os.path.join("data", "optimized_regime_params.json")
        if not os.path.exists(params_path):
            return
        try:
            import json
            with open(params_path, 'r') as f:
                data = json.load(f)
            trading_keys = ['gamma', 'kappa', 'skew_factor', 'price_tolerance',
                            'grid_spacing', 'order_size_mult', 'grid_layers', 'max_position_mult']
            for regime, params in data.get('regimes', {}).items():
                self._optimized_params[regime] = {k: params[k] for k in trading_keys if k in params}
            if self._optimized_params:
                print(f"GMM: Loaded optimized params for {list(self._optimized_params.keys())} "
                      f"(from {data.get('optimized_at', 'unknown')})")
        except Exception as e:
            print(f"GMM: Failed to load optimized params: {e}")
    
    def _calculate_features(self, df: pd.DataFrame) -> np.ndarray:
        """
        Calculate ENHANCED features for regime detection.
        
        Features (8 total):
        - returns: Price momentum (1-period)
        - volatility: Rolling std of returns (20-period)
        - atr_ratio: ATR relative to price
        - momentum: Price momentum (10-period)
        - rsi: Relative Strength Index (14-period)
        - macd: MACD line value
        - bb_width: Bollinger Band width
        - vol_spike: Volume spike ratio
        """
        df = df.copy()
        
        # Ensure numeric types
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # === Original Features ===
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
        
        # === NEW Enhanced Features ===
        # RSI (Relative Strength Index)
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss.replace(0, 1e-10)  # Avoid division by zero
        df['rsi'] = (100 - (100 / (1 + rs))) / 100  # Normalize to 0-1
        
        # MACD (Moving Average Convergence Divergence)
        ema12 = df['close'].ewm(span=12, adjust=False).mean()
        ema26 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = (ema12 - ema26) / df['close']  # Normalize by price
        
        # Bollinger Band Width
        sma20 = df['close'].rolling(20).mean()
        std20 = df['close'].rolling(20).std()
        df['bb_width'] = (2 * std20) / sma20
        
        # Volume Spike Ratio
        vol_sma = df['volume'].rolling(20).mean()
        df['vol_spike'] = df['volume'] / vol_sma.replace(0, 1)
        df['vol_spike'] = df['vol_spike'].clip(upper=5)  # Cap at 5x to avoid outliers
        
        # Drop NaN rows
        df = df.dropna()
        
        # Select ALL 8 features
        features = df[['returns', 'volatility', 'atr_ratio', 'momentum',
                       'rsi', 'macd', 'bb_width', 'vol_spike']].values
        
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
        
        # Gaussian Mixture Model (GMM) - probability-based clustering
        self.model = GaussianMixture(
            n_components=n_clusters, 
            random_state=42, 
            covariance_type='full',
            n_init=5
        )
        self.model.fit(features_scaled)
        labels = self.model.predict(features_scaled)
        
        # Analyze clusters to assign regime names
        self._analyze_clusters(features, labels)
        
        self.is_fitted = True
        print(f"GMM Model fitted on {len(features)} samples")
        
        return labels
    
    def _analyze_clusters(self, features, labels):
        """
        Analyze cluster characteristics to map to regime names.
        """
        df_analysis = pd.DataFrame(features, columns=[
            'returns', 'volatility', 'atr_ratio', 'momentum',
            'rsi', 'macd', 'bb_width', 'vol_spike'
        ])
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

    def predict_proba(self, df: pd.DataFrame) -> dict:
        """
        Predict regime probabilities for blended parameters.
        Returns dict: {regime_name: probability, ...}
        """
        if not self.is_fitted:
            return {}
        
        features, _ = self._calculate_features(df)
        if len(features) == 0:
            return {}
            
        latest_features = features[-1].reshape(1, -1)
        features_scaled = self.scaler.transform(latest_features)
        
        # Get probabilities for each cluster
        probs = self.model.predict_proba(features_scaled)[0]
        
        # Map cluster probs to regime names
        regime_probs = {}
        for cluster_idx, prob in enumerate(probs):
            regime_name = self.cluster_to_regime.get(cluster_idx, f"cluster_{cluster_idx}")
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
        """Get A&S parameters — optimized values override hardcoded defaults."""
        base = self.REGIME_PARAMS.get(regime, self.REGIME_PARAMS["low_vol"]).copy()
        if regime in self._optimized_params:
            base.update(self._optimized_params[regime])
        return base
    
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
