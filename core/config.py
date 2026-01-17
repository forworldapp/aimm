import os
import yaml
import logging
from dotenv import load_dotenv

load_dotenv()

class Config:
    """
    Central Configuration Class.
    Loads environment variables for secrets and YAML for strategy parameters.
    """
    # Secrets (Env Vars)
    GRVT_API_KEY = os.getenv("GRVT_API_KEY")
    GRVT_PRIVATE_KEY = os.getenv("GRVT_PRIVATE_KEY")
    
    # Defaults
    _config = {
        "exchange": {"env": "testnet", "symbol": "BTC-USDT"},
        "strategy": {"name": "market_maker", "spread_pct": 0.001, "order_amount": 0.001, "refresh_interval": 5},
        "risk": {"max_position_usd": 1000.0, "max_drawdown_pct": 0.05, "inventory_skew_factor": 0.0}
    }

    @classmethod
    def load(cls, path="config.yaml"):
        """Load configuration from a YAML file."""
        # Reload env vars to ensure we catch the .env file
        load_dotenv(override=True)
        cls.GRVT_API_KEY = os.getenv("GRVT_API_KEY")
        cls.GRVT_PRIVATE_KEY = os.getenv("GRVT_PRIVATE_KEY")
        
        if os.path.exists(path):
            try:
                with open(path, 'r') as f:
                    yaml_conf = yaml.safe_load(f)
                    if yaml_conf:
                        cls._update_dict(cls._config, yaml_conf)
                logging.info(f"Loaded config from {path}")
            except Exception as e:
                logging.error(f"Failed to load config: {e}")
        else:
            logging.warning(f"Config file {path} not found. Using defaults.")

    @classmethod
    def _update_dict(cls, original, update):
        """Recursively update dictionary."""
        for key, value in update.items():
            if isinstance(value, dict) and key in original:
                cls._update_dict(original[key], value)
            else:
                original[key] = value

    # Accessors
    @classmethod
    def get(cls, section, key, default=None):
        return cls._config.get(section, {}).get(key, default)

    @classmethod
    def set(cls, section, key, value):
        if section not in cls._config:
            if cls._config is None: cls._config = {}
            cls._config[section] = {}
        if cls._config.get(section) is None:
            cls._config[section] = {}
        cls._config[section][key] = value
