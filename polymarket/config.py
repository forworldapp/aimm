import os
from dotenv import load_dotenv

load_dotenv()

# Gamma API URL (Public Data & Querying)
GAMMA_API_URL = "https://gamma-api.polymarket.com"

# CLOB API URL (Orderbook & Trading)
CLOB_API_URL = "https://clob.polymarket.com"

# Threshold for Arb (e.g., 0.98 means 2% profit margin)
# If YES + NO < ARB_THRESHOLD, we signal an arb.
ARB_THRESHOLD = float(os.getenv("ARB_THRESHOLD", "0.98"))

# Polling Interval in seconds
POLL_INTERVAL = float(os.getenv("POLL_INTERVAL", "1.0"))

# Dry Run Mode (Default to True for safety)
DRY_RUN = os.getenv("DRY_RUN", "True").lower() == "true"
