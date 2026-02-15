"""Final test: find correct sub_account_id."""
import os, sys, json, requests
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv
load_dotenv()

api_key = os.environ.get('GRVT_API_KEY')

# Auth
session = requests.Session()
session.headers.update({"Content-Type": "application/json"})
resp = session.post("https://edge.grvt.io/auth/api_key/login", json={"api_key": api_key}, timeout=5)
from http.cookies import SimpleCookie
cookie = SimpleCookie()
cookie.load(resp.headers.get("Set-Cookie", ""))
session.cookies.update({"gravity": cookie["gravity"].value})
grvt_id = resp.headers.get("X-Grvt-Account-Id", "")
session.headers.update({"X-Grvt-Account-Id": grvt_id})
print(f"Authenticated. Account ID: {grvt_id}")

# Try to list sub-accounts
print("\n=== Get Sub Accounts ===")
for endpoint in [
    "https://edge.grvt.io/full/v1/get_sub_accounts",
    "https://trades.grvt.io/full/v1/get_sub_accounts",
    "https://edge.grvt.io/full/v1/sub_accounts",
    "https://trades.grvt.io/full/v1/sub_accounts",
]:
    try:
        r = session.post(endpoint, json={}, timeout=5)
        if r.status_code == 200:
            data = r.json()
            print(f"✅ {endpoint}")
            print(f"   Response: {json.dumps(data, indent=2)[:1000]}")
        else:
            print(f"❌ {endpoint} -> {r.status_code}")
    except Exception as e:
        print(f"❌ {endpoint} -> {e}")

# Try aggregated_account_summary - it had sub account data
print("\n=== Aggregated Summary (sub-accounts) ===")
r = session.post("https://trades.grvt.io/full/v1/aggregated_account_summary", json={}, timeout=5)
data = r.json()
result = data.get('result', {})
print(f"Main Account: {result.get('main_account_id')}")
print(f"Total Equity: {result.get('total_equity')}")
print(f"Sub Account Balance: {result.get('total_sub_account_balance')}")
print(f"All keys: {list(result.keys())}")

# The sub_account_id might be derivable. Try numeric patterns
# GRVT seems to use the ID format directly as string in API, but as uint64 in EIP712
# Let's check if the SDK signing code itself handles conversion
print("\n=== Test signing with string ID ===")
from pysdk.grvt_ccxt_utils import GrvtOrder, GrvtOrderLeg, GrvtSignature, OrderMetadata, TimeInForce, get_signable_message
from pysdk.grvt_ccxt_env import GrvtEnv
from decimal import Decimal

# Try using encode_typed_data directly with a string 
from eth_account.messages import encode_typed_data
domain = {"name": "GRVT Exchange", "version": "0", "chainId": 325}
types = {
    "Order": [
        {"name": "subAccountID", "type": "uint64"},
        {"name": "isMarket", "type": "bool"},
        {"name": "timeInForce", "type": "uint8"},
        {"name": "postOnly", "type": "bool"},
        {"name": "reduceOnly", "type": "bool"},
        {"name": "legs", "type": "OrderLeg[]"},
        {"name": "nonce", "type": "uint32"},
        {"name": "expiration", "type": "int64"},
    ],
    "OrderLeg": [
        {"name": "assetID", "type": "uint256"},
        {"name": "contractSize", "type": "uint64"},
        {"name": "limitPrice", "type": "uint64"},
        {"name": "isBuyingContract", "type": "bool"},
    ],
}

# The string "35IB75FKEUbGlw5MDW1azb05Iru" can't be int() 
# But what if it's actually base62 or base58 encoded?
import string
def decode_base62(s):
    chars = string.digits + string.ascii_uppercase + string.ascii_lowercase
    result = 0
    for c in s:
        result = result * 62 + chars.index(c)
    return result

try:
    numeric = decode_base62(grvt_id)
    print(f"Base62 decoded: {numeric}")
    
    # Try signing with this numeric value
    msg = {"subAccountID": numeric, "isMarket": False, "timeInForce": 1, 
           "postOnly": False, "reduceOnly": False, 
           "legs": [{"assetID": "0x030501", "contractSize": 2000000, "limitPrice": 68000000000000, "isBuyingContract": True}],
           "nonce": 12345, "expiration": "1771246078469815100"}
    signed = encode_typed_data(domain, types, msg)
    print(f"✅ Signing succeeded with base62 decoded ID: {numeric}")
except Exception as e:
    print(f"❌ Base62 decode/sign failed: {e}")

# Try base36
def decode_base36(s):
    return int(s, 36) if all(c in string.digits + string.ascii_lowercase for c in s.lower()) else None

try:
    numeric36 = int(grvt_id, 36)
    print(f"\nBase36 decoded: {numeric36}")
    msg["subAccountID"] = numeric36
    signed = encode_typed_data(domain, types, msg)
    print(f"✅ Signing succeeded with base36 decoded ID: {numeric36}")
except Exception as e:
    print(f"❌ Base36 failed: {e}")
