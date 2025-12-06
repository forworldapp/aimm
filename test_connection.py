import os
import asyncio
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

async def main():
    print("Testing GRVT Connection...")
    
    api_key = os.getenv("GRVT_API_KEY")
    private_key = os.getenv("GRVT_PRIVATE_KEY")
    
    if not api_key or not private_key:
        print("Error: API Key or Private Key not found in .env file.")
        print("Please copy .env.example to .env and fill in your credentials.")
        return

    try:
        # Attempting to import GRVT SDK
        # Note: The import name might differ slightly depending on the installed package version.
        # Common patterns: import grvt, from grvt_pysdk import ...
        # Based on search, it might be:
        from grvt_pysdk.exchange.grvt_ccxt import GrvtCcxt # Hypothetical import based on search
        
        print("SDK Imported successfully.")
        
        # Initialize Exchange (Hypothetical usage)
        exchange = GrvtCcxt({
            'apiKey': api_key,
            'secret': private_key,
            'sandbox': True if os.getenv("GRVT_ENV") == 'testnet' else False
        })
        
        # Fetch Balance
        print("Fetching Balance...")
        balance = await exchange.fetch_balance()
        print("Balance:", balance)
        
    except ImportError:
        print("Error: Could not import 'grvt-pysdk'.")
        print("Please ensure you have installed the requirements: pip install -r requirements.txt")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    asyncio.run(main())
