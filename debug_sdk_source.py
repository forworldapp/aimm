import inspect
import sys
try:
    from pysdk.grvt_ccxt import GrvtCcxt
    print("Successfully imported GrvtCcxt")
    
    # Print __init__ signature and docstring
    print("\n--- __init__ Signature ---")
    print(inspect.signature(GrvtCcxt.__init__))
    
    print("\n--- __init__ Source Code ---")
    print(inspect.getsource(GrvtCcxt.__init__))
    
    # Also check GrvtCcxtBase if it inherits from it
    print("\n--- MRO ---")
    print(GrvtCcxt.mro())

except ImportError as e:
    print(f"ImportError: {e}")
except Exception as e:
    print(f"Error: {e}")
