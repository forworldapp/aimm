import sys
try:
    import grvt_pysdk
    print(f"grvt_pysdk imported: {grvt_pysdk}")
    print(f"Dir: {dir(grvt_pysdk)}")
    
    try:
        from grvt_pysdk.exchange.grvt_ccxt import GrvtCcxt
        print("GrvtCcxt imported successfully")
    except ImportError as e:
        print(f"Failed to import GrvtCcxt: {e}")

except ImportError as e:
    print(f"Failed to import grvt_pysdk: {e}")
