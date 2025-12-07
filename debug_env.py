import os
from dotenv import load_dotenv

print(f"Current Working Directory: {os.getcwd()}")
print(f"Files in directory: {os.listdir('.')}")

load_dotenv(override=True)

api_key = os.getenv("GRVT_API_KEY")
private_key = os.getenv("GRVT_PRIVATE_KEY")

print(f"GRVT_API_KEY Loaded: {'YES' if api_key else 'NO'}")
if api_key:
    print(f"API Key Length: {len(api_key)}")
    print(f"API Key First 5 chars: {api_key[:5]}...")

print(f"GRVT_PRIVATE_KEY Loaded: {'YES' if private_key else 'NO'}")
