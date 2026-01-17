from core.config import Config
import os

# Mock config.yaml
with open("temp_config.yaml", "w") as f:
    f.write("strategy:\n  name: test\n")

Config.set("strategy", "custom_key", "custom_value")
print(f"Before Load: {Config.get('strategy', 'custom_key')}")

Config.load("temp_config.yaml")
print(f"After Load: {Config.get('strategy', 'custom_key')}")
print(f"After Load (existing): {Config.get('strategy', 'name')}")

# Clean up
try:
    os.remove("temp_config.yaml")
except:
    pass
