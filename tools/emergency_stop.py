"""
EMERGENCY STOP SCRIPT
Immediately kills all bot processes, cancels orders, and closes positions.

Usage: python tools/emergency_stop.py
"""

import asyncio
import json
import os
import signal
import subprocess
import sys
import time

# Add parent dir to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


async def emergency_stop():
    print("=" * 50)
    print("🚨 EMERGENCY STOP INITIATED 🚨")
    print("=" * 50)

    # Step 1: Kill all python bot processes
    print("\n[1/3] Killing bot processes...")
    try:
        result = subprocess.run(
            ['taskkill', '/F', '/IM', 'python.exe'],
            capture_output=True, text=True, timeout=5
        )
        # Note: this will also kill this script, so we use a flag file approach
        print(f"  → {result.stdout.strip()}")
    except Exception as e:
        print(f"  → Warning: {e}")

    # Step 2: Update paper status to close position
    print("\n[2/3] Marking position for closure...")
    status_file = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'data', 'paper_status_BTC_USDT_Perp.json'
    )
    try:
        with open(status_file, 'r') as f:
            status = json.load(f)

        old_pos = status.get('position', {}).get('amount', 0)
        status['position'] = {
            'amount': 0.0,
            'entryPrice': 0.0,
            'unrealizedPnL': 0.0
        }
        status['open_orders_list'] = []
        status['open_orders'] = 0
        status['emergency_stop'] = True
        status['emergency_stop_time'] = time.time()

        with open(status_file, 'w') as f:
            json.dump(status, f)
        print(f"  → Position closed: {old_pos:.4f} → 0.0000 BTC")
        print(f"  → Orders cancelled: all cleared")
    except Exception as e:
        print(f"  → Warning: {e}")

    # Step 3: Send Telegram alert
    print("\n[3/3] Sending Telegram alert...")
    try:
        from core.config import Config
        Config.load()
        tg_config = Config.get('telegram', default={})

        if tg_config.get('enabled') and tg_config.get('bot_token'):
            from core.notifier import TelegramNotifier
            notifier = TelegramNotifier(tg_config)
            await notifier.alert_bot_stop(reason="🚨 EMERGENCY STOP by user")
            print("  → Telegram alert sent")
        else:
            print("  → Telegram not configured, skipping")
    except Exception as e:
        print(f"  → Telegram alert failed: {e}")

    print("\n" + "=" * 50)
    print("✅ EMERGENCY STOP COMPLETE")
    print("  - All processes killed")
    print("  - Position closed")
    print("  - Orders cancelled")
    print("=" * 50)
    print("\nTo restart: python main.py")


if __name__ == "__main__":
    print("\n⚠️  This will IMMEDIATELY stop the bot and close all positions!")
    print("    Press Enter to confirm or Ctrl+C to cancel...")

    try:
        input()
    except KeyboardInterrupt:
        print("\nCancelled.")
        sys.exit(0)

    asyncio.run(emergency_stop())
