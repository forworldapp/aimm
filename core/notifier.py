"""
Telegram Notification Module for AIMM Bot.
Sends alerts for circuit breaker, daily PnL, kill switch, and bot start/stop events.
"""

import asyncio
import logging
import time
from datetime import datetime
from typing import Optional

logger = logging.getLogger("Notifier")


class TelegramNotifier:
    """Sends alerts to Telegram using Bot API (no external deps, uses urllib)."""

    @staticmethod
    def _resolve_env(value: str) -> str:
        """Resolve 'env:VAR_NAME' to actual environment variable value."""
        if isinstance(value, str) and value.startswith('env:'):
            import os
            # Load .env file if not already loaded
            env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env')
            if os.path.exists(env_path):
                with open(env_path, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('#') and '=' in line:
                            key, val = line.split('=', 1)
                            os.environ.setdefault(key.strip(), val.strip())
            var_name = value[4:]  # Remove 'env:' prefix
            return os.environ.get(var_name, '')
        return value

    def __init__(self, config: dict):
        self.enabled = config.get('enabled', False)
        self.bot_token = self._resolve_env(config.get('bot_token', ''))
        self.chat_id = self._resolve_env(config.get('chat_id', ''))
        self.alerts = config.get('alerts', {})
        self._last_daily_report = None

        if self.enabled and (not self.bot_token or not self.chat_id):
            logger.warning("Telegram enabled but bot_token or chat_id missing. Disabling.")
            self.enabled = False

        if self.enabled:
            logger.info("✅ Telegram Notifier initialized")

    def _send_sync(self, text: str, parse_mode: str = "HTML"):
        """Send message synchronously using urllib (no external deps)."""
        if not self.enabled:
            return False

        import urllib.request
        import urllib.parse
        import ssl
        import json

        url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
        data = urllib.parse.urlencode({
            'chat_id': self.chat_id,
            'text': text,
            'parse_mode': parse_mode,
        }).encode('utf-8')

        # Create SSL context with certifi if available, otherwise use default
        ctx = None
        try:
            import certifi
            ctx = ssl.create_default_context(cafile=certifi.where())
        except ImportError:
            ctx = ssl.create_default_context()
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE

        try:
            req = urllib.request.Request(url, data=data)
            with urllib.request.urlopen(req, timeout=10, context=ctx) as resp:
                result = json.loads(resp.read())
                if result.get('ok'):
                    logger.debug("Telegram message sent")
                    return True
                else:
                    logger.error(f"Telegram API error: {result}")
                    return False
        except Exception as e:
            logger.error(f"Telegram send failed: {e}")
            return False

    async def send(self, text: str, parse_mode: str = "HTML"):
        """Send message asynchronously."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._send_sync, text, parse_mode)

    # --- Alert Methods ---

    async def alert_bot_start(self, symbol: str, balance: float, mode: str = "paper"):
        """Alert when bot starts."""
        if not self.alerts.get('bot_start', True):
            return
        msg = (
            f"🟢 <b>AIMM Bot Started</b>\n"
            f"Mode: <code>{mode.upper()}</code>\n"
            f"Symbol: <code>{symbol}</code>\n"
            f"Balance: <code>${balance:,.2f}</code>\n"
            f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        )
        await self.send(msg)

    async def alert_bot_stop(self, reason: str = "Manual"):
        """Alert when bot stops."""
        if not self.alerts.get('bot_stop', True):
            return
        msg = (
            f"🔴 <b>AIMM Bot Stopped</b>\n"
            f"Reason: {reason}\n"
            f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        )
        await self.send(msg)

    async def alert_circuit_breaker(self, loss: float, max_loss: float, position: float, price: float):
        """Alert when circuit breaker triggers."""
        if not self.alerts.get('circuit_breaker', True):
            return
        msg = (
            f"🚨 <b>CIRCUIT BREAKER TRIGGERED</b>\n\n"
            f"Loss: <code>${abs(loss):.2f}</code> (max: ${max_loss:.2f})\n"
            f"Position: <code>{position:.4f} BTC</code>\n"
            f"Price: <code>${price:,.2f}</code>\n"
            f"Action: Positions liquidated, bot stopped\n"
            f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n"
            f"⚠️ Auto-reset in 24 hours"
        )
        await self.send(msg)

    async def alert_kill_switch(self, module: str, pnl: float, reason: str):
        """Alert when kill switch disables a module."""
        if not self.alerts.get('kill_switch', True):
            return
        msg = (
            f"🛑 <b>Kill Switch: {module}</b>\n"
            f"PnL: <code>${pnl:.2f}</code>\n"
            f"Reason: {reason}\n"
            f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        )
        await self.send(msg)

    async def alert_daily_pnl(self, pnl: float, balance: float, trades: int,
                               position: float = 0, price: float = 0):
        """Send daily PnL report."""
        if not self.alerts.get('daily_pnl', True):
            return

        # Prevent duplicate daily reports
        today = datetime.now().strftime('%Y-%m-%d')
        if self._last_daily_report == today:
            return
        self._last_daily_report = today

        emoji = "📈" if pnl >= 0 else "📉"
        sign = "+" if pnl >= 0 else ""
        msg = (
            f"{emoji} <b>Daily Report</b>\n\n"
            f"PnL: <code>{sign}${pnl:.2f}</code>\n"
            f"Balance: <code>${balance:,.2f}</code>\n"
            f"Trades: <code>{trades}</code>\n"
            f"Position: <code>{position:.4f} BTC</code>\n"
            f"BTC Price: <code>${price:,.2f}</code>\n"
            f"Date: {today}"
        )
        await self.send(msg)

    async def alert_fill(self, side: str, price: float, size: float,
                         grid_profit: float = 0, position: float = 0):
        """Alert when a bot order is filled."""
        if not self.alerts.get('fills', True):
            return
        emoji = "🟢" if side == 'buy' else "🔴"
        profit_str = ""
        if grid_profit != 0:
            p_emoji = "💰" if grid_profit > 0 else "💸"
            profit_str = f"\n{p_emoji} Grid P/L: <code>${grid_profit:+.4f}</code>"
        msg = (
            f"{emoji} <b>Fill: {side.upper()}</b>\n"
            f"Price: <code>${price:,.2f}</code>\n"
            f"Size: <code>{size:.4f} BTC</code>\n"
            f"Value: <code>${price * size:,.2f}</code>"
            f"{profit_str}\n"
            f"Position: <code>{position:.4f} BTC</code>\n"
            f"Time: {datetime.now().strftime('%H:%M:%S')}"
        )
        await self.send(msg)

    async def alert_position_change(self, old_pos: float, new_pos: float,
                                     price: float, unrealized_pnl: float = 0):
        """Alert when position changes significantly."""
        if not self.alerts.get('position_change', True):
            return
        if old_pos == 0 and new_pos == 0:
            return
        if old_pos == 0:
            action = "📥 Position Opened"
        elif new_pos == 0:
            action = "📤 Position Closed"
        else:
            action = "🔄 Position Changed"
        side_str = "LONG" if new_pos > 0 else "SHORT" if new_pos < 0 else "FLAT"
        msg = (
            f"<b>{action}</b>\n"
            f"Before: <code>{old_pos:.4f} BTC</code>\n"
            f"After: <code>{new_pos:.4f} BTC</code> ({side_str})\n"
            f"Price: <code>${price:,.2f}</code>\n"
            f"Unrealized P/L: <code>${unrealized_pnl:+.2f}</code>\n"
            f"Time: {datetime.now().strftime('%H:%M:%S')}"
        )
        await self.send(msg)

    async def alert_custom(self, title: str, message: str):
        """Send custom alert."""
        msg = f"📢 <b>{title}</b>\n{message}"
        await self.send(msg)
