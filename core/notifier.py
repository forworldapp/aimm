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

    def __init__(self, config: dict):
        self.enabled = config.get('enabled', False)
        self.bot_token = config.get('bot_token', '')
        self.chat_id = config.get('chat_id', '')
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
        import json

        url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
        data = urllib.parse.urlencode({
            'chat_id': self.chat_id,
            'text': text,
            'parse_mode': parse_mode,
        }).encode('utf-8')

        try:
            req = urllib.request.Request(url, data=data)
            with urllib.request.urlopen(req, timeout=10) as resp:
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

    async def alert_custom(self, title: str, message: str):
        """Send custom alert."""
        msg = f"📢 <b>{title}</b>\n{message}"
        await self.send(msg)
