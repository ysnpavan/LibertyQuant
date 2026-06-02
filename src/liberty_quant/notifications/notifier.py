"""Trade notifications.

``TelegramNotifier`` (default) sends messages through the Telegram Bot API.
``WhatsAppNotifier`` sends messages through Twilio's WhatsApp API.
``ConsoleNotifier`` is the no-credentials fallback.

Telegram setup (one-time, ~3 minutes):
1. On Telegram, message @BotFather -> /newbot -> follow prompts -> copy the bot token
2. Message @userinfobot -> it replies with your chat ID
3. Send any message to your new bot (bots can't message you first)
4. Put in .env:
     LQ_NOTIFICATIONS_ENABLED=true
     TELEGRAM_BOT_TOKEN=123456789:ABC...
     TELEGRAM_CHAT_ID=123456789

WhatsApp setup (alternative, via Twilio sandbox):
1. Create a free Twilio account: https://www.twilio.com/try-twilio
2. Console -> Messaging -> Try it out -> Send a WhatsApp message
3. Follow the sandbox instructions: send the join code from your phone
   to the sandbox number (+1 415 523 8886)
4. Put in .env:
     LQ_NOTIFICATIONS_ENABLED=true
     LQ_NOTIFICATION_CHANNEL=whatsapp
     TWILIO_ACCOUNT_SID=ACxxxxxxxx
     TWILIO_AUTH_TOKEN=xxxxxxxx
     TWILIO_WHATSAPP_FROM=whatsapp:+14155238886
     WHATSAPP_TO=whatsapp:+91XXXXXXXXXX
"""

import logging
from abc import ABC, abstractmethod
from typing import Any

import httpx

from ..config import get_settings

logger = logging.getLogger(__name__)


class Notifier(ABC):
    """Notification channel interface."""

    @abstractmethod
    def send(self, message: str) -> bool:
        """Send a message. Returns True on success."""

    # --- High-level event helpers (shared formatting) ----------------------

    def trade_executed(self, *, side: str, qty: float, symbol: str, price: float,
                       strategy: str | None, confidence: float | None) -> None:
        self.send(
            f"🟢 *{side.upper()}* {qty:.0f} {symbol} @ ${price:.2f}\n"
            f"Strategy: {strategy or '—'} | Confidence: {confidence or 0:.0%}"
        )

    def position_closed(self, *, symbol: str, qty: float, exit_price: float,
                        pnl: float, reason: str) -> None:
        emoji = "✅" if pnl >= 0 else "🔴"
        self.send(
            f"{emoji} *CLOSED* {qty:.0f} {symbol} @ ${exit_price:.2f}\n"
            f"P&L: ${pnl:+,.2f} | Reason: {reason.replace('_', ' ')}"
        )

    def kill_switch_engaged(self, *, reason: str) -> None:
        self.send(f"🚨 *KILL SWITCH ENGAGED*\n{reason}\nAll trading halted for today.")

    def daily_summary(self, *, equity: float, realized_pnl: float, trades: int,
                      open_positions: int) -> None:
        emoji = "📈" if realized_pnl >= 0 else "📉"
        self.send(
            f"{emoji} *Daily Summary*\n"
            f"Equity: ${equity:,.2f}\n"
            f"Realized P&L: ${realized_pnl:+,.2f}\n"
            f"Trades: {trades} | Open positions: {open_positions}"
        )


class ConsoleNotifier(Notifier):
    """Logs notifications instead of sending them (default when Twilio not configured)."""

    def send(self, message: str) -> bool:
        logger.info("[notify] %s", message.replace("\n", " | "))
        return True


class TelegramNotifier(Notifier):
    """Telegram messages via the Bot API (free, no third-party account needed)."""

    def __init__(self) -> None:
        settings = get_settings()
        if not (settings.telegram_bot_token and settings.telegram_chat_id):
            raise RuntimeError("Telegram credentials not configured (TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID)")
        token = settings.telegram_bot_token.get_secret_value()
        self._chat_id = settings.telegram_chat_id
        self._url = f"https://api.telegram.org/bot{token}/sendMessage"

    def send(self, message: str) -> bool:
        try:
            response = httpx.post(
                self._url,
                json={
                    "chat_id": self._chat_id,
                    "text": message,
                    "parse_mode": "Markdown",
                },
                timeout=10.0,
            )
            if response.status_code == 200:
                return True
            # Markdown parse errors (e.g. unescaped chars in symbols) -> retry as plain text
            if response.status_code == 400:
                retry = httpx.post(
                    self._url,
                    json={"chat_id": self._chat_id, "text": message},
                    timeout=10.0,
                )
                if retry.status_code == 200:
                    return True
            logger.error(
                "[notify] Telegram error %d: %s", response.status_code, response.text[:200]
            )
            return False
        except httpx.HTTPError as e:
            logger.error("[notify] Telegram send failed: %s", e)
            return False


class WhatsAppNotifier(Notifier):
    """WhatsApp messages via the Twilio REST API."""

    def __init__(self) -> None:
        settings = get_settings()
        if not (
            settings.twilio_account_sid
            and settings.twilio_auth_token
            and settings.twilio_whatsapp_from
            and settings.whatsapp_to
        ):
            raise RuntimeError("Twilio WhatsApp credentials not fully configured")
        self._sid = settings.twilio_account_sid
        self._token = settings.twilio_auth_token.get_secret_value()
        self._from = settings.twilio_whatsapp_from
        self._to = settings.whatsapp_to
        self._url = f"https://api.twilio.com/2010-04-01/Accounts/{self._sid}/Messages.json"

    def send(self, message: str) -> bool:
        try:
            response = httpx.post(
                self._url,
                auth=(self._sid, self._token),
                data={"From": self._from, "To": self._to, "Body": message},
                timeout=10.0,
            )
            if response.status_code in (200, 201):
                return True
            logger.error(
                "[notify] Twilio error %d: %s", response.status_code, response.text[:200]
            )
            return False
        except httpx.HTTPError as e:
            logger.error("[notify] WhatsApp send failed: %s", e)
            return False


class MultiNotifier(Notifier):
    """Fan-out to multiple channels; succeeds if any channel succeeds."""

    def __init__(self, notifiers: list[Notifier]) -> None:
        self.notifiers = notifiers

    def send(self, message: str) -> bool:
        results = [n.send(message) for n in self.notifiers]
        return any(results)


def create_notifier() -> Notifier:
    """Create the configured notifier (Telegram/WhatsApp + console, or console only)."""
    settings = get_settings()
    channels: list[Notifier] = [ConsoleNotifier()]
    if settings.notifications_enabled:
        try:
            if settings.notification_channel == "telegram":
                channels.append(TelegramNotifier())
                logger.info("Telegram notifications enabled")
            else:
                channels.append(WhatsAppNotifier())
                logger.info("WhatsApp notifications enabled")
        except Exception as e:
            logger.warning("%s notifier unavailable (%s); console only",
                           settings.notification_channel, e)
    return MultiNotifier(channels) if len(channels) > 1 else channels[0]


def notification_event(notifier: Notifier, event: str, payload: dict[str, Any]) -> None:
    """Dispatch a named event to the notifier (used by the trading loop)."""
    try:
        handler = getattr(notifier, event, None)
        if handler:
            handler(**payload)
    except Exception as e:
        # Notifications must never break the trading loop
        logger.error("[notify] event '%s' failed: %s", event, e)
