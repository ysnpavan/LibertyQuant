"""US market hours.

Primary source: Alpaca clock API (knows holidays and half-days).
Fallback: simple ET-hours check (Mon-Fri 9:30-16:00, ignores holidays).
"""

import logging
from datetime import datetime, time, timedelta
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)

ET = ZoneInfo("America/New_York")
MARKET_OPEN = time(9, 30)
MARKET_CLOSE = time(16, 0)


def is_market_open_fallback(now: datetime | None = None) -> bool:
    """Simple ET-hours check. Does NOT know about market holidays."""
    now_et = (now or datetime.now(tz=ET)).astimezone(ET)
    if now_et.weekday() >= 5:  # Sat/Sun
        return False
    return MARKET_OPEN <= now_et.time() < MARKET_CLOSE


def is_market_open(broker=None) -> bool:
    """Check whether the US equity market is open right now.

    Uses the Alpaca clock when an AlpacaBroker is provided (handles holidays
    and half-days); otherwise falls back to a plain ET-hours check.
    """
    if broker is not None and hasattr(broker, "is_market_open"):
        try:
            return broker.is_market_open()
        except Exception as e:
            logger.warning("Alpaca clock unavailable (%s); using ET-hours fallback", e)
    return is_market_open_fallback()


def minutes_to_close(broker=None, now: datetime | None = None) -> float | None:
    """Minutes until today's market close, or None if the market is closed.

    Uses the Alpaca clock when available (handles half-days); otherwise assumes
    a 16:00 ET close.
    """
    if broker is not None and hasattr(broker, "next_market_times"):
        try:
            from datetime import datetime as dt

            clock_open = broker.is_market_open()
            if not clock_open:
                return None
            _, next_close = broker.next_market_times()
            if next_close:
                close_dt = dt.fromisoformat(str(next_close).replace("Z", "+00:00"))
                now_utc = (now or datetime.now(tz=ET)).astimezone(close_dt.tzinfo)
                return max(0.0, (close_dt - now_utc).total_seconds() / 60)
        except Exception as e:
            logger.warning("Could not compute minutes to close from Alpaca: %s", e)

    now_et = (now or datetime.now(tz=ET)).astimezone(ET)
    if not is_market_open_fallback(now_et):
        return None
    close_et = now_et.replace(hour=16, minute=0, second=0, microsecond=0)
    return max(0.0, (close_et - now_et).total_seconds() / 60)


def seconds_to_next_open_estimate(now: datetime | None = None) -> int:
    """Rough seconds until the next weekday 9:30 ET open (for sleep scheduling).

    Ignores holidays — the loop re-checks the real clock when it wakes.
    """
    now_et = (now or datetime.now(tz=ET)).astimezone(ET)
    candidate = now_et.replace(hour=9, minute=30, second=0, microsecond=0)

    # If we're past today's open, start from tomorrow
    if now_et >= candidate:
        candidate += timedelta(days=1)
    # Skip weekend days
    while candidate.weekday() >= 5:
        candidate += timedelta(days=1)

    return max(60, int((candidate - now_et).total_seconds()))
