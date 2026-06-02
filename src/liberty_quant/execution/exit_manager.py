"""Exit manager — owns the lifecycle of every open position.

Entries are registered when orders fill; every cycle the manager checks:
1. Stop-loss hit          -> close
2. Take-profit hit        -> close
3. Trailing stop breached -> close (trail ratchets up from the high-water mark)
4. End-of-day flatten     -> close everything N minutes before the bell

Position metadata (stop/target/high-water mark) lives in SQLite so restarts
don't orphan positions.
"""

import logging
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from ..config import get_settings

logger = logging.getLogger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS managed_positions (
    symbol TEXT PRIMARY KEY,
    qty REAL NOT NULL,
    side TEXT NOT NULL DEFAULT 'long',
    entry_price REAL NOT NULL,
    stop_loss REAL,
    take_profit REAL,
    high_water_mark REAL,
    strategy TEXT,
    regime_at_entry TEXT,
    confidence REAL,
    cycle_id TEXT,
    opened_at TEXT NOT NULL
);
"""

ExitReason = Literal["stop_loss", "take_profit", "trailing_stop", "eod_flatten", "manual"]


@dataclass
class ExitOrder:
    """An exit the manager wants executed."""

    symbol: str
    qty: float
    side: Literal["buy", "sell"]  # order side to close (sell closes a long)
    reason: ExitReason
    current_price: float
    position: dict[str, Any]


class ExitManager:
    """Tracks open positions and decides when to close them."""

    def __init__(self, db_path: str | None = None) -> None:
        settings = get_settings()
        path = Path(db_path or settings.db_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(path))
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(SCHEMA)
        self._conn.commit()
        self._settings = settings

    def close(self) -> None:
        self._conn.close()

    # --- registration ------------------------------------------------------

    def register_entry(
        self,
        *,
        symbol: str,
        qty: float,
        entry_price: float,
        stop_loss: float | None,
        take_profit: float | None,
        strategy: str | None = None,
        regime: str | None = None,
        confidence: float | None = None,
        cycle_id: str | None = None,
    ) -> None:
        """Record a new (or added-to) position with its exit levels."""
        existing = self.get_position(symbol)
        if existing:
            # Position added to: blend entry, keep tightest stop / nearest target
            total_qty = existing["qty"] + qty
            blended_entry = (
                existing["entry_price"] * existing["qty"] + entry_price * qty
            ) / total_qty
            stop_loss = max(
                s for s in (existing["stop_loss"], stop_loss) if s is not None
            ) if (existing["stop_loss"] or stop_loss) else None
            take_profit = min(
                t for t in (existing["take_profit"], take_profit) if t is not None
            ) if (existing["take_profit"] or take_profit) else None
            qty, entry_price = total_qty, blended_entry

        self._conn.execute(
            """INSERT OR REPLACE INTO managed_positions
               (symbol, qty, side, entry_price, stop_loss, take_profit, high_water_mark,
                strategy, regime_at_entry, confidence, cycle_id, opened_at)
               VALUES (?, ?, 'long', ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                symbol,
                qty,
                entry_price,
                stop_loss,
                take_profit,
                entry_price,  # high-water mark starts at entry
                strategy,
                regime,
                confidence,
                cycle_id,
                datetime.now(UTC).isoformat(),
            ),
        )
        self._conn.commit()
        logger.info(
            "[exits] registered %s x%.0f @ %.2f (stop=%.2f target=%.2f)",
            symbol, qty, entry_price, stop_loss or 0, take_profit or 0,
        )

    def remove(self, symbol: str) -> dict[str, Any] | None:
        """Remove a position from management (after it's been closed)."""
        position = self.get_position(symbol)
        if position:
            self._conn.execute("DELETE FROM managed_positions WHERE symbol = ?", (symbol,))
            self._conn.commit()
        return position

    def get_position(self, symbol: str) -> dict[str, Any] | None:
        row = self._conn.execute(
            "SELECT * FROM managed_positions WHERE symbol = ?", (symbol,)
        ).fetchone()
        return dict(row) if row else None

    def all_positions(self) -> list[dict[str, Any]]:
        rows = self._conn.execute("SELECT * FROM managed_positions").fetchall()
        return [dict(r) for r in rows]

    # --- exit checks --------------------------------------------------------

    def check_exits(
        self, current_prices: dict[str, float], *, near_market_close: bool = False
    ) -> list[ExitOrder]:
        """Evaluate all managed positions against current prices.

        Args:
            current_prices: symbol -> latest price
            near_market_close: True when within the EOD flatten window

        Returns:
            Exit orders to execute (long positions close with a SELL).
        """
        exits: list[ExitOrder] = []
        settings = self._settings

        for position in self.all_positions():
            symbol = position["symbol"]
            price = current_prices.get(symbol)
            if price is None or price <= 0:
                continue

            # Ratchet the high-water mark
            hwm = max(position["high_water_mark"] or position["entry_price"], price)
            if hwm != position["high_water_mark"]:
                self._conn.execute(
                    "UPDATE managed_positions SET high_water_mark = ? WHERE symbol = ?",
                    (hwm, symbol),
                )
                self._conn.commit()
                position["high_water_mark"] = hwm

            reason: ExitReason | None = None

            if settings.eod_flatten and near_market_close:
                reason = "eod_flatten"
            elif position["stop_loss"] and price <= position["stop_loss"]:
                reason = "stop_loss"
            elif position["take_profit"] and price >= position["take_profit"]:
                reason = "take_profit"
            elif price <= hwm * (1 - settings.trailing_stop_pct) and price > position["entry_price"]:
                # Trailing stop only protects gains — never fires below entry
                # (the hard stop_loss handles downside)
                reason = "trailing_stop"

            if reason:
                exits.append(
                    ExitOrder(
                        symbol=symbol,
                        qty=position["qty"],
                        side="sell",
                        reason=reason,
                        current_price=price,
                        position=position,
                    )
                )

        if exits:
            logger.info(
                "[exits] %d exit(s) triggered: %s",
                len(exits),
                [(e.symbol, e.reason) for e in exits],
            )
        return exits

    # --- reconciliation -----------------------------------------------------

    def reconcile(self, broker_symbols: set[str]) -> list[str]:
        """Drop managed positions that no longer exist at the broker.

        (e.g. manually closed in the Alpaca dashboard). Returns dropped symbols.
        """
        dropped = []
        for position in self.all_positions():
            if position["symbol"] not in broker_symbols:
                self._conn.execute(
                    "DELETE FROM managed_positions WHERE symbol = ?", (position["symbol"],)
                )
                dropped.append(position["symbol"])
        if dropped:
            self._conn.commit()
            logger.warning("[exits] dropped orphaned positions: %s", dropped)
        return dropped
