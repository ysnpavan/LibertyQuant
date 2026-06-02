"""Trade journal — SQLite audit trail of every cycle and every order."""

import json
import logging
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ..config import get_settings

logger = logging.getLogger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS cycles (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    cycle_id TEXT NOT NULL,
    started_at TEXT NOT NULL,
    regime TEXT,
    regime_confidence REAL,
    active_strategies TEXT,
    signals_count INTEGER,
    validated_count INTEGER,
    approved_count INTEGER,
    kill_switch INTEGER DEFAULT 0,
    state_json TEXT
);

CREATE TABLE IF NOT EXISTS orders (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    cycle_id TEXT NOT NULL,
    symbol TEXT NOT NULL,
    side TEXT NOT NULL,
    qty REAL NOT NULL,
    status TEXT NOT NULL,
    filled_price REAL,
    order_id TEXT,
    error TEXT,
    strategy TEXT,
    confidence REAL,
    reasoning TEXT,
    submitted_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS daily_stats (
    trade_date TEXT PRIMARY KEY,
    trades_count INTEGER DEFAULT 0,
    realized_pnl REAL DEFAULT 0.0,
    starting_equity REAL
);
"""


class TradeJournal:
    """Append-only journal of cycles, orders, and daily statistics."""

    def __init__(self, db_path: str | None = None) -> None:
        settings = get_settings()
        path = Path(db_path or settings.db_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(path))
        self._conn.executescript(SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    # --- cycles -----------------------------------------------------------

    def log_cycle(self, state: dict[str, Any]) -> None:
        self._conn.execute(
            """INSERT INTO cycles
               (cycle_id, started_at, regime, regime_confidence, active_strategies,
                signals_count, validated_count, approved_count, kill_switch, state_json)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                state.get("cycle_id"),
                datetime.now(UTC).isoformat(),
                state.get("regime"),
                state.get("regime_confidence"),
                json.dumps(state.get("active_strategies", [])),
                len(state.get("signals", [])),
                len(state.get("validated_signals", [])),
                len(state.get("approved_trades", [])),
                int(state.get("kill_switch", False)),
                json.dumps(
                    {
                        k: v
                        for k, v in state.items()
                        if k in ("regime_reasoning", "strategy_reasoning", "risk_warnings", "errors")
                    },
                    default=str,
                ),
            ),
        )
        self._conn.commit()

    # --- orders -----------------------------------------------------------

    def log_order(self, cycle_id: str, trade: dict[str, Any], result: dict[str, Any]) -> None:
        self._conn.execute(
            """INSERT INTO orders
               (cycle_id, symbol, side, qty, status, filled_price, order_id, error,
                strategy, confidence, reasoning, submitted_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                cycle_id,
                result["symbol"],
                result["side"],
                result["qty"],
                result["status"],
                result.get("filled_price"),
                result.get("order_id"),
                result.get("error"),
                trade.get("strategy"),
                trade.get("confidence"),
                trade.get("risk_reasoning") or trade.get("validation_reasoning"),
                result.get("submitted_at", datetime.now(UTC).isoformat()),
            ),
        )
        self._conn.commit()

    # --- daily stats -------------------------------------------------------

    def get_daily_stats(self, starting_equity: float | None = None) -> dict[str, Any]:
        today = datetime.now(UTC).date().isoformat()
        row = self._conn.execute(
            "SELECT trades_count, realized_pnl, starting_equity FROM daily_stats WHERE trade_date = ?",
            (today,),
        ).fetchone()
        if row is None:
            self._conn.execute(
                "INSERT INTO daily_stats (trade_date, starting_equity) VALUES (?, ?)",
                (today, starting_equity),
            )
            self._conn.commit()
            return {"trades_count": 0, "realized_pnl": 0.0, "starting_equity": starting_equity}
        return {"trades_count": row[0], "realized_pnl": row[1], "starting_equity": row[2]}

    def record_trade_executed(self, realized_pnl_delta: float = 0.0) -> None:
        today = datetime.now(UTC).date().isoformat()
        self._conn.execute(
            """UPDATE daily_stats
               SET trades_count = trades_count + 1, realized_pnl = realized_pnl + ?
               WHERE trade_date = ?""",
            (realized_pnl_delta, today),
        )
        self._conn.commit()

    # --- reporting ----------------------------------------------------------

    def recent_orders(self, limit: int = 20) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            """SELECT symbol, side, qty, status, filled_price, strategy, confidence, submitted_at
               FROM orders ORDER BY id DESC LIMIT ?""",
            (limit,),
        ).fetchall()
        return [
            {
                "symbol": r[0], "side": r[1], "qty": r[2], "status": r[3],
                "filled_price": r[4], "strategy": r[5], "confidence": r[6], "submitted_at": r[7],
            }
            for r in rows
        ]
