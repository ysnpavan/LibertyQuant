"""Memory store — closed trades and lessons in SQLite."""

import json
import logging
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ..config import get_settings

logger = logging.getLogger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS closed_trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    strategy TEXT,
    side TEXT NOT NULL,
    qty REAL NOT NULL,
    entry_price REAL NOT NULL,
    exit_price REAL NOT NULL,
    pnl REAL NOT NULL,
    pnl_pct REAL NOT NULL,
    exit_reason TEXT,
    regime_at_entry TEXT,
    confidence REAL,
    opened_at TEXT,
    closed_at TEXT NOT NULL,
    cycle_id TEXT
);

CREATE TABLE IF NOT EXISTS lessons (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    category TEXT NOT NULL,
    severity TEXT NOT NULL,
    lesson TEXT NOT NULL,
    reasoning TEXT,
    context_json TEXT,
    closed_trade_id INTEGER REFERENCES closed_trades(id)
);

CREATE INDEX IF NOT EXISTS idx_lessons_category ON lessons(category);
CREATE INDEX IF NOT EXISTS idx_closed_trades_strategy ON closed_trades(strategy, regime_at_entry);
"""


class MemoryStore:
    """SQLite-backed store for trade history and learned lessons."""

    def __init__(self, db_path: str | None = None) -> None:
        settings = get_settings()
        path = Path(db_path or settings.db_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(path))
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    # --- closed trades ------------------------------------------------------

    def record_closed_trade(
        self,
        *,
        symbol: str,
        strategy: str | None,
        side: str,
        qty: float,
        entry_price: float,
        exit_price: float,
        exit_reason: str,
        regime_at_entry: str | None,
        confidence: float | None,
        opened_at: str | None,
        cycle_id: str | None,
    ) -> int:
        """Record a completed round-trip trade. Returns the row id."""
        if side == "long":
            pnl = (exit_price - entry_price) * qty
            pnl_pct = (exit_price - entry_price) / entry_price * 100
        else:  # short
            pnl = (entry_price - exit_price) * qty
            pnl_pct = (entry_price - exit_price) / entry_price * 100

        cursor = self._conn.execute(
            """INSERT INTO closed_trades
               (symbol, strategy, side, qty, entry_price, exit_price, pnl, pnl_pct,
                exit_reason, regime_at_entry, confidence, opened_at, closed_at, cycle_id)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                symbol, strategy, side, qty, entry_price, exit_price,
                round(pnl, 2), round(pnl_pct, 4), exit_reason, regime_at_entry,
                confidence, opened_at, datetime.now(UTC).isoformat(), cycle_id,
            ),
        )
        self._conn.commit()
        logger.info(
            "[memory] closed trade: %s %s pnl=$%.2f (%.2f%%) via %s",
            symbol, strategy, pnl, pnl_pct, exit_reason,
        )
        return cursor.lastrowid

    def get_closed_trade(self, trade_id: int) -> dict[str, Any] | None:
        row = self._conn.execute(
            "SELECT * FROM closed_trades WHERE id = ?", (trade_id,)
        ).fetchone()
        return dict(row) if row else None

    def recent_closed_trades(self, limit: int = 50) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT * FROM closed_trades ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in rows]

    # --- strategy performance (feeds Strategy Selection Agent) ---------------

    def strategy_performance(self, regime: str | None = None) -> dict[str, dict[str, Any]]:
        """Win rate and trade count per strategy, optionally filtered by regime."""
        query = """
            SELECT strategy,
                   COUNT(*) AS trades,
                   SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) AS wins,
                   ROUND(SUM(pnl), 2) AS total_pnl
            FROM closed_trades
            WHERE strategy IS NOT NULL
        """
        params: tuple = ()
        if regime:
            query += " AND regime_at_entry = ?"
            params = (regime,)
        query += " GROUP BY strategy"

        result = {}
        for row in self._conn.execute(query, params).fetchall():
            trades = row["trades"]
            result[row["strategy"]] = {
                "trades": trades,
                "win_rate": round(row["wins"] / trades, 2) if trades else 0.0,
                "total_pnl": row["total_pnl"],
            }
        return result

    # --- lessons --------------------------------------------------------------

    def add_lesson(
        self,
        *,
        category: str,
        severity: str,
        lesson: str,
        reasoning: str | None = None,
        context: dict[str, Any] | None = None,
        closed_trade_id: int | None = None,
    ) -> int:
        cursor = self._conn.execute(
            """INSERT INTO lessons (created_at, category, severity, lesson, reasoning, context_json, closed_trade_id)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                datetime.now(UTC).isoformat(),
                category,
                severity,
                lesson,
                reasoning,
                json.dumps(context or {}, sort_keys=True, default=str),
                closed_trade_id,
            ),
        )
        self._conn.commit()
        logger.info("[memory] lesson saved [%s/%s]: %s", category, severity, lesson[:80])
        return cursor.lastrowid

    def get_lessons(
        self,
        *,
        categories: list[str] | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """Most recent lessons, optionally filtered by category.

        Severity-weighted ordering: critical/high lessons surface first,
        then by recency.
        """
        settings = get_settings()
        limit = limit or settings.memory_top_n

        query = "SELECT * FROM lessons"
        params: list[Any] = []
        if categories:
            placeholders = ",".join("?" * len(categories))
            query += f" WHERE category IN ({placeholders})"
            params.extend(categories)
        query += """
            ORDER BY CASE severity
                WHEN 'critical' THEN 0 WHEN 'high' THEN 1 WHEN 'medium' THEN 2 ELSE 3
            END, id DESC
            LIMIT ?
        """
        params.append(limit)

        rows = self._conn.execute(query, params).fetchall()
        return [
            {
                "category": r["category"],
                "severity": r["severity"],
                "lesson": r["lesson"],
                "created_at": r["created_at"],
            }
            for r in rows
        ]
