"""Local paper trading engine — simulates fills with slippage, persists to disk."""

import json
import logging
from pathlib import Path
from typing import Any, Literal

from ..config import get_settings
from .broker import Broker, OrderResult, Position

logger = logging.getLogger(__name__)

SLIPPAGE_BPS = 5  # 5 basis points of simulated slippage per fill


class LocalPaperEngine(Broker):
    """Fully local simulated broker. State persists to a JSON file."""

    def __init__(self, state_path: str | None = None) -> None:
        settings = get_settings()
        self._path = Path(state_path or Path(settings.db_path).parent / "paper_wallet.json")
        self._state: dict[str, Any] = self._load()

    # --- persistence -----------------------------------------------------

    def _load(self) -> dict[str, Any]:
        if self._path.exists():
            try:
                return json.loads(self._path.read_text())
            except (json.JSONDecodeError, OSError) as e:
                logger.warning("Could not load paper wallet (%s); starting fresh", e)
        settings = get_settings()
        return {"cash": settings.starting_cash, "positions": {}, "realized_pnl": 0.0}

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps(self._state, indent=2, sort_keys=True))

    # --- Broker interface -------------------------------------------------

    def get_account(self) -> dict[str, Any]:
        positions_value = sum(
            p["qty"] * p["current_price"] for p in self._state["positions"].values()
        )
        equity = self._state["cash"] + positions_value
        return {
            "cash": self._state["cash"],
            "equity": equity,
            "buying_power": self._state["cash"],
            "realized_pnl": self._state.get("realized_pnl", 0.0),
        }

    def get_positions(self) -> list[Position]:
        return [
            Position(
                symbol=sym,
                qty=p["qty"],
                avg_entry_price=p["avg_entry_price"],
                current_price=p["current_price"],
            )
            for sym, p in self._state["positions"].items()
        ]

    def mark_to_market(self, prices: dict[str, float]) -> None:
        """Update current prices on open positions."""
        for sym, pos in self._state["positions"].items():
            if sym in prices:
                pos["current_price"] = prices[sym]
        self._save()

    def submit_market_order(
        self, *, symbol: str, side: Literal["buy", "sell"], qty: float, current_price: float
    ) -> OrderResult:
        slippage = current_price * SLIPPAGE_BPS / 10_000
        fill_price = current_price + slippage if side == "buy" else current_price - slippage
        notional = qty * fill_price

        positions = self._state["positions"]

        if side == "buy":
            if notional > self._state["cash"]:
                return OrderResult(
                    symbol=symbol, side=side, qty=qty, status="rejected",
                    error=f"insufficient cash: need ${notional:.2f}, have ${self._state['cash']:.2f}",
                )
            self._state["cash"] -= notional
            if symbol in positions:
                pos = positions[symbol]
                total_qty = pos["qty"] + qty
                pos["avg_entry_price"] = (
                    pos["avg_entry_price"] * pos["qty"] + fill_price * qty
                ) / total_qty
                pos["qty"] = total_qty
                pos["current_price"] = fill_price
            else:
                positions[symbol] = {
                    "qty": qty,
                    "avg_entry_price": fill_price,
                    "current_price": fill_price,
                }
        else:  # sell
            pos = positions.get(symbol)
            if pos is None or pos["qty"] < qty:
                return OrderResult(
                    symbol=symbol, side=side, qty=qty, status="rejected",
                    error=f"insufficient shares: have {pos['qty'] if pos else 0}, selling {qty}",
                )
            self._state["cash"] += notional
            realized = (fill_price - pos["avg_entry_price"]) * qty
            self._state["realized_pnl"] = self._state.get("realized_pnl", 0.0) + realized
            pos["qty"] -= qty
            if pos["qty"] <= 0:
                del positions[symbol]

        self._save()
        logger.info("[paper] %s %s %.0f @ %.2f", side.upper(), symbol, qty, fill_price)
        return OrderResult(
            symbol=symbol, side=side, qty=qty, status="filled",
            filled_price=round(fill_price, 4), order_id=f"paper-{symbol}-{side}",
        )
