"""Broker abstraction shared by the local paper engine and the Alpaca adapter."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal


@dataclass
class Position:
    symbol: str
    qty: float
    avg_entry_price: float
    current_price: float
    side: Literal["long", "short"] = "long"

    @property
    def market_value(self) -> float:
        return self.qty * self.current_price

    @property
    def unrealized_pnl(self) -> float:
        sign = 1 if self.side == "long" else -1
        return sign * (self.current_price - self.avg_entry_price) * self.qty

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "qty": self.qty,
            "side": self.side,
            "avg_entry_price": round(self.avg_entry_price, 2),
            "current_price": round(self.current_price, 2),
            "market_value": round(self.market_value, 2),
            "unrealized_pnl": round(self.unrealized_pnl, 2),
        }


@dataclass
class OrderResult:
    """Outcome of an order submission."""

    symbol: str
    side: Literal["buy", "sell"]
    qty: float
    status: Literal["filled", "submitted", "rejected"]
    filled_price: float | None = None
    order_id: str | None = None
    error: str | None = None
    submitted_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "side": self.side,
            "qty": self.qty,
            "status": self.status,
            "filled_price": self.filled_price,
            "order_id": self.order_id,
            "error": self.error,
            "submitted_at": self.submitted_at.isoformat(),
        }


class Broker(ABC):
    """Common interface for order execution backends."""

    @abstractmethod
    def get_account(self) -> dict[str, Any]:
        """Return {'cash': float, 'equity': float, 'buying_power': float}."""

    @abstractmethod
    def get_positions(self) -> list[Position]: ...

    @abstractmethod
    def submit_market_order(
        self, *, symbol: str, side: Literal["buy", "sell"], qty: float, current_price: float
    ) -> OrderResult: ...

    def get_portfolio_snapshot(self) -> dict[str, Any]:
        """Portfolio dict in the shape the agent pipeline expects."""
        account = self.get_account()
        positions = self.get_positions()
        return {
            "cash": round(account["cash"], 2),
            "equity": round(account["equity"], 2),
            "buying_power": round(account.get("buying_power", account["cash"]), 2),
            "positions": [p.to_dict() for p in positions],
        }
