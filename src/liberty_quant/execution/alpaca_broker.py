"""Alpaca broker adapter — routes orders to the Alpaca paper (or live) account."""

import logging
from typing import Any, Literal

from ..config import get_settings
from .broker import Broker, OrderResult, Position

logger = logging.getLogger(__name__)


class AlpacaBroker(Broker):
    """Order execution via the Alpaca Trading API.

    With ALPACA_PAPER=true (default) this hits the paper-trading endpoint:
    real order matching against real market data, fake money.
    """

    def __init__(self) -> None:
        from alpaca.trading.client import TradingClient

        settings = get_settings()
        if not (settings.alpaca_api_key and settings.alpaca_secret_key):
            raise RuntimeError("Alpaca credentials not configured")
        self._client = TradingClient(
            api_key=settings.alpaca_api_key,
            secret_key=settings.alpaca_secret_key.get_secret_value(),
            paper=settings.alpaca_paper,
        )
        mode = "PAPER" if settings.alpaca_paper else "LIVE"
        logger.info("Alpaca broker connected (%s)", mode)

    def get_account(self) -> dict[str, Any]:
        account = self._client.get_account()
        return {
            "cash": float(account.cash),
            "equity": float(account.equity),
            "buying_power": float(account.buying_power),
        }

    def get_positions(self) -> list[Position]:
        positions = []
        for p in self._client.get_all_positions():
            positions.append(
                Position(
                    symbol=p.symbol,
                    qty=float(p.qty),
                    avg_entry_price=float(p.avg_entry_price),
                    current_price=float(p.current_price or p.avg_entry_price),
                    side="long" if float(p.qty) >= 0 else "short",
                )
            )
        return positions

    def submit_market_order(
        self, *, symbol: str, side: Literal["buy", "sell"], qty: float, current_price: float
    ) -> OrderResult:
        from alpaca.trading.enums import OrderSide, TimeInForce
        from alpaca.trading.requests import MarketOrderRequest

        try:
            order = self._client.submit_order(
                MarketOrderRequest(
                    symbol=symbol,
                    qty=qty,
                    side=OrderSide.BUY if side == "buy" else OrderSide.SELL,
                    time_in_force=TimeInForce.DAY,
                )
            )
            logger.info("[alpaca] submitted %s %s x%.0f (order %s)", side.upper(), symbol, qty, order.id)
            return OrderResult(
                symbol=symbol,
                side=side,
                qty=qty,
                status="filled" if order.filled_at else "submitted",
                filled_price=float(order.filled_avg_price) if order.filled_avg_price else None,
                order_id=str(order.id),
            )
        except Exception as e:
            logger.error("[alpaca] order failed: %s %s x%.0f — %s", side, symbol, qty, e)
            return OrderResult(
                symbol=symbol, side=side, qty=qty, status="rejected", error=str(e)
            )

    def is_market_open(self) -> bool:
        try:
            return bool(self._client.get_clock().is_open)
        except Exception as e:
            logger.warning("Could not fetch Alpaca clock: %s", e)
            return False

    def next_market_times(self) -> tuple[str | None, str | None]:
        """(next_open, next_close) as ISO strings, or (None, None) on failure."""
        try:
            clock = self._client.get_clock()
            return str(clock.next_open), str(clock.next_close)
        except Exception:
            return None, None


def create_broker() -> Broker:
    """Create the configured execution backend."""
    settings = get_settings()
    if settings.execution_mode == "alpaca":
        return AlpacaBroker()
    from .paper_engine import LocalPaperEngine

    return LocalPaperEngine()
