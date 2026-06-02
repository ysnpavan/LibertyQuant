"""Position sizing — ATR/stop-distance risk-based sizing."""

import logging
from dataclasses import dataclass

from ..config import get_settings

logger = logging.getLogger(__name__)


@dataclass
class PositionSize:
    symbol: str
    shares: int
    notional: float
    risk_amount: float
    reason: str


def size_position(
    *,
    symbol: str,
    entry_price: float,
    stop_loss: float,
    equity: float,
    open_position_notional: float = 0.0,
) -> PositionSize:
    """Risk-based position sizing.

    Shares = (equity * risk_per_trade) / per-share risk, capped so the position
    notional never exceeds max_position_pct of equity.
    """
    settings = get_settings()

    per_share_risk = abs(entry_price - stop_loss)
    if per_share_risk <= 0:
        per_share_risk = entry_price * 0.02  # default 2% stop distance

    risk_budget = equity * settings.risk_per_trade
    shares_by_risk = int(risk_budget / per_share_risk)

    max_notional = equity * settings.max_position_pct
    shares_by_notional = int(max_notional / entry_price) if entry_price > 0 else 0

    shares = max(0, min(shares_by_risk, shares_by_notional))
    notional = shares * entry_price

    reason = (
        f"risk-based: {shares_by_risk} shares (risk ${risk_budget:.0f} / "
        f"${per_share_risk:.2f} per-share), notional cap: {shares_by_notional} shares"
    )

    return PositionSize(
        symbol=symbol,
        shares=shares,
        notional=notional,
        risk_amount=shares * per_share_risk,
        reason=reason,
    )
