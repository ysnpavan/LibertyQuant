"""Shared trading state passed between agents in the LangGraph workflow."""

from enum import StrEnum
from typing import Any, TypedDict


class MarketRegime(StrEnum):
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    RANGING = "ranging"
    VOLATILE = "volatile"
    UNKNOWN = "unknown"


class TradingState(TypedDict):
    """Blackboard state shared by all agents in one trading cycle."""

    # --- Inputs (from market layer) ---
    indicators: dict[str, dict[str, Any]]   # symbol -> indicator dict
    signals: list[dict[str, Any]]            # raw signals from SignalEngine
    portfolio: dict[str, Any]                # cash, equity, positions
    daily_stats: dict[str, Any]              # trades_count, realized_pnl, ...

    # --- Market Regime Agent ---
    regime: str
    regime_confidence: float
    regime_reasoning: str

    # --- Strategy Selection Agent ---
    active_strategies: list[str]
    strategy_reasoning: str

    # --- Signal Validation Agent ---
    validated_signals: list[dict[str, Any]]
    rejected_signals: list[dict[str, Any]]

    # --- Risk & Compliance Agent ---
    approved_trades: list[dict[str, Any]]
    risk_rejected: list[dict[str, Any]]
    risk_warnings: list[str]
    kill_switch: bool

    # --- Metadata ---
    cycle_id: str
    errors: list[str]


def create_initial_state(
    *,
    cycle_id: str,
    indicators: dict[str, dict[str, Any]],
    signals: list[dict[str, Any]],
    portfolio: dict[str, Any],
    daily_stats: dict[str, Any],
) -> TradingState:
    return TradingState(
        indicators=indicators,
        signals=signals,
        portfolio=portfolio,
        daily_stats=daily_stats,
        regime=MarketRegime.UNKNOWN.value,
        regime_confidence=0.0,
        regime_reasoning="",
        active_strategies=[],
        strategy_reasoning="",
        validated_signals=[],
        rejected_signals=[],
        approved_trades=[],
        risk_rejected=[],
        risk_warnings=[],
        kill_switch=False,
        cycle_id=cycle_id,
        errors=[],
    )
