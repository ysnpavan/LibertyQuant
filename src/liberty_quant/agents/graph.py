"""LangGraph workflow: regime -> strategy -> validation -> risk.

Mirrors the RakshaQuant pipeline with conditional short-circuits:
- Abort the cycle if regime confidence is too low
- Skip risk assessment when no signals survive validation
"""

import logging
from typing import Any, Literal

from langgraph.graph import END, START, StateGraph

from .nodes import (
    market_regime_node,
    risk_compliance_node,
    signal_validation_node,
    strategy_selection_node,
)
from .state import TradingState, create_initial_state

logger = logging.getLogger(__name__)

MIN_REGIME_CONFIDENCE = 0.3


def _after_regime(state: TradingState) -> Literal["strategy_selection", "end"]:
    if state["regime_confidence"] < MIN_REGIME_CONFIDENCE:
        logger.info(
            "Regime confidence %.2f below %.2f — skipping cycle",
            state["regime_confidence"],
            MIN_REGIME_CONFIDENCE,
        )
        return "end"
    return "strategy_selection"


def _after_strategy(state: TradingState) -> Literal["signal_validation", "end"]:
    if not state["active_strategies"]:
        logger.info("No active strategies selected — ending cycle")
        return "end"
    return "signal_validation"


def _after_validation(state: TradingState) -> Literal["risk_compliance", "end"]:
    if not state["validated_signals"]:
        logger.info("No validated signals — ending cycle")
        return "end"
    return "risk_compliance"


def create_trading_graph():
    """Compile the agent workflow graph."""
    workflow = StateGraph(TradingState)

    workflow.add_node("market_regime", market_regime_node)
    workflow.add_node("strategy_selection", strategy_selection_node)
    workflow.add_node("signal_validation", signal_validation_node)
    workflow.add_node("risk_compliance", risk_compliance_node)

    workflow.add_edge(START, "market_regime")
    workflow.add_conditional_edges(
        "market_regime",
        _after_regime,
        {"strategy_selection": "strategy_selection", "end": END},
    )
    workflow.add_conditional_edges(
        "strategy_selection",
        _after_strategy,
        {"signal_validation": "signal_validation", "end": END},
    )
    workflow.add_conditional_edges(
        "signal_validation",
        _after_validation,
        {"risk_compliance": "risk_compliance", "end": END},
    )
    workflow.add_edge("risk_compliance", END)

    return workflow.compile()


def run_trading_cycle(
    graph,
    *,
    cycle_id: str,
    indicators: dict[str, dict[str, Any]],
    signals: list[dict[str, Any]],
    portfolio: dict[str, Any],
    daily_stats: dict[str, Any],
) -> TradingState:
    """Run one full decision cycle through the agent graph."""
    state = create_initial_state(
        cycle_id=cycle_id,
        indicators=indicators,
        signals=signals,
        portfolio=portfolio,
        daily_stats=daily_stats,
    )

    logger.info("=== Cycle %s: %d signals in ===", cycle_id, len(signals))
    final_state: TradingState = graph.invoke(state)
    logger.info(
        "=== Cycle %s: regime=%s strategies=%s approved=%d ===",
        cycle_id,
        final_state.get("regime"),
        final_state.get("active_strategies"),
        len(final_state.get("approved_trades", [])),
    )
    return final_state
