"""LangGraph node functions for each agent.

Each node:
1. Builds a per-cycle context string (volatile data -> user message, never the system prompt)
2. Calls Claude via the structured LLM layer
3. Falls back to a deterministic heuristic if the LLM call fails — the trading
   loop must never die because of an API error
"""

import json
import logging
from typing import Any

from ..config import get_settings
from ..llm import get_llm
from ..market.universe import BENCHMARKS
from .prompts import (
    MARKET_REGIME_PROMPT,
    RISK_COMPLIANCE_PROMPT,
    SIGNAL_VALIDATION_PROMPT,
    STRATEGY_SELECTION_PROMPT,
)
from .schemas import (
    RegimeAssessment,
    RiskAssessment,
    SignalValidationResult,
    StrategySelection,
)
from .state import TradingState

logger = logging.getLogger(__name__)


def _fmt(obj: Any) -> str:
    """Deterministic, compact JSON for prompt context."""
    return json.dumps(obj, indent=1, sort_keys=True, default=str)


def _lessons_block(state: TradingState, agent: str) -> str:
    """Format the lessons this agent should heed (empty string if none)."""
    lessons = state.get("memory_lessons", {}).get(agent, [])
    if not lessons:
        return ""
    lines = [
        f"- [{lesson['severity']}] {lesson['lesson']}"
        for lesson in lessons
    ]
    return (
        "\n\n## Lessons from past mistakes (follow these — they were learned from real losses)\n"
        + "\n".join(lines)
    )


# =====================================================================
# 1. Market Regime Agent (fast model)
# =====================================================================

def market_regime_node(state: TradingState) -> dict[str, Any]:
    logger.info("[regime] classifying market regime...")
    indicators = state["indicators"]

    benchmarks = {s: ind for s, ind in indicators.items() if s in BENCHMARKS}
    stocks = {s: ind for s, ind in indicators.items() if s not in BENCHMARKS}

    context = (
        "## Benchmark indicators (SPY/QQQ/IWM)\n"
        f"{_fmt(benchmarks)}\n\n"
        "## Active stock indicators\n"
        f"{_fmt(stocks)}"
        f"{_lessons_block(state, 'regime')}\n\n"
        "Classify the current market regime."
    )

    try:
        result = get_llm().structured_call(
            system=MARKET_REGIME_PROMPT,
            user_content=context,
            output_model=RegimeAssessment,
            deep=False,
        )
        logger.info("[regime] %s (confidence %.2f)", result.regime, result.confidence)
        return {
            "regime": result.regime,
            "regime_confidence": max(0.0, min(1.0, result.confidence)),
            "regime_reasoning": result.reasoning,
        }
    except Exception as e:
        logger.error("[regime] LLM call failed: %s — using fallback", e)
        return _fallback_regime(state, str(e))


def _fallback_regime(state: TradingState, error: str) -> dict[str, Any]:
    """Deterministic fallback: infer regime from benchmark average change."""
    benchmarks = [
        ind for s, ind in state["indicators"].items() if s in BENCHMARKS
    ]
    if not benchmarks:
        return {
            "regime": "ranging",
            "regime_confidence": 0.0,
            "regime_reasoning": f"Fallback (no benchmark data). Error: {error}",
            "errors": state["errors"] + [f"regime fallback: {error}"],
        }
    avg_change = sum(b.get("change_percent", 0) for b in benchmarks) / len(benchmarks)
    avg_rsi = sum(b.get("rsi_14", 50) for b in benchmarks) / len(benchmarks)

    if avg_change > 0.5 and avg_rsi > 55:
        regime, conf = "trending_up", 0.45
    elif avg_change < -0.5 and avg_rsi < 45:
        regime, conf = "trending_down", 0.45
    else:
        regime, conf = "ranging", 0.4

    return {
        "regime": regime,
        "regime_confidence": conf,
        "regime_reasoning": f"Fallback heuristic: avg benchmark change {avg_change:.2f}%, RSI {avg_rsi:.0f}. Error: {error}",
        "errors": state["errors"] + [f"regime fallback: {error}"],
    }


# =====================================================================
# 2. Strategy Selection Agent (fast model)
# =====================================================================

def strategy_selection_node(state: TradingState) -> dict[str, Any]:
    logger.info("[strategy] selecting strategies for regime=%s...", state["regime"])

    perf = state.get("strategy_performance", {})
    perf_block = (
        f"\n\n## Realized strategy performance (from this system's own closed trades)\n{_fmt(perf)}"
        if perf
        else ""
    )

    context = (
        "## Current regime\n"
        f"- Regime: {state['regime']}\n"
        f"- Confidence: {state['regime_confidence']:.2f}\n"
        f"- Reasoning: {state['regime_reasoning']}\n\n"
        "## Today's stats\n"
        f"{_fmt(state['daily_stats'])}\n\n"
        "## Raw signal counts by strategy\n"
        f"{_fmt(_signal_counts(state['signals']))}"
        f"{perf_block}"
        f"{_lessons_block(state, 'strategy')}\n\n"
        "Select which strategies should be active this cycle."
    )

    try:
        result = get_llm().structured_call(
            system=STRATEGY_SELECTION_PROMPT,
            user_content=context,
            output_model=StrategySelection,
            deep=False,
        )
        logger.info("[strategy] active: %s", result.active_strategies)
        return {
            "active_strategies": list(dict.fromkeys(result.active_strategies)),
            "strategy_reasoning": result.reasoning,
        }
    except Exception as e:
        logger.error("[strategy] LLM call failed: %s — using fallback", e)
        fallback = {
            "trending_up": ["momentum", "trend_following"],
            "trending_down": ["trend_following"],
            "ranging": ["mean_reversion"],
            "volatile": ["breakout"],
        }.get(state["regime"], [])
        return {
            "active_strategies": fallback,
            "strategy_reasoning": f"Fallback mapping for {state['regime']}. Error: {e}",
            "errors": state["errors"] + [f"strategy fallback: {e}"],
        }


def _signal_counts(signals: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for s in signals:
        counts[s.get("strategy", "?")] = counts.get(s.get("strategy", "?"), 0) + 1
    return counts


# =====================================================================
# 3. Signal Validation Agent (deep model)
# =====================================================================

def signal_validation_node(state: TradingState) -> dict[str, Any]:
    signals = state["signals"]
    active = set(state["active_strategies"])

    # Pre-filter: only send signals whose strategy is active (saves tokens)
    candidates = [s for s in signals if s.get("strategy") in active]
    pre_rejected = [
        {**s, "rejection_reason": "strategy not active this cycle"}
        for s in signals
        if s.get("strategy") not in active
    ]

    if not candidates:
        logger.info("[validation] no candidate signals after strategy filter")
        return {"validated_signals": [], "rejected_signals": pre_rejected}

    logger.info("[validation] validating %d candidate signals...", len(candidates))

    context = (
        "## Market regime\n"
        f"- Regime: {state['regime']} (confidence {state['regime_confidence']:.2f})\n"
        f"- Active strategies: {sorted(active)}\n\n"
        "## Candidate signals\n"
        f"{_fmt(candidates)}\n\n"
        "## Current portfolio\n"
        f"{_fmt(state['portfolio'])}"
        f"{_lessons_block(state, 'validation')}\n\n"
        "Validate each candidate signal: approve or reject with adjusted confidence."
    )

    try:
        result = get_llm().structured_call(
            system=SIGNAL_VALIDATION_PROMPT,
            user_content=context,
            output_model=SignalValidationResult,
            deep=True,
            max_tokens=8192,
        )
    except Exception as e:
        logger.error("[validation] LLM call failed: %s — rejecting all signals (fail-safe)", e)
        return {
            "validated_signals": [],
            "rejected_signals": pre_rejected
            + [{**s, "rejection_reason": f"validation unavailable: {e}"} for s in candidates],
            "errors": state["errors"] + [f"validation fallback: {e}"],
        }

    # Join verdicts back to the original signal dicts
    by_key = {(s["symbol"], s["signal_type"], s["strategy"]): s for s in candidates}
    validated, rejected = [], list(pre_rejected)

    for verdict in result.verdicts:
        key = (verdict.symbol, verdict.signal_type, verdict.strategy)
        original = by_key.pop(key, None)
        if original is None:
            continue  # hallucinated symbol — drop it
        if verdict.verdict == "approve" and verdict.adjusted_confidence >= 0.3:
            validated.append(
                {
                    **original,
                    "confidence": verdict.adjusted_confidence,
                    "validation_reasoning": verdict.reasoning,
                }
            )
        else:
            rejected.append({**original, "rejection_reason": verdict.reasoning})

    # Anything the model didn't return a verdict for is rejected (fail-safe)
    for leftover in by_key.values():
        rejected.append({**leftover, "rejection_reason": "no verdict returned by validator"})

    logger.info("[validation] approved=%d rejected=%d", len(validated), len(rejected))
    return {"validated_signals": validated, "rejected_signals": rejected}


# =====================================================================
# 4. Risk & Compliance Agent (deep model)
# =====================================================================

def risk_compliance_node(state: TradingState) -> dict[str, Any]:
    validated = state["validated_signals"]
    settings = get_settings()

    # Deterministic hard checks first — these don't need an LLM
    hard_block = _hard_risk_checks(state)
    if hard_block is not None:
        return hard_block

    logger.info("[risk] assessing %d validated signals...", len(validated))

    risk_limits = {
        "max_daily_trades": settings.max_daily_trades,
        "daily_loss_limit_usd": settings.daily_loss_limit,
        "max_open_positions": settings.max_open_positions,
        "max_position_pct_of_equity": settings.max_position_pct,
        "risk_per_trade_pct_of_equity": settings.risk_per_trade,
    }

    context = (
        "## Validated signals awaiting final approval\n"
        f"{_fmt(validated)}\n\n"
        "## Portfolio\n"
        f"{_fmt(state['portfolio'])}\n\n"
        "## Today's stats\n"
        f"{_fmt(state['daily_stats'])}\n\n"
        "## Configured risk limits\n"
        f"{_fmt(risk_limits)}\n\n"
        f"## Market regime\n- {state['regime']} (confidence {state['regime_confidence']:.2f})"
        f"{_lessons_block(state, 'risk')}\n\n"
        "Give the final approve/deny for each trade and decide on the kill switch."
    )

    try:
        result = get_llm().structured_call(
            system=RISK_COMPLIANCE_PROMPT,
            user_content=context,
            output_model=RiskAssessment,
            deep=True,
            max_tokens=8192,
        )
    except Exception as e:
        logger.error("[risk] LLM call failed: %s — denying all trades (fail-safe)", e)
        return {
            "approved_trades": [],
            "risk_rejected": [
                {**s, "rejection_reason": f"risk agent unavailable: {e}"} for s in validated
            ],
            "risk_warnings": [f"Risk agent unavailable: {e}"],
            "kill_switch": False,
            "errors": state["errors"] + [f"risk fallback: {e}"],
        }

    by_key = {(s["symbol"], s["signal_type"]): s for s in validated}
    approved, denied = [], []

    for decision in result.decisions:
        original = by_key.pop((decision.symbol, decision.signal_type), None)
        if original is None:
            continue
        if decision.approved and not result.kill_switch:
            approved.append({**original, "risk_reasoning": decision.reasoning})
        else:
            denied.append({**original, "rejection_reason": decision.reasoning})

    for leftover in by_key.values():
        denied.append({**leftover, "rejection_reason": "no risk decision returned"})

    logger.info(
        "[risk] approved=%d denied=%d kill_switch=%s",
        len(approved),
        len(denied),
        result.kill_switch,
    )
    return {
        "approved_trades": approved,
        "risk_rejected": denied,
        "risk_warnings": result.warnings,
        "kill_switch": result.kill_switch,
    }


def _hard_risk_checks(state: TradingState) -> dict[str, Any] | None:
    """Deterministic limits that block trading regardless of what any LLM says."""
    settings = get_settings()
    stats = state["daily_stats"]
    validated = state["validated_signals"]

    if stats.get("trades_count", 0) >= settings.max_daily_trades:
        return _deny_all(state, "daily trade limit reached", kill=False)

    if stats.get("realized_pnl", 0.0) <= -settings.daily_loss_limit:
        return _deny_all(state, "daily loss limit breached — kill switch engaged", kill=True)

    open_positions = len(state["portfolio"].get("positions", []))
    if open_positions >= settings.max_open_positions and validated:
        buys = [s for s in validated if s.get("signal_type") == "buy"]
        if len(buys) == len(validated):
            return _deny_all(state, "max open positions reached", kill=False)

    return None


def _deny_all(state: TradingState, reason: str, kill: bool) -> dict[str, Any]:
    logger.warning("[risk] HARD BLOCK: %s", reason)
    return {
        "approved_trades": [],
        "risk_rejected": [
            {**s, "rejection_reason": f"hard limit: {reason}"}
            for s in state["validated_signals"]
        ],
        "risk_warnings": [reason],
        "kill_switch": kill,
    }
