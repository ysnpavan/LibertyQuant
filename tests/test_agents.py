"""Tests for the agent pipeline with mocked LLM calls.

These verify graph wiring, fallbacks, fail-safes, and hard risk checks
without hitting the Anthropic API.
"""

from unittest.mock import patch

from liberty_quant.agents.graph import create_trading_graph, run_trading_cycle
from liberty_quant.agents.nodes import _hard_risk_checks
from liberty_quant.agents.schemas import (
    RegimeAssessment,
    RiskAssessment,
    SignalValidationResult,
    SignalVerdict,
    StrategySelection,
    TradeRiskDecision,
)
from liberty_quant.agents.state import create_initial_state

BENCHMARK_INDICATORS = {
    "SPY": {"symbol": "SPY", "close": 600.0, "change_percent": 1.0, "rsi_14": 60.0},
    "QQQ": {"symbol": "QQQ", "close": 530.0, "change_percent": 1.2, "rsi_14": 62.0},
}

SIGNALS = [
    {
        "symbol": "NVDA", "signal_type": "buy", "strategy": "momentum",
        "confidence": 0.6, "entry_price": 190.0, "stop_loss": 180.0, "take_profit": 210.0,
        "rationale": "test",
    }
]

PORTFOLIO = {"cash": 100_000.0, "equity": 100_000.0, "buying_power": 100_000.0, "positions": []}
STATS = {"trades_count": 0, "realized_pnl": 0.0}


class FakeLLM:
    """Configurable fake for AgentLLM.structured_call."""

    def __init__(self, responses: dict):
        self.responses = responses
        self.calls = []

    def structured_call(self, *, output_model, **kwargs):
        self.calls.append(output_model.__name__)
        result = self.responses.get(output_model.__name__)
        if isinstance(result, Exception):
            raise result
        return result


def make_fake_llm(
    regime="trending_up",
    regime_confidence=0.8,
    strategies=("momentum",),
    approve_validation=True,
    approve_risk=True,
    kill_switch=False,
):
    return FakeLLM(
        {
            "RegimeAssessment": RegimeAssessment(
                regime=regime,
                confidence=regime_confidence,
                reasoning="test regime",
                key_factors=["test"],
            ),
            "StrategySelection": StrategySelection(
                active_strategies=list(strategies), reasoning="test strategies"
            ),
            "SignalValidationResult": SignalValidationResult(
                verdicts=[
                    SignalVerdict(
                        symbol="NVDA",
                        strategy="momentum",
                        signal_type="buy",
                        verdict="approve" if approve_validation else "reject",
                        adjusted_confidence=0.7 if approve_validation else 0.1,
                        reasoning="test validation",
                    )
                ],
                overall_assessment="test",
            ),
            "RiskAssessment": RiskAssessment(
                decisions=[
                    TradeRiskDecision(
                        symbol="NVDA",
                        signal_type="buy",
                        approved=approve_risk,
                        reasoning="test risk",
                    )
                ],
                kill_switch=kill_switch,
                warnings=[],
                reasoning="test",
            ),
        }
    )


def run_pipeline(fake_llm):
    graph = create_trading_graph()
    with patch("liberty_quant.agents.nodes.get_llm", return_value=fake_llm):
        return run_trading_cycle(
            graph,
            cycle_id="TEST",
            indicators=BENCHMARK_INDICATORS,
            signals=SIGNALS,
            portfolio=PORTFOLIO,
            daily_stats=STATS,
        )


def test_full_pipeline_approves_trade():
    state = run_pipeline(make_fake_llm())
    assert state["regime"] == "trending_up"
    assert state["active_strategies"] == ["momentum"]
    assert len(state["validated_signals"]) == 1
    assert len(state["approved_trades"]) == 1
    assert state["approved_trades"][0]["symbol"] == "NVDA"


def test_low_regime_confidence_aborts_cycle():
    fake = make_fake_llm(regime_confidence=0.2)
    state = run_pipeline(fake)
    # Pipeline should stop after regime — no strategy/validation calls
    assert state["active_strategies"] == []
    assert state["approved_trades"] == []
    assert fake.calls == ["RegimeAssessment"]


def test_no_strategies_ends_cycle():
    fake = make_fake_llm(strategies=())
    state = run_pipeline(fake)
    assert state["approved_trades"] == []
    assert "SignalValidationResult" not in fake.calls


def test_validation_rejection_skips_risk():
    fake = make_fake_llm(approve_validation=False)
    state = run_pipeline(fake)
    assert state["validated_signals"] == []
    assert len(state["rejected_signals"]) >= 1
    assert "RiskAssessment" not in fake.calls


def test_risk_denial():
    state = run_pipeline(make_fake_llm(approve_risk=False))
    assert state["approved_trades"] == []
    assert len(state["risk_rejected"]) == 1


def test_kill_switch_blocks_all_trades():
    state = run_pipeline(make_fake_llm(kill_switch=True))
    assert state["approved_trades"] == []
    assert state["kill_switch"] is True


def test_llm_failure_falls_back_safely():
    """If every LLM call fails, the pipeline must complete with zero trades."""
    fake = FakeLLM(
        {
            "RegimeAssessment": RuntimeError("API down"),
            "StrategySelection": RuntimeError("API down"),
            "SignalValidationResult": RuntimeError("API down"),
            "RiskAssessment": RuntimeError("API down"),
        }
    )
    state = run_pipeline(fake)
    # Fallback regime still classifies, but validation fail-safe rejects everything
    assert state["approved_trades"] == []
    assert len(state["errors"]) >= 1


# --- Hard risk checks (no LLM involved) ---------------------------------


def make_state(**overrides):
    state = create_initial_state(
        cycle_id="T",
        indicators={},
        signals=[],
        portfolio=dict(PORTFOLIO),
        daily_stats=dict(STATS),
    )
    state["validated_signals"] = [dict(s) for s in SIGNALS]
    state.update(overrides)
    return state


def test_hard_check_daily_trade_limit():
    state = make_state(daily_stats={"trades_count": 999, "realized_pnl": 0.0})
    result = _hard_risk_checks(state)
    assert result is not None
    assert result["approved_trades"] == []
    assert result["kill_switch"] is False


def test_hard_check_daily_loss_limit_engages_kill_switch():
    state = make_state(daily_stats={"trades_count": 0, "realized_pnl": -999_999.0})
    result = _hard_risk_checks(state)
    assert result is not None
    assert result["kill_switch"] is True


def test_hard_check_passes_clean_state():
    state = make_state()
    assert _hard_risk_checks(state) is None
