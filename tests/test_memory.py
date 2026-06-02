"""Tests for the memory store and learning loop."""

from unittest.mock import patch

import pytest

from liberty_quant.agents.schemas import TradePostMortem
from liberty_quant.memory.learning import TradeLearner, get_lessons_for_agents
from liberty_quant.memory.store import MemoryStore


@pytest.fixture
def store(tmp_path):
    s = MemoryStore(db_path=str(tmp_path / "memory.db"))
    yield s
    s.close()


def make_position(**overrides):
    defaults = dict(
        symbol="NVDA",
        qty=10,
        side="long",
        entry_price=200.0,
        stop_loss=190.0,
        take_profit=220.0,
        strategy="momentum",
        regime_at_entry="trending_up",
        confidence=0.7,
        opened_at="2026-06-02T14:00:00+00:00",
        cycle_id="TEST",
    )
    defaults.update(overrides)
    return defaults


# --- MemoryStore ---------------------------------------------------------


def test_record_winning_trade(store):
    trade_id = store.record_closed_trade(
        symbol="NVDA", strategy="momentum", side="long", qty=10,
        entry_price=200.0, exit_price=220.0, exit_reason="take_profit",
        regime_at_entry="trending_up", confidence=0.7,
        opened_at=None, cycle_id="T",
    )
    trade = store.get_closed_trade(trade_id)
    assert trade["pnl"] == pytest.approx(200.0)
    assert trade["pnl_pct"] == pytest.approx(10.0)


def test_record_losing_trade(store):
    trade_id = store.record_closed_trade(
        symbol="AMD", strategy="breakout", side="long", qty=20,
        entry_price=100.0, exit_price=95.0, exit_reason="stop_loss",
        regime_at_entry="volatile", confidence=0.5,
        opened_at=None, cycle_id="T",
    )
    trade = store.get_closed_trade(trade_id)
    assert trade["pnl"] == pytest.approx(-100.0)
    assert trade["pnl_pct"] == pytest.approx(-5.0)


def test_strategy_performance(store):
    # 2 momentum wins, 1 momentum loss, 1 breakout loss
    for entry, exit_, strat in [
        (100, 110, "momentum"), (100, 105, "momentum"),
        (100, 95, "momentum"), (100, 90, "breakout"),
    ]:
        store.record_closed_trade(
            symbol="X", strategy=strat, side="long", qty=1,
            entry_price=entry, exit_price=exit_, exit_reason="manual",
            regime_at_entry="trending_up", confidence=0.5, opened_at=None, cycle_id="T",
        )
    perf = store.strategy_performance()
    assert perf["momentum"]["trades"] == 3
    assert perf["momentum"]["win_rate"] == pytest.approx(0.67, abs=0.01)
    assert perf["breakout"]["win_rate"] == 0.0


def test_lessons_severity_ordering(store):
    store.add_lesson(category="poor_timing", severity="low", lesson="low lesson")
    store.add_lesson(category="regime_mismatch", severity="critical", lesson="critical lesson")
    store.add_lesson(category="wrong_sizing", severity="medium", lesson="medium lesson")

    lessons = store.get_lessons(limit=10)
    assert lessons[0]["severity"] == "critical"
    assert lessons[-1]["severity"] == "low"


def test_lessons_category_filter(store):
    store.add_lesson(category="regime_mismatch", severity="high", lesson="regime lesson")
    store.add_lesson(category="wrong_sizing", severity="high", lesson="sizing lesson")

    lessons = store.get_lessons(categories=["regime_mismatch"], limit=10)
    assert len(lessons) == 1
    assert lessons[0]["lesson"] == "regime lesson"


# --- TradeLearner ---------------------------------------------------------


def postmortem(is_mistake=True, category="regime_mismatch", severity="high"):
    return TradePostMortem(
        is_mistake=is_mistake,
        category=category,
        severity=severity,
        lesson="Do not take momentum trades in a ranging regime",
        reasoning="test",
    )


class FakeLLM:
    def __init__(self, result):
        self.result = result
        self.called = False

    def structured_call(self, **kwargs):
        self.called = True
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


def test_losing_trade_creates_lesson(store):
    learner = TradeLearner(store)
    fake = FakeLLM(postmortem())

    with patch("liberty_quant.memory.learning.get_llm", return_value=fake):
        learner.process_closed_trade(
            position=make_position(),
            exit_price=185.0,  # loss
            exit_reason="stop_loss",
            cycle_id="T",
        )

    assert fake.called
    lessons = store.get_lessons(limit=10)
    assert len(lessons) == 1
    assert lessons[0]["category"] == "regime_mismatch"


def test_no_mistake_creates_no_lesson(store):
    learner = TradeLearner(store)
    fake = FakeLLM(postmortem(is_mistake=False, category="no_mistake"))

    with patch("liberty_quant.memory.learning.get_llm", return_value=fake):
        learner.process_closed_trade(
            position=make_position(),
            exit_price=185.0,
            exit_reason="stop_loss",
            cycle_id="T",
        )

    assert store.get_lessons(limit=10) == []


def test_routine_winner_skips_postmortem(store):
    learner = TradeLearner(store)
    fake = FakeLLM(postmortem())

    with patch("liberty_quant.memory.learning.get_llm", return_value=fake):
        learner.process_closed_trade(
            position=make_position(),
            exit_price=220.0,  # clean take-profit win
            exit_reason="take_profit",
            cycle_id="T",
        )

    assert not fake.called  # no post-mortem needed
    # But the trade is still recorded
    assert len(store.recent_closed_trades()) == 1


def test_postmortem_failure_does_not_break_recording(store):
    learner = TradeLearner(store)
    fake = FakeLLM(RuntimeError("API down"))

    with patch("liberty_quant.memory.learning.get_llm", return_value=fake):
        trade_id = learner.process_closed_trade(
            position=make_position(),
            exit_price=185.0,
            exit_reason="stop_loss",
            cycle_id="T",
        )

    # Trade recorded despite the classifier failing
    assert store.get_closed_trade(trade_id) is not None
    assert store.get_lessons(limit=10) == []


def test_lessons_routed_to_agents(store):
    store.add_lesson(category="regime_mismatch", severity="high", lesson="regime lesson")
    store.add_lesson(category="wrong_sizing", severity="high", lesson="sizing lesson")
    store.add_lesson(category="poor_timing", severity="high", lesson="timing lesson")

    routed = get_lessons_for_agents(store)
    assert any(lesson["lesson"] == "regime lesson" for lesson in routed["regime"])
    assert any(lesson["lesson"] == "sizing lesson" for lesson in routed["risk"])
    assert any(lesson["lesson"] == "timing lesson" for lesson in routed["validation"])
    # Sizing lessons don't go to the regime agent
    assert not any(lesson["lesson"] == "sizing lesson" for lesson in routed["regime"])
