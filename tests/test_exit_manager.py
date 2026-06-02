"""Tests for the exit manager."""

import pytest

from liberty_quant.execution.exit_manager import ExitManager


@pytest.fixture
def manager(tmp_path):
    m = ExitManager(db_path=str(tmp_path / "exits.db"))
    yield m
    m.close()


def register_test_position(manager, **overrides):
    defaults = dict(
        symbol="NVDA",
        qty=10,
        entry_price=200.0,
        stop_loss=190.0,
        take_profit=220.0,
        strategy="momentum",
        regime="trending_up",
        confidence=0.7,
        cycle_id="TEST",
    )
    defaults.update(overrides)
    manager.register_entry(**defaults)
    return defaults


def test_register_and_get(manager):
    register_test_position(manager)
    position = manager.get_position("NVDA")
    assert position is not None
    assert position["qty"] == 10
    assert position["stop_loss"] == 190.0
    assert position["high_water_mark"] == 200.0


def test_no_exit_when_price_in_range(manager):
    register_test_position(manager)
    exits = manager.check_exits({"NVDA": 205.0})
    assert exits == []


def test_stop_loss_exit(manager):
    register_test_position(manager)
    exits = manager.check_exits({"NVDA": 189.0})
    assert len(exits) == 1
    assert exits[0].reason == "stop_loss"
    assert exits[0].side == "sell"
    assert exits[0].qty == 10


def test_take_profit_exit(manager):
    register_test_position(manager)
    exits = manager.check_exits({"NVDA": 221.0})
    assert len(exits) == 1
    assert exits[0].reason == "take_profit"


def test_trailing_stop_only_after_gains(manager):
    register_test_position(manager, stop_loss=180.0, take_profit=300.0)

    # Price runs up to 215 — high-water mark ratchets
    assert manager.check_exits({"NVDA": 215.0}) == []
    assert manager.get_position("NVDA")["high_water_mark"] == 215.0

    # Price pulls back 3%+ from the high but still above entry -> trailing stop
    exits = manager.check_exits({"NVDA": 208.0})  # 215 * 0.97 = 208.55
    assert len(exits) == 1
    assert exits[0].reason == "trailing_stop"


def test_trailing_stop_never_fires_below_entry(manager):
    register_test_position(manager, stop_loss=150.0, take_profit=300.0)
    # Price drops below entry but above the hard stop — no trailing stop exit
    exits = manager.check_exits({"NVDA": 195.0})
    assert exits == []


def test_eod_flatten(manager):
    register_test_position(manager)
    exits = manager.check_exits({"NVDA": 205.0}, near_market_close=True)
    assert len(exits) == 1
    assert exits[0].reason == "eod_flatten"


def test_remove_position(manager):
    register_test_position(manager)
    removed = manager.remove("NVDA")
    assert removed["symbol"] == "NVDA"
    assert manager.get_position("NVDA") is None
    assert manager.all_positions() == []


def test_adding_to_position_blends_entry(manager):
    register_test_position(manager, qty=10, entry_price=200.0)
    register_test_position(manager, qty=10, entry_price=210.0)
    position = manager.get_position("NVDA")
    assert position["qty"] == 20
    assert position["entry_price"] == pytest.approx(205.0)


def test_reconcile_drops_orphans(manager):
    register_test_position(manager, symbol="NVDA")
    register_test_position(manager, symbol="AMD")
    # Broker only has NVDA — AMD was closed manually
    dropped = manager.reconcile({"NVDA"})
    assert dropped == ["AMD"]
    assert manager.get_position("AMD") is None
    assert manager.get_position("NVDA") is not None


def test_persistence_across_instances(tmp_path):
    path = str(tmp_path / "exits.db")
    m1 = ExitManager(db_path=path)
    register_test_position(m1)
    m1.close()

    m2 = ExitManager(db_path=path)
    assert m2.get_position("NVDA") is not None
    m2.close()
