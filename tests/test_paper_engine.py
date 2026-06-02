"""Tests for the local paper trading engine."""

import pytest

from liberty_quant.execution.paper_engine import LocalPaperEngine


@pytest.fixture
def engine(tmp_path):
    return LocalPaperEngine(state_path=str(tmp_path / "wallet.json"))


def test_starting_account(engine):
    account = engine.get_account()
    assert account["cash"] == 100_000.0
    assert account["equity"] == 100_000.0
    assert engine.get_positions() == []


def test_buy_creates_position(engine):
    result = engine.submit_market_order(symbol="AAPL", side="buy", qty=10, current_price=200.0)
    assert result.status == "filled"
    assert result.filled_price > 200.0  # slippage on buys raises price

    positions = engine.get_positions()
    assert len(positions) == 1
    assert positions[0].symbol == "AAPL"
    assert positions[0].qty == 10

    account = engine.get_account()
    assert account["cash"] < 100_000.0


def test_buy_insufficient_cash_rejected(engine):
    result = engine.submit_market_order(
        symbol="BRK.A", side="buy", qty=1000, current_price=700_000.0
    )
    assert result.status == "rejected"
    assert "insufficient cash" in result.error


def test_sell_without_position_rejected(engine):
    result = engine.submit_market_order(symbol="AAPL", side="sell", qty=5, current_price=200.0)
    assert result.status == "rejected"
    assert "insufficient shares" in result.error


def test_round_trip_pnl(engine):
    engine.submit_market_order(symbol="MSFT", side="buy", qty=10, current_price=400.0)
    # Price rises; sell at 420
    result = engine.submit_market_order(symbol="MSFT", side="sell", qty=10, current_price=420.0)
    assert result.status == "filled"
    assert engine.get_positions() == []

    account = engine.get_account()
    assert account["realized_pnl"] > 0
    # Equity should be above start (gain) minus slippage costs
    assert account["equity"] > 100_000.0


def test_state_persists_across_instances(tmp_path):
    path = str(tmp_path / "wallet.json")
    e1 = LocalPaperEngine(state_path=path)
    e1.submit_market_order(symbol="NVDA", side="buy", qty=5, current_price=190.0)

    e2 = LocalPaperEngine(state_path=path)
    positions = e2.get_positions()
    assert len(positions) == 1
    assert positions[0].symbol == "NVDA"


def test_mark_to_market(engine):
    engine.submit_market_order(symbol="AMD", side="buy", qty=10, current_price=160.0)
    engine.mark_to_market({"AMD": 170.0})
    positions = engine.get_positions()
    assert positions[0].current_price == 170.0
    assert positions[0].unrealized_pnl > 0
