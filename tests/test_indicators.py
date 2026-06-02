"""Tests for the indicator engine."""

import numpy as np
import pandas as pd
import pytest

from liberty_quant.market.indicators import (
    bollinger_bands,
    compute_indicators,
    ema,
    macd,
    rsi,
)


@pytest.fixture
def trending_up_df() -> pd.DataFrame:
    """60 bars of steadily rising prices."""
    rng = np.random.default_rng(42)
    closes = 100 + np.cumsum(rng.normal(0.5, 0.5, 60))
    return pd.DataFrame(
        {
            "open": closes - 0.2,
            "high": closes + 0.5,
            "low": closes - 0.5,
            "close": closes,
            "volume": rng.integers(1_000_000, 2_000_000, 60).astype(float),
        },
        index=pd.date_range("2026-01-01", periods=60, freq="D"),
    )


def test_ema_follows_price(trending_up_df):
    e = ema(trending_up_df["close"], 9)
    assert len(e) == 60
    # EMA should be below the last close in an uptrend
    assert e.iloc[-1] < trending_up_df["close"].iloc[-1] + 1


def test_rsi_in_uptrend_is_high(trending_up_df):
    r = rsi(trending_up_df["close"])
    assert 50 < r.iloc[-1] <= 100


def test_rsi_bounds():
    # Strictly increasing prices -> RSI should approach 100
    closes = pd.Series(np.arange(100, 160, dtype=float))
    r = rsi(closes)
    assert r.iloc[-1] > 90
    # Strictly decreasing -> RSI should approach 0
    closes_down = pd.Series(np.arange(160, 100, -1, dtype=float))
    r_down = rsi(closes_down)
    assert r_down.iloc[-1] < 10


def test_macd_positive_in_uptrend(trending_up_df):
    m = macd(trending_up_df["close"])
    assert m["macd"].iloc[-1] > 0


def test_bollinger_band_ordering(trending_up_df):
    bb = bollinger_bands(trending_up_df["close"])
    last = bb.iloc[-1]
    assert last["lower"] < last["middle"] < last["upper"]
    assert 0 <= last["percent"] <= 1


def test_compute_indicators_full_set(trending_up_df):
    ind = compute_indicators("TEST", trending_up_df)
    assert ind is not None
    assert ind.symbol == "TEST"
    assert ind.close == pytest.approx(trending_up_df["close"].iloc[-1])
    assert ind.ema_9 > ind.ema_21  # uptrend: fast EMA above slow
    assert ind.rsi_14 > 50
    assert ind.atr_14 > 0
    d = ind.to_dict()
    assert set(d) >= {"symbol", "close", "rsi_14", "macd", "atr_14", "bb_percent"}


def test_compute_indicators_insufficient_data():
    short_df = pd.DataFrame(
        {"open": [1, 2], "high": [1, 2], "low": [1, 2], "close": [1, 2], "volume": [1, 2]}
    )
    assert compute_indicators("X", short_df) is None
