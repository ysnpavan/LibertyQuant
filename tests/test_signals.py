"""Tests for the signal engine."""

from liberty_quant.market.indicators import IndicatorSet
from liberty_quant.market.signals import SignalEngine, SignalType, Strategy


def make_indicators(**overrides) -> IndicatorSet:
    defaults = dict(
        symbol="TEST",
        close=100.0,
        change_percent=0.0,
        ema_9=100.0,
        ema_21=100.0,
        sma_50=100.0,
        adx=20.0,
        rsi_14=50.0,
        macd=0.0,
        macd_signal=0.0,
        macd_histogram=0.0,
        atr_14=2.0,
        bb_upper=104.0,
        bb_middle=100.0,
        bb_lower=96.0,
        bb_percent=0.5,
        volume=1_000_000,
        volume_ratio=1.0,
        vwap=100.0,
    )
    defaults.update(overrides)
    return IndicatorSet(**defaults)


def test_no_signals_on_neutral_tape():
    engine = SignalEngine()
    signals = engine.generate({"TEST": make_indicators()})
    assert signals == []


def test_rsi_oversold_generates_buy():
    engine = SignalEngine()
    ind = make_indicators(rsi_14=25.0)
    signals = engine.generate({"TEST": ind})
    rsi_signals = [s for s in signals if s.strategy == Strategy.MEAN_REVERSION]
    assert len(rsi_signals) == 1
    assert rsi_signals[0].signal_type == SignalType.BUY
    assert rsi_signals[0].stop_loss < rsi_signals[0].entry_price


def test_rsi_overbought_generates_sell():
    engine = SignalEngine()
    ind = make_indicators(rsi_14=78.0)
    signals = engine.generate({"TEST": ind})
    rsi_signals = [s for s in signals if s.strategy == Strategy.MEAN_REVERSION]
    assert len(rsi_signals) == 1
    assert rsi_signals[0].signal_type == SignalType.SELL


def test_bollinger_breakout_needs_volume():
    engine = SignalEngine()
    # Breakout above band but LOW volume -> no breakout signal
    quiet = make_indicators(close=105.0, volume_ratio=1.0)
    breakouts = [
        s for s in engine.generate({"TEST": quiet}) if s.strategy == Strategy.BREAKOUT
    ]
    assert breakouts == []

    # Same breakout WITH volume -> signal
    loud = make_indicators(close=105.0, volume_ratio=2.0)
    breakouts = [
        s for s in engine.generate({"TEST": loud}) if s.strategy == Strategy.BREAKOUT
    ]
    assert len(breakouts) == 1
    assert breakouts[0].signal_type == SignalType.BUY


def test_ema_crossover_trend_following():
    engine = SignalEngine()
    ind = make_indicators(close=101.5, ema_9=101.0, ema_21=100.0)
    signals = [
        s for s in engine.generate({"TEST": ind}) if s.strategy == Strategy.TREND_FOLLOWING
    ]
    assert len(signals) == 1
    assert signals[0].signal_type == SignalType.BUY


def test_signal_to_dict_roundtrip():
    engine = SignalEngine()
    signals = engine.generate({"TEST": make_indicators(rsi_14=25.0)})
    d = signals[0].to_dict()
    assert d["symbol"] == "TEST"
    assert d["signal_type"] == "buy"
    assert d["strategy"] == "mean_reversion"
    assert isinstance(d["confidence"], float)
