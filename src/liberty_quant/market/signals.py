"""Deterministic signal generation from computed indicators.

Raw signals are cheap, rule-based observations. The agent pipeline (regime ->
strategy -> validation -> risk) decides which, if any, become trades.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .indicators import IndicatorSet


class SignalType(Enum):
    BUY = "buy"
    SELL = "sell"


class Strategy(Enum):
    MOMENTUM = "momentum"
    MEAN_REVERSION = "mean_reversion"
    BREAKOUT = "breakout"
    TREND_FOLLOWING = "trend_following"


@dataclass
class Signal:
    """A raw trading signal produced by the rule engine."""

    symbol: str
    signal_type: SignalType
    strategy: Strategy
    confidence: float  # 0-1, rule-based heuristic
    entry_price: float
    stop_loss: float
    take_profit: float
    rationale: str
    indicators: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "signal_type": self.signal_type.value,
            "strategy": self.strategy.value,
            "confidence": round(self.confidence, 2),
            "entry_price": round(self.entry_price, 2),
            "stop_loss": round(self.stop_loss, 2),
            "take_profit": round(self.take_profit, 2),
            "rationale": self.rationale,
        }


class SignalEngine:
    """Generates raw signals from indicator sets using deterministic rules."""

    def generate(self, indicators: dict[str, IndicatorSet]) -> list[Signal]:
        signals: list[Signal] = []
        for symbol, ind in indicators.items():
            if ind is None:
                continue
            signals.extend(self._ema_crossover(ind))
            signals.extend(self._rsi_extreme(ind))
            signals.extend(self._macd_cross(ind))
            signals.extend(self._bollinger(ind))
        return signals

    # --- Strategy rules -------------------------------------------------

    def _ema_crossover(self, ind: IndicatorSet) -> list[Signal]:
        """EMA-9 vs EMA-21 — momentum / trend following."""
        out = []
        atr = max(ind.atr_14, ind.close * 0.005)
        spread_pct = (ind.ema_9 - ind.ema_21) / ind.close * 100

        if 0 < spread_pct < 1.5 and ind.close > ind.ema_9:
            # Recent bullish cross, price above fast EMA
            out.append(
                Signal(
                    symbol=ind.symbol,
                    signal_type=SignalType.BUY,
                    strategy=Strategy.TREND_FOLLOWING,
                    confidence=min(0.5 + abs(spread_pct) * 0.2, 0.8),
                    entry_price=ind.close,
                    stop_loss=ind.close - 2 * atr,
                    take_profit=ind.close + 3 * atr,
                    rationale=f"EMA9 ({ind.ema_9:.2f}) crossed above EMA21 ({ind.ema_21:.2f}), price above EMA9",
                    indicators=ind.to_dict(),
                )
            )
        elif -1.5 < spread_pct < 0 and ind.close < ind.ema_9:
            out.append(
                Signal(
                    symbol=ind.symbol,
                    signal_type=SignalType.SELL,
                    strategy=Strategy.TREND_FOLLOWING,
                    confidence=min(0.5 + abs(spread_pct) * 0.2, 0.8),
                    entry_price=ind.close,
                    stop_loss=ind.close + 2 * atr,
                    take_profit=ind.close - 3 * atr,
                    rationale=f"EMA9 ({ind.ema_9:.2f}) crossed below EMA21 ({ind.ema_21:.2f}), price below EMA9",
                    indicators=ind.to_dict(),
                )
            )
        return out

    def _rsi_extreme(self, ind: IndicatorSet) -> list[Signal]:
        """RSI oversold/overbought — mean reversion."""
        out = []
        atr = max(ind.atr_14, ind.close * 0.005)

        if ind.rsi_14 < 30:
            out.append(
                Signal(
                    symbol=ind.symbol,
                    signal_type=SignalType.BUY,
                    strategy=Strategy.MEAN_REVERSION,
                    confidence=min(0.4 + (30 - ind.rsi_14) * 0.02, 0.85),
                    entry_price=ind.close,
                    stop_loss=ind.close - 1.5 * atr,
                    take_profit=ind.bb_middle,
                    rationale=f"RSI oversold at {ind.rsi_14:.1f}",
                    indicators=ind.to_dict(),
                )
            )
        elif ind.rsi_14 > 70:
            out.append(
                Signal(
                    symbol=ind.symbol,
                    signal_type=SignalType.SELL,
                    strategy=Strategy.MEAN_REVERSION,
                    confidence=min(0.4 + (ind.rsi_14 - 70) * 0.02, 0.85),
                    entry_price=ind.close,
                    stop_loss=ind.close + 1.5 * atr,
                    take_profit=ind.bb_middle,
                    rationale=f"RSI overbought at {ind.rsi_14:.1f}",
                    indicators=ind.to_dict(),
                )
            )
        return out

    def _macd_cross(self, ind: IndicatorSet) -> list[Signal]:
        """MACD histogram flip — momentum."""
        out = []
        atr = max(ind.atr_14, ind.close * 0.005)
        # Histogram near zero and positive => fresh bullish cross
        hist_pct = ind.macd_histogram / ind.close * 100

        if 0 < hist_pct < 0.3 and ind.macd > 0:
            out.append(
                Signal(
                    symbol=ind.symbol,
                    signal_type=SignalType.BUY,
                    strategy=Strategy.MOMENTUM,
                    confidence=0.55,
                    entry_price=ind.close,
                    stop_loss=ind.close - 2 * atr,
                    take_profit=ind.close + 3 * atr,
                    rationale=f"MACD bullish cross (hist={ind.macd_histogram:.3f}) above zero line",
                    indicators=ind.to_dict(),
                )
            )
        elif -0.3 < hist_pct < 0 and ind.macd < 0:
            out.append(
                Signal(
                    symbol=ind.symbol,
                    signal_type=SignalType.SELL,
                    strategy=Strategy.MOMENTUM,
                    confidence=0.55,
                    entry_price=ind.close,
                    stop_loss=ind.close + 2 * atr,
                    take_profit=ind.close - 3 * atr,
                    rationale=f"MACD bearish cross (hist={ind.macd_histogram:.3f}) below zero line",
                    indicators=ind.to_dict(),
                )
            )
        return out

    def _bollinger(self, ind: IndicatorSet) -> list[Signal]:
        """Bollinger band breakout with volume confirmation."""
        out = []
        atr = max(ind.atr_14, ind.close * 0.005)

        if ind.close > ind.bb_upper and ind.volume_ratio > 1.5:
            out.append(
                Signal(
                    symbol=ind.symbol,
                    signal_type=SignalType.BUY,
                    strategy=Strategy.BREAKOUT,
                    confidence=min(0.5 + (ind.volume_ratio - 1.5) * 0.1, 0.85),
                    entry_price=ind.close,
                    stop_loss=ind.bb_middle,
                    take_profit=ind.close + 2 * atr,
                    rationale=(
                        f"Price broke above upper Bollinger band on {ind.volume_ratio:.1f}x volume"
                    ),
                    indicators=ind.to_dict(),
                )
            )
        elif ind.close < ind.bb_lower and ind.volume_ratio > 1.5:
            out.append(
                Signal(
                    symbol=ind.symbol,
                    signal_type=SignalType.SELL,
                    strategy=Strategy.BREAKOUT,
                    confidence=min(0.5 + (ind.volume_ratio - 1.5) * 0.1, 0.85),
                    entry_price=ind.close,
                    stop_loss=ind.bb_middle,
                    take_profit=ind.close - 2 * atr,
                    rationale=(
                        f"Price broke below lower Bollinger band on {ind.volume_ratio:.1f}x volume"
                    ),
                    indicators=ind.to_dict(),
                )
            )
        return out
