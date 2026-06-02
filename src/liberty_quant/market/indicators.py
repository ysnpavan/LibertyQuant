"""Technical indicators computed from OHLCV DataFrames.

Pure pandas/numpy — no TA-Lib dependency. Input DataFrames must have columns:
open, high, low, close, volume (lowercase) indexed by timestamp ascending.
"""

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class IndicatorSet:
    """All indicators for one symbol at the latest bar."""

    symbol: str
    close: float
    change_percent: float

    # Trend
    ema_9: float
    ema_21: float
    sma_50: float | None
    adx: float | None

    # Momentum
    rsi_14: float
    macd: float
    macd_signal: float
    macd_histogram: float

    # Volatility
    atr_14: float
    bb_upper: float
    bb_middle: float
    bb_lower: float
    bb_percent: float  # position of close within bands: 0 = lower, 1 = upper

    # Volume
    volume: float
    volume_ratio: float  # current volume / 20-bar average
    vwap: float | None

    extras: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "close": round(self.close, 2),
            "change_percent": round(self.change_percent, 2),
            "ema_9": round(self.ema_9, 2),
            "ema_21": round(self.ema_21, 2),
            "sma_50": round(self.sma_50, 2) if self.sma_50 is not None else None,
            "adx": round(self.adx, 1) if self.adx is not None else None,
            "rsi_14": round(self.rsi_14, 1),
            "macd": round(self.macd, 4),
            "macd_signal": round(self.macd_signal, 4),
            "macd_histogram": round(self.macd_histogram, 4),
            "atr_14": round(self.atr_14, 2),
            "bb_upper": round(self.bb_upper, 2),
            "bb_middle": round(self.bb_middle, 2),
            "bb_lower": round(self.bb_lower, 2),
            "bb_percent": round(self.bb_percent, 2),
            "volume": int(self.volume),
            "volume_ratio": round(self.volume_ratio, 2),
            "vwap": round(self.vwap, 2) if self.vwap is not None else None,
        }


def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(window=period).mean()


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    # RSI = 100 * gain / (gain + loss) — handles all-gain (=100) and all-loss (=0)
    denom = (gain + loss).replace(0, np.nan)
    return (100 * gain / denom).fillna(50.0)


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    macd_line = ema(close, fast) - ema(close, slow)
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return pd.DataFrame(
        {"macd": macd_line, "signal": signal_line, "histogram": macd_line - signal_line}
    )


def bollinger_bands(close: pd.Series, period: int = 20, num_std: float = 2.0) -> pd.DataFrame:
    middle = sma(close, period)
    std = close.rolling(window=period).std()
    upper = middle + num_std * std
    lower = middle - num_std * std
    band_range = (upper - lower).replace(0, np.nan)
    percent = ((close - lower) / band_range).clip(0, 1)
    return pd.DataFrame({"upper": upper, "middle": middle, "lower": lower, "percent": percent})


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low = df["high"], df["low"]
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=df.index)
    minus_dm = pd.Series(
        np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=df.index
    )
    atr_series = atr(df, period).replace(0, np.nan)
    plus_di = 100 * plus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr_series
    minus_di = 100 * minus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr_series
    di_sum = (plus_di + minus_di).replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / di_sum
    return dx.ewm(alpha=1 / period, adjust=False).mean()


def vwap(df: pd.DataFrame) -> pd.Series:
    typical = (df["high"] + df["low"] + df["close"]) / 3
    cum_vol = df["volume"].cumsum().replace(0, np.nan)
    return (typical * df["volume"]).cumsum() / cum_vol


def compute_indicators(symbol: str, df: pd.DataFrame) -> IndicatorSet | None:
    """Compute the full indicator set for a symbol from its OHLCV history.

    Returns None if there isn't enough history (< 30 bars).
    """
    if df is None or len(df) < 30:
        return None

    close = df["close"]
    latest = df.iloc[-1]
    prev_close = close.iloc[-2]

    macd_df = macd(close)
    bb_df = bollinger_bands(close)
    atr_series = atr(df)
    vol_avg = df["volume"].rolling(20).mean().iloc[-1]

    has_50 = len(df) >= 50

    return IndicatorSet(
        symbol=symbol,
        close=float(latest["close"]),
        change_percent=float((latest["close"] - prev_close) / prev_close * 100),
        ema_9=float(ema(close, 9).iloc[-1]),
        ema_21=float(ema(close, 21).iloc[-1]),
        sma_50=float(sma(close, 50).iloc[-1]) if has_50 else None,
        adx=float(adx(df).iloc[-1]) if len(df) >= 28 else None,
        rsi_14=float(rsi(close).iloc[-1]),
        macd=float(macd_df["macd"].iloc[-1]),
        macd_signal=float(macd_df["signal"].iloc[-1]),
        macd_histogram=float(macd_df["histogram"].iloc[-1]),
        atr_14=float(atr_series.iloc[-1]),
        bb_upper=float(bb_df["upper"].iloc[-1]),
        bb_middle=float(bb_df["middle"].iloc[-1]),
        bb_lower=float(bb_df["lower"].iloc[-1]),
        bb_percent=float(bb_df["percent"].iloc[-1]) if not np.isnan(bb_df["percent"].iloc[-1]) else 0.5,
        volume=float(latest["volume"]),
        volume_ratio=float(latest["volume"] / vol_avg) if vol_avg and vol_avg > 0 else 1.0,
        vwap=float(vwap(df).iloc[-1]),
    )
