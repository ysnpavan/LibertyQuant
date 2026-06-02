"""Market data feeds.

``DataFeed`` is the interface; ``AlpacaFeed`` (primary, real-time IEX data) and
``YFinanceFeed`` (free fallback) are the implementations. The manager picks
whichever is configured/available.

All feeds return OHLCV DataFrames with lowercase columns
(open, high, low, close, volume) indexed by timestamp ascending.
"""

import logging
from abc import ABC, abstractmethod
from datetime import UTC, datetime, timedelta

import pandas as pd

from ..config import get_settings

logger = logging.getLogger(__name__)


class DataFeed(ABC):
    """Abstract market data feed."""

    @abstractmethod
    def get_bars(self, symbols: list[str], lookback_days: int = 90) -> dict[str, pd.DataFrame]:
        """Daily OHLCV history per symbol."""

    @abstractmethod
    def get_intraday_bars(
        self, symbols: list[str], minutes: int = 390
    ) -> dict[str, pd.DataFrame]:
        """Intraday (1-minute) OHLCV per symbol for roughly the last trading day."""

    @abstractmethod
    def is_available(self) -> bool:
        """Whether this feed can currently serve data."""


class AlpacaFeed(DataFeed):
    """Alpaca market data (free IEX feed with a paper account)."""

    def __init__(self) -> None:
        from alpaca.data.historical import StockHistoricalDataClient

        settings = get_settings()
        if not (settings.alpaca_api_key and settings.alpaca_secret_key):
            raise RuntimeError("Alpaca credentials not configured")
        self._client = StockHistoricalDataClient(
            api_key=settings.alpaca_api_key,
            secret_key=settings.alpaca_secret_key.get_secret_value(),
        )

    def is_available(self) -> bool:
        return True

    def _bars_request(self, symbols: list[str], timeframe, start: datetime):
        from alpaca.data.enums import DataFeed as AlpacaDataFeed
        from alpaca.data.requests import StockBarsRequest

        return StockBarsRequest(
            symbol_or_symbols=symbols,
            timeframe=timeframe,
            start=start,
            feed=AlpacaDataFeed.IEX,  # free tier
        )

    def _to_frames(self, bars) -> dict[str, pd.DataFrame]:
        """Convert an Alpaca BarSet to per-symbol OHLCV DataFrames."""
        frames: dict[str, pd.DataFrame] = {}
        df = bars.df
        if df is None or df.empty:
            return frames
        # Alpaca returns a MultiIndex (symbol, timestamp) DataFrame
        for symbol in df.index.get_level_values(0).unique():
            sym_df = df.xs(symbol, level=0)[["open", "high", "low", "close", "volume"]].copy()
            sym_df.sort_index(inplace=True)
            frames[str(symbol)] = sym_df
        return frames

    def get_bars(self, symbols: list[str], lookback_days: int = 90) -> dict[str, pd.DataFrame]:
        from alpaca.data.timeframe import TimeFrame

        start = datetime.now(UTC) - timedelta(days=lookback_days)
        try:
            bars = self._client.get_stock_bars(
                self._bars_request(symbols, TimeFrame.Day, start)
            )
            return self._to_frames(bars)
        except Exception as e:
            logger.error("Alpaca daily bars fetch failed: %s", e)
            return {}

    def get_intraday_bars(
        self, symbols: list[str], minutes: int = 390
    ) -> dict[str, pd.DataFrame]:
        from alpaca.data.timeframe import TimeFrame

        # Go back far enough to cover weekends/holidays
        start = datetime.now(UTC) - timedelta(days=4)
        try:
            bars = self._client.get_stock_bars(
                self._bars_request(symbols, TimeFrame.Minute, start)
            )
            frames = self._to_frames(bars)
            # Keep only the most recent `minutes` bars per symbol
            return {sym: df.tail(minutes) for sym, df in frames.items()}
        except Exception as e:
            logger.error("Alpaca intraday bars fetch failed: %s", e)
            return {}


class YFinanceFeed(DataFeed):
    """Yahoo Finance fallback feed (free, delayed, rate-limited)."""

    def is_available(self) -> bool:
        return True

    def get_bars(self, symbols: list[str], lookback_days: int = 90) -> dict[str, pd.DataFrame]:
        return self._download(symbols, period=f"{lookback_days}d", interval="1d")

    def get_intraday_bars(
        self, symbols: list[str], minutes: int = 390
    ) -> dict[str, pd.DataFrame]:
        frames = self._download(symbols, period="2d", interval="1m")
        return {sym: df.tail(minutes) for sym, df in frames.items()}

    def _download(self, symbols: list[str], period: str, interval: str) -> dict[str, pd.DataFrame]:
        import yfinance as yf

        frames: dict[str, pd.DataFrame] = {}
        try:
            # yfinance uses '-' instead of '.' for class shares (BRK.B -> BRK-B)
            yf_symbols = [s.replace(".", "-") for s in symbols]
            data = yf.download(
                tickers=" ".join(yf_symbols),
                period=period,
                interval=interval,
                group_by="ticker",
                auto_adjust=True,
                progress=False,
                threads=True,
            )
            if data is None or data.empty:
                return frames

            for orig, yf_sym in zip(symbols, yf_symbols):
                try:
                    sym_df = data[yf_sym] if len(symbols) > 1 else data
                    sym_df = sym_df.rename(
                        columns={
                            "Open": "open",
                            "High": "high",
                            "Low": "low",
                            "Close": "close",
                            "Volume": "volume",
                        }
                    )[["open", "high", "low", "close", "volume"]].dropna()
                    if not sym_df.empty:
                        frames[orig] = sym_df
                except (KeyError, TypeError):
                    continue
        except Exception as e:
            logger.error("yfinance download failed: %s", e)
        return frames


def create_feed() -> DataFeed:
    """Create the best available feed: Alpaca if credentialed, else yfinance."""
    settings = get_settings()
    if settings.alpaca_api_key and settings.alpaca_secret_key:
        try:
            feed = AlpacaFeed()
            logger.info("Using Alpaca market data feed (IEX)")
            return feed
        except Exception as e:
            logger.warning("Alpaca feed unavailable (%s); falling back to yfinance", e)
    logger.info("Using yfinance market data feed")
    return YFinanceFeed()
