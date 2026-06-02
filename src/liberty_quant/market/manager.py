"""MarketDataManager — orchestrates feeds, discovery, indicators, and signals.

Each trading cycle:
1. Refresh daily bars for the whole universe (cached, refreshed periodically)
2. Select the most "interesting" symbols (top movers + volume spikes + benchmarks)
3. Compute indicators on the active set
4. Generate raw signals
"""

import logging
import time

import pandas as pd

from ..config import get_settings
from .feed import DataFeed, create_feed
from .indicators import IndicatorSet, compute_indicators
from .signals import Signal, SignalEngine
from .universe import BENCHMARKS, get_universe

logger = logging.getLogger(__name__)

# How long the universe-wide daily scan stays fresh (seconds)
UNIVERSE_SCAN_TTL = 15 * 60


class MarketDataManager:
    """Single entry point for all market data used by the trading loop."""

    def __init__(self, feed: DataFeed | None = None) -> None:
        settings = get_settings()
        self.feed = feed or create_feed()
        self.universe = get_universe(settings.universe)
        self.max_symbols = settings.max_symbols
        self.signal_engine = SignalEngine()

        self._daily_bars: dict[str, pd.DataFrame] = {}
        self._daily_bars_at: float = 0.0

    # --- Universe scan ----------------------------------------------------

    def _refresh_universe_bars(self) -> None:
        """Fetch daily bars for the full universe (throttled by TTL)."""
        if time.monotonic() - self._daily_bars_at < UNIVERSE_SCAN_TTL and self._daily_bars:
            return
        logger.info("Scanning universe (%d symbols) for daily bars...", len(self.universe))
        self._daily_bars = self.feed.get_bars(self.universe, lookback_days=90)
        self._daily_bars_at = time.monotonic()
        logger.info("Got daily bars for %d symbols", len(self._daily_bars))

    def select_active_symbols(self) -> list[str]:
        """Dynamic discovery: rank universe by abs(%change) * volume ratio.

        Benchmarks (SPY/QQQ/IWM) are always included for regime context.
        """
        self._refresh_universe_bars()

        scores: list[tuple[str, float]] = []
        for symbol, df in self._daily_bars.items():
            if len(df) < 21:
                continue
            close = df["close"]
            change_pct = abs((close.iloc[-1] - close.iloc[-2]) / close.iloc[-2] * 100)
            vol_avg = df["volume"].rolling(20).mean().iloc[-1]
            vol_ratio = df["volume"].iloc[-1] / vol_avg if vol_avg > 0 else 1.0
            scores.append((symbol, change_pct * max(vol_ratio, 0.5)))

        scores.sort(key=lambda x: x[1], reverse=True)
        movers = [s for s, _ in scores[: self.max_symbols]]

        active = list(dict.fromkeys(BENCHMARKS + movers))  # dedupe, keep order
        logger.info("Active symbols this cycle: %s", active)
        return active

    # --- Snapshot for one cycle -------------------------------------------

    def get_cycle_snapshot(
        self, symbols: list[str] | None = None
    ) -> tuple[dict[str, IndicatorSet], list[Signal]]:
        """Compute indicators + signals for the active symbol set.

        Uses daily bars (robust regardless of market hours). Intraday bars can be
        layered in later for finer-grained entries.
        """
        if symbols is None:
            symbols = self.select_active_symbols()

        self._refresh_universe_bars()

        indicators: dict[str, IndicatorSet] = {}
        for symbol in symbols:
            df = self._daily_bars.get(symbol)
            if df is None:
                continue
            ind = compute_indicators(symbol, df)
            if ind is not None:
                indicators[symbol] = ind

        signals = self.signal_engine.generate(
            {s: i for s, i in indicators.items() if s not in BENCHMARKS or s in symbols}
        )
        logger.info(
            "Cycle snapshot: %d symbols with indicators, %d raw signals",
            len(indicators),
            len(signals),
        )
        return indicators, signals
