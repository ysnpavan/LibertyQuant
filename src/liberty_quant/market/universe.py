"""Tradeable universe definitions for US markets.

Mirrors RakshaQuant's NIFTY-50 + midcap universe, but for US large caps and ETFs.
The discovery step (manager.select_active_symbols) narrows this universe to the
most interesting symbols each cycle based on movement and volume.
"""

# S&P 100 — large, liquid US names (subset; the most actively traded)
SP100 = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "BRK.B", "JPM", "V",
    "UNH", "XOM", "LLY", "JNJ", "PG", "MA", "AVGO", "HD", "CVX", "MRK",
    "ABBV", "COST", "PEP", "ADBE", "KO", "WMT", "CRM", "BAC", "TMO", "MCD",
    "CSCO", "ACN", "ABT", "NFLX", "AMD", "LIN", "DHR", "ORCL", "DIS", "WFC",
    "TXN", "PM", "VZ", "INTU", "CAT", "NEE", "IBM", "AMGN", "GE", "QCOM",
    "HON", "UNP", "BA", "PFE", "GS", "MS", "RTX", "LOW", "INTC", "UPS",
    "SPGI", "BLK", "T", "ELV", "BKNG", "PLD", "AXP", "DE", "MDT", "GILD",
    "SBUX", "LMT", "TJX", "SYK", "ADP", "MDLZ", "MMC", "VRTX", "ISRG", "CVS",
    "CI", "ZTS", "REGN", "SO", "MO", "PGR", "DUK", "SCHW", "BSX", "CB",
    "C", "FI", "EOG", "BMY", "EQIX", "AON", "ITW", "APD", "CSX", "CL",
]

# Nasdaq-100 additions not already in S&P 100 (high-beta tech / growth)
NASDAQ100_EXTRA = [
    "PLTR", "MU", "AMAT", "LRCX", "PANW", "SNPS", "CDNS", "KLAC", "MELI", "CRWD",
    "MAR", "CTAS", "ORLY", "MRVL", "DASH", "FTNT", "ADSK", "ROP", "WDAY", "TTD",
    "NXPI", "PCAR", "CPRT", "PAYX", "ROST", "ODFL", "FAST", "MNST", "ABNB", "COIN",
]

# Highly liquid ETFs — index, sector, and volatility
ETFS = [
    "SPY",   # S&P 500
    "QQQ",   # Nasdaq-100
    "IWM",   # Russell 2000
    "DIA",   # Dow Jones
    "XLK",   # Technology
    "XLF",   # Financials
    "XLE",   # Energy
    "XLV",   # Healthcare
    "XLI",   # Industrials
    "XLP",   # Consumer staples
    "XLY",   # Consumer discretionary
    "XLU",   # Utilities
    "XLB",   # Materials
    "XLRE",  # Real estate
    "SMH",   # Semiconductors
    "GLD",   # Gold
    "TLT",   # 20+ yr treasuries
]

# Benchmark symbols always fetched for regime context (even if not traded)
BENCHMARKS = ["SPY", "QQQ", "IWM"]


def get_universe(preset: str) -> list[str]:
    """Return the symbol universe for a preset name."""
    presets = {
        "sp100": SP100,
        "nasdaq100": sorted(set(SP100[:50] + NASDAQ100_EXTRA)),
        "etfs": ETFS,
        "all": sorted(set(SP100 + NASDAQ100_EXTRA + ETFS)),
    }
    if preset not in presets:
        raise ValueError(f"Unknown universe preset: {preset!r} (use {list(presets)})")
    return presets[preset]
