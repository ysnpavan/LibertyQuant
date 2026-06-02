#!/usr/bin/env python
"""Run one agent-pipeline cycle on synthetic market data.

No Alpaca account or open market needed — only ANTHROPIC_API_KEY.
This exercises the full Claude agent pipeline end-to-end:
regime -> strategy -> validation -> risk.

Usage:
    uv run scripts/demo_cycle.py
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rich.console import Console
from rich.logging import RichHandler

console = Console()

logging.basicConfig(
    level="INFO",
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(console=console, show_path=False)],
)
logging.getLogger("httpx").setLevel(logging.WARNING)

# Synthetic indicator snapshots: a clean "trending up" tape
DEMO_INDICATORS = {
    "SPY": {
        "symbol": "SPY", "close": 612.40, "change_percent": 1.1,
        "ema_9": 608.2, "ema_21": 601.5, "sma_50": 590.3, "adx": 28.4,
        "rsi_14": 63.2, "macd": 4.1, "macd_signal": 3.2, "macd_histogram": 0.9,
        "atr_14": 5.8, "bb_upper": 618.0, "bb_middle": 602.0, "bb_lower": 586.0,
        "bb_percent": 0.83, "volume": 78_000_000, "volume_ratio": 1.3, "vwap": 610.1,
    },
    "QQQ": {
        "symbol": "QQQ", "close": 540.10, "change_percent": 1.6,
        "ema_9": 535.0, "ema_21": 527.8, "sma_50": 515.2, "adx": 31.0,
        "rsi_14": 66.8, "macd": 5.2, "macd_signal": 4.0, "macd_histogram": 1.2,
        "atr_14": 6.9, "bb_upper": 546.0, "bb_middle": 528.0, "bb_lower": 510.0,
        "bb_percent": 0.83, "volume": 49_000_000, "volume_ratio": 1.4, "vwap": 538.2,
    },
    "IWM": {
        "symbol": "IWM", "close": 232.50, "change_percent": 0.8,
        "ema_9": 231.1, "ema_21": 229.0, "sma_50": 224.7, "adx": 22.1,
        "rsi_14": 58.9, "macd": 1.4, "macd_signal": 1.1, "macd_histogram": 0.3,
        "atr_14": 3.1, "bb_upper": 236.0, "bb_middle": 229.5, "bb_lower": 223.0,
        "bb_percent": 0.73, "volume": 31_000_000, "volume_ratio": 1.1, "vwap": 232.0,
    },
    "NVDA": {
        "symbol": "NVDA", "close": 198.20, "change_percent": 3.4,
        "ema_9": 192.5, "ema_21": 187.2, "sma_50": 178.0, "adx": 35.6,
        "rsi_14": 68.0, "macd": 3.8, "macd_signal": 2.5, "macd_histogram": 1.3,
        "atr_14": 5.5, "bb_upper": 199.5, "bb_middle": 188.0, "bb_lower": 176.5,
        "bb_percent": 0.94, "volume": 320_000_000, "volume_ratio": 2.1, "vwap": 195.8,
    },
    "AMD": {
        "symbol": "AMD", "close": 168.90, "change_percent": 2.8,
        "ema_9": 165.2, "ema_21": 161.8, "sma_50": 155.4, "adx": 29.8,
        "rsi_14": 64.5, "macd": 2.9, "macd_signal": 2.1, "macd_histogram": 0.8,
        "atr_14": 4.8, "bb_upper": 172.0, "bb_middle": 162.5, "bb_lower": 153.0,
        "bb_percent": 0.84, "volume": 95_000_000, "volume_ratio": 1.8, "vwap": 167.2,
    },
}

DEMO_SIGNALS = [
    {
        "symbol": "NVDA", "signal_type": "buy", "strategy": "momentum",
        "confidence": 0.65, "entry_price": 198.20, "stop_loss": 187.20, "take_profit": 214.70,
        "rationale": "MACD bullish cross above zero line with strong volume",
    },
    {
        "symbol": "NVDA", "signal_type": "buy", "strategy": "breakout",
        "confidence": 0.62, "entry_price": 198.20, "stop_loss": 188.00, "take_profit": 209.20,
        "rationale": "Price broke above upper Bollinger band on 2.1x volume",
    },
    {
        "symbol": "AMD", "signal_type": "buy", "strategy": "trend_following",
        "confidence": 0.58, "entry_price": 168.90, "stop_loss": 159.30, "take_profit": 183.30,
        "rationale": "EMA9 crossed above EMA21, price above EMA9",
    },
    {
        "symbol": "AMD", "signal_type": "sell", "strategy": "mean_reversion",
        "confidence": 0.45, "entry_price": 168.90, "stop_loss": 176.10, "take_profit": 162.50,
        "rationale": "RSI approaching overbought territory",
    },
]

DEMO_PORTFOLIO = {
    "cash": 100_000.0,
    "equity": 100_000.0,
    "buying_power": 100_000.0,
    "positions": [],
}

DEMO_DAILY_STATS = {"trades_count": 0, "realized_pnl": 0.0, "starting_equity": 100_000.0}


def main() -> int:
    console.rule("[bold blue]LibertyQuant — Demo Cycle (synthetic data)")
    console.print(
        "Scenario: broad trending-up tape; NVDA and AMD showing momentum/breakout signals.\n"
    )

    from liberty_quant.agents.graph import create_trading_graph, run_trading_cycle
    from liberty_quant.llm import get_usage_totals

    graph = create_trading_graph()
    state = run_trading_cycle(
        graph,
        cycle_id="DEMO",
        indicators=DEMO_INDICATORS,
        signals=DEMO_SIGNALS,
        portfolio=DEMO_PORTFOLIO,
        daily_stats=DEMO_DAILY_STATS,
    )

    console.rule("[bold]Results")
    console.print(f"[bold]Regime:[/] {state['regime']} ({state['regime_confidence']:.2f})")
    console.print(f"[dim]{state['regime_reasoning']}[/]\n")
    console.print(f"[bold]Active strategies:[/] {state['active_strategies']}")
    console.print(f"[dim]{state['strategy_reasoning']}[/]\n")

    console.print(f"[bold]Validated signals:[/] {len(state['validated_signals'])}")
    for s in state["validated_signals"]:
        console.print(
            f"  [green]✓[/] {s['signal_type'].upper()} {s['symbol']} [{s['strategy']}] "
            f"conf={s['confidence']:.2f}"
        )
        console.print(f"    [dim]{s.get('validation_reasoning', '')}[/]")
    for s in state["rejected_signals"]:
        console.print(
            f"  [red]✗[/] {s['signal_type'].upper()} {s['symbol']} [{s['strategy']}] — "
            f"[dim]{s.get('rejection_reason', '')}[/]"
        )

    console.print(f"\n[bold]Final approved trades:[/] {len(state['approved_trades'])}")
    for t in state["approved_trades"]:
        console.print(
            f"  [bold green]APPROVED[/] {t['signal_type'].upper()} {t['symbol']} "
            f"@ ${t['entry_price']:.2f} (stop ${t['stop_loss']:.2f}, target ${t['take_profit']:.2f})"
        )
        console.print(f"    [dim]{t.get('risk_reasoning', '')}[/]")
    for t in state["risk_rejected"]:
        console.print(
            f"  [red]DENIED[/] {t['signal_type'].upper()} {t['symbol']} — "
            f"[dim]{t.get('rejection_reason', '')}[/]"
        )

    if state["risk_warnings"]:
        console.print(f"\n[yellow]Warnings: {'; '.join(state['risk_warnings'])}[/]")

    usage = get_usage_totals()
    console.print(
        f"\n[dim]Token usage: in={usage['input_tokens']:,} out={usage['output_tokens']:,} "
        f"cache_read={usage['cache_read_input_tokens']:,} "
        f"cache_write={usage['cache_creation_input_tokens']:,}[/]"
    )
    console.rule("[bold green]Demo complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
