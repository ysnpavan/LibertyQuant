#!/usr/bin/env python
"""LibertyQuant main trading loop.

Usage:
    uv run scripts/run_trading.py              # trade during market hours
    uv run scripts/run_trading.py --once       # run a single cycle and exit
    uv run scripts/run_trading.py --force      # ignore market-hours check (for testing)

Observe -> Analyze -> Decide (agents) -> Execute -> Journal, every N seconds.
"""

import argparse
import logging
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table

from liberty_quant.agents.graph import create_trading_graph, run_trading_cycle
from liberty_quant.config import get_settings
from liberty_quant.execution.alpaca_broker import create_broker
from liberty_quant.execution.journal import TradeJournal
from liberty_quant.llm import get_usage_totals
from liberty_quant.market.manager import MarketDataManager
from liberty_quant.market.sizing import size_position
from liberty_quant.utils.market_hours import is_market_open, seconds_to_next_open_estimate

console = Console()
logger = logging.getLogger("liberty_quant")


def setup_logging() -> None:
    settings = get_settings()
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    logging.basicConfig(
        level=settings.log_level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[
            RichHandler(console=console, rich_tracebacks=True, show_path=False),
            logging.FileHandler(logs_dir / f"trading_{datetime.now().strftime('%Y%m%d')}.log"),
        ],
    )
    # Quiet noisy libraries
    for noisy in ("httpx", "urllib3", "yfinance", "peewee", "anthropic._base_client"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def execute_approved_trades(
    *, broker, journal: TradeJournal, state: dict, indicators: dict
) -> int:
    """Size and submit every approved trade. Returns the number executed."""
    approved = state.get("approved_trades", [])
    if not approved:
        return 0

    portfolio = broker.get_portfolio_snapshot()
    equity = portfolio["equity"]
    executed = 0

    for trade in approved:
        symbol = trade["symbol"]
        side = trade["signal_type"]
        entry = trade.get("entry_price") or indicators.get(symbol, {}).get("close")
        stop = trade.get("stop_loss", entry * 0.98 if entry else None)

        if not entry:
            logger.warning("Skipping %s — no entry price available", symbol)
            continue

        sizing = size_position(
            symbol=symbol, entry_price=entry, stop_loss=stop, equity=equity
        )
        if sizing.shares <= 0:
            logger.info("Skipping %s — sizing produced 0 shares (%s)", symbol, sizing.reason)
            continue

        result = broker.submit_market_order(
            symbol=symbol, side=side, qty=sizing.shares, current_price=entry
        )
        journal.log_order(state["cycle_id"], trade, result.to_dict())

        if result.status in ("filled", "submitted"):
            journal.record_trade_executed()
            executed += 1
            console.print(
                f"  [bold green]EXECUTED[/] {side.upper()} {sizing.shares} {symbol} "
                f"@ ~${entry:.2f} [{trade.get('strategy')}] conf={trade.get('confidence', 0):.2f}"
            )
        else:
            console.print(f"  [bold red]REJECTED[/] {side.upper()} {symbol}: {result.error}")

    return executed


def print_cycle_summary(state: dict, executed: int) -> None:
    table = Table(title=f"Cycle {state['cycle_id']}", show_header=False, border_style="dim")
    table.add_column("k", style="cyan")
    table.add_column("v")
    table.add_row("Regime", f"{state.get('regime')} ({state.get('regime_confidence', 0):.2f})")
    table.add_row("Strategies", ", ".join(state.get("active_strategies", [])) or "—")
    table.add_row("Raw signals", str(len(state.get("signals", []))))
    table.add_row("Validated", str(len(state.get("validated_signals", []))))
    table.add_row("Approved", str(len(state.get("approved_trades", []))))
    table.add_row("Executed", str(executed))
    if state.get("kill_switch"):
        table.add_row("KILL SWITCH", "[bold red]ENGAGED[/]")
    if state.get("risk_warnings"):
        table.add_row("Warnings", "; ".join(state["risk_warnings"]))

    usage = get_usage_totals()
    table.add_row(
        "Tokens (session)",
        f"in={usage['input_tokens']:,} out={usage['output_tokens']:,} "
        f"cached={usage['cache_read_input_tokens']:,}",
    )
    console.print(table)


def run_one_cycle(*, graph, manager: MarketDataManager, broker, journal: TradeJournal) -> dict:
    """One full observe -> decide -> execute cycle."""
    cycle_id = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")

    # 1. Observe
    indicators, signals = manager.get_cycle_snapshot()
    indicator_dicts = {s: ind.to_dict() for s, ind in indicators.items()}
    signal_dicts = [s.to_dict() for s in signals]

    # 2. Portfolio + daily stats
    portfolio = broker.get_portfolio_snapshot()
    daily_stats = journal.get_daily_stats(starting_equity=portfolio["equity"])

    # 3. Decide (agent pipeline)
    state = run_trading_cycle(
        graph,
        cycle_id=cycle_id,
        indicators=indicator_dicts,
        signals=signal_dicts,
        portfolio=portfolio,
        daily_stats=daily_stats,
    )

    # 4. Execute
    executed = execute_approved_trades(
        broker=broker, journal=journal, state=dict(state), indicators=indicator_dicts
    )

    # 5. Journal
    journal.log_cycle(dict(state))
    print_cycle_summary(dict(state), executed)
    return dict(state)


def main() -> int:
    parser = argparse.ArgumentParser(description="LibertyQuant trading loop")
    parser.add_argument("--once", action="store_true", help="Run a single cycle and exit")
    parser.add_argument(
        "--force", action="store_true", help="Run even when the market is closed"
    )
    args = parser.parse_args()

    setup_logging()
    settings = get_settings()

    console.rule("[bold blue]LibertyQuant — Agentic US Trading System")
    console.print(f"Execution mode : [bold]{settings.execution_mode}[/]")
    console.print(f"Universe       : {settings.universe} (top {settings.max_symbols} movers)")
    console.print(f"Deep model     : {settings.model_deep}")
    console.print(f"Fast model     : {settings.model_fast} (effort={settings.fast_effort})")
    console.print(f"Cycle interval : {settings.cycle_seconds}s")
    console.rule()

    broker = create_broker()
    manager = MarketDataManager()
    journal = TradeJournal()
    graph = create_trading_graph()

    portfolio = broker.get_portfolio_snapshot()
    console.print(
        f"Account: equity=[bold]${portfolio['equity']:,.2f}[/] "
        f"cash=${portfolio['cash']:,.2f} positions={len(portfolio['positions'])}\n"
    )

    try:
        while True:
            kill_switch = False

            if args.force or is_market_open(broker):
                try:
                    state = run_one_cycle(
                        graph=graph, manager=manager, broker=broker, journal=journal
                    )
                    kill_switch = state.get("kill_switch", False)
                except KeyboardInterrupt:
                    raise
                except Exception:
                    logger.exception("Cycle failed — continuing to next cycle")

                if args.once:
                    break
                if kill_switch:
                    console.print(
                        "[bold red]Kill switch engaged — halting trading for today.[/]"
                    )
                    break
                time.sleep(settings.cycle_seconds)
            else:
                if args.once:
                    console.print("[yellow]Market is closed. Use --force to run anyway.[/]")
                    break
                wait = min(seconds_to_next_open_estimate(), 30 * 60)
                console.print(
                    f"[dim]Market closed. Sleeping {wait // 60} minutes "
                    f"(Ctrl-C to exit)...[/]"
                )
                time.sleep(wait)
    except KeyboardInterrupt:
        console.print("\n[yellow]Stopped by user.[/]")
    finally:
        journal.close()

    usage = get_usage_totals()
    console.print(
        f"\nSession token usage: in={usage['input_tokens']:,} out={usage['output_tokens']:,} "
        f"cache_read={usage['cache_read_input_tokens']:,}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
