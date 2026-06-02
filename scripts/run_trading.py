#!/usr/bin/env python
"""LibertyQuant main trading loop.

Usage:
    uv run scripts/run_trading.py              # trade during market hours
    uv run scripts/run_trading.py --once       # run a single cycle and exit
    uv run scripts/run_trading.py --force      # ignore market-hours check (for testing)

Each cycle:
    Exits -> Observe -> Decide (agents) -> Execute -> Learn -> Journal -> Notify
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

from liberty_quant.agents.graph import create_trading_graph, run_trading_cycle
from liberty_quant.config import get_settings
from liberty_quant.dashboard import render_cycle
from liberty_quant.execution.alpaca_broker import create_broker
from liberty_quant.execution.exit_manager import ExitManager
from liberty_quant.execution.journal import TradeJournal
from liberty_quant.llm import get_usage_totals
from liberty_quant.market.manager import MarketDataManager
from liberty_quant.market.sizing import size_position
from liberty_quant.memory.learning import TradeLearner, get_lessons_for_agents
from liberty_quant.memory.store import MemoryStore
from liberty_quant.notifications.notifier import create_notifier, notification_event
from liberty_quant.utils.market_hours import (
    is_market_open,
    minutes_to_close,
    seconds_to_next_open_estimate,
)

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
    for noisy in ("httpx", "urllib3", "yfinance", "peewee", "anthropic._base_client"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


class TradingSystem:
    """Owns all components and runs the cycle loop."""

    def __init__(self) -> None:
        self.settings = get_settings()
        self.broker = create_broker()
        self.manager = MarketDataManager()
        self.journal = TradeJournal()
        self.exit_manager = ExitManager()
        self.memory = MemoryStore()
        self.learner = TradeLearner(self.memory)
        self.notifier = create_notifier()
        self.graph = create_trading_graph()

    def shutdown(self) -> None:
        self.journal.close()
        self.exit_manager.close()
        self.memory.close()

    # ------------------------------------------------------------------
    # Exits
    # ------------------------------------------------------------------

    def process_exits(self, cycle_id: str) -> list[dict]:
        """Check and execute exits for managed positions. Returns exit summaries."""
        managed = self.exit_manager.all_positions()
        if not managed:
            return []

        # Reconcile with broker (handles positions closed manually in Alpaca UI)
        broker_positions = {p.symbol for p in self.broker.get_positions()}
        self.exit_manager.reconcile(broker_positions)
        managed = self.exit_manager.all_positions()
        if not managed:
            return []

        # Current prices for managed symbols
        symbols = [p["symbol"] for p in managed]
        bars = self.manager.feed.get_bars(symbols, lookback_days=5)
        prices = {
            sym: float(df["close"].iloc[-1]) for sym, df in bars.items() if not df.empty
        }

        # EOD flatten window?
        mins = minutes_to_close(self.broker)
        near_close = (
            self.settings.eod_flatten
            and mins is not None
            and mins <= self.settings.eod_flatten_minutes
        )

        exits = self.exit_manager.check_exits(prices, near_market_close=near_close)
        results = []

        for exit_order in exits:
            result = self.broker.submit_market_order(
                symbol=exit_order.symbol,
                side=exit_order.side,
                qty=exit_order.qty,
                current_price=exit_order.current_price,
            )
            self.journal.log_order(
                cycle_id,
                {"strategy": exit_order.position.get("strategy"), "confidence": None},
                result.to_dict(),
            )

            if result.status in ("filled", "submitted"):
                exit_price = result.filled_price or exit_order.current_price
                position = self.exit_manager.remove(exit_order.symbol)

                # Learn from the closed trade (records outcome + post-mortem)
                pnl = (exit_price - position["entry_price"]) * position["qty"]
                self.learner.process_closed_trade(
                    position=position,
                    exit_price=exit_price,
                    exit_reason=exit_order.reason,
                    cycle_id=cycle_id,
                )
                self.journal.record_trade_executed(realized_pnl_delta=pnl)

                notification_event(
                    self.notifier,
                    "position_closed",
                    {
                        "symbol": exit_order.symbol,
                        "qty": exit_order.qty,
                        "exit_price": exit_price,
                        "pnl": pnl,
                        "reason": exit_order.reason,
                    },
                )
                results.append(
                    {
                        "symbol": exit_order.symbol,
                        "reason": exit_order.reason,
                        "exit_price": exit_price,
                        "pnl": pnl,
                    }
                )
            else:
                logger.error(
                    "Exit order failed for %s: %s — will retry next cycle",
                    exit_order.symbol,
                    result.error,
                )
        return results

    # ------------------------------------------------------------------
    # Entries
    # ------------------------------------------------------------------

    def execute_entries(self, state: dict, indicators: dict, cycle_id: str) -> int:
        """Size and submit approved trades; register them with the exit manager."""
        approved = state.get("approved_trades", [])
        if not approved:
            return 0

        portfolio = self.broker.get_portfolio_snapshot()
        equity = portfolio["equity"]
        executed = 0

        for trade in approved:
            symbol = trade["symbol"]
            side = trade["signal_type"]
            entry = trade.get("entry_price") or indicators.get(symbol, {}).get("close")
            stop = trade.get("stop_loss") or (entry * 0.98 if entry else None)
            target = trade.get("take_profit")

            if not entry:
                logger.warning("Skipping %s — no entry price available", symbol)
                continue

            # MVP supports long entries only; SELL signals close longs via exits
            if side == "sell" and not self.exit_manager.get_position(symbol):
                logger.info("Skipping SELL %s — no long position to close (shorting not enabled)", symbol)
                continue

            sizing = size_position(
                symbol=symbol, entry_price=entry, stop_loss=stop, equity=equity
            )
            if sizing.shares <= 0:
                logger.info("Skipping %s — sizing produced 0 shares", symbol)
                continue

            result = self.broker.submit_market_order(
                symbol=symbol, side=side, qty=sizing.shares, current_price=entry
            )
            self.journal.log_order(cycle_id, trade, result.to_dict())

            if result.status in ("filled", "submitted"):
                fill_price = result.filled_price or entry
                self.journal.record_trade_executed()
                executed += 1

                if side == "buy":
                    self.exit_manager.register_entry(
                        symbol=symbol,
                        qty=sizing.shares,
                        entry_price=fill_price,
                        stop_loss=stop,
                        take_profit=target,
                        strategy=trade.get("strategy"),
                        regime=state.get("regime"),
                        confidence=trade.get("confidence"),
                        cycle_id=cycle_id,
                    )
                else:
                    # SELL closing an existing long
                    position = self.exit_manager.remove(symbol)
                    if position:
                        self.learner.process_closed_trade(
                            position=position,
                            exit_price=fill_price,
                            exit_reason="manual",
                            cycle_id=cycle_id,
                        )

                notification_event(
                    self.notifier,
                    "trade_executed",
                    {
                        "side": side,
                        "qty": sizing.shares,
                        "symbol": symbol,
                        "price": fill_price,
                        "strategy": trade.get("strategy"),
                        "confidence": trade.get("confidence"),
                    },
                )
            else:
                console.print(f"  [bold red]ORDER REJECTED[/] {side.upper()} {symbol}: {result.error}")

        return executed

    # ------------------------------------------------------------------
    # One cycle
    # ------------------------------------------------------------------

    def run_cycle(self) -> dict:
        cycle_id = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")

        # 1. Exits first — protect existing positions before adding new ones
        exit_results = self.process_exits(cycle_id)

        # 2. Observe
        indicators, signals = self.manager.get_cycle_snapshot()
        indicator_dicts = {s: ind.to_dict() for s, ind in indicators.items()}
        signal_dicts = [s.to_dict() for s in signals]

        # 3. Portfolio + stats + memory
        portfolio = self.broker.get_portfolio_snapshot()
        daily_stats = self.journal.get_daily_stats(starting_equity=portfolio["equity"])
        lessons = (
            get_lessons_for_agents(self.memory) if self.settings.memory_enabled else {}
        )
        strategy_perf = self.memory.strategy_performance() if self.settings.memory_enabled else {}

        # 4. Decide
        state = run_trading_cycle(
            self.graph,
            cycle_id=cycle_id,
            indicators=indicator_dicts,
            signals=signal_dicts,
            portfolio=portfolio,
            daily_stats=daily_stats,
            memory_lessons=lessons,
            strategy_performance=strategy_perf,
        )

        # 5. Execute entries
        self.execute_entries(dict(state), indicator_dicts, cycle_id)

        # 6. Kill switch notification
        if state.get("kill_switch"):
            notification_event(
                self.notifier,
                "kill_switch_engaged",
                {"reason": "; ".join(state.get("risk_warnings", ["risk limit breached"]))},
            )

        # 7. Journal + dashboard
        self.journal.log_cycle(dict(state))
        portfolio = self.broker.get_portfolio_snapshot()  # refresh after fills
        render_cycle(
            console,
            state=dict(state),
            portfolio=portfolio,
            daily_stats=self.journal.get_daily_stats(),
            exit_results=exit_results,
        )
        return dict(state)

    def send_daily_summary(self) -> None:
        portfolio = self.broker.get_portfolio_snapshot()
        stats = self.journal.get_daily_stats()
        notification_event(
            self.notifier,
            "daily_summary",
            {
                "equity": portfolio["equity"],
                "realized_pnl": stats.get("realized_pnl", 0.0),
                "trades": stats.get("trades_count", 0),
                "open_positions": len(portfolio["positions"]),
            },
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="LibertyQuant trading loop")
    parser.add_argument("--once", action="store_true", help="Run a single cycle and exit")
    parser.add_argument("--force", action="store_true", help="Run even when the market is closed")
    args = parser.parse_args()

    setup_logging()
    settings = get_settings()

    console.rule("[bold blue]LibertyQuant — Agentic US Trading System")
    console.print(f"Execution mode : [bold]{settings.execution_mode}[/]")
    console.print(f"Universe       : {settings.universe} (top {settings.max_symbols} movers)")
    console.print(f"Deep model     : {settings.model_deep}")
    console.print(f"Fast model     : {settings.model_fast} (effort={settings.fast_effort})")
    console.print(f"Cycle interval : {settings.cycle_seconds}s")
    console.print(f"Memory         : {'enabled' if settings.memory_enabled else 'disabled'}")
    console.print(
        f"Notifications  : "
        f"{settings.notification_channel if settings.notifications_enabled else 'console only'}"
    )
    console.rule()

    system = TradingSystem()

    portfolio = system.broker.get_portfolio_snapshot()
    console.print(
        f"Account: equity=[bold]${portfolio['equity']:,.2f}[/] "
        f"cash=${portfolio['cash']:,.2f} positions={len(portfolio['positions'])} "
        f"managed={len(system.exit_manager.all_positions())}\n"
    )

    market_was_open = False
    try:
        while True:
            market_open = args.force or is_market_open(system.broker)

            if market_open:
                market_was_open = True
                kill_switch = False
                try:
                    state = system.run_cycle()
                    kill_switch = state.get("kill_switch", False)
                except KeyboardInterrupt:
                    raise
                except Exception:
                    logger.exception("Cycle failed — continuing to next cycle")

                if args.once:
                    break
                if kill_switch:
                    console.print("[bold red]Kill switch engaged — halting for today.[/]")
                    system.send_daily_summary()
                    break
                time.sleep(settings.cycle_seconds)
            else:
                if market_was_open:
                    # Market just closed — send the daily wrap-up
                    system.send_daily_summary()
                    market_was_open = False
                if args.once:
                    console.print("[yellow]Market is closed. Use --force to run anyway.[/]")
                    break
                wait = min(seconds_to_next_open_estimate(), 30 * 60)
                console.print(f"[dim]Market closed. Sleeping {wait // 60} minutes (Ctrl-C to exit)...[/]")
                time.sleep(wait)
    except KeyboardInterrupt:
        console.print("\n[yellow]Stopped by user.[/]")
    finally:
        system.shutdown()

    usage = get_usage_totals()
    console.print(
        f"\nSession token usage: in={usage['input_tokens']:,} out={usage['output_tokens']:,} "
        f"cache_read={usage['cache_read_input_tokens']:,}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
