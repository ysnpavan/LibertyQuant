"""Terminal dashboard — rich renderables for the trading loop."""

from typing import Any

from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from .llm import get_usage_totals


def money(value: float, signed: bool = False) -> Text:
    """Colored money text: green positive, red negative."""
    style = "green" if value >= 0 else "red"
    prefix = "+" if signed and value > 0 else ""
    return Text(f"{prefix}${value:,.2f}", style=style)


def account_panel(portfolio: dict[str, Any], daily_stats: dict[str, Any]) -> Panel:
    table = Table.grid(padding=(0, 2))
    table.add_column(style="cyan")
    table.add_column(justify="right")
    table.add_row("Equity", money(portfolio.get("equity", 0)))
    table.add_row("Cash", money(portfolio.get("cash", 0)))
    table.add_row("Realized P&L (today)", money(daily_stats.get("realized_pnl", 0), signed=True))
    table.add_row("Trades today", str(daily_stats.get("trades_count", 0)))
    return Panel(table, title="💰 Account", border_style="blue")


def positions_table(positions: list[dict[str, Any]]) -> Panel:
    if not positions:
        return Panel(Text("No open positions", style="dim"), title="📊 Positions", border_style="blue")

    table = Table(expand=True)
    table.add_column("Symbol", style="bold")
    table.add_column("Qty", justify="right")
    table.add_column("Entry", justify="right")
    table.add_column("Now", justify="right")
    table.add_column("Value", justify="right")
    table.add_column("Unrealized", justify="right")

    for p in positions:
        table.add_row(
            p["symbol"],
            f"{p['qty']:.0f}",
            f"${p['avg_entry_price']:.2f}",
            f"${p['current_price']:.2f}",
            f"${p['market_value']:,.2f}",
            money(p["unrealized_pnl"], signed=True),
        )
    return Panel(table, title="📊 Positions", border_style="blue")


def agents_panel(state: dict[str, Any]) -> Panel:
    table = Table.grid(padding=(0, 2))
    table.add_column(style="cyan", width=18)
    table.add_column()

    regime_text = Text()
    regime_text.append(f"{state.get('regime', '?')} ", style="bold")
    regime_text.append(f"({state.get('regime_confidence', 0):.2f})", style="dim")
    table.add_row("🏔️  Regime", regime_text)

    strategies = state.get("active_strategies", [])
    table.add_row("📋 Strategies", ", ".join(strategies) if strategies else Text("none", style="dim"))

    table.add_row(
        "✅ Validation",
        f"{len(state.get('validated_signals', []))} approved / "
        f"{len(state.get('rejected_signals', []))} rejected",
    )
    table.add_row(
        "🛡️  Risk",
        f"{len(state.get('approved_trades', []))} approved / "
        f"{len(state.get('risk_rejected', []))} denied",
    )

    if state.get("kill_switch"):
        table.add_row("", Text("🚨 KILL SWITCH ENGAGED", style="bold red"))

    reasoning = state.get("regime_reasoning", "")
    if reasoning:
        table.add_row("💭 Reasoning", Text(reasoning[:200], style="dim italic"))

    return Panel(table, title="🤖 Agent Pipeline", border_style="magenta")


def exits_panel(exit_results: list[dict[str, Any]]) -> Panel | None:
    if not exit_results:
        return None
    table = Table(expand=True)
    table.add_column("Symbol", style="bold")
    table.add_column("Reason")
    table.add_column("Exit", justify="right")
    table.add_column("P&L", justify="right")
    for e in exit_results:
        table.add_row(
            e["symbol"],
            e["reason"].replace("_", " "),
            f"${e['exit_price']:.2f}",
            money(e.get("pnl", 0), signed=True),
        )
    return Panel(table, title="🚪 Exits This Cycle", border_style="yellow")


def usage_footer() -> Text:
    usage = get_usage_totals()
    # Rough cost estimate: blended Opus/Sonnet pricing
    est_cost = (
        usage["input_tokens"] * 4 / 1_000_000
        + usage["output_tokens"] * 20 / 1_000_000
        + usage["cache_read_input_tokens"] * 0.4 / 1_000_000
    )
    return Text(
        f"Tokens: {usage['input_tokens']:,} in / {usage['output_tokens']:,} out / "
        f"{usage['cache_read_input_tokens']:,} cached  •  est. cost ${est_cost:.2f}",
        style="dim",
    )


def render_cycle(
    console: Console,
    *,
    state: dict[str, Any],
    portfolio: dict[str, Any],
    daily_stats: dict[str, Any],
    exit_results: list[dict[str, Any]],
) -> None:
    """Render the full post-cycle dashboard."""
    panels: list[Any] = [
        account_panel(portfolio, daily_stats),
        positions_table(portfolio.get("positions", [])),
        agents_panel(state),
    ]
    exits = exits_panel(exit_results)
    if exits:
        panels.append(exits)
    panels.append(usage_footer())

    console.print(
        Panel(
            Group(*panels),
            title=f"[bold]LibertyQuant — Cycle {state.get('cycle_id', '?')}[/]",
            border_style="bright_blue",
        )
    )
