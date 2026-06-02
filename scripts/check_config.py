#!/usr/bin/env python
"""Verify LibertyQuant configuration: env vars, Anthropic API, Alpaca connectivity.

Usage:
    uv run scripts/check_config.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rich.console import Console

console = Console()


def check(label: str, fn) -> bool:
    try:
        detail = fn()
        console.print(f"  [green]✓[/] {label}" + (f" — {detail}" if detail else ""))
        return True
    except Exception as e:
        console.print(f"  [red]✗[/] {label} — {e}")
        return False


def main() -> int:
    console.rule("[bold blue]LibertyQuant Configuration Check")
    ok = True

    # 1. Settings load
    console.print("\n[bold]1. Settings (.env)[/]")

    def load_settings():
        from liberty_quant.config import get_settings

        s = get_settings()
        return f"execution={s.execution_mode}, universe={s.universe}"

    ok &= check("Settings load", load_settings)
    if not ok:
        console.print("\n[red]Fix .env first (copy .env.example to .env and fill in keys).[/]")
        return 1

    from liberty_quant.config import get_settings

    settings = get_settings()

    # 2. Anthropic
    console.print("\n[bold]2. Anthropic API[/]")

    def check_anthropic():
        import anthropic

        client = anthropic.Anthropic(api_key=settings.anthropic_api_key.get_secret_value())
        response = client.messages.create(
            model=settings.model_fast,
            max_tokens=32,
            messages=[{"role": "user", "content": "Reply with exactly: OK"}],
        )
        text = next((b.text for b in response.content if b.type == "text"), "")
        return f"{settings.model_fast} responded: {text.strip()[:20]}"

    ok &= check("Claude API call", check_anthropic)

    # 3. Alpaca
    console.print("\n[bold]3. Alpaca[/]")
    if settings.alpaca_api_key and settings.alpaca_secret_key:

        def check_alpaca_account():
            from alpaca.trading.client import TradingClient

            client = TradingClient(
                api_key=settings.alpaca_api_key,
                secret_key=settings.alpaca_secret_key.get_secret_value(),
                paper=settings.alpaca_paper,
            )
            account = client.get_account()
            mode = "PAPER" if settings.alpaca_paper else "LIVE"
            return f"{mode} account, equity=${float(account.equity):,.2f}"

        def check_alpaca_data():
            from liberty_quant.market.feed import AlpacaFeed

            feed = AlpacaFeed()
            bars = feed.get_bars(["SPY"], lookback_days=10)
            if "SPY" not in bars or bars["SPY"].empty:
                raise RuntimeError("no SPY bars returned")
            return f"SPY bars: {len(bars['SPY'])} days, last close ${bars['SPY']['close'].iloc[-1]:.2f}"

        def check_alpaca_clock():
            from alpaca.trading.client import TradingClient

            client = TradingClient(
                api_key=settings.alpaca_api_key,
                secret_key=settings.alpaca_secret_key.get_secret_value(),
                paper=settings.alpaca_paper,
            )
            clock = client.get_clock()
            status = "OPEN" if clock.is_open else "CLOSED"
            return f"market is {status}, next open: {clock.next_open}"

        ok &= check("Trading account", check_alpaca_account)
        ok &= check("Market data (IEX)", check_alpaca_data)
        ok &= check("Market clock", check_alpaca_clock)
    else:
        console.print("  [yellow]—[/] Alpaca keys not set (local_paper mode + yfinance only)")

    # 4. yfinance fallback
    console.print("\n[bold]4. yfinance (fallback feed)[/]")

    def check_yf():
        from liberty_quant.market.feed import YFinanceFeed

        feed = YFinanceFeed()
        bars = feed.get_bars(["SPY"], lookback_days=10)
        if "SPY" not in bars or bars["SPY"].empty:
            raise RuntimeError("no SPY bars returned")
        return f"SPY bars: {len(bars['SPY'])} days"

    ok &= check("yfinance download", check_yf)

    console.rule()
    if ok:
        console.print("[bold green]All checks passed. Run: uv run scripts/run_trading.py[/]")
        return 0
    console.print("[bold red]Some checks failed — fix the issues above before trading.[/]")
    return 1


if __name__ == "__main__":
    sys.exit(main())
