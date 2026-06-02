# LibertyQuant — Claude Code instructions

## What this is

Agentic US-market paper-trading system: market data → technical signals → LangGraph pipeline of Claude agents (regime → strategy → validation → risk) → Alpaca paper execution → SQLite journal.

## Commands

```bash
uv sync --all-extras              # install
uv run pytest                     # tests (no network/API needed — LLM is mocked)
uv run scripts/check_config.py    # verify .env, Anthropic, Alpaca connectivity
uv run scripts/demo_cycle.py      # agent pipeline on synthetic data (needs ANTHROPIC_API_KEY)
uv run scripts/run_trading.py --once --force   # one real cycle, ignore market hours
uv run scripts/run_trading.py     # full loop
uv run ruff check src tests       # lint
```

## Architecture rules

- **All Anthropic API calls go through `src/liberty_quant/llm.py`** (`AgentLLM.structured_call`). Never call the SDK directly from agent nodes.
- **Agent outputs are Pydantic schemas** (`agents/schemas.py`) parsed via `messages.parse()`. Never parse JSON from text.
- **System prompts are frozen** (`agents/prompts.py`) — no f-strings, no timestamps, no per-cycle data. Volatile context goes in the user message. This preserves prompt caching.
- **Fail-safe principle**: any LLM failure must result in *fewer* trades, never more. Validation failure → reject all. Risk failure → deny all. Deterministic hard limits (`_hard_risk_checks`) run before the risk agent and are non-negotiable.
- **Model routing**: `deep=True` → Opus (validation, risk); `deep=False` → Sonnet (regime, strategy). Configured in `.env`, defaults in `config.py`.

## Conventions

- Python 3.11+, ruff for linting, line length 100
- OHLCV DataFrames: lowercase columns (open/high/low/close/volume), timestamp index ascending
- Money is USD floats rounded to 2dp at display/serialization boundaries only
- Tests must not hit the network — mock `get_llm()` (see `tests/test_agents.py::FakeLLM`)

## Phase 2 backlog (not yet built)

Memory/learning loop, options trading, exit manager (trailing stops/EOD flatten), WhatsApp notifications, live dashboard, backtesting harness.
