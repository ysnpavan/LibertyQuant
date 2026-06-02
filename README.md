# LibertyQuant 🦅

**Agentic AI trading system for US markets** — a pipeline of Claude-powered agents (Market Regime → Strategy Selection → Signal Validation → Risk & Compliance) that observes US equities/ETFs, generates technical signals, and paper-trades through Alpaca.

Inspired by [RakshaQuant](https://github.com/HimanshuMohanty-Git24/RakshaQuant) (Indian NSE), rebuilt for US markets with Claude as the decision engine.

> ⚠️ **Paper trading only.** This is an educational/research system. Nothing here is financial advice. Do not point it at real money.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│ MARKET LAYER                                                     │
│  Universe (S&P100 + NDX + ETFs) → Discovery (top movers)        │
│  → Alpaca/yfinance bars → Indicators (EMA/RSI/MACD/BB/ATR/ADX)  │
│  → Signal Engine (crossovers, RSI extremes, breakouts)          │
└────────────────────────────┬────────────────────────────────────┘
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│ AGENT LAYER (LangGraph + Claude)          ◄── 🧠 MEMORY          │
│                                               (lessons from      │
│  🏔️ Market Regime Agent (Sonnet 4.6)          past mistakes      │
│           ▼                                    injected into     │
│  📋 Strategy Selection Agent (Sonnet 4.6)      every cycle)      │
│           ▼                                                      │
│  ✅ Signal Validation Agent (Opus 4.8)                            │
│           ▼                                                      │
│  🛡️ Risk & Compliance Agent (Opus 4.8) + deterministic limits   │
└────────────────────────────┬────────────────────────────────────┘
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│ EXECUTION LAYER                                                  │
│  Position sizing → Alpaca paper / local paper → SQLite journal  │
│                                                                  │
│  🚪 EXIT MANAGER (every cycle, before new entries):              │
│     stop-loss · take-profit · trailing stop · EOD flatten       │
└────────────────────────────┬────────────────────────────────────┘
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│ LEARNING LOOP                              📱 NOTIFICATIONS      │
│  Closed trade → Opus post-mortem →            Telegram (default) │
│  mistake classification → lesson →            or WhatsApp:       │
│  injected into future agent cycles            fills, exits, EOD  │
└──────────────────────────────────────────────────────────────────┘
```

**Key design choices**

- **Structured outputs everywhere** — every agent response is parsed into a Pydantic schema via the Anthropic SDK's `messages.parse()`. No JSON string parsing, no malformed responses.
- **Model routing** — fast/cheap calls (regime, strategy) run on Sonnet 4.6 at low effort; high-stakes calls (validation, risk, trade post-mortems) run on Opus 4.8 with adaptive thinking.
- **Prompt caching** — agent system prompts are frozen and marked `cache_control: ephemeral`, so the trading loop reuses cached prefixes.
- **Fail-safe by default** — if any LLM call fails, signals are *rejected*, never approved. Deterministic hard limits (daily loss, trade count, position caps) run before the risk agent and cannot be overridden by it.
- **Exits before entries** — every cycle checks stops/targets/trailing stops on open positions *before* considering new trades. Positions are flattened before the close (configurable).
- **Self-improving** — losing trades get an Opus post-mortem that distinguishes *decision errors* from *bad luck*. Real mistakes become one-sentence lessons that future agent cycles must follow.

## Setup

### 1. Prerequisites

- Python 3.11+ and [uv](https://docs.astral.sh/uv/) (`brew install uv`)
- **Anthropic API key** — [console.anthropic.com](https://console.anthropic.com/settings/keys)
- **Alpaca paper account** (free) — [alpaca.markets](https://app.alpaca.markets/signup). After signup, go to the paper-trading dashboard → "API Keys" → generate.

### 2. Install

```bash
cd LibertyQuant
uv sync --all-extras
```

### 3. Configure

```bash
cp .env.example .env
# Edit .env: set ANTHROPIC_API_KEY, ALPACA_API_KEY, ALPACA_SECRET_KEY
```

### 4. Verify

```bash
uv run scripts/check_config.py
```

### 5. Run

```bash
# Demo: run the agent pipeline on synthetic data (no Alpaca needed, market can be closed)
uv run scripts/demo_cycle.py

# Single real cycle (market must be open, or add --force)
uv run scripts/run_trading.py --once --force

# Full trading loop (sleeps when market is closed, trades 9:30–16:00 ET)
uv run scripts/run_trading.py
```

### Tests

```bash
uv run pytest
```

## Configuration reference

All knobs live in `.env` (see `.env.example`):

| Variable | Default | Meaning |
|---|---|---|
| `LQ_MODEL_DEEP` | `claude-opus-4-8` | Model for validation + risk agents |
| `LQ_MODEL_FAST` | `claude-sonnet-4-6` | Model for regime + strategy agents |
| `LQ_FAST_EFFORT` | `low` | Effort for fast-loop calls (low/medium/high) |
| `LQ_EXECUTION_MODE` | `alpaca` | `alpaca` (paper account) or `local_paper` (offline sim) |
| `LQ_UNIVERSE` | `all` | `sp100`, `nasdaq100`, `etfs`, or `all` |
| `LQ_MAX_SYMBOLS` | `10` | Top movers analyzed per cycle |
| `LQ_CYCLE_SECONDS` | `60` | Seconds between cycles |
| `LQ_MAX_POSITION_PCT` | `0.10` | Max position size as fraction of equity |
| `LQ_RISK_PER_TRADE` | `0.02` | Risk budget per trade as fraction of equity |
| `LQ_MAX_DAILY_TRADES` | `20` | Hard daily trade cap |
| `LQ_DAILY_LOSS_LIMIT` | `2000` | Daily loss (USD) that triggers the kill switch |
| `LQ_MAX_OPEN_POSITIONS` | `8` | Max simultaneous positions |
| `LQ_TRAILING_STOP_PCT` | `0.03` | Trailing stop distance from the position's high |
| `LQ_EOD_FLATTEN` | `true` | Close all positions before the market close |
| `LQ_EOD_FLATTEN_MINUTES` | `10` | Minutes before close to start flattening |
| `LQ_MEMORY_ENABLED` | `true` | Learn from closed trades and inject lessons |
| `LQ_MEMORY_TOP_N` | `5` | Lessons injected per agent per cycle |
| `LQ_NOTIFICATIONS_ENABLED` | `false` | Push alerts to your phone (see below) |
| `LQ_NOTIFICATION_CHANNEL` | `telegram` | `telegram` or `whatsapp` |

## Cost expectations

With default settings (60s cycle, ~6.5 trading hours/day ≈ 390 cycles):

- Most cycles end early (low confidence / no signals) → 1–2 Sonnet calls at low effort
- Cycles with signals add 1–2 Opus calls
- Prompt caching reduces repeated input costs significantly

Expect roughly **$3–10/day** of API usage depending on market activity. Reduce by raising `LQ_CYCLE_SECONDS` or setting `LQ_MODEL_DEEP=claude-sonnet-4-6`.

## Telegram notifications setup (optional, ~3 min)

1. On Telegram, message **@BotFather** → `/newbot` → follow the prompts → copy the **bot token**
2. Message **@userinfobot** → it replies with your **chat ID**
3. Send any message to your new bot once (bots can't message you first)
4. In `.env`:
   ```
   LQ_NOTIFICATIONS_ENABLED=true
   TELEGRAM_BOT_TOKEN=123456789:ABC...
   TELEGRAM_CHAT_ID=123456789
   ```

You'll get Telegram messages for: every fill, every position close (with P&L), kill-switch events, and a daily summary at the close.

<details>
<summary>Alternative: WhatsApp via Twilio</summary>

1. Free Twilio account → Console → Messaging → Try it out → Send a WhatsApp message
2. Join the sandbox from your phone (send the join code to +1 415 523 8886)
3. In `.env`: `LQ_NOTIFICATION_CHANNEL=whatsapp` plus the four `TWILIO_*`/`WHATSAPP_TO` values from `.env.example`
</details>

## Roadmap (Phase 3)

- 📈 **Options trading** — Alpaca options paper trading, chain selection by delta/DTE
- 🔁 **Backtesting harness** — replay historical days through the full agent pipeline
- 🌐 **Web dashboard** — live positions/P&L/agent reasoning in the browser

## Disclaimer

This software is for educational purposes. Markets involve risk of loss. The authors accept no liability for trading losses. Always paper trade first; never deploy capital you cannot afford to lose.
