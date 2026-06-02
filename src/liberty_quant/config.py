"""Application settings loaded from environment / .env via pydantic-settings."""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """LibertyQuant configuration.

    All trading-specific knobs use the LQ_ prefix; provider credentials use
    their conventional names (ANTHROPIC_API_KEY, ALPACA_API_KEY, ...).
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ===========================================
    # Anthropic / Claude
    # ===========================================
    anthropic_api_key: SecretStr = Field(..., description="Anthropic API key")
    model_deep: str = Field(
        default="claude-opus-4-8",
        alias="LQ_MODEL_DEEP",
        description="Model for deep-reasoning agents (signal validation, risk & compliance)",
    )
    model_fast: str = Field(
        default="claude-sonnet-4-6",
        alias="LQ_MODEL_FAST",
        description="Model for fast-loop agents (market regime, strategy selection)",
    )
    fast_effort: Literal["low", "medium", "high"] = Field(
        default="low",
        alias="LQ_FAST_EFFORT",
        description="Effort level for fast-loop agent calls",
    )

    # ===========================================
    # Alpaca
    # ===========================================
    alpaca_api_key: str | None = Field(default=None, description="Alpaca API key ID")
    alpaca_secret_key: SecretStr | None = Field(default=None, description="Alpaca API secret")
    alpaca_paper: bool = Field(default=True, description="Use Alpaca paper trading endpoint")

    # ===========================================
    # Trading configuration
    # ===========================================
    execution_mode: Literal["local_paper", "alpaca"] = Field(
        default="alpaca",
        alias="LQ_EXECUTION_MODE",
        description="local_paper = simulate fills locally; alpaca = route to Alpaca paper account",
    )
    universe: Literal["sp100", "nasdaq100", "etfs", "all"] = Field(
        default="all", alias="LQ_UNIVERSE"
    )
    max_symbols: int = Field(default=10, alias="LQ_MAX_SYMBOLS", ge=1, le=50)
    cycle_seconds: int = Field(default=60, alias="LQ_CYCLE_SECONDS", ge=15)

    # ===========================================
    # Risk limits (USD)
    # ===========================================
    starting_cash: float = Field(default=100_000.0, alias="LQ_STARTING_CASH", gt=0)
    max_position_pct: float = Field(default=0.10, alias="LQ_MAX_POSITION_PCT", gt=0, le=1)
    risk_per_trade: float = Field(default=0.02, alias="LQ_RISK_PER_TRADE", gt=0, le=1)
    max_daily_trades: int = Field(default=20, alias="LQ_MAX_DAILY_TRADES", ge=1)
    daily_loss_limit: float = Field(default=2_000.0, alias="LQ_DAILY_LOSS_LIMIT", gt=0)
    max_open_positions: int = Field(default=8, alias="LQ_MAX_OPEN_POSITIONS", ge=1)

    # ===========================================
    # Exit management
    # ===========================================
    trailing_stop_pct: float = Field(
        default=0.03,
        alias="LQ_TRAILING_STOP_PCT",
        gt=0,
        le=0.5,
        description="Trailing stop distance as fraction of the position's high-water price",
    )
    eod_flatten: bool = Field(
        default=True,
        alias="LQ_EOD_FLATTEN",
        description="Close all positions near the end of the trading day",
    )
    eod_flatten_minutes: int = Field(
        default=10,
        alias="LQ_EOD_FLATTEN_MINUTES",
        ge=1,
        le=60,
        description="Minutes before market close to flatten positions",
    )

    # ===========================================
    # Memory / learning
    # ===========================================
    memory_enabled: bool = Field(default=True, alias="LQ_MEMORY_ENABLED")
    memory_top_n: int = Field(
        default=5,
        alias="LQ_MEMORY_TOP_N",
        ge=1,
        le=20,
        description="Number of lessons injected into agent context per cycle",
    )

    # ===========================================
    # Notifications
    # ===========================================
    notifications_enabled: bool = Field(default=False, alias="LQ_NOTIFICATIONS_ENABLED")
    notification_channel: Literal["telegram", "whatsapp"] = Field(
        default="telegram",
        alias="LQ_NOTIFICATION_CHANNEL",
        description="Which channel to send alerts through",
    )

    # --- Telegram (default channel) ---
    telegram_bot_token: SecretStr | None = Field(
        default=None, description="Telegram bot token from @BotFather"
    )
    telegram_chat_id: str | None = Field(
        default=None, description="Your Telegram chat ID (from @userinfobot)"
    )

    # --- WhatsApp via Twilio (alternative channel) ---
    twilio_account_sid: str | None = Field(default=None, description="Twilio account SID")
    twilio_auth_token: SecretStr | None = Field(default=None, description="Twilio auth token")
    twilio_whatsapp_from: str | None = Field(
        default=None,
        description="Twilio WhatsApp sender, e.g. 'whatsapp:+14155238886' (sandbox number)",
    )
    whatsapp_to: str | None = Field(
        default=None,
        description="Your WhatsApp number, e.g. 'whatsapp:+91XXXXXXXXXX'",
    )

    # ===========================================
    # Storage / logging
    # ===========================================
    db_path: str = Field(default="data/liberty_quant.db", alias="LQ_DB_PATH")
    log_level: str = Field(default="INFO", alias="LQ_LOG_LEVEL")

    @model_validator(mode="after")
    def _validate(self) -> "Settings":
        if self.execution_mode == "alpaca" and not (self.alpaca_api_key and self.alpaca_secret_key):
            raise ValueError(
                "execution_mode=alpaca requires ALPACA_API_KEY and ALPACA_SECRET_KEY "
                "(or set LQ_EXECUTION_MODE=local_paper)"
            )
        if self.risk_per_trade > self.max_position_pct:
            raise ValueError(
                f"LQ_RISK_PER_TRADE ({self.risk_per_trade}) must not exceed "
                f"LQ_MAX_POSITION_PCT ({self.max_position_pct})"
            )
        if self.notifications_enabled:
            if self.notification_channel == "telegram":
                required = (
                    ("TELEGRAM_BOT_TOKEN", self.telegram_bot_token),
                    ("TELEGRAM_CHAT_ID", self.telegram_chat_id),
                )
            else:  # whatsapp
                required = (
                    ("TWILIO_ACCOUNT_SID", self.twilio_account_sid),
                    ("TWILIO_AUTH_TOKEN", self.twilio_auth_token),
                    ("TWILIO_WHATSAPP_FROM", self.twilio_whatsapp_from),
                    ("WHATSAPP_TO", self.whatsapp_to),
                )
            missing = [name for name, value in required if not value]
            if missing:
                raise ValueError(
                    f"LQ_NOTIFICATIONS_ENABLED=true with channel={self.notification_channel} "
                    f"requires: {', '.join(missing)}"
                )
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        return self


@lru_cache
def get_settings() -> Settings:
    """Cached settings singleton."""
    return Settings()


def reload_settings() -> Settings:
    """Reload settings, clearing the cache (used by tests)."""
    get_settings.cache_clear()
    return get_settings()
