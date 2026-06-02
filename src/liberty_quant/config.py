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
