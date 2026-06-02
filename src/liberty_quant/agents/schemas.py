"""Pydantic schemas for structured agent outputs.

Every Claude agent call is forced into one of these schemas via
``client.messages.parse`` — there is no free-text JSON parsing anywhere.
"""

from typing import Literal

from pydantic import BaseModel, Field

StrategyName = Literal["momentum", "mean_reversion", "breakout", "trend_following"]
RegimeName = Literal["trending_up", "trending_down", "ranging", "volatile"]


class RegimeAssessment(BaseModel):
    """Output of the Market Regime Agent."""

    regime: RegimeName
    confidence: float = Field(description="0.0 to 1.0 — how certain the classification is")
    reasoning: str = Field(description="Brief explanation grounded in the provided indicators")
    key_factors: list[str] = Field(description="The 2-4 indicator readings that drove the call")


class StrategySelection(BaseModel):
    """Output of the Strategy Selection Agent."""

    active_strategies: list[StrategyName] = Field(
        description="Strategies to activate this cycle (1-2 is usually better than all 4)"
    )
    reasoning: str


class SignalVerdict(BaseModel):
    """Per-signal verdict from the Signal Validation Agent."""

    symbol: str
    strategy: StrategyName
    signal_type: Literal["buy", "sell"]
    verdict: Literal["approve", "reject"]
    adjusted_confidence: float = Field(description="0.0 to 1.0 after validation")
    reasoning: str


class SignalValidationResult(BaseModel):
    """Output of the Signal Validation Agent."""

    verdicts: list[SignalVerdict]
    overall_assessment: str


class TradeRiskDecision(BaseModel):
    """Per-trade decision from the Risk & Compliance Agent."""

    symbol: str
    signal_type: Literal["buy", "sell"]
    approved: bool
    reasoning: str


class RiskAssessment(BaseModel):
    """Output of the Risk & Compliance Agent."""

    decisions: list[TradeRiskDecision]
    kill_switch: bool = Field(
        description="True ONLY if trading should halt entirely for the day (e.g. daily loss limit breached)"
    )
    warnings: list[str]
    reasoning: str


MistakeCategory = Literal[
    "regime_mismatch",     # traded against the actual market regime
    "strategy_mismatch",   # wrong strategy for the conditions
    "poor_timing",         # right idea, bad entry timing
    "bad_entry",           # entry contradicted by indicators
    "wrong_sizing",        # position too large/small for the setup
    "overtrading",         # too many trades / correlated positions
    "bad_exit",            # exit logic failed (stop too tight/wide, gave back gains)
    "no_mistake",          # sound decision, unlucky outcome
]


class TradePostMortem(BaseModel):
    """Output of the trade post-mortem (learning loop)."""

    is_mistake: bool = Field(
        description="True if a decision error was made; False if the loss was just variance on a sound decision"
    )
    category: MistakeCategory
    severity: Literal["low", "medium", "high", "critical"]
    lesson: str = Field(
        description="One actionable sentence future agents should follow, e.g. 'Do not take mean-reversion BUYs when the regime is trending_down with confidence > 0.7'"
    )
    reasoning: str
