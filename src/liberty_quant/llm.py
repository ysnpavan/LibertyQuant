"""Claude LLM layer.

Centralizes every Anthropic API call made by the agent pipeline:

- Model routing: deep-reasoning agents (risk, validation) run on Opus 4.8 with
  adaptive thinking; fast-loop agents (regime, strategy) run on Sonnet 4.6 with
  a configurable effort level.
- Structured outputs: every agent response is parsed into a Pydantic model via
  ``client.messages.parse`` — agents cannot return malformed JSON.
- Prompt caching: agent system prompts are frozen strings marked with
  ``cache_control: ephemeral``. In the trading loop the same prompts repeat
  every cycle, so reads come from cache at ~0.1x input price.
"""

import logging
from typing import TypeVar

import anthropic
from pydantic import BaseModel

from .config import get_settings

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class LLMOutputError(RuntimeError):
    """Raised when the model response cannot be parsed into the expected schema."""

# Token accounting across the session (useful for cost visibility in the dashboard)
_usage_totals = {
    "input_tokens": 0,
    "output_tokens": 0,
    "cache_read_input_tokens": 0,
    "cache_creation_input_tokens": 0,
}


def get_usage_totals() -> dict[str, int]:
    """Cumulative token usage for this process."""
    return dict(_usage_totals)


class AgentLLM:
    """Thin wrapper over the Anthropic SDK for agent calls."""

    def __init__(self) -> None:
        settings = get_settings()
        self._client = anthropic.Anthropic(
            api_key=settings.anthropic_api_key.get_secret_value(),
            max_retries=3,
        )
        self._model_deep = settings.model_deep
        self._model_fast = settings.model_fast
        self._fast_effort = settings.fast_effort

    def _record_usage(self, usage) -> None:
        _usage_totals["input_tokens"] += usage.input_tokens or 0
        _usage_totals["output_tokens"] += usage.output_tokens or 0
        _usage_totals["cache_read_input_tokens"] += getattr(usage, "cache_read_input_tokens", 0) or 0
        _usage_totals["cache_creation_input_tokens"] += (
            getattr(usage, "cache_creation_input_tokens", 0) or 0
        )

    def structured_call(
        self,
        *,
        system: str,
        user_content: str,
        output_model: type[T],
        deep: bool = False,
        max_tokens: int = 4096,
    ) -> T:
        """Call Claude and parse the response into ``output_model``.

        Args:
            system: Frozen system prompt (cached across calls — never interpolate
                timestamps or per-cycle data into it; that belongs in user_content).
            user_content: Per-cycle context (market data, signals, portfolio).
            output_model: Pydantic model the response must conform to.
            deep: True routes to the deep model (Opus 4.8, adaptive thinking);
                False routes to the fast model (Sonnet 4.6, configured effort).
            max_tokens: Response token cap.

        Returns:
            Validated instance of ``output_model``.

        Raises:
            anthropic.APIError: On unrecoverable API failure.
            LLMOutputError: If the response cannot be parsed into the schema.
            (Callers must provide deterministic fallbacks — the trading loop
            never dies because an LLM call failed.)
        """
        model = self._model_deep if deep else self._model_fast

        kwargs: dict = {
            "model": model,
            "max_tokens": max_tokens,
            # System prompt is the stable prefix — cache it. Volatile content
            # (market data) goes in the user message, after the breakpoint.
            "system": [
                {
                    "type": "text",
                    "text": system,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            "messages": [{"role": "user", "content": user_content}],
            "output_format": output_model,
        }

        if deep:
            # Opus 4.8: adaptive thinking; sampling params are not supported.
            kwargs["thinking"] = {"type": "adaptive"}
        else:
            # Sonnet 4.6 fast loop: skip thinking, run at configured effort.
            kwargs["thinking"] = {"type": "disabled"}
            kwargs["output_config"] = {"effort": self._fast_effort}

        response = self._client.messages.parse(**kwargs)
        self._record_usage(response.usage)

        cache_read = getattr(response.usage, "cache_read_input_tokens", 0) or 0
        logger.debug(
            "LLM call model=%s output=%s in=%d out=%d cache_read=%d",
            model,
            output_model.__name__,
            response.usage.input_tokens,
            response.usage.output_tokens,
            cache_read,
        )

        parsed = response.parsed_output
        if parsed is None:
            raise LLMOutputError(
                f"Model returned unparseable output for {output_model.__name__}"
            )
        return parsed


_llm: AgentLLM | None = None


def get_llm() -> AgentLLM:
    """Process-wide AgentLLM singleton."""
    global _llm
    if _llm is None:
        _llm = AgentLLM()
    return _llm
