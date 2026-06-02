"""Trade learning loop: closed trade -> Opus post-mortem -> lesson.

Every closed trade gets recorded. Losing trades (and surprising winners) get a
post-mortem from the deep model, which classifies what went wrong and writes an
actionable lesson. Lessons are injected into future agent cycles.
"""

import json
import logging
from typing import Any

from ..agents.prompts import TRADE_POSTMORTEM_PROMPT
from ..agents.schemas import TradePostMortem
from ..llm import get_llm
from .store import MemoryStore

logger = logging.getLogger(__name__)

# Lesson categories routed to each agent (which lessons each agent sees)
AGENT_LESSON_CATEGORIES = {
    "regime": ["regime_mismatch"],
    "strategy": ["regime_mismatch", "overtrading", "strategy_mismatch"],
    "validation": ["poor_timing", "ignored_indicator", "bad_entry"],
    "risk": ["wrong_sizing", "overtrading", "bad_exit"],
}


class TradeLearner:
    """Processes closed trades into recorded outcomes and lessons."""

    def __init__(self, store: MemoryStore) -> None:
        self.store = store

    def process_closed_trade(
        self,
        *,
        position: dict[str, Any],
        exit_price: float,
        exit_reason: str,
        cycle_id: str | None = None,
    ) -> int:
        """Record the outcome of a closed position and learn from it.

        Args:
            position: managed position dict (from ExitManager)
            exit_price: fill price of the closing order
            exit_reason: stop_loss / take_profit / trailing_stop / eod_flatten / manual
            cycle_id: current cycle

        Returns:
            The closed_trades row id.
        """
        trade_id = self.store.record_closed_trade(
            symbol=position["symbol"],
            strategy=position.get("strategy"),
            side="long" if position.get("side", "long") == "long" else "short",
            qty=position["qty"],
            entry_price=position["entry_price"],
            exit_price=exit_price,
            exit_reason=exit_reason,
            regime_at_entry=position.get("regime_at_entry"),
            confidence=position.get("confidence"),
            opened_at=position.get("opened_at"),
            cycle_id=cycle_id,
        )

        trade = self.store.get_closed_trade(trade_id)
        if trade and self._deserves_postmortem(trade):
            try:
                self._run_postmortem(trade)
            except Exception as e:
                # Learning is best-effort — never let it break the trading loop
                logger.error("[memory] post-mortem failed for trade %d: %s", trade_id, e)

        return trade_id

    @staticmethod
    def _deserves_postmortem(trade: dict[str, Any]) -> bool:
        """Post-mortem losing trades and stop-outs; skip routine small winners."""
        if trade["pnl"] < 0:
            return True
        if trade["exit_reason"] in ("stop_loss", "eod_flatten"):
            return True
        return False

    def _run_postmortem(self, trade: dict[str, Any]) -> None:
        """Ask the deep model what went wrong and store the lesson."""
        logger.info("[memory] running post-mortem on %s trade (pnl=$%.2f)...", trade["symbol"], trade["pnl"])

        context = (
            "## Closed trade\n"
            f"{json.dumps(trade, indent=1, sort_keys=True, default=str)}\n\n"
            "## Recent strategy performance (all regimes)\n"
            f"{json.dumps(self.store.strategy_performance(), indent=1, sort_keys=True)}\n\n"
            "Analyze this trade and decide whether a mistake was made."
        )

        result = get_llm().structured_call(
            system=TRADE_POSTMORTEM_PROMPT,
            user_content=context,
            output_model=TradePostMortem,
            deep=True,
        )

        if result.is_mistake and result.category != "no_mistake":
            self.store.add_lesson(
                category=result.category,
                severity=result.severity,
                lesson=result.lesson,
                reasoning=result.reasoning,
                context={
                    "symbol": trade["symbol"],
                    "strategy": trade["strategy"],
                    "regime_at_entry": trade["regime_at_entry"],
                    "pnl": trade["pnl"],
                    "exit_reason": trade["exit_reason"],
                },
                closed_trade_id=trade["id"],
            )
        else:
            logger.info("[memory] post-mortem: no mistake (%s)", result.reasoning[:80])


def get_lessons_for_agents(store: MemoryStore) -> dict[str, list[dict[str, Any]]]:
    """Fetch the lesson sets each agent should see this cycle."""
    return {
        agent: store.get_lessons(categories=categories)
        for agent, categories in AGENT_LESSON_CATEGORIES.items()
    }
