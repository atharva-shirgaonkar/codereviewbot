"""Local token and estimated cost tracking."""

import logging

from .agent.types import Usage
from .config import Settings, get_settings


logger = logging.getLogger(__name__)


class UsageTracker:
    """Accumulate token totals and estimate cost when prices are configured."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._prompt_tokens = 0
        self._completion_tokens = 0

    def record(self, usage: Usage) -> None:
        """Add token counts from one model response."""
        self._prompt_tokens += usage.prompt_tokens
        self._completion_tokens += usage.completion_tokens

    def summary(self) -> dict[str, int | float | None]:
        """Return totals and log an estimated cost, if rates are known."""
        cost_usd: float | None = None
        input_price = self._settings.llm_input_price_per_m
        output_price = self._settings.llm_output_price_per_m
        if input_price is not None and output_price is not None:
            cost_usd = (
                self._prompt_tokens * input_price
                + self._completion_tokens * output_price
            ) / 1_000_000

        result: dict[str, int | float | None] = {
            "prompt_tokens": self._prompt_tokens,
            "completion_tokens": self._completion_tokens,
            "total_tokens": self._prompt_tokens + self._completion_tokens,
            "cost_usd": cost_usd,
        }
        logger.info(
            "LLM usage: prompt_tokens=%d completion_tokens=%d total_tokens=%d cost_usd=%s",
            self._prompt_tokens,
            self._completion_tokens,
            result["total_tokens"],
            cost_usd if cost_usd is not None else "unknown",
        )
        return result
