"""Tests for token and cost tracking."""

import logging

import pytest

from codereviewbot.agent.types import Usage
from codereviewbot.config import Settings
from codereviewbot.usage import UsageTracker


def test_usage_tracker_accumulates_known_cost(caplog: pytest.LogCaptureFixture) -> None:
    """Configured per-million prices produce an estimated cost and a log."""
    settings = Settings(
        llm_input_price_per_m=2.0,
        llm_output_price_per_m=8.0,
    )
    tracker = UsageTracker(settings=settings)
    tracker.record(Usage(prompt_tokens=1_000_000, completion_tokens=0))
    tracker.record(Usage(prompt_tokens=0, completion_tokens=500_000))

    with caplog.at_level(logging.INFO, logger="codereviewbot.usage"):
        summary = tracker.summary()

    assert summary["prompt_tokens"] == 1_000_000
    assert summary["completion_tokens"] == 500_000
    assert summary["total_tokens"] == 1_500_000
    assert summary["cost_usd"] == pytest.approx(6.0)
    assert "cost_usd=6.0" in caplog.text


def test_usage_tracker_leaves_unknown_cost_as_none() -> None:
    """Missing prices leave cost unknown instead of inventing a rate."""
    tracker = UsageTracker(settings=Settings())
    tracker.record(Usage(prompt_tokens=10, completion_tokens=5))

    summary = tracker.summary()

    assert summary["total_tokens"] == 15
    assert summary["cost_usd"] is None


def test_settings_read_optional_prices(monkeypatch: pytest.MonkeyPatch) -> None:
    """Configured per-million prices are loaded from environment variables."""
    from codereviewbot import config
    from codereviewbot.config import get_settings

    monkeypatch.setattr(config, "load_dotenv", lambda: False)
    monkeypatch.setenv("LLM_INPUT_PRICE_PER_M", "1.5")
    monkeypatch.setenv("LLM_OUTPUT_PRICE_PER_M", "4.0")

    settings = get_settings()

    assert settings.llm_input_price_per_m == 1.5
    assert settings.llm_output_price_per_m == 4.0
