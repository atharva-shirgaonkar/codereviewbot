"""Tests for environment-based configuration."""

import pytest

from codereviewbot import config
from codereviewbot.config import get_settings


def test_settings_read_environment_variables(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Settings should use the values supplied in the process environment."""
    monkeypatch.setattr(config, "load_dotenv", lambda: False)
    monkeypatch.setenv("LLM_API_KEY", "test-only-key")
    monkeypatch.setenv("LLM_BASE_URL", "https://example.test/v1")
    monkeypatch.setenv("LLM_MODEL", "example-model")
    monkeypatch.setenv("LLM_API_STYLE", "chat_completions")
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")

    settings = get_settings()

    assert settings.llm_api_key is not None
    assert settings.llm_api_key.get_secret_value() == "test-only-key"
    assert settings.llm_base_url == "https://example.test/v1"
    assert settings.llm_model == "example-model"
    assert settings.llm_api_style == "chat_completions"
    assert settings.log_level == "DEBUG"


def test_settings_use_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    """Optional settings should have safe defaults when variables are absent."""
    monkeypatch.setattr(config, "load_dotenv", lambda: False)
    for name in ("LLM_API_KEY", "LLM_BASE_URL", "LLM_MODEL", "LLM_API_STYLE", "LOG_LEVEL"):
        monkeypatch.delenv(name, raising=False)

    settings = get_settings()

    assert settings.llm_api_key is None
    assert settings.llm_base_url is None
    assert settings.llm_model is None
    assert settings.llm_api_style == "responses"
    assert settings.log_level == "INFO"
