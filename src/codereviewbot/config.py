"""Application settings loaded from environment variables."""

import os

from dotenv import load_dotenv
from pydantic import BaseModel, SecretStr


class Settings(BaseModel):
    """Settings used to configure CodeReviewBot."""

    llm_api_key: SecretStr | None = None
    llm_base_url: str | None = None
    llm_model: str | None = None
    log_level: str = "INFO"

    def require_llm_api_key(self) -> str:
        """Return the API key or raise when a caller explicitly requires it."""
        if self.llm_api_key is None:
            raise ValueError("LLM_API_KEY is required for this operation.")
        return self.llm_api_key.get_secret_value()


def get_settings() -> Settings:
    """Load optional local .env values and return current settings."""
    load_dotenv()
    return Settings(
        llm_api_key=os.getenv("LLM_API_KEY"),
        llm_base_url=os.getenv("LLM_BASE_URL"),
        llm_model=os.getenv("LLM_MODEL"),
        log_level=os.getenv("LOG_LEVEL", "INFO"),
    )
