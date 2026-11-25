"""Run reviews against cases and save per-case and aggregate results."""

import json
import logging
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from ..agent.loop import run_review
from ..agent.types import LLMClient
from ..config import Settings, get_settings
from ..models import ReviewResult
from ..usage import UsageTracker
from .dataset import EvalCase
from .judge import Judge
from .scoring import CaseScore, aggregate, score_case


logger = logging.getLogger(__name__)


def _safe_error(error: Exception, settings: Settings) -> str:
    """Return a short error message with configured credentials removed."""
    message = " ".join(str(error).split())
    for secret in (settings.llm_api_key, settings.github_token):
        if secret is not None:
            value = secret.get_secret_value()
            if value:
                message = message.replace(value, "[redacted]")
    return f"{type(error).__name__}: {message[:300]}"


def _case_filename(index: int, case_id: str) -> str:
    """Make a readable result filename that cannot escape the output folder."""
    safe_id = re.sub(r"[^A-Za-z0-9._-]", "_", case_id).strip(".") or "case"
    return f"case-{index:04d}-{safe_id}.json"


def _write_json(path: Path, data: dict[str, Any], settings: Settings) -> None:
    """Write JSON while removing configured credentials from stored text."""
    content = json.dumps(data, indent=2, ensure_ascii=False)
    for secret in (settings.llm_api_key, settings.github_token):
