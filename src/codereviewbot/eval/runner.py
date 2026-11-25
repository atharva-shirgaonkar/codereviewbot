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
        if secret is not None:
            value = secret.get_secret_value()
            if value:
                content = content.replace(value, "[redacted]")
    path.write_text(content + "\n", encoding="utf-8")


def run_eval(
    cases: Sequence[EvalCase],
    client: LLMClient,
    judge: Judge,
    out_dir: str | Path,
    max_steps: int = 8,
) -> dict[str, Any]:
    """Run each case independently, recording errors and aggregate metrics."""
    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    settings = get_settings()
    case_scores: list[CaseScore] = []
    error_count = 0

    for index, case in enumerate(cases, start=1):
        tracker = UsageTracker(settings)
        started = time.perf_counter()
        review: ReviewResult | None = None
        error_text: str | None = None
        try:
            review = run_review(
                case.repo,
                case.pr_number,
                client,
                max_steps=max_steps,
                tracker=tracker,
                dry_run=True,
            )
            elapsed = time.perf_counter() - started
            usage = tracker.summary()
            case_score = score_case(
