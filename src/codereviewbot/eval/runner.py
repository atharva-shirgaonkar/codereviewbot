"""Run reviews against cases and save per-case and aggregate results."""

import json
import logging
import re
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from ..agent.loop import run_review
from ..agent.types import LLMClient
from ..config import Settings, get_settings
from ..models import ReviewResult
from ..usage import UsageTracker
from .dataset import EvalCase
from .judge import Judge, JudgeDecision, LLMJudge, RecordingJudge
from .noise import LLMUsefulnessJudge, UsefulnessJudge, noise_summary
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
    usefulness_judge: UsefulnessJudge | None = None,
) -> dict[str, Any]:
    """Run each case independently, recording errors and aggregate metrics."""
    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    settings = get_settings()
    noise_judge = usefulness_judge if usefulness_judge is not None else LLMUsefulnessJudge(client)
    case_scores: list[CaseScore] = []
    noise_totals = {
        "total_comments": 0,
        "matched": 0,
        "unmatched_useful": 0,
        "unmatched_useless": 0,
    }
    error_count = 0

    for index, case in enumerate(cases, start=1):
        tracker = UsageTracker(settings)
        started = time.perf_counter()
        review: ReviewResult | None = None
        error_text: str | None = None
        case_judge = RecordingJudge(judge, case.case_id)
        case_score: CaseScore | None = None
        case_noise: dict[str, int | float] | None = None
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
                review.comments,
                case.issues,
                case_judge,
                cost_usd=usage["cost_usd"],
                latency_seconds=elapsed,
                case_id=case.case_id,
            )
            case_noise = noise_summary(review.comments, case_judge.decisions, noise_judge)
        except Exception as error:
            error_count += 1
            elapsed = time.perf_counter() - started
            error_text = _safe_error(error, settings)
            logger.warning("Evaluation case %s failed: %s", case.case_id, error_text)
            usage = tracker.summary()
            if case_score is None:
                false_positives = len(review.comments) if review is not None else 0
                false_negatives = len(case.issues)
                case_score = CaseScore(
                    true_positives=0,
                    false_positives=false_positives,
                    false_negatives=false_negatives,
                    precision=0.0,
                    recall=0.0,
                    f1=0.0,
                    cost_usd=usage["cost_usd"],
                    latency_seconds=elapsed,
                )
            if case_noise is None:
                total = len(review.comments) if review is not None else 0
                matched_indices = {
                    decision.comment_index
                    for decision in case_judge.decisions
                    if decision.match
                }
                matched = len(matched_indices)
                case_noise = {
                    "total_comments": total,
                    "matched": matched,
                    "unmatched_useful": 0,
                    "unmatched_useless": total - matched,
                    "noise_rate": (total - matched) / total if total else 0.0,
                }

        case_scores.append(case_score)
        for key in noise_totals:
            noise_totals[key] += int(case_noise[key])
        case_noise["noise_rate"] = (
            case_noise["unmatched_useless"] / case_noise["total_comments"]
            if case_noise["total_comments"]
            else 0.0
        )
        filename = _case_filename(index, case.case_id)
        _write_json(
            output_dir / filename,
            {
                "case_id": case.case_id,
                "repo": case.repo,
                "pr_number": case.pr_number,
                "language": case.language,
                "review": review.model_dump(mode="json") if review is not None else None,
                "score": case_score.__dict__,
                "judge_decisions": [asdict(decision) for decision in case_judge.decisions],
                "noise": case_noise,
                "usage": usage,
                "error": error_text,
            },
            settings,
        )

    total_comments = noise_totals["total_comments"]
    total_useless = noise_totals["unmatched_useless"]
    run_summary: dict[str, Any] = {
        "run_timestamp": datetime.now(timezone.utc).isoformat(),
        "model": settings.llm_model,
        "case_count": len(cases),
        "error_count": error_count,
        "command": list(sys.argv),
        "noise": {
            **noise_totals,
            "noise_rate": total_useless / total_comments if total_comments else 0.0,
        },
        "aggregate": aggregate(case_scores),
    }
    _write_json(output_dir / "summary.json", run_summary, settings)
    return run_summary
