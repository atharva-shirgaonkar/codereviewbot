"""Normalized JSONL dataset models and loader."""

import json
from pathlib import Path

from pydantic import BaseModel, ValidationError


class GroundTruthIssue(BaseModel):
    """One expected review finding in a benchmark case."""

    path: str
    line: int
    description: str
    category: str | None = None


class EvalCase(BaseModel):
    """A pull request and its expected findings."""

    case_id: str
    repo: str
    pr_number: int
    language: str
    issues: list[GroundTruthIssue]


def load_cases(path: str | Path) -> list[EvalCase]:
    """Load one JSON object per line, reporting the bad line clearly."""
    cases: list[EvalCase] = []
    with Path(path).open(encoding="utf-8") as dataset_file:
        for line_number, line in enumerate(dataset_file, start=1):
            if not line.strip():
                continue
            try:
                data = json.loads(line)
