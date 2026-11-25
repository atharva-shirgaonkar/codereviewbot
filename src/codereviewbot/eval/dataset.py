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
