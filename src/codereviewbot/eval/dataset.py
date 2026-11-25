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

