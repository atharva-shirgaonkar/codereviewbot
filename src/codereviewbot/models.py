"""Pydantic models and parsing for review results."""

import json
from typing import Literal

from pydantic import BaseModel, ValidationError


class ReviewParseError(ValueError):
    """Raised when model output is not a valid review result."""


class ReviewComment(BaseModel):
    """One finding tied to a file and line."""

    path: str
    line: int
    severity: Literal["critical", "major", "minor", "nit"]
    message: str


class ReviewResult(BaseModel):
    """The summary and findings from a pull request review."""

    summary: str
    comments: list[ReviewComment]


def parse_review_json(text: str) -> ReviewResult:
    """Parse JSON review output, optionally wrapped in Markdown fences."""
    source = text.strip()
    if source.startswith("```"):
        lines = source.splitlines()
        if len(lines) < 2 or lines[-1].strip() != "```":
            raise ReviewParseError("Review output has an unclosed Markdown code fence.")
        source = "\n".join(lines[1:-1]).strip()

    try:
        data = json.loads(source)
    except json.JSONDecodeError:
        raise ReviewParseError("Review output is not valid JSON.") from None

    try:
        return ReviewResult.model_validate(data)
    except ValidationError:
        raise ReviewParseError("Review JSON does not match the expected schema.") from None
