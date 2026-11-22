"""Pydantic models used by code review tools."""

from pydantic import BaseModel


class ReviewComment(BaseModel):
    """One finding tied to a file and line."""

    path: str
    line: int
    severity: str
    message: str


class ReviewResult(BaseModel):
    """The summary and findings from a pull request review."""

    summary: str
    comments: list[ReviewComment]
