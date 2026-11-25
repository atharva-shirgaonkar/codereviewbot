"""One-to-one finding matching and evaluation metrics."""

from dataclasses import dataclass
from statistics import fmean
from typing import Protocol, Sequence

from ..models import ReviewComment
from .dataset import GroundTruthIssue


class IssueJudge(Protocol):
    """Decide whether a review comment describes an expected issue."""

    def matches(self, comment_message: str, issue_description: str) -> bool:
        """Return whether the two descriptions refer to the same issue."""
        ...


@dataclass
class CaseScore:
    """Counts and derived metrics for one evaluated pull request."""

    true_positives: int
    false_positives: int
    false_negatives: int
    precision: float
    recall: float
    f1: float
    cost_usd: float | None = None
    latency_seconds: float = 0.0


def match_comments(
    comments: Sequence[ReviewComment],
    issues: Sequence[GroundTruthIssue],
    judge: IssueJudge,
    line_tolerance: int = 5,
) -> list[tuple[int, int]]:
    """Greedily match comments to unused same-path issues within line tolerance.

    Comments and issues are considered in input order. Each is used at most
    once, so duplicate comments cannot inflate the true-positive count.
    """
    if line_tolerance < 0:
        raise ValueError("line_tolerance must be zero or greater.")

    matches: list[tuple[int, int]] = []
    used_issues: set[int] = set()
    for comment_index, comment in enumerate(comments):
        for issue_index, issue in enumerate(issues):
            if issue_index in used_issues:
                continue
            if comment.path != issue.path:
                continue
            if abs(comment.line - issue.line) > line_tolerance:
                continue
            if judge.matches(comment.message, issue.description):
                matches.append((comment_index, issue_index))
                used_issues.add(issue_index)
                break
    return matches


def _ratio(numerator: int, denominator: int) -> float:
    """Return a metric ratio, using zero when no denominator exists."""
    return numerator / denominator if denominator else 0.0


def score_case(
