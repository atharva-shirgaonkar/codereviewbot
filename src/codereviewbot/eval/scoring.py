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
