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
    comments: Sequence[ReviewComment],
    issues: Sequence[GroundTruthIssue],
    judge: IssueJudge,
    line_tolerance: int = 5,
    cost_usd: float | None = None,
    latency_seconds: float = 0.0,
) -> CaseScore:
    """Score findings by one-to-one matches, extras, and missed issues."""
    matches = match_comments(comments, issues, judge, line_tolerance)
    true_positives = len(matches)
    false_positives = len(comments) - true_positives
    false_negatives = len(issues) - true_positives
    precision = _ratio(true_positives, true_positives + false_positives)
    recall = _ratio(true_positives, true_positives + false_negatives)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return CaseScore(
        true_positives=true_positives,
        false_positives=false_positives,
        false_negatives=false_negatives,
        precision=precision,
        recall=recall,
        f1=f1,
        cost_usd=cost_usd,
        latency_seconds=latency_seconds,
    )


def aggregate(case_scores: Sequence[CaseScore]) -> dict[str, int | float | None]:
    """Compute micro metrics and means across the supplied case scores."""
    true_positives = sum(score.true_positives for score in case_scores)
    false_positives = sum(score.false_positives for score in case_scores)
    false_negatives = sum(score.false_negatives for score in case_scores)
    precision = _ratio(true_positives, true_positives + false_positives)
    recall = _ratio(true_positives, true_positives + false_negatives)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    known_costs = [score.cost_usd for score in case_scores if score.cost_usd is not None]
    latencies = [score.latency_seconds for score in case_scores]
    return {
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "micro_precision": precision,
        "micro_recall": recall,
        "micro_f1": f1,
        "mean_cost_usd": fmean(known_costs) if known_costs else None,
        "mean_latency_seconds": fmean(latencies) if latencies else 0.0,
    }
