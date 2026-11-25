"""Tests for finding matching and metric calculations."""

import pytest

from codereviewbot.eval.dataset import GroundTruthIssue
from codereviewbot.eval.judge import FakeJudge
from codereviewbot.eval.scoring import CaseScore, aggregate, match_comments, score_case
from codereviewbot.models import ReviewComment


def _comment(line: int, message: str = "same issue", path: str = "src/a.py") -> ReviewComment:
    """Create a small review comment for scoring tests."""
    return ReviewComment(path=path, line=line, severity="minor", message=message)


def _issue(line: int, description: str = "expected issue", path: str = "src/a.py") -> GroundTruthIssue:
    """Create a small expected finding for scoring tests."""
    return GroundTruthIssue(path=path, line=line, description=description)


def test_perfect_match_has_perfect_metrics() -> None:
    """A single correctly matched finding scores one across metrics."""
    score = score_case([_comment(10)], [_issue(10)], FakeJudge())

    assert (score.true_positives, score.false_positives, score.false_negatives) == (1, 0, 0)
    assert (score.precision, score.recall, score.f1) == (1.0, 1.0, 1.0)


def test_no_semantic_match_is_false_positive_and_missed_issue() -> None:
    """A nearby candidate rejected by the judge is both FP and FN."""
    score = score_case([_comment(10)], [_issue(10)], FakeJudge(False))

    assert (score.true_positives, score.false_positives, score.false_negatives) == (0, 1, 1)


def test_extra_comments_are_false_positives() -> None:
    """Additional unmatched comments count as false positives."""
    score = score_case(
        [_comment(10), _comment(40, "extra")], [_issue(10)], FakeJudge()
    )

    assert (score.true_positives, score.false_positives, score.false_negatives) == (1, 1, 0)


def test_missed_issues_are_false_negatives() -> None:
    """Ground-truth issues with no comments count as false negatives."""
    score = score_case([], [_issue(10)], FakeJudge())

    assert (score.true_positives, score.false_positives, score.false_negatives) == (0, 0, 1)
    assert score.recall == 0.0


def test_line_tolerance_includes_boundary_only() -> None:
    """The configured distance is inclusive and the next line is excluded."""
    comments = [_comment(15), _comment(16)]
    issues = [_issue(10), _issue(10)]

    assert match_comments(comments, issues, FakeJudge(), line_tolerance=5) == [(0, 0)]
    assert match_comments([_comment(16)], [_issue(10)], FakeJudge()) == []


def test_matching_is_one_to_one() -> None:
    """A second duplicate comment cannot reuse the already matched issue."""
    matches = match_comments(
        [_comment(10), _comment(10)], [_issue(10)], FakeJudge(), line_tolerance=0
    )

    assert matches == [(0, 0)]


def test_zero_division_metrics_are_zero() -> None:
    """An empty prediction and target set returns zero metrics safely."""
    score = score_case([], [], FakeJudge())
    totals = aggregate([score])

    assert (score.precision, score.recall, score.f1) == (0.0, 0.0, 0.0)
    assert totals["micro_precision"] == 0.0
    assert totals["micro_recall"] == 0.0
    assert totals["micro_f1"] == 0.0


def test_aggregate_uses_micro_counts_and_mean_resources() -> None:
    """Aggregate metrics use count totals and arithmetic cost/latency means."""
    totals = aggregate(
        [
            CaseScore(1, 0, 0, 1.0, 1.0, 1.0, cost_usd=0.2, latency_seconds=2.0),
            CaseScore(1, 1, 0, 0.5, 1.0, 2 / 3, cost_usd=0.4, latency_seconds=4.0),
        ]
    )

    assert totals["micro_precision"] == 2 / 3
    assert totals["micro_recall"] == 1.0
    assert totals["micro_f1"] == 0.8
    assert totals["mean_cost_usd"] == pytest.approx(0.3)
    assert totals["mean_latency_seconds"] == 3.0
