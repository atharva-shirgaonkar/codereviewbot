"""Tests for usefulness judging and comment noise metrics."""

from codereviewbot.eval.judge import JudgeDecision
from codereviewbot.eval.noise import FakeUsefulnessJudge, LLMUsefulnessJudge, noise_summary
from codereviewbot.agent.fake_client import FakeLLMClient
from codereviewbot.agent.types import ModelResponse, Usage
from codereviewbot.models import ReviewComment


def _comment(message: str) -> ReviewComment:
    """Build a minimal review comment for noise checks."""
    return ReviewComment(path="src/a.py", line=1, severity="minor", message=message)


def test_noise_rate_with_zero_comments() -> None:
    """An empty review has zero noise and does not call the usefulness judge."""
    judge = FakeUsefulnessJudge([])

    result = noise_summary([], [], judge)

    assert result == {
        "total_comments": 0,
        "matched": 0,
        "unmatched_useful": 0,
        "unmatched_useless": 0,
        "noise_rate": 0.0,
    }
    assert judge.comments == []


def test_matched_comments_are_useful_and_noise_counts_unmatched() -> None:
    """Only unmatched comments are classified for usefulness."""
    comments = [_comment("matched"), _comment("useful extra"), _comment("noise")]
    decisions = [JudgeDecision("case", 0, 0, "matched", "issue", True, "Same issue.")]
    judge = FakeUsefulnessJudge([True, False])

    result = noise_summary(comments, decisions, judge)

    assert result == {
        "total_comments": 3,
        "matched": 1,
        "unmatched_useful": 1,
        "unmatched_useless": 1,
        "noise_rate": 1 / 3,
    }
    assert judge.comments == ["useful extra", "noise"]


def test_llm_usefulness_judge_uses_fake_client_without_tools() -> None:
    """Parse strict usefulness JSON through the fake client only."""
    client = FakeLLMClient(
        [
            ModelResponse(
                content='{"useful": true, "reason": "actionable"}',
                usage=Usage(prompt_tokens=1, completion_tokens=1),
            )
        ]
    )

    assert LLMUsefulnessJudge(client).is_useful("review finding") is True
    assert client.messages_received[0] and client.messages_received[0][0]["role"] == "system"
