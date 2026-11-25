"""Judge usefulness of unmatched comments and summarize review noise."""

import json
import logging
from typing import Protocol, Sequence

from ..agent.types import LLMClient
from ..models import ReviewComment
from .judge import JudgeDecision


logger = logging.getLogger(__name__)


class UsefulnessJudge(Protocol):
    """Decide whether an unmatched review comment is useful."""

    def is_useful(self, comment_message: str) -> bool:
        """Return whether a comment has actionable review value."""
        ...


class LLMUsefulnessJudge:
    """Ask an LLM for a strict JSON usefulness decision without tools."""

    def __init__(self, client: LLMClient) -> None:
        self._client = client

    def is_useful(self, comment_message: str) -> bool:
        """Return false for malformed judge output and log a generic warning."""
        response = self._client.chat(
            [
                {
                    "role": "system",
                    "content": 'Decide whether this code review comment is useful and actionable. '
                    'Return ONLY JSON: {"useful": true or false, "reason": "..."}.',
                },
                {"role": "user", "content": json.dumps({"comment": comment_message})},
            ],
            [],
        )
        try:
            if response.tool_calls or response.content is None:
                raise ValueError
            data = json.loads(response.content)
            if (
                not isinstance(data, dict)
                or set(data) != {"useful", "reason"}
                or type(data["useful"]) is not bool
                or not isinstance(data["reason"], str)
            ):
                raise ValueError
            return data["useful"]
        except (ValueError, TypeError, KeyError):
            logger.warning("Usefulness judge returned invalid JSON; treating it as useless.")
            return False


class FakeUsefulnessJudge:
    """Return fixed or scripted usefulness values for tests."""

    def __init__(self, useful: bool | list[bool] = True) -> None:
        self._values = useful if isinstance(useful, list) else [useful]
        self._next = 0
        self.comments: list[str] = []

    def is_useful(self, comment_message: str) -> bool:
        """Return the next scripted result or fail when exhausted."""
        self.comments.append(comment_message)
        if self._next >= len(self._values):
            raise RuntimeError("FakeUsefulnessJudge has no scripted decisions left.")
        result = self._values[self._next]
        self._next += 1
        return result


def noise_summary(
    comments: Sequence[ReviewComment],
    decisions: Sequence[JudgeDecision],
    judge: UsefulnessJudge,
) -> dict[str, int | float]:
    """Count matched and useful/unuseful unmatched comments.

    A comment matched to ground truth is useful by definition. Only unmatched
    comments are sent to the usefulness judge.
    """
    matched_indices = {decision.comment_index for decision in decisions if decision.match}
    matched = len(matched_indices)
    unmatched_useful = 0
    unmatched_useless = 0
    for index, comment in enumerate(comments):
        if index in matched_indices:
            continue
        if judge.is_useful(comment.message):
            unmatched_useful += 1
        else:
            unmatched_useless += 1
    total = len(comments)
    return {
        "total_comments": total,
        "matched": matched,
        "unmatched_useful": unmatched_useful,
        "unmatched_useless": unmatched_useless,
        "noise_rate": unmatched_useless / total if total else 0.0,
    }
