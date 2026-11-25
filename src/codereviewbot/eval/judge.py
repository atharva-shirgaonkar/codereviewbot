"""LLM-backed and scripted judges for finding comparisons."""

import json
import logging
from dataclasses import dataclass
from typing import Protocol

from ..agent.types import LLMClient


logger = logging.getLogger(__name__)


@dataclass
class JudgeDecision:
    """Auditable result for one comment and expected-issue comparison."""

    case_id: str
    comment_index: int
    issue_index: int
    comment_message: str
    issue_description: str
    match: bool
    reason: str


class Judge(Protocol):
    """Compare a predicted comment with one expected issue."""

    def matches(self, comment_message: str, issue_description: str) -> bool:
        """Return whether the comment identifies the expected issue."""
        ...


class LLMJudge:
    """Use an LLM client to make strict-JSON semantic match decisions."""

    def __init__(self, client: LLMClient) -> None:
        self._client = client
        self.last_reason = ""

    def matches(self, comment_message: str, issue_description: str) -> bool:
        """Return false and log safely if the judge response is invalid."""
        response = self._client.chat(
            [
                {
                    "role": "system",
                    "content": 'Decide if the review comment describes the expected issue. '
                    'Return ONLY JSON: {"match": true or false, "reason": "..."}.',
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "comment": comment_message,
                            "expected_issue": issue_description,
                        }
                    ),
                },
            ],
            [],
        )
        try:
            if response.tool_calls or response.content is None:
                raise ValueError
            data = json.loads(response.content)
            if (
                not isinstance(data, dict)
                or set(data) != {"match", "reason"}
                or type(data["match"]) is not bool
                or not isinstance(data["reason"], str)
            ):
                raise ValueError
            self.last_reason = data["reason"]
            return data["match"]
        except (ValueError, TypeError, KeyError):
            self.last_reason = "Invalid judge response."
            logger.warning("LLM judge returned invalid JSON; treating it as no match.")
            return False


class FakeJudge:
    """Return fixed or scripted match decisions and record compared text."""

    def __init__(self, matches: bool | list[bool] = True) -> None:
        self._matches = matches if isinstance(matches, list) else [matches]
        self._next = 0
        self.comparisons: list[tuple[str, str]] = []
        self.last_reason = ""

    def matches(self, comment_message: str, issue_description: str) -> bool:
        """Return the next scripted decision and fail if none remain."""
        self.comparisons.append((comment_message, issue_description))
        if self._next >= len(self._matches):
            raise RuntimeError("FakeJudge has no scripted decisions left.")
        decision = self._matches[self._next]
        self._next += 1
        self.last_reason = f"Fake judge decision: {str(decision).lower()}."
        return decision


class RecordingJudge:
    """Add pair context and decision history around any Judge implementation."""

    def __init__(self, judge: Judge, case_id: str = "") -> None:
        self._judge = judge
        self._case_id = case_id
        self._comment_index = -1
        self._issue_index = -1
        self.decisions: list[JudgeDecision] = []

    def set_context(self, case_id: str, comment_index: int, issue_index: int) -> None:
        """Set the finding pair that the next judge call will compare."""
        self._case_id = case_id
        self._comment_index = comment_index
        self._issue_index = issue_index

    def matches(self, comment_message: str, issue_description: str) -> bool:
        """Delegate a comparison and store its result and safe reason."""
        try:
            match = self._judge.matches(comment_message, issue_description)
        except Exception:
            self.decisions.append(
                JudgeDecision(
                    self._case_id,
                    self._comment_index,
                    self._issue_index,
                    comment_message,
                    issue_description,
                    False,
                    "Judge call failed.",
                )
            )
            raise

        self.decisions.append(
            JudgeDecision(
                self._case_id,
                self._comment_index,
                self._issue_index,
                comment_message,
                issue_description,
                match,
                getattr(self._judge, "last_reason", ""),
            )
        )
        return match
