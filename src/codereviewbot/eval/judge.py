"""LLM-backed and scripted judges for finding comparisons."""

import json
import logging
from typing import Protocol

from ..agent.types import LLMClient


logger = logging.getLogger(__name__)


class Judge(Protocol):
    """Compare a predicted comment with one expected issue."""

    def matches(self, comment_message: str, issue_description: str) -> bool:
        """Return whether the comment identifies the expected issue."""
        ...


class LLMJudge:
    """Use an LLM client to make strict-JSON semantic match decisions."""

    def __init__(self, client: LLMClient) -> None:
        self._client = client

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
