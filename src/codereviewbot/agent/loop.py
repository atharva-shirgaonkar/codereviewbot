"""Bounded tool-calling loop for reviewing one pull request."""

import json
import logging
from typing import Any

from ..models import ReviewParseError, ReviewResult, parse_review_json
from ..usage import UsageTracker
from . import tools
from .prompts import SYSTEM_PROMPT
from .schemas import TOOL_SCHEMAS
from .types import LLMClient, ToolCall


logger = logging.getLogger(__name__)


class StepLimitError(RuntimeError):
    """Raised when a review needs more model turns than allowed."""


def _initial_messages(repo: str, pr_number: int) -> list[dict[str, Any]]:
    """Create the initial Responses API input history."""
