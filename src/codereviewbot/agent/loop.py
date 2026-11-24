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

