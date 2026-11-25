"""Judge usefulness of unmatched comments and summarize review noise."""

import json
import logging
from typing import Protocol, Sequence

from ..agent.types import LLMClient
from ..models import ReviewComment
from .judge import JudgeDecision

