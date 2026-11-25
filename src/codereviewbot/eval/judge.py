"""LLM-backed and scripted judges for finding comparisons."""

import json
import logging
from typing import Protocol

from ..agent.types import LLMClient


logger = logging.getLogger(__name__)


class Judge(Protocol):
    """Compare a predicted comment with one expected issue."""

