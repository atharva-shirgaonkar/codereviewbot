"""Bounded tool-calling loop for reviewing one pull request."""

import json
import logging
from typing import Any

from ..models import ReviewParseError, ReviewResult, parse_review_json
