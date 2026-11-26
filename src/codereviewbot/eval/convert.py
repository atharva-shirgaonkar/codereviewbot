"""Generic CSV/JSON record conversion to the normalized case format."""

import csv
import json
from pathlib import Path
from typing import Any, Sequence

from pydantic import ValidationError

from .dataset import EvalCase


MAPPING_KEYS = (
    "case_id",
    "repo",
    "pr_number",
    "language",
    "issues",
    "path",
    "line",
    "description",
)
