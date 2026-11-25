"""Render saved evaluation summaries as a compact Markdown table."""

import json
from pathlib import Path
from typing import Any


def _cell(value: Any) -> str:
    """Format one Markdown cell without breaking the table."""
    return str(value).replace("|", "\\|").replace("\n", " ")


