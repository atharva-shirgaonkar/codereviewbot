"""Export judge decisions for human review and measure agreement."""

import csv
import json
import random
from pathlib import Path
from typing import Any


CSV_COLUMNS = ["id", "comment", "issue", "judge_match", "judge_reason", "human_match"]


def _read_decisions(results_dir: str | Path) -> list[dict[str, Any]]:
    """Read recorded comparison decisions from per-case result JSON files."""
    decisions: list[dict[str, Any]] = []
    for result_path in sorted(Path(results_dir).glob("case-*.json")):
        try:
            case = json.loads(result_path.read_text(encoding="utf-8"))
            decisions.extend(case.get("judge_decisions", []))
        except (json.JSONDecodeError, OSError, TypeError) as error:
            raise ValueError(f"Could not read judge decisions from {result_path.name}.") from error
    return decisions
