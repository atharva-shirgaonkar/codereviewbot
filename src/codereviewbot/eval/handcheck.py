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


def export_handcheck(
    results_dir: str | Path,
    out_csv: str | Path,
    n: int = 20,
    seed: int = 42,
) -> int:
    """Write a reproducible random sample of decisions for manual labels."""
    if n < 0:
        raise ValueError("n must be zero or greater.")
    decisions = _read_decisions(results_dir)
    sample = random.Random(seed).sample(decisions, min(n, len(decisions)))
    output_path = Path(out_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for decision in sample:
            writer.writerow(
                {
                    "id": (
                        f"{decision['case_id']}:{decision['comment_index']}"
                        f":{decision['issue_index']}"
                    ),
                    "comment": decision["comment_message"],
                    "issue": decision["issue_description"],
                    "judge_match": str(decision["match"]).lower(),
                    "judge_reason": decision["reason"],
                    "human_match": "",
                }
            )
    return len(sample)
