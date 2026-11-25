"""Render saved evaluation summaries as a compact Markdown table."""

import json
from pathlib import Path
from typing import Any


def _cell(value: Any) -> str:
    """Format one Markdown cell without breaking the table."""
    return str(value).replace("|", "\\|").replace("\n", " ")


def results_table(summary_path: str | Path) -> str:
    """Read a run summary and render its key quality and cost metrics."""
    try:
        summary = json.loads(Path(summary_path).read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as error:
        raise ValueError("Could not read evaluation summary JSON.") from error

    aggregate = summary.get("aggregate", {})
    rows = [
        ("Precision", aggregate.get("micro_precision", "unknown")),
        ("Recall", aggregate.get("micro_recall", "unknown")),
        ("F1", aggregate.get("micro_f1", "unknown")),
    ]
    noise = summary.get("noise")
    if isinstance(noise, dict) and "noise_rate" in noise:
        rows.append(("Noise rate", noise["noise_rate"]))
    cost = aggregate.get("mean_cost_usd")
    rows.extend(
        [
            ("Mean cost per PR (USD)", "unknown" if cost is None else cost),
            ("Mean latency (seconds)", aggregate.get("mean_latency_seconds", "unknown")),
            ("Model", summary.get("model") or "unknown"),
            ("Date", str(summary.get("run_timestamp", "unknown")).split("T", 1)[0]),
            ("Command", json.dumps(summary.get("command", []), ensure_ascii=False)),
        ]
    )
    lines = ["| Metric | Value |", "| --- | --- |"]
    lines.extend(f"| {_cell(name)} | {_cell(value)} |" for name, value in rows)
    return "\n".join(lines)
