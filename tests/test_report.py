"""Tests for saved evaluation summary tables."""

import json
from pathlib import Path
from collections.abc import Iterator
from tempfile import TemporaryDirectory

import pytest

from codereviewbot.eval.report import results_table


@pytest.fixture
def tmp_path() -> Iterator[Path]:
    """Use project-local test files if the Windows temp root is inaccessible."""
    with TemporaryDirectory(dir=Path.cwd()) as temp_dir:
        yield Path(temp_dir)


def test_report_formats_unknown_cost_and_optional_noise(tmp_path: Path) -> None:
    """Show unknown costs plainly and include saved run metadata."""
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "run_timestamp": "2026-11-25T10:30:00+00:00",
                "model": "fake-model",
                "command": ["python", "-m", "codereviewbot.eval.cli"],
                "aggregate": {
                    "micro_precision": 0.8,
                    "micro_recall": 0.5,
                    "micro_f1": 0.615,
                    "mean_cost_usd": None,
                    "mean_latency_seconds": 1.25,
                },
                "noise": {"noise_rate": 0.1},
            }
        ),
        encoding="utf-8",
    )

    table = results_table(summary_path)

    assert "| Precision | 0.8 |" in table
    assert "| Noise rate | 0.1 |" in table
    assert "| Mean cost per PR (USD) | unknown |" in table
    assert "| Model | fake-model |" in table
    assert "| Date | 2026-11-25 |" in table
    assert "codereviewbot.eval.cli" in table
