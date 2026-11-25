"""Tests for the normalized evaluation dataset loader."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from codereviewbot.eval.dataset import EvalCase, load_cases


def test_load_cases_reads_jsonl() -> None:
    """Load valid rows and preserve optional issue categories."""
    row = {
        "case_id": "case-1",
        "repo": "owner/project",
        "pr_number": 7,
        "language": "python",
        "issues": [
            {"path": "src/a.py", "line": 3, "description": "Expected bug", "category": "bug"}
        ],
    }
    with TemporaryDirectory(dir=Path.cwd()) as temp_dir:
        dataset = Path(temp_dir) / "cases.jsonl"
        dataset.write_text(json.dumps(row) + "\n", encoding="utf-8")

        cases = load_cases(dataset)

        assert len(cases) == 1
        assert isinstance(cases[0], EvalCase)
        assert cases[0].issues[0].category == "bug"


def test_load_cases_reports_bad_jsonl_line() -> None:
    """Identify the exact line when a JSONL record cannot be parsed."""
    first_row = {
        "case_id": "first",
        "repo": "owner/project",
        "pr_number": 1,
        "language": "python",
        "issues": [],
    }
    with TemporaryDirectory(dir=Path.cwd()) as temp_dir:
        dataset = Path(temp_dir) / "cases.jsonl"
        dataset.write_text(json.dumps(first_row) + "\nnot-json\n", encoding="utf-8")

        with pytest.raises(ValueError, match="dataset line 2"):
            load_cases(dataset)


def test_load_cases_reports_schema_error() -> None:
    """Explain which line has an invalid normalized-case shape."""
    with TemporaryDirectory(dir=Path.cwd()) as temp_dir:
        dataset = Path(temp_dir) / "cases.jsonl"
        dataset.write_text("{}\n", encoding="utf-8")

        with pytest.raises(ValueError, match="line 1 does not match"):
            load_cases(dataset)
