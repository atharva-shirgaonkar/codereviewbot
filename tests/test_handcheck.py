"""Tests for deterministic human-check exports and agreement scoring."""

import csv
import json
from pathlib import Path
from collections.abc import Iterator
from tempfile import TemporaryDirectory

import pytest

from codereviewbot.eval.handcheck import agreement, export_handcheck


@pytest.fixture
def tmp_path() -> Iterator[Path]:
    """Use project-local test files if the Windows temp root is inaccessible."""
    with TemporaryDirectory(dir=Path.cwd()) as temp_dir:
        yield Path(temp_dir)


def _write_case(path: Path) -> None:
    """Write a small saved evaluation case with candidate judge decisions."""
    decisions = [
        {
            "case_id": "case-a",
            "comment_index": index,
            "issue_index": index,
            "comment_message": f"comment {index}",
            "issue_description": f"issue {index}",
            "match": index % 2 == 0,
            "reason": f"reason {index}",
        }
        for index in range(8)
    ]
    path.write_text(json.dumps({"judge_decisions": decisions}), encoding="utf-8")


def test_seeded_handcheck_export_is_reproducible(tmp_path: Path) -> None:
    """The same seed exports the same rows and leaves human labels blank."""
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    _write_case(results_dir / "case-0001-a.json")
    first = tmp_path / "first.csv"
    second = tmp_path / "second.csv"

    assert export_handcheck(results_dir, first, n=4, seed=15) == 4
    export_handcheck(results_dir, second, n=4, seed=15)

    assert first.read_text(encoding="utf-8") == second.read_text(encoding="utf-8")
    with first.open(newline="", encoding="utf-8") as csv_file:
        rows = list(csv.DictReader(csv_file))
    assert len(rows) == 4
    assert all(row["human_match"] == "" for row in rows)


def test_agreement_counts_mismatches_and_ignores_empty_human_rows(tmp_path: Path) -> None:
    """Compute agreement and false-match counts for labeled rows only."""
    csv_path = tmp_path / "labels.csv"
    csv_path.write_text(
        "id,comment,issue,judge_match,judge_reason,human_match\n"
        "1,a,x,true,reason,false\n"
        "2,b,y,false,reason,true\n"
        "3,c,z,true,reason,true\n"
        "4,d,w,false,reason,\n"
        ",,,,,\n",
        encoding="utf-8",
    )

    result = agreement(csv_path)

    assert result == {
        "n_labeled": 3,
        "agreement_rate": 1 / 3,
        "false_matches": 1,
        "false_non_matches": 1,
    }


def test_agreement_rejects_bad_boolean_values(tmp_path: Path) -> None:
    """Reject labels outside the documented true/false values."""
    csv_path = tmp_path / "labels.csv"
    csv_path.write_text(
        "id,comment,issue,judge_match,judge_reason,human_match\n"
        "1,a,x,maybe,reason,true\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="judge_match value on CSV row 2"):
        agreement(csv_path)
