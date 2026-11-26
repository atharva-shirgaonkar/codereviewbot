"""Tests for mapping-driven benchmark record conversion."""

import csv
import json
import sys
from collections.abc import Iterator
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from codereviewbot.eval import cli
from codereviewbot.eval.convert import (
    convert_records,
    filter_language,
    read_records,
    write_cases,
)
from codereviewbot.eval.dataset import load_cases


@pytest.fixture
def tmp_path() -> Iterator[Path]:
    """Use project-local files when Windows temp ACLs are restricted."""
    with TemporaryDirectory(dir=Path.cwd()) as temp_dir:
        yield Path(temp_dir)


MAPPING = {
    "case_id": "id",
    "repo": "repository",
    "pr_number": "number",
    "language": "lang",
    "issues": "findings",
    "path": "file",
    "line": "line_number",
    "description": "detail",
}


def _record(case_id: str, language: str = "python") -> dict[str, object]:
    """Build one source record in an arbitrary, test-only layout."""
    return {
        "id": case_id,
        "repository": "owner/project",
        "number": 23,
        "lang": language,
        "findings": [
            {"file": "src/app.py", "line_number": 9, "detail": "A fake issue."}
        ],
    }


def test_convert_records_uses_explicit_field_mapping() -> None:
    """Convert case and nested issue fields using only supplied names."""
    cases = convert_records([_record("case-1")], MAPPING)

    assert cases[0].case_id == "case-1"
    assert cases[0].repo == "owner/project"
    assert cases[0].issues[0].path == "src/app.py"
    assert cases[0].issues[0].line == 9


def test_convert_records_reports_missing_source_field() -> None:
    """Identify the record and expected normalized field when input is incomplete."""
    row = _record("case-1")
    del row["repository"]

    with pytest.raises(ValueError, match="Record 1 is missing source field 'repository' for 'repo'"):
        convert_records([row], MAPPING)


def test_filter_language_matches_case_insensitively() -> None:
    """Keep only records with the selected language."""
    cases = convert_records([_record("py"), _record("js", "JavaScript")], MAPPING)

    assert [case.case_id for case in filter_language(cases, "python")] == ["py"]


def test_csv_conversion_jsonl_round_trip(tmp_path: Path) -> None:
    """Read an explicitly mapped CSV and round-trip normalized JSONL."""
    source_path = tmp_path / "source.csv"
    with source_path.open("w", newline="", encoding="utf-8") as source_file:
        writer = csv.DictWriter(source_file, fieldnames=list(_record("case-1")))
        writer.writeheader()
        writer.writerow({**_record("case-1"), "findings": json.dumps(_record("x")["findings"])})

    records = read_records(source_path, issues_field="findings")
    cases = convert_records(records, MAPPING)
    output_path = tmp_path / "cases.jsonl"
    write_cases(cases, output_path)

    loaded = load_cases(output_path)
    assert len(loaded) == 1
    assert loaded[0].model_dump() == cases[0].model_dump()


def test_convert_cli_applies_language_and_limit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """CLI conversion filters the language before applying its limit."""
    source_path = tmp_path / "source.json"
    source_path.write_text(json.dumps([_record("one"), _record("two", "javascript")]), encoding="utf-8")
    mapping_path = tmp_path / "mapping.json"
    mapping_path.write_text(json.dumps(MAPPING), encoding="utf-8")
    output_path = tmp_path / "converted.jsonl"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "codereviewbot.eval.cli",
            "--convert",
            str(source_path),
            "--mapping",
            str(mapping_path),
            "--out",
            str(output_path),
            "--language",
            "python",
            "--limit",
            "1",
        ],
    )

    cli.main()

    assert [case.case_id for case in load_cases(output_path)] == ["one"]
