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


def _source_name(mapping: dict[str, str], field: str) -> str:
    """Return the mapped source key or explain what the mapping lacks."""
    source_name = mapping.get(field)
    if not isinstance(source_name, str) or not source_name:
        raise ValueError(f"Mapping must provide a source field for '{field}'.")
    return source_name


def convert_records(
    records: Sequence[dict[str, Any]],
    mapping: dict[str, str],
) -> list[EvalCase]:
    """Convert records using explicit source names for case and issue fields.

    Mapping keys are ``case_id``, ``repo``, ``pr_number``, ``language``, and
    ``issues`` for case fields, plus ``path``, ``line``, and ``description``
    for fields within each issue. No source schema is inferred.
    """
    source_fields = {key: _source_name(mapping, key) for key in MAPPING_KEYS}
    cases: list[EvalCase] = []
    for record_number, record in enumerate(records, start=1):
        values: dict[str, Any] = {}
        for field in ("case_id", "repo", "pr_number", "language"):
            source_name = source_fields[field]
            if source_name not in record:
                raise ValueError(
                    f"Record {record_number} is missing source field '{source_name}' for '{field}'."
                )
            values[field] = record[source_name]

        issues_source = source_fields["issues"]
        if issues_source not in record:
            raise ValueError(
                f"Record {record_number} is missing source field '{issues_source}' for 'issues'."
            )
        raw_issues = record[issues_source]
        if isinstance(raw_issues, str):
            try:
                raw_issues = json.loads(raw_issues)
            except json.JSONDecodeError:
                raise ValueError(
                    f"Record {record_number} field '{issues_source}' must contain a JSON issue list."
                ) from None
        if not isinstance(raw_issues, list):
            raise ValueError(f"Record {record_number} field '{issues_source}' must be a list.")

        normalized_issues: list[dict[str, Any]] = []
        for issue_number, raw_issue in enumerate(raw_issues, start=1):
            if not isinstance(raw_issue, dict):
                raise ValueError(f"Record {record_number} issue {issue_number} must be an object.")
            normalized_issue: dict[str, Any] = {}
            for field in ("path", "line", "description"):
                source_name = source_fields[field]
                if source_name not in raw_issue:
                    raise ValueError(
                        f"Record {record_number} issue {issue_number} is missing "
                        f"source field '{source_name}' for '{field}'."
                    )
                normalized_issue[field] = raw_issue[source_name]
            normalized_issues.append(normalized_issue)
        values["issues"] = normalized_issues
        try:
            cases.append(EvalCase.model_validate(values))
        except ValidationError:
