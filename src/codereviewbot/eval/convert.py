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
