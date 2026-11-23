"""Tests for parsing validated review results."""

import pytest

from codereviewbot.models import ReviewParseError, parse_review_json


VALID_REVIEW = (
    '{"summary":"Looks good.","comments":['
    '{"path":"src/app.py","line":8,"severity":"minor","message":"Check this."}'
    ']}'
)


def test_parse_valid_review_json() -> None:
    """Valid JSON becomes typed review models."""
    result = parse_review_json(VALID_REVIEW)

    assert result.summary == "Looks good."
    assert result.comments[0].severity == "minor"


def test_parse_fenced_review_json() -> None:
    """JSON in a Markdown code fence is accepted."""
    result = parse_review_json(f"```json\n{VALID_REVIEW}\n```")

    assert result.summary == "Looks good."


def test_parse_invalid_json_raises_short_error() -> None:
    """Bad JSON fails without copying model output into the error."""
    with pytest.raises(ReviewParseError, match="not valid JSON") as error:
        parse_review_json('{"private":"do not repeat this text"')

    assert "do not repeat this text" not in str(error.value)


def test_parse_wrong_severity_raises_schema_error() -> None:
    """Severity values outside the allowed set are rejected."""
    invalid_review = VALID_REVIEW.replace('"minor"', '"urgent"')

    with pytest.raises(ReviewParseError, match="does not match the expected schema") as error:
        parse_review_json(invalid_review)

    assert "urgent" not in str(error.value)


def test_parse_missing_fields_raises_schema_error() -> None:
    """Required review fields cannot be omitted."""
    with pytest.raises(ReviewParseError, match="does not match the expected schema"):
        parse_review_json('{"summary":"Missing comments."}')
