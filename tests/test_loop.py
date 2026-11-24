"""Tests for the bounded review loop using scripted model responses."""

import json

import pytest

from codereviewbot.agent import loop
from codereviewbot.agent.fake_client import FakeLLMClient
from codereviewbot.agent.loop import StepLimitError, run_review
from codereviewbot.agent.types import ModelResponse, ToolCall, Usage
from codereviewbot.usage import UsageTracker


def _response(content: str | None = None, calls: list[ToolCall] | None = None) -> ModelResponse:
    """Build a model response with predictable token counts."""
    return ModelResponse(
        content=content,
        tool_calls=calls or [],
        usage=Usage(prompt_tokens=3, completion_tokens=2),
    )


def _review_json(summary: str = "No issues found.") -> str:
    """Return a minimal valid review result."""
    return json.dumps({"summary": summary, "comments": []})


def test_tool_call_then_final_answer(monkeypatch: pytest.MonkeyPatch) -> None:
    """Run the requested diff tool and send linked output on the next turn."""
    def fake_diff(repo: str, pr_number: int) -> str:
        assert repo == "owner/project"
        assert pr_number == 12
        return "diff text"

    monkeypatch.setattr(loop.tools, "get_diff", fake_diff)
    client = FakeLLMClient(
        [
            _response(calls=[ToolCall(
                id="call-1",
                name="get_diff",
                arguments={"repo": "owner/project", "pr_number": 12},
            )]),
            _response(_review_json("Reviewed diff.")),
        ]
    )

    result = run_review("owner/project", 12, client)

    assert result.summary == "Reviewed diff."
    history = client.messages_received[1]
    assert history[2] == {
        "type": "function_call",
        "call_id": "call-1",
        "name": "get_diff",
        "arguments": '{"repo": "owner/project", "pr_number": 12}',
    }
    output = json.loads(history[3]["output"])
    assert history[3]["call_id"] == "call-1"
    assert output.startswith("diff text")
    assert client.messages_received[0] == history[:2]


def test_plain_final_answer() -> None:
    """Accept a valid result when the model does not request a tool."""
    result = run_review("owner/project", 12, FakeLLMClient([_response(_review_json())]))

    assert result.comments == []


def test_wrong_repository_tool_call_is_rejected() -> None:
    """Return a model-visible error without calling a mismatched repository."""
    client = FakeLLMClient(
        [
            _response(calls=[ToolCall(
                id="call-1",
                name="get_diff",
                arguments={"repo": "other/project", "pr_number": 12},
            )]),
            _response(_review_json()),
        ]
    )

    run_review("owner/project", 12, client)

    assert json.loads(client.messages_received[1][3]["output"]) == {
        "error": "Tool repository must match the pull request repository."
    }


def test_tool_error_is_passed_to_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """Convert a tool exception to a short output item for the next turn."""
    def fail_diff(repo: str, pr_number: int) -> str:
        raise RuntimeError("diff unavailable")

    monkeypatch.setattr(loop.tools, "get_diff", fail_diff)
    client = FakeLLMClient(
        [
            _response(calls=[ToolCall(
                id="call-1",
                name="get_diff",
                arguments={"repo": "owner/project", "pr_number": 12},
            )]),
            _response(_review_json()),
        ]
    )

    run_review("owner/project", 12, client)

    assert json.loads(client.messages_received[1][3]["output"]) == {
        "error": "diff unavailable"
    }


def test_bad_json_then_fixed() -> None:
    """Ask once for corrected JSON and accept the corrected review."""
    client = FakeLLMClient([_response("not json"), _response(_review_json("Fixed."))])

    result = run_review("owner/project", 12, client)

    assert result.summary == "Fixed."
    assert "Return a corrected JSON" in client.messages_received[1][-1]["content"]


def test_bad_json_twice_raises() -> None:
    """Raise the safe parse error when the correction is invalid too."""
    client = FakeLLMClient([_response("secret model text"), _response("still bad")])

    with pytest.raises(ValueError, match="not valid JSON") as error:
        run_review("owner/project", 12, client)

    assert "secret model text" not in str(error.value)


def test_step_cap_raises() -> None:
    """Stop when the model keeps requesting tools beyond the configured cap."""
    response = _response(calls=[ToolCall(
        id="call-1",
        name="unknown_tool",
        arguments={"repo": "owner/project"},
    )])

    with pytest.raises(StepLimitError, match="limit of 1"):
        run_review("owner/project", 12, FakeLLMClient([response]), max_steps=1)


def test_usage_is_recorded() -> None:
    """Accumulate usage from each model turn, including the final turn."""
    tracker = UsageTracker()
    client = FakeLLMClient([_response("not json"), _response(_review_json())])

    run_review("owner/project", 12, client, tracker=tracker)

    assert tracker.summary()["prompt_tokens"] == 6
    assert tracker.summary()["completion_tokens"] == 4
