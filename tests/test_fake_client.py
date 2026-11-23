"""Tests for the scripted fake model client."""

import pytest

from codereviewbot.agent.fake_client import FakeLLMClient
from codereviewbot.agent.types import ModelResponse, ToolCall, Usage


def test_fake_client_returns_scripted_responses_in_order() -> None:
    """The fake returns responses in order and records each message list."""
    first = ModelResponse(
        content=None,
        tool_calls=[ToolCall(id="call-1", name="get_diff", arguments={"repo": "a/b"})],
        usage=Usage(prompt_tokens=2, completion_tokens=1),
    )
    second = ModelResponse(
        content="Done.",
        usage=Usage(prompt_tokens=3, completion_tokens=2),
    )
    client = FakeLLMClient([first, second])
    first_messages = [{"role": "user", "content": "Review PR 1."}]
    second_messages = [{"role": "user", "content": "Review PR 2."}]

    assert client.chat(first_messages, []) is first
    assert client.chat(second_messages, []) is second
    assert client.messages_received == [first_messages, second_messages]


def test_fake_client_errors_when_script_is_exhausted() -> None:
    """Calling beyond the scripted responses raises a readable error."""
    response = ModelResponse(
        content="Done.",
        usage=Usage(prompt_tokens=1, completion_tokens=1),
    )
    client = FakeLLMClient([response])
    client.chat([], [])

    with pytest.raises(RuntimeError, match="no scripted responses left"):
        client.chat([], [])
