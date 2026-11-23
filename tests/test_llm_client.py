"""Tests for the HTTP LLM client using only mock transports."""

import json

import httpx
import pytest
from pydantic import SecretStr

from codereviewbot.agent import llm_client
from codereviewbot.agent.llm_client import HttpLLMClient, LLMClientError
from codereviewbot.config import Settings


@pytest.fixture
def configured_client(monkeypatch: pytest.MonkeyPatch) -> None:
    """Provide test settings without loading a local .env file."""
    monkeypatch.setattr(
        llm_client,
        "get_settings",
        lambda: Settings(
            llm_api_key=SecretStr("test-only-key"),
            llm_base_url="https://llm.example/v1",
            llm_model="test-model",
        ),
    )


def response_body(output: list[dict[str, object]]) -> dict[str, object]:
    """Wrap output items in a Responses API response."""
    return {
        "output": output,
        "usage": {"input_tokens": 12, "output_tokens": 5},
    }


def test_chat_parses_tool_call(configured_client: None) -> None:
    """The client sends a Responses request and parses function arguments."""
    tool_schema = {"type": "function", "name": "read_file"}
    messages = [{"role": "user", "content": "Review this file."}]

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.url == "https://llm.example/v1/responses"
        assert request.headers["Authorization"] == "Bearer test-only-key"
        assert request.extensions["timeout"]["read"] == llm_client.DEFAULT_TIMEOUT
        body = json.loads(request.content)
        assert body["model"] == "test-model"
        assert body["input"] == messages
        assert body["tools"] == [tool_schema]
        return httpx.Response(
            200,
            json=response_body(
                [
                    {
                        "type": "function_call",
                        "call_id": "call-1",
                        "name": "read_file",
                        "arguments": '{"repo":"a/b","path":"x.py","ref":"main"}',
                    }
                ]
            ),
        )

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        result = HttpLLMClient(client=client).chat(messages, [tool_schema])

    assert result.content is None
    assert result.tool_calls[0].id == "call-1"
    assert result.tool_calls[0].name == "read_file"
    assert result.tool_calls[0].arguments == {
        "repo": "a/b",
        "path": "x.py",
        "ref": "main",
    }
    assert result.usage.prompt_tokens == 12
    assert result.usage.completion_tokens == 5


def test_chat_parses_plain_answer(configured_client: None) -> None:
    """The client extracts output text when there are no function calls."""
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=response_body(
                [
                    {
                        "type": "message",
                        "content": [{"type": "output_text", "text": "No issues found."}],
                    }
                ]
            ),
        )

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        result = HttpLLMClient(client=client).chat([], [])

    assert result.content == "No issues found."
    assert result.tool_calls == []


def test_chat_rejects_invalid_tool_arguments(configured_client: None) -> None:
    """Invalid JSON in a function call raises a clear client error."""
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=response_body(
                [
                    {
                        "type": "function_call",
                        "call_id": "call-1",
                        "name": "read_file",
                        "arguments": "{bad",
                    }
                ]
            ),
        )

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(LLMClientError, match="invalid JSON tool arguments"):
            HttpLLMClient(client=client).chat([], [])


def test_chat_non_200_error_hides_headers_and_key(
    configured_client: None,
) -> None:
    """HTTP errors identify status and path without exposing the API key."""
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, text="test-only-key should not be copied")

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(LLMClientError) as error:
            HttpLLMClient(client=client).chat([], [])

    assert "HTTP 503" in str(error.value)
    assert "/v1/responses" in str(error.value)
    assert "test-only-key" not in str(error.value)
    assert "Authorization" not in str(error.value)


def test_chat_requires_key_only_when_called(monkeypatch: pytest.MonkeyPatch) -> None:
    """A missing key errors on chat without sending a request."""
    requests: list[httpx.Request] = []
    monkeypatch.setattr(
        llm_client,
        "get_settings",
        lambda: Settings(llm_base_url="https://llm.example/v1", llm_model="test-model"),
    )

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={})

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(ValueError, match="LLM_API_KEY is required") as error:
            HttpLLMClient(client=client).chat([], [])

    assert requests == []
    assert "test-only-key" not in str(error.value)


def test_tool_schemas_use_responses_format() -> None:
    """Tool definitions match the Responses API and hide dry_run."""
    from codereviewbot.agent.schemas import TOOL_SCHEMAS

    assert [tool["name"] for tool in TOOL_SCHEMAS] == [
        "get_diff",
        "read_file",
        "post_comment",
    ]
    post_schema = TOOL_SCHEMAS[2]
    properties = post_schema["parameters"]["properties"]
    assert "dry_run" not in properties
