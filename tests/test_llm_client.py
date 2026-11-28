"""Tests for the HTTP LLM client using only mock transports."""

import json

import httpx
import pytest
from pydantic import SecretStr

from codereviewbot.agent import llm_client
from codereviewbot.agent.llm_client import HttpLLMClient, LLMClientError
from codereviewbot import config
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


def set_api_style(monkeypatch: pytest.MonkeyPatch, style: str, api_key: str | None = "test-only-key") -> None:
    """Set isolated test configuration for one API style."""
    settings = Settings(
        llm_api_key=SecretStr(api_key) if api_key is not None else None,
        llm_base_url="https://llm.example/v1",
        llm_model="test-model",
        llm_api_style=style,
    )
    monkeypatch.setattr(llm_client, "get_settings", lambda: settings)


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


def test_chat_completions_parses_plain_answer(monkeypatch: pytest.MonkeyPatch) -> None:
    """Send a Chat Completions request and parse its text and token usage."""
    set_api_style(monkeypatch, "chat_completions")

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/chat/completions"
        body = json.loads(request.content)
        assert body["messages"] == [{"role": "user", "content": "Review this."}]
        assert body["tools"] == []
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"role": "assistant", "content": "Looks good."}}],
                "usage": {"prompt_tokens": 8, "completion_tokens": 3},
            },
        )

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        response = HttpLLMClient(client=client).chat(
            [{"role": "user", "content": "Review this."}], []
        )

    assert response.content == "Looks good."
    assert response.tool_calls == []
    assert response.usage.prompt_tokens == 8
    assert response.usage.completion_tokens == 3


def test_chat_completions_parses_tool_call(monkeypatch: pytest.MonkeyPatch) -> None:
    """Convert tool schemas and parse a Chat Completions function call."""
    set_api_style(monkeypatch, "chat_completions")
    schema = {
        "type": "function",
        "name": "read_file",
        "description": "Read a file",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}}},
    }

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert body["tools"] == [
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": "Read a file",
                    "parameters": schema["parameters"],
                },
            }
        ]
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call-1",
                                    "type": "function",
                                    "function": {
                                        "name": "read_file",
                                        "arguments": '{"path":"README.md"}',
                                    },
                                }
                            ],
                        }
                    }
                ],
                "usage": {"prompt_tokens": 9, "completion_tokens": 4},
            },
        )

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        response = HttpLLMClient(client=client).chat([], [schema])

    assert response.content is None
    assert response.tool_calls[0].id == "call-1"
    assert response.tool_calls[0].name == "read_file"
    assert response.tool_calls[0].arguments == {"path": "README.md"}


def test_chat_completions_converts_grouped_tool_history(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Group internal function items into one assistant call message."""
    set_api_style(monkeypatch, "chat_completions")
    messages = [
        {"role": "system", "content": "Review."},
        {"role": "user", "content": "Inspect PR."},
        {
            "type": "function_call",
            "call_id": "call-a",
            "name": "get_diff",
            "arguments": '{"repo":"o/r","pr_number":1}',
        },
        {
            "type": "function_call",
            "call_id": "call-b",
            "name": "read_file",
            "arguments": '{"repo":"o/r","path":"a.py","ref":"main"}',
        },
        {"type": "function_call_output", "call_id": "call-a", "output": "diff"},
        {"type": "function_call_output", "call_id": "call-b", "output": "file"},
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert body["messages"] == [
            {"role": "system", "content": "Review."},
            {"role": "user", "content": "Inspect PR."},
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call-a",
                        "type": "function",
                        "function": {
                            "name": "get_diff",
                            "arguments": '{"repo":"o/r","pr_number":1}',
                        },
                    },
                    {
                        "id": "call-b",
                        "type": "function",
                        "function": {
                            "name": "read_file",
                            "arguments": '{"repo":"o/r","path":"a.py","ref":"main"}',
                        },
                    },
                ],
            },
            {"role": "tool", "tool_call_id": "call-a", "content": "diff"},
            {"role": "tool", "tool_call_id": "call-b", "content": "file"},
        ]
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"role": "assistant", "content": "done"}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            },
        )

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        result = HttpLLMClient(client=client).chat(messages, [])

    assert result.content == "done"


def test_chat_completions_rejects_invalid_arguments(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reject malformed tool argument JSON without returning source text."""
    set_api_style(monkeypatch, "chat_completions")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call-a",
                                    "function": {"name": "read_file", "arguments": "private-{"},
                                }
                            ],
                        }
                    }
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            },
        )

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(LLMClientError, match="invalid JSON tool arguments") as error:
            HttpLLMClient(client=client).chat([], [])

    assert "private" not in str(error.value)


def test_chat_completions_http_error_hides_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Chat endpoint HTTP errors expose only status and path."""
    set_api_style(monkeypatch, "chat_completions")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, text="test-only-key must not be shown")

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(LLMClientError) as error:
            HttpLLMClient(client=client).chat([], [])

    assert "HTTP 401" in str(error.value)
    assert "/v1/chat/completions" in str(error.value)
    assert "test-only-key" not in str(error.value)
    assert "Authorization" not in str(error.value)


def test_chat_completions_missing_key_errors_without_network(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A Chat Completions call requires a key only when invoked."""
    set_api_style(monkeypatch, "chat_completions", api_key=None)
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200)

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(ValueError, match="LLM_API_KEY is required"):
            HttpLLMClient(client=client).chat([], [])

    assert requests == []


def test_local_ollama_chat_completions_needs_no_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Local Ollama receives no Authorization header and needs no key."""
    settings = Settings(
        llm_api_key=SecretStr(""),
        llm_base_url="http://localhost:11434/v1",
        llm_model="local-model",
        llm_api_style="chat_completions",
    )
    monkeypatch.setattr(llm_client, "get_settings", lambda: settings)

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url == "http://localhost:11434/v1/chat/completions"
        assert "Authorization" not in request.headers
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "Local response."}}],
                "usage": {"prompt_tokens": 2, "completion_tokens": 2},
            },
        )

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        result = HttpLLMClient(client=client).chat([], [])

    assert result.content == "Local response."


def test_remote_chat_completions_without_key_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Remote Chat Completions still requires a non-empty API key."""
    set_api_style(monkeypatch, "chat_completions", api_key=None)
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200)

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(ValueError, match="LLM_API_KEY is required"):
            HttpLLMClient(client=client).chat([], [])

    assert requests == []


def test_local_host_uses_120_second_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """Local endpoints get more time for model startup and generation."""
    monkeypatch.setattr(
        llm_client,
        "get_settings",
        lambda: Settings(
            llm_api_key=SecretStr("test-only-key"),
            llm_base_url="http://127.0.0.1:11434/v1",
            llm_model="local-model",
        ),
    )

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.extensions["timeout"]["read"] == 120.0
        return httpx.Response(200, json=response_body([]))

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        HttpLLMClient(client=client).chat([], [])

def test_unknown_api_style_fails_without_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """An unsupported API style raises a clear error when the client is used."""
    set_api_style(monkeypatch, "mystery")
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200)

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(LLMClientError, match="LLM_API_STYLE must be"):
            HttpLLMClient(client=client).chat([], [])

    assert requests == []
