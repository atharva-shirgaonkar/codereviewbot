"""HTTP client for xAI's OpenAI-compatible Responses API."""

import json
from typing import Any

import httpx
from pydantic import ValidationError

from ..config import get_settings
from .types import ModelResponse, ToolCall, Usage


DEFAULT_TIMEOUT = 30.0


class LLMClientError(RuntimeError):
    """Raised for API errors or invalid model responses."""


class HttpLLMClient:
    """Send Responses or Chat Completions requests using configured settings."""

    def __init__(
        self,
        client: httpx.Client | None = None,
        timeout: float = DEFAULT_TIMEOUT,
    ) -> None:
        self._client = client
        self._timeout = timeout

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> ModelResponse:
        """Request and normalize one model response."""
        settings = get_settings()
        style = settings.llm_api_style.strip().lower()
        if style not in {"responses", "chat_completions"}:
            raise LLMClientError(
                "LLM_API_STYLE must be 'responses' or 'chat_completions'."
            )
        api_key = settings.require_llm_api_key()
        if not settings.llm_base_url:
            raise LLMClientError("LLM_BASE_URL must be set before calling the model.")
        if not settings.llm_model:
            raise LLMClientError("LLM_MODEL must be set before calling the model.")

        if style == "chat_completions":
            url = f"{settings.llm_base_url.rstrip('/')}/chat/completions"
            payload = {
                "model": settings.llm_model,
                "messages": self._chat_messages(messages),
                "tools": self._chat_tools(tools),
            }
        else:
            url = f"{settings.llm_base_url.rstrip('/')}/responses"
            payload = {"model": settings.llm_model, "input": messages, "tools": tools}
        path = httpx.URL(url).path
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        response = self._post(url, headers, payload)
        if response.status_code != 200:
            raise LLMClientError(
                f"LLM API returned HTTP {response.status_code} for {path}."
            )
        try:
            response_data = response.json()
        except ValueError:
            raise LLMClientError(f"LLM API returned invalid JSON for {path}.") from None
        return self._parse_response(response_data)

    def _post(
        self,
        url: str,
        headers: dict[str, str],
        payload: dict[str, Any],
    ) -> httpx.Response:
        """Post JSON with a timeout, reusing an injected client when given."""
        if self._client is not None:
            return self._client.post(
                url,
                headers=headers,
                json=payload,
                timeout=self._timeout,
            )
        with httpx.Client(timeout=self._timeout) as client:
            return client.post(url, headers=headers, json=payload)

    @staticmethod
    def _parse_response(response_data: Any) -> ModelResponse:
        """Convert a Responses API result to the shared response model."""
        try:
            output = response_data["output"]
            usage_data = response_data["usage"]
            usage = Usage(
                prompt_tokens=usage_data["input_tokens"],
                completion_tokens=usage_data["output_tokens"],
            )
        except (KeyError, TypeError, ValidationError):
            raise LLMClientError("LLM API returned an incomplete response.") from None
        if not isinstance(output, list):
            raise LLMClientError("LLM API returned invalid response output.")

        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        for item in output:
            if not isinstance(item, dict):
                raise LLMClientError("LLM API returned an invalid output item.")
            if item.get("type") == "function_call":
                tool_calls.append(HttpLLMClient._parse_tool_call(item))
            elif item.get("type") == "message":
                text_parts.extend(HttpLLMClient._message_text(item))
        content = "\n".join(text_parts) or None
        return ModelResponse(content=content, tool_calls=tool_calls, usage=usage)

    @staticmethod
    def _message_text(item: dict[str, Any]) -> list[str]:
        """Extract text blocks from an assistant message output item."""
        content = item.get("content", [])
        if not isinstance(content, list):
            raise LLMClientError("LLM API returned invalid message content.")
        text_parts: list[str] = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "output_text":
                text = block.get("text")
                if isinstance(text, str):
                    text_parts.append(text)
        return text_parts

    @staticmethod
    def _parse_tool_call(item: dict[str, Any]) -> ToolCall:
        """Decode one function call and require arguments to be a JSON object."""
        try:
            call_id = item["call_id"]
            name = item["name"]
            arguments = json.loads(item["arguments"])
        except (KeyError, TypeError, ValueError):
            raise LLMClientError("LLM API returned invalid JSON tool arguments.") from None
        if not isinstance(arguments, dict):
            raise LLMClientError("LLM API tool arguments must decode to a JSON object.")
        return ToolCall(id=call_id, name=name, arguments=arguments)
