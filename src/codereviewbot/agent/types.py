"""Shared types for language-model clients."""

from typing import Any, Protocol

from pydantic import BaseModel, Field


class ToolCall(BaseModel):
    """A requested function call returned by a model."""

    id: str
    name: str
    arguments: dict[str, Any]


class Usage(BaseModel):
    """Token counts for one model response."""

    prompt_tokens: int
    completion_tokens: int


class ModelResponse(BaseModel):
    """Normalized text, tool calls, and usage from a model."""

    content: str | None
    tool_calls: list[ToolCall] = Field(default_factory=list)
    usage: Usage


class LLMClient(Protocol):
    """Interface shared by real and fake model clients."""

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> ModelResponse:
        """Return a model response for the supplied conversation."""
        ...
