"""Scripted model client for tests and local experiments."""

from copy import deepcopy
from typing import Any

from .types import ModelResponse


class FakeLLMClient:
    """Return prebuilt responses in order and record received messages."""

    def __init__(self, scripted: list[ModelResponse]) -> None:
        self._scripted = list(scripted)
        self._next_response = 0
        self.messages_received: list[list[dict[str, Any]]] = []

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> ModelResponse:
        """Return the next response or fail when the script is exhausted."""
        self.messages_received.append(deepcopy(messages))
        if self._next_response >= len(self._scripted):
            raise RuntimeError("FakeLLMClient has no scripted responses left.")
        response = self._scripted[self._next_response]
        self._next_response += 1
        return response
