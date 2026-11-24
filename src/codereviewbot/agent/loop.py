"""Bounded tool-calling loop for reviewing one pull request."""

import json
import logging
from typing import Any

from ..models import ReviewParseError, ReviewResult, parse_review_json
from ..usage import UsageTracker
from . import tools
from .prompts import SYSTEM_PROMPT
from .schemas import TOOL_SCHEMAS
from .types import LLMClient, ToolCall


logger = logging.getLogger(__name__)


class StepLimitError(RuntimeError):
    """Raised when a review needs more model turns than allowed."""


def _initial_messages(repo: str, pr_number: int) -> list[dict[str, Any]]:
    """Create the initial Responses API input history."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"Review pull request #{pr_number} in {repo}.",
        },
    ]


def _run_tool(call: ToolCall, repo: str, dry_run: bool) -> Any:
    """Validate a requested tool and run it with the loop's safety settings."""
    arguments = dict(call.arguments)
    if arguments.get("repo") != repo:
        return {"error": "Tool repository must match the pull request repository."}

    if call.name == "get_diff":
        return tools.truncate_diff(tools.get_diff(**arguments))
    if call.name == "read_file":
        return tools.read_file(**arguments)
    if call.name == "post_comment":
        arguments["dry_run"] = dry_run
        return tools.post_comment(**arguments)
    return {"error": f"Unknown tool: {call.name}"}


def _tool_output(call: ToolCall, repo: str, dry_run: bool) -> str:
    """Run one call and encode either its value or a short error for the model."""
    try:
        result = _run_tool(call, repo, dry_run)
    except Exception as error:
        message = " ".join(str(error).split())[:300]
        result = {"error": message or "Tool failed."}
    return json.dumps(result, ensure_ascii=False, default=str)


def _append_tool_turn(
    messages: list[dict[str, Any]],
    calls: list[ToolCall],
    repo: str,
    dry_run: bool,
) -> None:
    """Append function-call and corresponding output items to full history."""
    for call in calls:
        messages.append(
            {
                "type": "function_call",
                "call_id": call.id,
                "name": call.name,
                "arguments": json.dumps(call.arguments),
            }
        )
    for call in calls:
        messages.append(
            {
                "type": "function_call_output",
                "call_id": call.id,
                "output": _tool_output(call, repo, dry_run),
            }
        )


def run_review(
    repo: str,
    pr_number: int,
    client: LLMClient,
    max_steps: int = 8,
    tracker: UsageTracker | None = None,
    dry_run: bool = True,
) -> ReviewResult:
    """Run a bounded review and return the validated findings."""
    if max_steps < 1:
        raise ValueError("max_steps must be at least 1.")

    messages = _initial_messages(repo, pr_number)
    usage_tracker = tracker or UsageTracker()
    parse_failures = 0

    for step in range(1, max_steps + 1):
        logger.info("Review step %d", step)
        response = client.chat(messages, TOOL_SCHEMAS)
        usage_tracker.record(response.usage)

        if response.tool_calls:
            for call in response.tool_calls:
                logger.info("Review step %d requested tool %s", step, call.name)
            _append_tool_turn(messages, response.tool_calls, repo, dry_run)
            continue

        try:
            return parse_review_json(response.content or "")
        except ReviewParseError:
            parse_failures += 1
            if parse_failures >= 2:
                raise
            messages.extend(
                [
                    {"role": "assistant", "content": response.content or ""},
                    {
                        "role": "user",
                        "content": "Your response was not valid review JSON. "
                        "Return a corrected JSON object matching ReviewResult only.",
                    },
                ]
            )

    raise StepLimitError(f"Review reached the limit of {max_steps} model steps.")
