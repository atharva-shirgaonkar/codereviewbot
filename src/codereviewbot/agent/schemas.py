"""Function tool definitions for xAI's Responses API."""

TOOL_SCHEMAS: list[dict[str, object]] = [
    {
        "type": "function",
        "name": "get_diff",
        "description": "Fetch the unified diff for a GitHub pull request.",
        "parameters": {
            "type": "object",
            "properties": {
                "repo": {"type": "string", "description": "Repository as owner/name."},
                "pr_number": {"type": "integer", "description": "Pull request number."},
            },
            "required": ["repo", "pr_number"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "read_file",
        "description": "Read a repository file at a branch, tag, or commit ref.",
        "parameters": {
            "type": "object",
            "properties": {
                "repo": {"type": "string", "description": "Repository as owner/name."},
                "path": {"type": "string", "description": "Repository-relative file path."},
                "ref": {"type": "string", "description": "Branch, tag, or commit SHA."},
            },
            "required": ["repo", "path", "ref"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "post_comment",
        "description": "Post or preview an inline pull request review comment.",
        "parameters": {
            "type": "object",
            "properties": {
                "repo": {"type": "string", "description": "Repository as owner/name."},
                "pr_number": {"type": "integer", "description": "Pull request number."},
                "path": {"type": "string", "description": "Repository-relative file path."},
                "line": {"type": "integer", "description": "Line number in the diff."},
                "body": {"type": "string", "description": "Review comment text."},
            },
            "required": ["repo", "pr_number", "path", "line", "body"],
            "additionalProperties": False,
        },
    },
]
