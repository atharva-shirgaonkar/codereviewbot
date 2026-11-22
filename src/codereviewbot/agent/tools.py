"""Small GitHub REST API tools for pull request review."""

import logging
from typing import Any
from urllib.parse import quote

import httpx

from ..config import get_settings


API_BASE_URL = "https://api.github.com"
API_VERSION = "2026-03-10"
REQUEST_TIMEOUT = 15.0
logger = logging.getLogger(__name__)


class GitHubApiError(RuntimeError):
    """Raised when GitHub returns an unexpected response."""


def _repository_path(repo: str) -> str:
    """Return a URL-safe API path for an owner/repository name."""
    parts = repo.strip("/").split("/")
    if len(parts) != 2 or not all(parts):
        raise ValueError("repo must use the 'owner/repository' format.")
    owner, name = parts
    return f"/repos/{quote(owner, safe='')}/{quote(name, safe='')}"


def _headers(accept: str) -> dict[str, str]:
    """Build GitHub headers without exposing the optional token."""
    headers = {
        "Accept": accept,
        "User-Agent": "CodeReviewBot",
        "X-GitHub-Api-Version": API_VERSION,
    }
    token = get_settings().github_token
    if token is not None:
        headers["Authorization"] = f"Bearer {token.get_secret_value()}"
    return headers


def _request(
    method: str,
    url: str,
    headers: dict[str, str],
    client: httpx.Client | None,
    params: dict[str, str] | None = None,
    json_body: dict[str, Any] | None = None,
) -> httpx.Response:
    """Send a request with a timeout, reusing an injected client when given."""
    if client is not None:
        return client.request(
            method,
            url,
            headers=headers,
            params=params,
            json=json_body,
            timeout=REQUEST_TIMEOUT,
        )

    with httpx.Client(timeout=REQUEST_TIMEOUT) as owned_client:
        return owned_client.request(
            method,
            url,
            headers=headers,
            params=params,
            json=json_body,
            timeout=REQUEST_TIMEOUT,
        )


def _check_status(response: httpx.Response, expected_status: int = 200) -> None:
    """Raise a readable error for any unexpected HTTP status."""
    if response.status_code != expected_status:
        raise GitHubApiError(
            f"GitHub API returned HTTP {response.status_code} "
            f"for {response.request.method} {response.request.url.path}."
        )


def get_diff(
    repo: str,
    pr_number: int,
    client: httpx.Client | None = None,
) -> str:
    """Fetch a pull request diff as text."""
    url = f"{API_BASE_URL}{_repository_path(repo)}/pulls/{pr_number}"
    response = _request(
        "GET",
        url,
        _headers("application/vnd.github.diff"),
        client,
    )
    _check_status(response)
    return response.text


def read_file(
    repo: str,
    path: str,
    ref: str,
    client: httpx.Client | None = None,
) -> str:
    """Fetch raw repository file contents at a branch, tag, or commit."""
    encoded_path = "/".join(quote(part, safe="") for part in path.split("/"))
    url = f"{API_BASE_URL}{_repository_path(repo)}/contents/{encoded_path}"
    response = _request(
        "GET",
        url,
        _headers("application/vnd.github.raw+json"),
        client,
        params={"ref": ref},
    )
    _check_status(response)
    return response.text


def post_comment(
    repo: str,
    pr_number: int,
    path: str,
    line: int,
    body: str,
    dry_run: bool = True,
    client: httpx.Client | None = None,
) -> dict[str, Any]:
    """Log a proposed inline comment, or post it when dry_run is disabled."""
    if dry_run:
        logger.info(
            "Dry run: would post review comment to %s pull request #%s at %s:%s: %s",
            repo,
            pr_number,
            path,
            line,
            body,
        )
        return {
            "dry_run": True,
            "repo": repo,
            "pr_number": pr_number,
            "path": path,
            "line": line,
            "body": body,
        }

    repository_path = _repository_path(repo)
    headers = _headers("application/vnd.github+json")
    pull_url = f"{API_BASE_URL}{repository_path}/pulls/{pr_number}"
    pull_response = _request("GET", pull_url, headers, client)
    _check_status(pull_response)
    pull_data = pull_response.json()
    try:
        commit_id = pull_data["head"]["sha"]
    except (KeyError, TypeError) as error:
        raise GitHubApiError("GitHub pull request response did not include head.sha.") from error

    comment_url = f"{API_BASE_URL}{repository_path}/pulls/{pr_number}/comments"
    comment_response = _request(
        "POST",
        comment_url,
        headers,
        client,
        json_body={
            "body": body,
            "commit_id": commit_id,
            "path": path,
            "line": line,
            "side": "RIGHT",
        },
    )
    _check_status(comment_response, expected_status=201)
    comment_data = comment_response.json()
    if not isinstance(comment_data, dict):
        raise GitHubApiError("GitHub API returned an invalid review comment response.")
    return comment_data
