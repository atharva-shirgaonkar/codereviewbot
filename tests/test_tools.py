"""Tests for GitHub tools and GitHub token configuration."""

import logging

import httpx
import pytest

from codereviewbot import config
from codereviewbot.agent import tools
from codereviewbot.config import Settings, get_settings


@pytest.fixture(autouse=True)
def settings_without_dotenv(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep tool tests independent of any local .env file."""
    monkeypatch.setattr(tools, "get_settings", lambda: Settings())


def make_client(handler: httpx.MockTransport) -> httpx.Client:
    """Create a client that routes requests only to a mock transport."""
    return httpx.Client(transport=handler)


def test_get_diff_success() -> None:
    """The diff tool uses GitHub's diff media type and returns text."""
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "GET"
        assert request.url.path == "/repos/example/project/pulls/7"
        assert request.headers["Accept"] == "application/vnd.github.diff"
        assert request.headers["User-Agent"] == "CodeReviewBot"
        assert request.headers["X-GitHub-Api-Version"] == tools.API_VERSION
        assert request.extensions["timeout"]["read"] == tools.REQUEST_TIMEOUT
        return httpx.Response(200, text="diff --git a/file.py b/file.py")

    with make_client(httpx.MockTransport(handler)) as client:
        result = tools.get_diff("example/project", 7, client=client)

    assert result == "diff --git a/file.py b/file.py"


def test_read_file_success() -> None:
    """The file tool requests raw content and passes the requested ref."""
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "GET"
        assert request.url.path == "/repos/example/project/contents/src/app.py"
        assert request.url.params["ref"] == "feature-branch"
        assert request.headers["Accept"] == "application/vnd.github.raw+json"
        return httpx.Response(200, text="print('hello')")

    with make_client(httpx.MockTransport(handler)) as client:
        result = tools.read_file(
            "example/project",
            "src/app.py",
            "feature-branch",
            client=client,
        )

    assert result == "print('hello')"


def test_get_diff_raises_clear_error_on_404() -> None:
    """A missing pull request produces a clear API error."""
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404)

    with make_client(httpx.MockTransport(handler)) as client:
        with pytest.raises(tools.GitHubApiError, match="HTTP 404"):
            tools.get_diff("example/project", 7, client=client)


def test_read_file_raises_clear_error_on_404() -> None:
    """A missing file produces a clear API error."""
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404)

    with make_client(httpx.MockTransport(handler)) as client:
        with pytest.raises(tools.GitHubApiError, match="HTTP 404"):
            tools.read_file("example/project", "missing.py", "main", client=client)


def test_post_comment_dry_run_does_not_call_network(caplog: pytest.LogCaptureFixture) -> None:
    """Dry-run logs the proposed comment and never reaches the transport."""
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(201, json={})

    with make_client(httpx.MockTransport(handler)) as client:
        with caplog.at_level(logging.INFO, logger=tools.__name__):
            result = tools.post_comment(
                "example/project",
                7,
                "src/app.py",
                12,
                "Consider handling this case.",
                client=client,
            )

    assert result["dry_run"] is True
    assert "src/app.py:12" in caplog.text
    assert "Consider handling this case." in caplog.text
    assert requests == []


def test_post_comment_success_uses_head_commit() -> None:
    """Posting an inline comment fetches and uses the pull request head SHA."""
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.method == "GET":
            return httpx.Response(200, json={"head": {"sha": "abc123"}})
        return httpx.Response(201, json={"id": 42, "body": "Looks good."})

    with make_client(httpx.MockTransport(handler)) as client:
        result = tools.post_comment(
            "example/project",
            7,
            "src/app.py",
            12,
            "Looks good.",
            dry_run=False,
            client=client,
        )

    assert [request.method for request in requests] == ["GET", "POST"]
    assert requests[1].url.path == "/repos/example/project/pulls/7/comments"
    assert requests[1].read() == (
        b'{"body":"Looks good.","commit_id":"abc123","path":"src/app.py",'
        b'"line":12,"side":"RIGHT"}'
    )
    assert result["id"] == 42


def test_github_token_is_loaded_as_secret(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """GITHUB_TOKEN is available to callers without being exposed by repr."""
    monkeypatch.setattr(config, "load_dotenv", lambda: False)
    monkeypatch.setenv("GITHUB_TOKEN", "test-only-token")

    settings = get_settings()

    assert settings.github_token is not None
    assert settings.github_token.get_secret_value() == "test-only-token"
    assert "test-only-token" not in repr(settings)


def test_post_comment_raises_clear_error_on_404() -> None:
    """A rejected comment post reports the GitHub status clearly."""
    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            return httpx.Response(200, json={"head": {"sha": "abc123"}})
        return httpx.Response(404)

    with make_client(httpx.MockTransport(handler)) as client:
        with pytest.raises(tools.GitHubApiError, match="HTTP 404"):
            tools.post_comment(
                "example/project",
                7,
                "src/app.py",
                12,
                "Please fix this.",
                dry_run=False,
                client=client,
            )


def test_github_error_includes_message_and_truncates() -> None:
    """GitHub JSON error messages are included up to 200 characters."""
    message = "GitHub refused the request: " + "x" * 220 + " END_MARKER"

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(403, json={"message": message})

    with make_client(httpx.MockTransport(handler)) as client:
        with pytest.raises(tools.GitHubApiError) as error:
            tools.get_diff("example/project", 7, client=client)

    assert message[:200] in str(error.value)
    assert "END_MARKER" not in str(error.value)


def test_github_error_redacts_authorization_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A token echoed by an error response is redacted from the exception."""
    from pydantic import SecretStr

    secret = "mock-github-token"
    monkeypatch.setattr(
        tools,
        "get_settings",
        lambda: Settings(github_token=SecretStr(secret)),
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"message": f"Invalid token {secret}."})

    with make_client(httpx.MockTransport(handler)) as client:
        with pytest.raises(tools.GitHubApiError) as error:
            tools.get_diff("example/project", 7, client=client)

    assert secret not in str(error.value)
    assert "Authorization" not in str(error.value)


def test_post_comment_commit_id_skips_pull_request_get() -> None:
    """An explicit commit SHA avoids the extra pull request lookup."""
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(201, json={"id": 42})

    with make_client(httpx.MockTransport(handler)) as client:
        result = tools.post_comment(
            "example/project",
            7,
            "src/app.py",
            12,
            "Looks good.",
            dry_run=False,
            client=client,
            commit_id="explicit-sha",
        )

    assert len(requests) == 1
    assert requests[0].method == "POST"
    assert requests[0].read() == (
        b'{"body":"Looks good.","commit_id":"explicit-sha",'
        b'"path":"src/app.py","line":12,"side":"RIGHT"}'
    )
    assert result["id"] == 42


def test_truncate_diff_returns_short_diff_unchanged() -> None:
    """A short diff is returned without a truncation marker."""
    diff = "short diff"

    assert tools.truncate_diff(diff, max_chars=20) == diff


def test_truncate_diff_marks_removed_character_count() -> None:
    """A long diff is cut and reports exactly how many characters were removed."""
    result = tools.truncate_diff("abcdefghij", max_chars=4)

    assert result.startswith("abcd\n\n")
    assert "6 characters removed" in result
