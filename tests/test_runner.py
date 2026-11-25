"""Tests for case execution, result files, and the pluggable judge."""

import json
from pathlib import Path
from collections.abc import Iterator
from typing import Any
from tempfile import TemporaryDirectory

import pytest

from codereviewbot.agent.fake_client import FakeLLMClient
from codereviewbot.agent.types import ModelResponse, Usage
from codereviewbot.config import Settings
from codereviewbot.eval import runner
from codereviewbot.eval.dataset import EvalCase, GroundTruthIssue
from codereviewbot.eval.judge import FakeJudge, LLMJudge
from codereviewbot.eval.noise import FakeUsefulnessJudge
from codereviewbot.models import ReviewComment, ReviewResult


@pytest.fixture
def tmp_path() -> Iterator[Path]:
    """Use project-local temporary output when Windows temp ACLs are restricted."""
    with TemporaryDirectory(dir=Path.cwd()) as temp_dir:
        yield Path(temp_dir)


def _case(case_id: str, repo: str = "owner/project") -> EvalCase:
    """Build a case with one known finding."""
    return EvalCase(
        case_id=case_id,
        repo=repo,
        pr_number=1,
        language="python",
        issues=[GroundTruthIssue(path="src/a.py", line=4, description="expected issue")],
    )


def test_runner_records_case_error_and_writes_summary(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Keep going after one failed case and save safe result files."""
    settings = Settings(llm_model="fake-model", llm_api_key="test-secret")
    monkeypatch.setattr(runner, "get_settings", lambda: settings)
    monkeypatch.setattr(runner.sys, "argv", ["python", "-m", "codereviewbot.eval.cli"])

    def fake_review(
        repo: str,
        pr_number: int,
        client: FakeLLMClient,
        **kwargs: Any,
    ) -> ReviewResult:
        if repo == "owner/failure":
            raise RuntimeError("failure included test-secret")
        return ReviewResult(
            summary="one match test-secret",
            comments=[
                ReviewComment(
                    path="src/a.py", line=4, severity="minor", message="test-secret expected issue"
                )
            ],
        )

    monkeypatch.setattr(runner, "run_review", fake_review)
    client = FakeLLMClient([])

    output_dir = tmp_path
    summary = runner.run_eval(
        [_case("success"), _case("failure", "owner/failure")],
        client,
        FakeJudge(),
        output_dir,
        usefulness_judge=FakeUsefulnessJudge(False),
    )

    assert summary["model"] == "fake-model"
    assert summary["case_count"] == 2
    assert summary["error_count"] == 1
    assert summary["command"] == ["python", "-m", "codereviewbot.eval.cli"]
    assert summary["noise"]["matched"] == 1
    assert (output_dir / "summary.json").exists()
    case_files = sorted(output_dir.glob("case-*.json"))
    assert len(case_files) == 2
    failed = json.loads(case_files[1].read_text(encoding="utf-8"))
    assert failed["error"].startswith("RuntimeError:")
    successful = json.loads(case_files[0].read_text(encoding="utf-8"))
    assert successful["judge_decisions"][0]["case_id"] == "success"
    assert successful["judge_decisions"][0]["comment_index"] == 0
    assert successful["judge_decisions"][0]["match"] is True
    assert "test-secret" not in json.dumps(summary) + "".join(
        path.read_text(encoding="utf-8") for path in case_files
    )
    assert json.loads((output_dir / "summary.json").read_text(encoding="utf-8")) == summary


def test_llm_judge_uses_fake_client_without_tools() -> None:
    """The judge accepts strict JSON and makes a tool-free fake request."""
    response = ModelResponse(
        content='{"match": true, "reason": "same issue"}',
        usage=Usage(prompt_tokens=1, completion_tokens=1),
    )
    client = FakeLLMClient([response])
    judge = LLMJudge(client)

    assert judge.matches("comment", "issue") is True
    assert client.messages_received[0][1]["role"] == "user"


def test_invalid_llm_judge_json_is_no_match(caplog: pytest.LogCaptureFixture) -> None:
    """Invalid judge JSON logs a generic warning and returns false."""
    response = ModelResponse(
        content="not json with private text",
        usage=Usage(prompt_tokens=1, completion_tokens=1),
    )
    judge = LLMJudge(FakeLLMClient([response]))

    assert judge.matches("comment", "issue") is False
    assert "private text" not in caplog.text
    assert "invalid JSON" in caplog.text
