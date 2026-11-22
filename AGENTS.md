# AGENTS.md

## Project overview
CodeReviewBot is a Python agent that reviews GitHub pull requests and is scored on public benchmarks.

## Stack
Python 3.11+, Pydantic, HTTPX, python-dotenv, and pytest. Keep later frameworks and LLM integration out until requested.

## Run and test (Windows PowerShell)
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e ".[dev]"
pytest -v
```

## Code and security
- Use type hints and small functions. Use logging instead of `print`.
- Never commit secrets. Store local values in `.env` and keep it ignored by Git.
- Do not run destructive commands. Never delete outside the project, force-push, or run `git reset --hard`.
- Do not modify files outside the brief.

## Learning rule
The owner writes the agent loop, scoring function, and webhook signature check. Do not write these unless a brief explicitly asks.

## Response rule
Always end with a plain-English summary of at most 10 lines.
