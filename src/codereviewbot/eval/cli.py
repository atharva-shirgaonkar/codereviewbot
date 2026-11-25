"""Command-line runner for local pull request evaluations."""

import argparse
import json

from ..agent.llm_client import HttpLLMClient
from ..config import get_settings
from .dataset import load_cases
from .handcheck import agreement, export_handcheck
from .judge import LLMJudge
from .noise import LLMUsefulnessJudge
from .report import results_table
from .runner import run_eval


def main() -> None:
    """Load the selected cases, run them, and print aggregate metrics."""
    parser = argparse.ArgumentParser(description="Evaluate CodeReviewBot reviews.")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument(
        "--handcheck-export",
    parser.add_argument("--limit", type=int, help="Evaluate at most N cases")
    parser.add_argument("--max-steps", type=int, default=8, help="Model turns per review")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be at least 1")

    cases = load_cases(args.cases)
    if args.limit is not None:
        cases = cases[: args.limit]
    settings = get_settings()
    client = HttpLLMClient()
    summary = run_eval(
        cases,
        client,
        LLMJudge(client),
        args.out,
        max_steps=args.max_steps,
    )
    metrics = summary["aggregate"]
    print("Metric                 Value")
    print(f"Micro precision        {metrics['micro_precision']:.3f}")
    print(f"Micro recall           {metrics['micro_recall']:.3f}")
    print(f"Micro F1               {metrics['micro_f1']:.3f}")
    cost = metrics["mean_cost_usd"]
    if cost is None:
        cost_text = "unknown"
    else:
        cost_text = f"{cost:.6f}"
    print(f"Mean cost (USD)        {cost_text}")
    print(f"Mean latency (seconds) {metrics['mean_latency_seconds']:.3f}")
    print(f"Model                  {settings.llm_model or 'not configured'}")


if __name__ == "__main__":
    main()
