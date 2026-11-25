"""Command-line runner for local pull request evaluations."""

import argparse

from ..agent.llm_client import HttpLLMClient
from ..config import get_settings
from .dataset import load_cases
from .judge import LLMJudge
from .runner import run_eval


def main() -> None:
    """Load the selected cases, run them, and print aggregate metrics."""
    parser = argparse.ArgumentParser(description="Evaluate CodeReviewBot reviews.")
    parser.add_argument("--cases", required=True, help="Path to the JSONL case dataset")
    parser.add_argument("--out", required=True, help="Directory for evaluation results")
    parser.add_argument("--limit", type=int, help="Evaluate at most N cases")
    parser.add_argument("--max-steps", type=int, default=8, help="Model turns per review")
    args = parser.parse_args()
