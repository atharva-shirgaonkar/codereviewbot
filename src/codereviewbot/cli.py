"""Command-line entry point for reviewing a GitHub pull request."""

import argparse
import json
import logging

from .agent.llm_client import HttpLLMClient
from .agent.loop import run_review
from .config import get_settings
from .logging_setup import setup_logging
from .usage import UsageTracker


def main() -> None:
    """Parse command-line options, run a review, and print its result."""
    parser = argparse.ArgumentParser(description="Review a GitHub pull request.")
    parser.add_argument("repo", help="Repository in owner/name form")
    parser.add_argument("pr_number", type=int, help="Pull request number")
    parser.add_argument(
        "--post-comments",
        action="store_true",
        help="Allow requested review comments to be posted",
    )
    args = parser.parse_args()

    settings = get_settings()
    setup_logging(settings.log_level)
    tracker = UsageTracker(settings)
    try:
        result = run_review(
            args.repo,
            args.pr_number,
            HttpLLMClient(),
            tracker=tracker,
            dry_run=not args.post_comments,
        )
    except Exception as error:
        logging.getLogger(__name__).error("Review failed: %s", error)
        raise SystemExit(1) from None

    print(result.model_dump_json(indent=2))
    print(json.dumps(tracker.summary(), indent=2))


if __name__ == "__main__":
    main()
