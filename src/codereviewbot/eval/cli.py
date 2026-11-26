"""Command-line runner for local pull request evaluations."""

import argparse
import json
from pathlib import Path

from ..agent.llm_client import HttpLLMClient
from ..config import get_settings
from .dataset import load_cases
from .convert import convert_records, filter_language, read_records, write_cases
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
        nargs=2,
        metavar=("RESULTS_DIR", "OUT_CSV"),
        help="Export a seeded sample of saved judge decisions",
    )
    modes.add_argument("--convert", metavar="SOURCE", help="Convert a CSV or JSON export to cases JSONL")
    modes.add_argument("--handcheck-score", metavar="CSV", help="Score completed hand-check labels")
    modes.add_argument("--report", metavar="SUMMARY_JSON", help="Render a saved run summary")
    parser.add_argument("--cases", help="Path to the JSONL case dataset")
    parser.add_argument(
        "--mapping",
        help="JSON mapping for case_id, repo, pr_number, language, issues, path, line, description",
    )
    parser.add_argument("--language", help="Keep only cases for this language when converting")
    parser.add_argument("--out", help="Evaluation directory or converted JSONL output path")
    parser.add_argument("--limit", type=int, help="Evaluate at most N cases")
    parser.add_argument("--max-steps", type=int, default=8, help="Model turns per review")
    parser.add_argument("--seed", type=int, default=42, help="Hand-check sample seed")
    args = parser.parse_args()

    if args.convert:
        if not args.mapping or not args.out:
            parser.error("--convert requires --mapping and --out.")
        if args.limit is not None and args.limit < 1:
            parser.error("--limit must be at least 1")
        try:
            mapping = json.loads(Path(args.mapping).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            parser.error("--mapping must point to a readable JSON object.")
        if not isinstance(mapping, dict):
            parser.error("--mapping must contain a JSON object.")
        issues_field = mapping.get("issues")
        records = read_records(
            args.convert,
            issues_field=issues_field if isinstance(issues_field, str) else None,
        )
        converted = convert_records(records, mapping)
        if args.language:
            converted = filter_language(converted, args.language)
        if args.limit is not None:
            converted = converted[: args.limit]
    if args.handcheck_export:
        exported = export_handcheck(*args.handcheck_export, seed=args.seed)
        print(f"Exported {exported} judge decisions.")
        return
    if args.handcheck_score:
        print(json.dumps(agreement(args.handcheck_score), indent=2))
        return
    if args.report:
        print(results_table(args.report))
        return
    if not args.cases or not args.out:
        parser.error("--cases and --out are required when running an evaluation.")
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
        usefulness_judge=LLMUsefulnessJudge(client),
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
