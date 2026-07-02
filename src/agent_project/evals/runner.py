from __future__ import annotations

import argparse
from pathlib import Path

from agent_project.evals.dataset import load_dataset
from agent_project.evals.metrics import score_example, summarize_results
from agent_project.evals.reports import write_report


def run_eval(dataset_path: str | Path, report_path: str | Path) -> dict:
    examples = load_dataset(dataset_path)
    results = [score_example(example) for example in examples]
    summary = summarize_results(results)
    write_report(report_path, summary, results)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the offline Agent Project eval suite.")
    parser.add_argument("--dataset", required=True, help="Path to JSONL eval dataset.")
    parser.add_argument("--report", required=True, help="Path to JSON or Markdown report output.")
    args = parser.parse_args()
    summary = run_eval(args.dataset, args.report)
    print(
        "Eval complete: "
        f"count={summary['count']} "
        f"route_accuracy={summary['route_accuracy']} "
        f"task_success_rate={summary['task_success_rate']} "
        f"report={args.report}"
    )


if __name__ == "__main__":
    main()
