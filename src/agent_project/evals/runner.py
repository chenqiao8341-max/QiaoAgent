from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Callable

from agent_project.agent import invoke_agent
from agent_project.evals.dataset import load_dataset
from agent_project.evals.metrics import score_example, summarize_results
from agent_project.evals.reports import write_report
from agent_project.evals.schemas import EvalExample
from agent_project.tools.storage import connect
from agent_project.tracing import TraceStore


def run_eval(dataset_path: str | Path, report_path: str | Path) -> dict:
    examples = load_dataset(dataset_path)
    results = [score_example(example) for example in examples]
    summary = summarize_results(results)
    write_report(report_path, summary, results)
    return summary


def run_live_eval(
    dataset_path: str | Path,
    report_path: str | Path,
    invoke_fn: Callable[[str], str] | None = None,
) -> dict:
    invoke = invoke_fn or invoke_agent
    completed_examples: list[EvalExample] = []
    for example in load_dataset(dataset_path):
        before = _latest_trace_id()
        answer = invoke(example.input)
        trace = _latest_trace_after(before, example.input)
        completed_examples.append(_example_with_actuals(example, answer, trace))

    results = [score_example(example) for example in completed_examples]
    summary = summarize_results(results)
    write_report(report_path, summary, results)
    return summary


def _latest_trace_id() -> str:
    with connect() as connection:
        row = connection.execute(
            "SELECT trace_id FROM agent_traces ORDER BY started_at DESC, rowid DESC LIMIT 1"
        ).fetchone()
    return str(row["trace_id"]) if row else ""


def _latest_trace_after(previous_trace_id: str, user_input: str) -> dict | None:
    with connect() as connection:
        if previous_trace_id:
            row = connection.execute(
                """
                SELECT trace_id FROM agent_traces
                WHERE trace_id != ? AND user_input = ?
                ORDER BY started_at DESC, rowid DESC
                LIMIT 1
                """,
                (previous_trace_id, user_input),
            ).fetchone()
        else:
            row = connection.execute(
                """
                SELECT trace_id FROM agent_traces
                WHERE user_input = ?
                ORDER BY started_at DESC, rowid DESC
                LIMIT 1
                """,
                (user_input,),
            ).fetchone()
    if row is None:
        return None
    return TraceStore.get_trace(str(row["trace_id"]))


def _example_with_actuals(example: EvalExample, answer: str, trace: dict | None) -> EvalExample:
    if trace is None:
        return example.model_copy(update={"actual_answer": answer})

    steps = trace.get("steps", [])
    route = _extract_route(steps)
    tools = [
        str(step.get("tool", ""))
        for step in steps
        if step.get("type") == "tool_call" and step.get("tool")
    ]
    human_intervention_count = sum(
        1
        for step in steps
        if "human approval" in str(step.get("content", "")).lower()
        or "needs_human=true" in str(step.get("content", "")).lower()
    )
    latency_seconds = float(trace.get("latency_ms", 0)) / 1000.0
    return example.model_copy(
        update={
            "actual_route": route,
            "actual_tools": tools,
            "actual_answer": answer,
            "success": bool(answer.strip()) if example.success is None else example.success,
            "latency_seconds": latency_seconds,
            "tokens_or_prompt_chars": len(example.input) + len(answer),
            "human_intervention_count": human_intervention_count,
        }
    )


def _extract_route(steps: list[dict]) -> str:
    for step in steps:
        if step.get("type") != "router":
            continue
        match = re.search(r"\broute=([a-z_]+)", str(step.get("content", "")))
        if match:
            return match.group(1)
    return ""


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the offline Agent Project eval suite.")
    parser.add_argument("--dataset", required=True, help="Path to JSONL eval dataset.")
    parser.add_argument("--report", required=True, help="Path to JSON or Markdown report output.")
    parser.add_argument(
        "--live",
        action="store_true",
        help="Actually invoke the agent and score the generated trace outputs.",
    )
    args = parser.parse_args()
    summary = run_live_eval(args.dataset, args.report) if args.live else run_eval(args.dataset, args.report)
    print(
        "Eval complete: "
        f"count={summary['count']} "
        f"route_accuracy={summary['route_accuracy']} "
        f"task_success_rate={summary['task_success_rate']} "
        f"report={args.report}"
    )


if __name__ == "__main__":
    main()
