from __future__ import annotations

import json
from pathlib import Path

from agent_project.evals.schemas import EvalResult


def write_report(path: str | Path, summary: dict, results: list[EvalResult]) -> None:
    output_path = Path(path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix.lower() == ".md":
        output_path.write_text(_markdown(summary, results), encoding="utf-8")
    else:
        payload = {
            "summary": summary,
            "results": [result.model_dump() for result in results],
        }
        output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _format_rate(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.3f}"


def _markdown(summary: dict, results: list[EvalResult]) -> str:
    lines = [
        "# Agent Eval Report",
        "",
        f"- count: {summary['count']}",
        f"- route_accuracy: {_format_rate(summary['route_accuracy'])}",
        f"- tool_call_accuracy: {_format_rate(summary['tool_call_accuracy'])}",
        f"- task_success_rate: {_format_rate(summary['task_success_rate'])}",
        f"- groundedness: {_format_rate(summary['groundedness'])}",
        f"- latency_seconds: {summary['latency_seconds']:.3f}",
        f"- tokens_or_prompt_chars: {summary['tokens_or_prompt_chars']:.1f}",
        f"- human_intervention_count: {summary['human_intervention_count']}",
        "",
        "## By Category",
        "",
    ]
    for category, category_summary in summary["by_category"].items():
        lines.append(
            "- "
            f"{category}: count={category_summary['count']}, "
            f"route={_format_rate(category_summary['route_accuracy'])}, "
            f"tools={_format_rate(category_summary['tool_call_accuracy'])}, "
            f"success={_format_rate(category_summary['task_success_rate'])}, "
            f"grounded={_format_rate(category_summary['groundedness'])}"
        )
    lines.extend(["", "## Results", ""])
    for result in results:
        lines.append(
            "- "
            f"{result.example_id} [{result.category}] "
            f"route={result.route_correct} tools={result.tool_call_correct} "
            f"success={result.task_success} grounded={result.grounded}"
        )
    return "\n".join(lines) + "\n"
