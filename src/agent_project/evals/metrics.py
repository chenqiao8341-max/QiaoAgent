from __future__ import annotations

from collections.abc import Iterable

from agent_project.evals.schemas import EvalExample, EvalResult


def score_example(example: EvalExample) -> EvalResult:
    route_correct = None
    if example.expected_route:
        route_correct = example.actual_route == example.expected_route

    tool_call_correct = None
    if example.expected_tools:
        tool_call_correct = all(tool in example.actual_tools for tool in example.expected_tools)

    task_success = example.success
    if task_success is None:
        task_success = _infer_task_success(example, route_correct, tool_call_correct)

    grounded = example.grounded
    if grounded is None and example.expected_grounding:
        grounded = all(fragment in example.actual_answer for fragment in example.expected_grounding)

    return EvalResult(
        example_id=example.id,
        category=example.category,
        route_correct=route_correct,
        tool_call_correct=tool_call_correct,
        task_success=task_success,
        grounded=grounded,
        latency_seconds=example.latency_seconds,
        tokens_or_prompt_chars=example.tokens_or_prompt_chars or len(example.input),
        human_intervention_count=example.human_intervention_count,
        notes=_result_notes(example),
    )


def _result_notes(example: EvalExample) -> str:
    trace_id = example.metadata.get("trace_id")
    if trace_id:
        return f"trace_id={trace_id}"
    return ""


def _infer_task_success(
    example: EvalExample,
    route_correct: bool | None,
    tool_call_correct: bool | None,
) -> bool | None:
    if example.category == "route" and route_correct is not None:
        return route_correct
    if example.category in {"rag_qa", "local_file_suggestion", "codex_routing"}:
        checks = [
            value
            for value in (
                route_correct,
                tool_call_correct,
                _contains_expected_answer(example),
                example.grounded,
            )
            if value is not None
        ]
        return all(checks) if checks else None
    if example.category == "planning":
        checks = [
            value
            for value in (
                _contains_expected_answer(example),
                tool_call_correct,
            )
            if value is not None
        ]
        return all(checks) if checks else None
    return _contains_expected_answer(example)


def _contains_expected_answer(example: EvalExample) -> bool | None:
    if not example.expected_answer_contains:
        return None
    return all(fragment in example.actual_answer for fragment in example.expected_answer_contains)


def _rate(values: Iterable[bool | None]) -> float | None:
    materialized = [value for value in values if value is not None]
    if not materialized:
        return None
    return sum(1 for value in materialized if value) / len(materialized)


def summarize_results(results: list[EvalResult]) -> dict:
    summary = _summarize_flat(results)
    summary["by_category"] = _summarize_by_category(results)
    return summary


def _summarize_flat(results: list[EvalResult]) -> dict:
    return {
        "count": len(results),
        "route_accuracy": _rate(result.route_correct for result in results),
        "tool_call_accuracy": _rate(result.tool_call_correct for result in results),
        "task_success_rate": _rate(result.task_success for result in results),
        "groundedness": _rate(result.grounded for result in results),
        "latency_seconds": (
            sum(result.latency_seconds for result in results) / len(results) if results else 0.0
        ),
        "tokens_or_prompt_chars": (
            sum(result.tokens_or_prompt_chars for result in results) / len(results) if results else 0.0
        ),
        "human_intervention_count": sum(result.human_intervention_count for result in results),
    }


def _summarize_by_category(results: list[EvalResult]) -> dict:
    categories = sorted({result.category for result in results})
    return {
        category: _summarize_flat([result for result in results if result.category == category])
        for category in categories
    }
