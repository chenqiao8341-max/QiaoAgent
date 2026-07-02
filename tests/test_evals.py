from __future__ import annotations

import json

from agent_project.evals.dataset import load_dataset
from agent_project.evals.metrics import score_example, summarize_results
from agent_project.evals.runner import run_eval
from agent_project.evals.schemas import EvalExample


def test_eval_metrics_score_expected_fields() -> None:
    example = EvalExample(
        id="case-1",
        category="route",
        input="hello",
        expected_route="self",
        expected_tools=["capture_work_message"],
        expected_answer_contains=["done"],
        expected_grounding=["source"],
        actual_route="self",
        actual_tools=["capture_work_message"],
        actual_answer="done with source",
    )

    result = score_example(example)
    summary = summarize_results([result])

    assert result.route_correct is True
    assert result.tool_call_correct is True
    assert result.task_success is True
    assert result.grounded is True
    assert summary["route_accuracy"] == 1.0


def test_eval_runner_writes_json_report(tmp_path) -> None:
    dataset_path = tmp_path / "dataset.jsonl"
    report_path = tmp_path / "report.json"
    dataset_path.write_text(
        EvalExample(
            id="case-1",
            category="route",
            input="route this",
            expected_route="self",
            actual_route="self",
            success=True,
        ).model_dump_json()
        + "\n",
        encoding="utf-8",
    )

    summary = run_eval(dataset_path, report_path)
    payload = json.loads(report_path.read_text(encoding="utf-8"))

    assert summary["count"] == 1
    assert payload["summary"]["route_accuracy"] == 1.0
    assert load_dataset(dataset_path)[0].id == "case-1"
