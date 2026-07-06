from __future__ import annotations

import json

from agent_project.evals.dataset import load_dataset
from agent_project.evals.metrics import score_example, summarize_results
from agent_project.evals.runner import run_eval, run_live_eval
from agent_project.evals.schemas import EvalExample
from agent_project.tracing import TraceEvent, TraceStore


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


def test_live_eval_invokes_agent_and_scores_trace(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("AGENT_STATE_DB_PATH", str(tmp_path / "agent.sqlite3"))
    dataset_path = tmp_path / "dataset.jsonl"
    report_path = tmp_path / "report.json"
    dataset_path.write_text(
        EvalExample(
            id="case-live",
            category="route",
            input="route this live case",
            expected_route="self",
            expected_tools=["search_knowledge"],
            expected_answer_contains=["done"],
        ).model_dump_json()
        + "\n",
        encoding="utf-8",
    )

    def fake_invoke(user_input: str) -> str:
        trace = TraceStore()
        trace.start(user_input=user_input, model="fake")
        trace.add_event(TraceEvent(event_type="router", content="task_type=chat route=self"))
        trace.add_event(
            TraceEvent(
                event_type="tool_call",
                tool="search_knowledge",
                args={"query": "x"},
                ok=True,
            )
        )
        trace.finish("done", success=True)
        return "done"

    summary = run_live_eval(
        dataset_path,
        report_path,
        invoke_fn=fake_invoke,
        limit=1,
        category="route",
        save_traces=True,
    )
    payload = json.loads(report_path.read_text(encoding="utf-8"))

    assert summary["route_accuracy"] == 1.0
    assert summary["tool_call_accuracy"] == 1.0
    assert payload["summary"]["task_success_rate"] == 1.0
    assert payload["results"][0]["notes"]


def test_live_eval_filters_category_and_limit(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("AGENT_STATE_DB_PATH", str(tmp_path / "agent.sqlite3"))
    dataset_path = tmp_path / "dataset.jsonl"
    report_path = tmp_path / "report.json"
    dataset_path.write_text(
        "\n".join(
            [
                EvalExample(id="route-1", category="route", input="route", expected_route="self").model_dump_json(),
                EvalExample(id="rag-1", category="rag_qa", input="rag", expected_tools=["search_knowledge"]).model_dump_json(),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    calls: list[str] = []

    def fake_invoke(user_input: str) -> str:
        calls.append(user_input)
        trace = TraceStore()
        trace.start(user_input=user_input, model="fake")
        trace.add_event(TraceEvent(event_type="router", content="task_type=chat route=self"))
        trace.finish("answer", success=True)
        return "answer"

    summary = run_live_eval(
        dataset_path,
        report_path,
        invoke_fn=fake_invoke,
        limit=1,
        category="rag_qa",
        fresh_db=True,
    )

    assert calls == ["rag"]
    assert summary["count"] == 1


def test_eval_success_is_not_just_non_empty_answer() -> None:
    route_case = EvalExample(
        id="route-fail",
        category="route",
        input="route",
        expected_route="codex",
        actual_route="self",
        actual_answer="non-empty",
    )
    rag_case = EvalExample(
        id="rag-fail",
        category="rag_qa",
        input="rag",
        expected_tools=["search_knowledge"],
        actual_tools=[],
        actual_answer="non-empty",
    )

    assert score_example(route_case).task_success is False
    assert score_example(rag_case).task_success is False
