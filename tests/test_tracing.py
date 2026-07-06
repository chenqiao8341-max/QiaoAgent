from __future__ import annotations

from agent_project.tracing import TraceEvent, TraceStore


def test_agent_result_status_can_mark_trace_failure() -> None:
    from agent_project.agent import _trace_success_from_result

    assert _trace_success_from_result({"status": "done"}) is True
    assert _trace_success_from_result({"status": "failed"}) is False


def test_trace_store_round_trip(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("AGENT_STATE_DB_PATH", str(tmp_path / "agent.sqlite3"))
    trace = TraceStore(trace_id="trace-test")

    trace.start(user_input="hello", model="test-model", task_id="task-1")
    trace.add_event(
        TraceEvent(
            event_type="tool_call",
            tool="search_memories",
            args={"query": "x"},
            ok=True,
            duration_ms=12,
            node="executor",
            tool_call_id="call-1",
            prompt_chars=42,
            raw_error="",
        )
    )
    trace.finish("done", success=True)

    payload = TraceStore.get_trace("trace-test")
    assert payload is not None
    assert payload["task_id"] == "task-1"
    assert payload["model"] == "test-model"
    assert payload["final_answer"] == "done"
    assert payload["success"] is True
    assert payload["steps"][1]["tool"] == "search_memories"
    assert payload["steps"][1]["duration_ms"] == 12
    assert payload["steps"][1]["node"] == "executor"
    assert payload["steps"][1]["tool_call_id"] == "call-1"
    assert payload["steps"][1]["prompt_chars"] == 42
