from __future__ import annotations

from agent_project.tracing import TraceEvent, TraceStore


def test_trace_store_round_trip(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("AGENT_STATE_DB_PATH", str(tmp_path / "agent.sqlite3"))
    trace = TraceStore(trace_id="trace-test")

    trace.start(user_input="hello", model="test-model", task_id="task-1")
    trace.add_event(TraceEvent(event_type="tool_call", tool="search_memories", args={"query": "x"}, ok=True))
    trace.finish("done", success=True)

    payload = TraceStore.get_trace("trace-test")
    assert payload is not None
    assert payload["task_id"] == "task-1"
    assert payload["model"] == "test-model"
    assert payload["final_answer"] == "done"
    assert payload["success"] is True
    assert payload["steps"][1]["tool"] == "search_memories"
