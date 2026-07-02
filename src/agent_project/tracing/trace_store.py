from __future__ import annotations

import json
import time
import uuid
from datetime import datetime, timezone
from typing import Any

from agent_project.tools.storage import connect
from agent_project.tracing.event import TraceEvent


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class TraceStore:
    def __init__(self, trace_id: str | None = None):
        self.trace_id = trace_id or uuid.uuid4().hex
        self._started_monotonic = time.monotonic()
        self._event_index = 0

    def start(self, user_input: str, model: str = "", task_id: str = "") -> str:
        with connect() as connection:
            connection.execute(
                """
                INSERT INTO agent_traces(
                    trace_id, task_id, user_input, model, started_at
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (self.trace_id, task_id, user_input, model, _now()),
            )
        self.add_event(TraceEvent(event_type="user_input", content=user_input))
        return self.trace_id

    def add_event(self, event: TraceEvent) -> None:
        with connect() as connection:
            connection.execute(
                """
                INSERT INTO agent_trace_events(
                    trace_id, event_index, event_type, content, tool, args_json, ok, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    self.trace_id,
                    self._event_index,
                    event.event_type,
                    event.content,
                    event.tool,
                    json.dumps(event.args, ensure_ascii=False, sort_keys=True),
                    None if event.ok is None else int(event.ok),
                    _now(),
                ),
            )
        self._event_index += 1

    def finish(self, final_answer: str, success: bool = True, error_type: str = "") -> None:
        latency_ms = int((time.monotonic() - self._started_monotonic) * 1000)
        self.add_event(TraceEvent(event_type="final_answer", content=final_answer, ok=success))
        with connect() as connection:
            connection.execute(
                """
                UPDATE agent_traces
                SET final_answer = ?, latency_ms = ?, success = ?, error_type = ?, completed_at = ?
                WHERE trace_id = ?
                """,
                (
                    final_answer,
                    latency_ms,
                    int(success),
                    error_type,
                    _now(),
                    self.trace_id,
                ),
            )

    @staticmethod
    def get_trace(trace_id: str) -> dict[str, Any] | None:
        with connect() as connection:
            trace = connection.execute(
                "SELECT * FROM agent_traces WHERE trace_id = ?",
                (trace_id,),
            ).fetchone()
            if trace is None:
                return None
            events = connection.execute(
                """
                SELECT * FROM agent_trace_events
                WHERE trace_id = ?
                ORDER BY event_index ASC, id ASC
                """,
                (trace_id,),
            ).fetchall()

        return {
            "trace_id": trace["trace_id"],
            "task_id": trace["task_id"],
            "user_input": trace["user_input"],
            "model": trace["model"],
            "steps": [
                {
                    "type": event["event_type"],
                    "content": event["content"],
                    "tool": event["tool"],
                    "args": json.loads(event["args_json"] or "{}"),
                    "ok": None if event["ok"] is None else bool(event["ok"]),
                }
                for event in events
            ],
            "final_answer": trace["final_answer"],
            "latency_ms": trace["latency_ms"],
            "success": bool(trace["success"]),
            "error_type": trace["error_type"],
        }
