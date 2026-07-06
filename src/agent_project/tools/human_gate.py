from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import Any

from langchain_core.tools import tool

from agent_project.tools.progress import emit_progress
from agent_project.tools.storage import connect


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "model_dump"):
        return _jsonable(value.model_dump())
    if hasattr(value, "__dict__"):
        return _jsonable(value.__dict__)
    return str(value)


def create_human_gate_request(user_input: str, reason: str, state: dict[str, Any]) -> str:
    gate_id = uuid.uuid4().hex
    now = _now()
    state_json = json.dumps(_jsonable(state), ensure_ascii=False, sort_keys=True)
    with connect() as connection:
        connection.execute(
            """
            INSERT INTO human_gate_requests(
                gate_id, status, user_input, reason, state_json, created_at, updated_at
            ) VALUES (?, 'pending', ?, ?, ?, ?, ?)
            """,
            (gate_id, user_input, reason, state_json, now, now),
        )
    return gate_id


def get_human_gate_record(gate_id: str) -> dict[str, Any] | None:
    clean_gate_id = gate_id.strip()
    if not clean_gate_id:
        return None
    with connect() as connection:
        row = connection.execute(
            """
            SELECT *
            FROM human_gate_requests
            WHERE gate_id = ?
            """,
            (clean_gate_id,),
        ).fetchone()
    return dict(row) if row else None


def approve_human_gate_record(gate_id: str, response: str = "approved") -> tuple[bool, str]:
    clean_gate_id = gate_id.strip()
    if not clean_gate_id:
        return False, "gate_id is required."

    record = get_human_gate_record(clean_gate_id)
    if record is None:
        return False, f"Human gate not found: {clean_gate_id}"
    if record["status"] not in {"pending", "approved"}:
        return False, f"Human gate {clean_gate_id} is not resumable: status={record['status']}"
    if record["status"] == "approved":
        return True, f"Human gate {clean_gate_id} was already approved."

    now = _now()
    with connect() as connection:
        connection.execute(
            """
            UPDATE human_gate_requests
            SET status = 'approved', response = ?, updated_at = ?, resolved_at = ?
            WHERE gate_id = ? AND status = 'pending'
            """,
            (response.strip() or "approved", now, now, clean_gate_id),
        )
    return True, f"Approved human gate {clean_gate_id}."


def load_human_gate_state(gate_id: str) -> dict[str, Any]:
    record = get_human_gate_record(gate_id)
    if record is None:
        raise ValueError(f"Human gate not found: {gate_id}")
    try:
        state = json.loads(record["state_json"] or "{}")
    except json.JSONDecodeError as exc:
        raise ValueError(f"Human gate state is invalid JSON: {gate_id}") from exc
    if not isinstance(state, dict):
        raise ValueError(f"Human gate state is not an object: {gate_id}")
    return state


def build_human_gate_resume_state(gate_id: str, response: str = "approved") -> dict[str, Any]:
    record = get_human_gate_record(gate_id)
    if record is None:
        raise ValueError(f"Human gate not found: {gate_id}")
    if record["status"] not in {"approved", "resumed"}:
        raise ValueError(
            f"Human gate must be approved before resume: {gate_id} status={record['status']}"
        )
    state = load_human_gate_state(gate_id)
    approval_response = response.strip() or record["response"] or "approved"
    return {
        **state,
        "status": "running",
        "needs_human": False,
        "human_gate_reason": "",
        "human_gate_id": gate_id,
        "approved_human_gate_id": gate_id,
        "approved_human_gate_reason": record["reason"],
        "human_gate_response": approval_response,
    }


def mark_human_gate_resumed(gate_id: str) -> None:
    now = _now()
    with connect() as connection:
        connection.execute(
            """
            UPDATE human_gate_requests
            SET status = 'resumed', updated_at = ?
            WHERE gate_id = ? AND status = 'approved'
            """,
            (now, gate_id.strip()),
        )


def mark_human_gate_completed(gate_id: str, status: str = "completed") -> None:
    clean_status = status.strip() or "completed"
    now = _now()
    with connect() as connection:
        connection.execute(
            """
            UPDATE human_gate_requests
            SET status = ?, updated_at = ?, resolved_at = COALESCE(resolved_at, ?)
            WHERE gate_id = ?
            """,
            (clean_status, now, now, gate_id.strip()),
        )


@tool
def list_human_gates(status: str = "pending", limit: int = 20) -> str:
    """List human gate approval requests stored in SQLite."""
    clean_status = status.strip() or "pending"
    limit = max(1, min(limit, 100))
    emit_progress(f"listing human gates status={clean_status}")
    with connect() as connection:
        rows = connection.execute(
            """
            SELECT gate_id, status, user_input, reason, created_at, resolved_at
            FROM human_gate_requests
            WHERE status = ?
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (clean_status, limit),
        ).fetchall()
    if not rows:
        return f"No human gates found with status={clean_status}."
    lines = [f"Human gates status={clean_status}:"]
    for row in rows:
        lines.append(
            f"- {row['gate_id']} status={row['status']} created={row['created_at']} "
            f"reason={row['reason']} input={row['user_input'][:120]}"
        )
    return "\n".join(lines)


@tool
def approve_human_gate(gate_id: str, response: str = "approved") -> str:
    """Mark a human gate request as approved. Resume execution is handled by the caller."""
    ok, message = approve_human_gate_record(gate_id, response=response)
    if not ok:
        return message
    return f"{message} Run `agent-chat approve {gate_id.strip()}` to resume the saved workflow."
