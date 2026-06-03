from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from uuid import uuid4

from langchain_core.tools import tool

from agent_project.tools.progress import emit_progress
from agent_project.tools.storage import connect, state_db_path


VALID_STATUSES = {"pending", "in_progress", "completed", "blocked"}


@dataclass
class TaskStep:
    description: str
    status: str = "pending"
    note: str = ""


@dataclass
class TaskQueue:
    queue_id: str
    title: str
    steps: list[TaskStep]
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")
    )


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _format_queue(queue: TaskQueue) -> str:
    lines = [f"Queue {queue.queue_id}: {queue.title}", f"Created: {queue.created_at}"]
    for index, step in enumerate(queue.steps, start=1):
        note = f" | note: {step.note}" if step.note else ""
        lines.append(f"{index}. [{step.status}] {step.description}{note}")
    return "\n".join(lines)


def _load_queue(queue_id: str) -> TaskQueue | None:
    with connect() as connection:
        queue_row = connection.execute(
            "SELECT * FROM task_queues WHERE queue_id = ?",
            (queue_id,),
        ).fetchone()
        if queue_row is None:
            return None

        step_rows = connection.execute(
            "SELECT * FROM task_steps WHERE queue_id = ? ORDER BY step_index ASC",
            (queue_id,),
        ).fetchall()

    return TaskQueue(
        queue_id=queue_row["queue_id"],
        title=queue_row["title"],
        created_at=queue_row["created_at"],
        steps=[
            TaskStep(
                description=row["description"],
                status=row["status"],
                note=row["note"],
            )
            for row in step_rows
        ],
    )


@tool
def create_task_queue(title: str, steps: list[str]) -> str:
    """Create a persistent SQLite-backed multi-step task queue."""
    emit_progress(f"creating task queue: {title} ({len(steps)} proposed steps)")
    clean_steps = [step.strip() for step in steps if step.strip()]
    if not clean_steps:
        emit_progress("task queue creation skipped: no valid steps")
        return "Cannot create a task queue without at least one step."

    queue_id = uuid4().hex[:8]
    queue_title = title.strip() or "Untitled task queue"
    now = _now()
    with connect() as connection:
        connection.execute(
            """
            INSERT INTO task_queues(queue_id, title, created_at, updated_at)
            VALUES (?, ?, ?, ?)
            """,
            (queue_id, queue_title, now, now),
        )
        connection.executemany(
            """
            INSERT INTO task_steps(queue_id, step_index, description, status, note, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            [
                (queue_id, index, step, "pending", "", now)
                for index, step in enumerate(clean_steps, start=1)
            ],
        )

    queue = _load_queue(queue_id)
    emit_progress(f"task queue created: {queue_id} ({len(clean_steps)} steps, {state_db_path()})")
    return _format_queue(queue) if queue else f"Task queue created: {queue_id}"


@tool
def update_task_step(queue_id: str, step_index: int, status: str, note: str = "") -> str:
    """Update one persistent task queue step.

    Status: pending, in_progress, completed, or blocked.
    """
    emit_progress(f"updating task queue {queue_id}: step {step_index} -> {status}")
    normalized_status = status.strip().lower()
    if normalized_status not in VALID_STATUSES:
        emit_progress(f"task queue update failed, invalid status: {status}")
        return f"Invalid status. Use one of: {', '.join(sorted(VALID_STATUSES))}."

    now = _now()
    with connect() as connection:
        queue_exists = connection.execute(
            "SELECT 1 FROM task_queues WHERE queue_id = ?",
            (queue_id,),
        ).fetchone()
        if queue_exists is None:
            emit_progress(f"task queue update failed, not found: {queue_id}")
            return f"Task queue not found: {queue_id}"

        cursor = connection.execute(
            """
            UPDATE task_steps
            SET status = ?, note = ?, updated_at = ?
            WHERE queue_id = ? AND step_index = ?
            """,
            (normalized_status, note.strip(), now, queue_id, step_index),
        )
        if cursor.rowcount == 0:
            count = connection.execute(
                "SELECT COUNT(*) AS count FROM task_steps WHERE queue_id = ?",
                (queue_id,),
            ).fetchone()["count"]
            emit_progress(f"task queue update failed, invalid step: {step_index}")
            return f"Invalid step_index. Use a value from 1 to {count}."

        connection.execute(
            "UPDATE task_queues SET updated_at = ? WHERE queue_id = ?",
            (now, queue_id),
        )

    queue = _load_queue(queue_id)
    emit_progress(f"task step updated: {queue_id} step {step_index} is {normalized_status}")
    return _format_queue(queue) if queue else f"Task queue not found: {queue_id}"


@tool
def get_task_queue(queue_id: str) -> str:
    """Return the current state of one persistent task queue."""
    emit_progress(f"reading task queue: {queue_id}")
    queue = _load_queue(queue_id)
    if queue is None:
        emit_progress(f"task queue read failed, not found: {queue_id}")
        return f"Task queue not found: {queue_id}"
    return _format_queue(queue)


@tool
def list_task_queues() -> str:
    """List persistent task queues stored in SQLite."""
    emit_progress(f"listing task queues from {state_db_path()}")
    with connect() as connection:
        rows = connection.execute(
            """
            SELECT
                q.queue_id,
                q.title,
                COUNT(s.step_index) AS total,
                SUM(CASE WHEN s.status = ? THEN 1 ELSE 0 END) AS completed
            FROM task_queues q
            LEFT JOIN task_steps s ON s.queue_id = q.queue_id
            GROUP BY q.queue_id, q.title, q.updated_at
            ORDER BY q.updated_at DESC, q.created_at DESC
            """,
            ("completed",),
        ).fetchall()

    if not rows:
        return "No task queues exist in SQLite storage."

    lines = []
    for row in rows:
        completed = row["completed"] or 0
        lines.append(f"{row['queue_id']}: {row['title']} ({completed}/{row['total']} completed)")
    return "\n".join(lines)
