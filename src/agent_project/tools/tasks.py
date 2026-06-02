from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from uuid import uuid4

from langchain_core.tools import tool


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
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds"))


_TASK_QUEUES: dict[str, TaskQueue] = {}


def _format_queue(queue: TaskQueue) -> str:
    lines = [f"Queue {queue.queue_id}: {queue.title}", f"Created: {queue.created_at}"]
    for index, step in enumerate(queue.steps, start=1):
        note = f" | note: {step.note}" if step.note else ""
        lines.append(f"{index}. [{step.status}] {step.description}{note}")
    return "\n".join(lines)


@tool
def create_task_queue(title: str, steps: list[str]) -> str:
    """Create an in-memory multi-step task queue for planning and progress tracking."""
    clean_steps = [step.strip() for step in steps if step.strip()]
    if not clean_steps:
        return "Cannot create a task queue without at least one step."

    queue = TaskQueue(
        queue_id=uuid4().hex[:8],
        title=title.strip() or "Untitled task queue",
        steps=[TaskStep(description=step) for step in clean_steps],
    )
    _TASK_QUEUES[queue.queue_id] = queue
    return _format_queue(queue)


@tool
def update_task_step(queue_id: str, step_index: int, status: str, note: str = "") -> str:
    """Update one step in a task queue. Status: pending, in_progress, completed, or blocked."""
    queue = _TASK_QUEUES.get(queue_id)
    if queue is None:
        return f"Task queue not found: {queue_id}"

    normalized_status = status.strip().lower()
    if normalized_status not in VALID_STATUSES:
        return f"Invalid status. Use one of: {', '.join(sorted(VALID_STATUSES))}."

    if step_index < 1 or step_index > len(queue.steps):
        return f"Invalid step_index. Use a value from 1 to {len(queue.steps)}."

    step = queue.steps[step_index - 1]
    step.status = normalized_status
    step.note = note.strip()
    return _format_queue(queue)


@tool
def get_task_queue(queue_id: str) -> str:
    """Return the current state of one task queue."""
    queue = _TASK_QUEUES.get(queue_id)
    if queue is None:
        return f"Task queue not found: {queue_id}"
    return _format_queue(queue)


@tool
def list_task_queues() -> str:
    """List all in-memory task queues in this process."""
    if not _TASK_QUEUES:
        return "No task queues exist in this process."

    lines = []
    for queue in _TASK_QUEUES.values():
        completed = sum(1 for step in queue.steps if step.status == "completed")
        lines.append(f"{queue.queue_id}: {queue.title} ({completed}/{len(queue.steps)} completed)")
    return "\n".join(lines)
