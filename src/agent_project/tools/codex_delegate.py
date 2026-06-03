from __future__ import annotations

import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from langchain_core.tools import tool

from agent_project.tools.progress import emit_progress
from agent_project.tools.storage import connect


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _enabled() -> bool:
    return os.getenv("AGENT_ENABLE_CODEX_DELEGATION", "false").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _workspace_root() -> Path:
    configured = os.getenv("AGENT_WORKSPACE_ROOT")
    if configured:
        return Path(configured).expanduser().resolve()
    return Path.cwd().resolve()


def rewrite_task_text(task: str, context: str = "", expected_output: str = "") -> str:
    task = task.strip()
    context = context.strip()
    expected_output = expected_output.strip()
    parts = [
        "You are Codex working as a coding agent for Qiao.",
        "Goal:",
        task,
        "Working style:",
        "- Inspect the repository before editing.",
        "- Keep changes scoped to the requested task.",
        "- Preserve unrelated user changes.",
        "- Run focused verification when possible.",
        "- Report changed files, verification results, and any blockers.",
    ]
    if context:
        parts.extend(["Additional context:", context])
    if expected_output:
        parts.extend(["Expected output:", expected_output])
    return "\n".join(parts)


@tool
def rewrite_task_for_codex(task: str, context: str = "", expected_output: str = "") -> str:
    """Rewrite a user task into clear instructions suitable for Codex."""
    emit_progress("rewriting task for Codex")
    return rewrite_task_text(task, context=context, expected_output=expected_output)


@tool
def run_codex_task(
    task: str,
    context: str = "",
    expected_output: str = "",
    cwd: str = "",
    timeout_seconds: int = 1800,
    sandbox: str = "workspace-write",
    approval_policy: str = "never",
    model: str = "",
    dry_run: bool = False,
) -> str:
    """Rewrite a task, send it to codex exec, and store the result in SQLite."""
    rewritten = rewrite_task_text(task, context=context, expected_output=expected_output)
    workdir = Path(cwd).expanduser().resolve() if cwd else _workspace_root()
    created_at = _now()

    if dry_run:
        return rewritten
    if not _enabled():
        return "Codex delegation is disabled by AGENT_ENABLE_CODEX_DELEGATION."
    if not workdir.exists() or not workdir.is_dir():
        return f"Codex working directory is invalid: {workdir}"

    command = [
        "codex",
        "exec",
        "-C",
        str(workdir),
        "-s",
        sandbox,
        "-a",
        approval_policy,
    ]
    if model.strip():
        command.extend(["-m", model.strip()])
    command.append(rewritten)

    emit_progress(f"starting Codex task in {workdir}: {task[:80]}")
    status = "running"
    with connect() as connection:
        cursor = connection.execute(
            """
            INSERT INTO codex_tasks(original_task, rewritten_task, cwd, status, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (task, rewritten, str(workdir), status, created_at),
        )
        task_id = cursor.lastrowid

    timeout = max(30, min(timeout_seconds, 24 * 60 * 60))
    try:
        completed = subprocess.run(
            command,
            cwd=workdir,
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
        status = "completed" if completed.returncode == 0 else "failed"
        stdout = completed.stdout or ""
        stderr = completed.stderr or ""
        exit_code = completed.returncode
    except subprocess.TimeoutExpired as exc:
        status = "timeout"
        stdout = exc.stdout or ""
        stderr = exc.stderr or ""
        exit_code = None

    completed_at = _now()
    with connect() as connection:
        connection.execute(
            """
            UPDATE codex_tasks
            SET status = ?, exit_code = ?, stdout = ?, stderr = ?, completed_at = ?
            WHERE id = ?
            """,
            (status, exit_code, stdout, stderr, completed_at, task_id),
        )

    emit_progress(f"Codex task {task_id} finished with status={status}")
    output = stdout.strip() or stderr.strip() or "(no output)"
    if len(output) > 20000:
        output = output[:20000] + "\n\n[truncated after 20000 characters]"
    return f"Codex task {task_id} status={status} exit_code={exit_code}\n\n{output}"


@tool
def list_codex_tasks(limit: int = 10) -> str:
    """List recent delegated Codex tasks."""
    limit = max(1, min(limit, 50))
    with connect() as connection:
        rows = connection.execute(
            """
            SELECT id, status, cwd, created_at, completed_at, original_task
            FROM codex_tasks
            ORDER BY created_at DESC, id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    if not rows:
        return "No Codex tasks stored."
    return "\n".join(
        f"{row['id']} [{row['status']}] {row['created_at']} {row['original_task'][:120]}"
        for row in rows
    )


@tool
def get_codex_task(task_id: int, include_output: bool = True) -> str:
    """Return one delegated Codex task record."""
    with connect() as connection:
        row = connection.execute("SELECT * FROM codex_tasks WHERE id = ?", (task_id,)).fetchone()
    if row is None:
        return f"Codex task not found: {task_id}"

    lines = [
        f"Codex task {row['id']} [{row['status']}]",
        f"cwd: {row['cwd']}",
        f"created_at: {row['created_at']}",
        f"completed_at: {row['completed_at']}",
        f"original_task:\n{row['original_task']}",
        f"rewritten_task:\n{row['rewritten_task']}",
    ]
    if include_output:
        lines.append(f"stdout:\n{row['stdout']}")
        if row["stderr"]:
            lines.append(f"stderr:\n{row['stderr']}")
    return "\n\n".join(lines)
