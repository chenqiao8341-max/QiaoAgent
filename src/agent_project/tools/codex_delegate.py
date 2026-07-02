from __future__ import annotations

import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from langchain_core.tools import tool

from agent_project.tools.progress import emit_progress
from agent_project.tools.storage import connect

CODEX_COMMAND_PREFIXES = ("codex", "codex-proxy-")


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


def _coerce_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _configured_codex_command() -> str:
    return os.getenv("AGENT_CODEX_COMMAND", "codex-proxy-1").strip() or "codex-proxy-1"


def _resolve_codex_command(command_name: str) -> str | None:
    command_name = command_name.strip() or _configured_codex_command()
    if "/" in command_name:
        path = Path(command_name).expanduser()
        if path.is_file() and os.access(path, os.X_OK):
            return str(path)
        return None
    return shutil.which(command_name)


def _looks_like_codex_command(value: str) -> bool:
    name = Path(value.strip()).name
    return name == "codex" or name.startswith("codex-proxy-")


def _normalize_command_and_model(codex_command: str, model: str) -> tuple[str, str, str]:
    command_name = codex_command.strip() or _configured_codex_command()
    model_name = model.strip()
    warning = ""
    if model_name and _looks_like_codex_command(model_name):
        if not codex_command.strip():
            command_name = model_name
        model_name = ""
        warning = (
            "model looked like a Codex command, so it was treated as codex_command "
            "and was not passed with -m"
        )
    return command_name, model_name, warning


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
    codex_command: str = "",
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

    command_name, model_name, command_warning = _normalize_command_and_model(
        codex_command, model
    )
    command_path = _resolve_codex_command(command_name)
    if command_path is None:
        return f"Codex command not found or not executable: {command_name}"

    command = [
        command_path,
        "exec",
        "-C",
        str(workdir),
        "-s",
        sandbox,
        "--color",
        "never",
    ]
    if model_name:
        command.extend(["-m", model_name])
    command.append(rewritten)

    emit_progress(f"starting Codex task via {command_name} in {workdir}: {task[:80]}")
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
            input="",
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
        stdout = _coerce_text(exc.stdout)
        stderr = _coerce_text(exc.stderr)
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
    notes = []
    if command_warning:
        notes.append(command_warning)
    note = ""
    if approval_policy.strip() and approval_policy.strip() != "never":
        notes.append(
            "approval_policy is kept for compatibility, but this Codex CLI "
            "does not support -a/--approval-policy; the value was not passed."
        )
    if notes:
        note = "\n\nNote: " + " ".join(notes)
    return (
        f"Codex task {task_id} status={status} exit_code={exit_code} "
        f"command={command_name}\n\n{output}{note}"
    )



def _extract_session_id(*outputs: str) -> str:
    for output in outputs:
        for line in output.splitlines():
            line = line.strip()
            if line.lower().startswith("session id:"):
                return line.split(":", 1)[1].strip()
    return ""


def _trim_output(output: str, max_chars: int) -> str:
    if max_chars <= 0 or len(output) <= max_chars:
        return output
    return output[:max_chars] + f"\n\n[truncated after {max_chars} characters]"


def _run_codex_subprocess(
    command: list[str], workdir: Path, timeout_seconds: int
) -> tuple[str, int | None, str, str]:
    try:
        completed = subprocess.run(
            command,
            cwd=workdir,
            input="",
            text=True,
            capture_output=True,
            timeout=timeout_seconds,
            check=False,
        )
        status = "completed" if completed.returncode == 0 else "failed"
        return status, completed.returncode, completed.stdout or "", completed.stderr or ""
    except subprocess.TimeoutExpired as exc:
        return "timeout", None, _coerce_text(exc.stdout), _coerce_text(exc.stderr)


def _log_codex_session_interaction(
    *,
    action: str,
    session_id: str,
    command_name: str,
    cwd: Path,
    prompt: str,
    status: str,
    exit_code: int | None,
    stdout: str,
    stderr: str,
) -> None:
    with connect() as connection:
        connection.execute(
            """
            INSERT INTO codex_session_interactions(
                action, codex_session_id, command_name, cwd, prompt, status,
                exit_code, stdout, stderr, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                action,
                session_id,
                command_name,
                str(cwd),
                prompt,
                status,
                exit_code,
                stdout,
                stderr,
                _now(),
            ),
        )


def _format_codex_session_result(
    *,
    action: str,
    command_name: str,
    session_id: str,
    status: str,
    exit_code: int | None,
    stdout: str,
    stderr: str,
    command_warning: str,
    max_output_chars: int,
) -> str:
    output = ""
    if stdout.strip():
        output += f"stdout:\n{stdout.strip()}\n"
    if stderr.strip():
        output += f"stderr:\n{stderr.strip()}\n"
    output = output.strip() or "(no output)"
    output = _trim_output(output, max(1000, min(max_output_chars, 200000)))
    note = f"\n\nNote: {command_warning}" if command_warning else ""
    return (
        f"Codex {action} status={status} exit_code={exit_code} "
        f"command={command_name} session_id={session_id or '(unknown)'}\n\n"
        f"{output}{note}"
    )


@tool
def start_codex_session(
    prompt: str,
    cwd: str = "",
    timeout_seconds: int = 1800,
    sandbox: str = "workspace-write",
    model: str = "",
    codex_command: str = "",
    max_output_chars: int = 60000,
) -> str:
    """Start a Codex exec session and return its session_id plus output."""
    if not _enabled():
        return "Codex delegation is disabled by AGENT_ENABLE_CODEX_DELEGATION."
    workdir = Path(cwd).expanduser().resolve() if cwd else _workspace_root()
    if not workdir.exists() or not workdir.is_dir():
        return f"Codex working directory is invalid: {workdir}"

    command_name, model_name, command_warning = _normalize_command_and_model(
        codex_command, model
    )
    command_path = _resolve_codex_command(command_name)
    if command_path is None:
        return f"Codex command not found or not executable: {command_name}"

    command = [
        command_path,
        "exec",
        "-C",
        str(workdir),
        "-s",
        sandbox,
        "--color",
        "never",
    ]
    if model_name:
        command.extend(["-m", model_name])
    command.append(prompt)

    timeout = max(30, min(timeout_seconds, 24 * 60 * 60))
    emit_progress(f"starting Codex session via {command_name}: {prompt[:80]}")
    status, exit_code, stdout, stderr = _run_codex_subprocess(command, workdir, timeout)
    session_id = _extract_session_id(stderr, stdout)
    _log_codex_session_interaction(
        action="session_start",
        session_id=session_id,
        command_name=command_name,
        cwd=workdir,
        prompt=prompt,
        status=status,
        exit_code=exit_code,
        stdout=stdout,
        stderr=stderr,
    )
    emit_progress(
        f"Codex session start finished status={status} "
        f"session_id={session_id or 'unknown'}"
    )
    return _format_codex_session_result(
        action="session_start",
        command_name=command_name,
        session_id=session_id,
        status=status,
        exit_code=exit_code,
        stdout=stdout,
        stderr=stderr,
        command_warning=command_warning,
        max_output_chars=max_output_chars,
    )


@tool
def continue_codex_session(
    session_id: str,
    prompt: str,
    cwd: str = "",
    timeout_seconds: int = 1800,
    model: str = "",
    codex_command: str = "",
    max_output_chars: int = 60000,
) -> str:
    """Continue a specific Codex session and return output for the agent to decide the next step."""
    if not _enabled():
        return "Codex delegation is disabled by AGENT_ENABLE_CODEX_DELEGATION."
    session_id = session_id.strip()
    if not session_id:
        return "Codex session_id is required. Call start_codex_session first."
    workdir = Path(cwd).expanduser().resolve() if cwd else _workspace_root()
    if not workdir.exists() or not workdir.is_dir():
        return f"Codex working directory is invalid: {workdir}"

    command_name, model_name, command_warning = _normalize_command_and_model(
        codex_command, model
    )
    command_path = _resolve_codex_command(command_name)
    if command_path is None:
        return f"Codex command not found or not executable: {command_name}"

    command = [command_path, "exec", "resume"]
    if model_name:
        command.extend(["-m", model_name])
    command.extend([session_id, prompt])

    timeout = max(30, min(timeout_seconds, 24 * 60 * 60))
    emit_progress(f"continuing Codex session {session_id} via {command_name}: {prompt[:80]}")
    status, exit_code, stdout, stderr = _run_codex_subprocess(command, workdir, timeout)
    returned_session_id = _extract_session_id(stderr, stdout) or session_id
    _log_codex_session_interaction(
        action="session_continue",
        session_id=returned_session_id,
        command_name=command_name,
        cwd=workdir,
        prompt=prompt,
        status=status,
        exit_code=exit_code,
        stdout=stdout,
        stderr=stderr,
    )
    emit_progress(
        f"Codex session continue finished status={status} session_id={returned_session_id}"
    )
    return _format_codex_session_result(
        action="session_continue",
        command_name=command_name,
        session_id=returned_session_id,
        status=status,
        exit_code=exit_code,
        stdout=stdout,
        stderr=stderr,
        command_warning=command_warning,
        max_output_chars=max_output_chars,
    )



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
