from __future__ import annotations

import os
import subprocess
from pathlib import Path

from langchain_core.tools import tool

from agent_project.tools.progress import emit_progress


def _workspace_root() -> Path:
    configured = os.getenv("AGENT_WORKSPACE_ROOT")
    if configured:
        return Path(configured).expanduser().resolve()
    return Path.cwd().resolve()


def _resolve_cwd(cwd: str) -> Path:
    return Path(cwd).expanduser().resolve()


def _is_inside(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _approval_enabled() -> bool:
    return os.getenv("AGENT_ENABLE_HUMAN_APPROVAL", "true").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _shell_enabled() -> bool:
    return os.getenv("AGENT_ENABLE_SHELL_COMMANDS", "true").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _ask_human_approval(command: str, cwd: Path, reason: str) -> bool:
    if not _approval_enabled():
        return False

    print("\n[approval required]")
    print("Action: execute shell command")
    print(f"Command: {command}")
    print(f"CWD: {cwd}")
    print(f"Reason: {reason}")
    answer = input("Allow this operation? Type yes to approve: ").strip().lower()
    return answer == "yes"


@tool
def execute_shell_command(
    command: str,
    cwd: str = ".",
    timeout_seconds: int = 30,
    max_chars: int = 20000,
) -> str:
    """Execute a shell command and return stdout/stderr. Approval depends on config."""
    emit_progress(f"preparing shell command: {command}")
    if not _shell_enabled():
        emit_progress("shell command skipped: disabled by AGENT_ENABLE_SHELL_COMMANDS")
        return "Shell command execution is disabled by AGENT_ENABLE_SHELL_COMMANDS."

    workdir = _resolve_cwd(cwd)
    if not workdir.exists():
        emit_progress(f"shell command rejected, cwd does not exist: {workdir}")
        return f"Working directory does not exist: {workdir}"
    if not workdir.is_dir():
        emit_progress(f"shell command rejected, cwd is not a directory: {workdir}")
        return f"Working directory is not a directory: {workdir}"

    root = _workspace_root()
    inside_workspace = _is_inside(workdir, root)
    if inside_workspace and not _approval_enabled():
        reason = ""
    elif inside_workspace:
        reason = "shell commands require explicit terminal approval"
    else:
        reason = f"cwd is outside AGENT_WORKSPACE_ROOT={root}; terminal approval is required"

    if reason and not _ask_human_approval(command, workdir, reason):
        emit_progress(f"shell command denied: {command}")
        if _approval_enabled():
            return f"Shell command denied: {reason}. The user must type yes in the terminal prompt."
        return f"Shell command denied: {reason}. AGENT_ENABLE_HUMAN_APPROVAL=false, so operations outside the workspace are not prompted."

    timeout = max(1, min(timeout_seconds, 120))
    emit_progress(f"running shell command in {workdir}: {command}")
    try:
        completed = subprocess.run(
            command,
            cwd=workdir,
            shell=True,
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        emit_progress(f"shell command timed out after {timeout}s: {command}")
        output = ""
        if exc.stdout:
            output += f"stdout:\n{exc.stdout}\n"
        if exc.stderr:
            output += f"stderr:\n{exc.stderr}\n"
        if max_chars > 0 and len(output) > max_chars:
            output = output[:max_chars] + f"\n\n[truncated after {max_chars} characters]"
        return f"Command timed out after {timeout} seconds.\n{output}".strip()

    emit_progress(f"shell command finished with exit code {completed.returncode}: {command}")
    output_parts = [f"exit_code: {completed.returncode}"]
    if completed.stdout:
        output_parts.append(f"stdout:\n{completed.stdout}")
    if completed.stderr:
        output_parts.append(f"stderr:\n{completed.stderr}")
    output = "\n".join(output_parts)

    if max_chars > 0 and len(output) > max_chars:
        return output[:max_chars] + f"\n\n[truncated after {max_chars} characters]"
    return output
