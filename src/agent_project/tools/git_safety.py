from __future__ import annotations

import re
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

from langchain_core.tools import tool

from agent_project.tools.progress import emit_progress


def _run_git(args: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=cwd,
        text=True,
        capture_output=True,
        timeout=15,
        check=False,
    )


def _slugify_task(task_slug: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "-", task_slug.strip().lower()).strip("-._")
    return slug[:40] or "work"


def _unique_branch_name(cwd: Path, base: str) -> str:
    candidate = base
    suffix = 2
    while _run_git(["rev-parse", "--verify", "--quiet", candidate], cwd).returncode == 0:
        candidate = f"{base}-{suffix}"
        suffix += 1
    return candidate


def ensure_work_branch_state(
    task_slug: str = "",
    base_branch: str = "main",
    dry_run: bool = False,
    cwd: str | Path | None = None,
) -> dict[str, Any]:
    root = Path(cwd).expanduser().resolve() if cwd else Path.cwd().resolve()
    repo_check = _run_git(["rev-parse", "--is-inside-work-tree"], root)
    if repo_check.returncode != 0 or repo_check.stdout.strip() != "true":
        return {
            "ok": False,
            "reason": "not_a_git_repository",
            "cwd": str(root),
            "do_not_auto_merge": True,
        }

    top_level = _run_git(["rev-parse", "--show-toplevel"], root)
    if top_level.returncode == 0 and top_level.stdout.strip():
        root = Path(top_level.stdout.strip())

    current = _run_git(["branch", "--show-current"], root).stdout.strip()
    status = _run_git(["status", "--short"], root).stdout.strip()
    clean_base = base_branch.strip() or "main"
    safe_slug = _slugify_task(task_slug)
    date_prefix = datetime.now().strftime("%Y%m%d")
    branch_base = f"agent/{date_prefix}-{safe_slug}"
    new_branch = _unique_branch_name(root, branch_base)

    if current and current not in {clean_base, "master", "main"}:
        return {
            "ok": True,
            "action": "already_on_work_branch",
            "cwd": str(root),
            "current_branch": current,
            "dirty": bool(status),
            "status": status,
            "do_not_auto_merge": True,
        }

    if dry_run:
        return {
            "ok": True,
            "action": "would_create_branch",
            "cwd": str(root),
            "current_branch": current,
            "new_branch": new_branch,
            "dirty": bool(status),
            "status": status,
            "do_not_auto_merge": True,
        }

    switch = _run_git(["switch", "-c", new_branch], root)
    if switch.returncode != 0:
        return {
            "ok": False,
            "reason": "branch_create_failed",
            "stderr": switch.stderr.strip(),
            "cwd": str(root),
            "current_branch": current,
            "new_branch": new_branch,
            "dirty": bool(status),
            "status": status,
            "do_not_auto_merge": True,
        }

    return {
        "ok": True,
        "action": "created_branch",
        "cwd": str(root),
        "previous_branch": current,
        "current_branch": new_branch,
        "dirty": bool(status),
        "status": status,
        "do_not_auto_merge": True,
    }


def _format_state(state: dict[str, Any]) -> str:
    lines = [
        f"GitSafety ok={state.get('ok')} action={state.get('action', state.get('reason', ''))}",
        f"cwd: {state.get('cwd', '')}",
    ]
    if state.get("previous_branch"):
        lines.append(f"previous_branch: {state['previous_branch']}")
    if state.get("current_branch"):
        lines.append(f"current_branch: {state['current_branch']}")
    if state.get("new_branch"):
        lines.append(f"new_branch: {state['new_branch']}")
    lines.append("do_not_auto_merge: true")
    if state.get("dirty"):
        lines.append("dirty: true")
    if state.get("stderr"):
        lines.append(f"stderr: {state['stderr']}")
    return "\n".join(lines)


@tool
def ensure_work_branch(task_slug: str = "", base_branch: str = "main", dry_run: bool = False) -> str:
    """Ensure repo work happens on an isolated branch and never auto-merges to main."""
    emit_progress("checking git work branch")
    return _format_state(
        ensure_work_branch_state(
            task_slug=task_slug,
            base_branch=base_branch,
            dry_run=dry_run,
        )
    )
