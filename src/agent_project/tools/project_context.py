from __future__ import annotations

import json
import subprocess
from pathlib import Path

from langchain_core.tools import tool

from agent_project.tools.progress import emit_progress


PROJECT_CONTEXT_FILES = (
    "README.md",
    "current_status.md",
    "todo.md",
    "last_report.json",
)


def _trim(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + f"\n\n[truncated after {max_chars} characters]"


def _read_optional_file(path: Path, max_chars: int) -> str:
    if not path.exists() or not path.is_file():
        return ""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return f"[read error: {exc}]"
    return _trim(text.strip(), max_chars)


def _git_output(project_path: Path, args: list[str], max_chars: int) -> str:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=project_path,
            text=True,
            capture_output=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return f"[git unavailable: {exc}]"
    output = (completed.stdout or completed.stderr or "").strip()
    if completed.returncode != 0 and not output:
        output = f"[git exited with {completed.returncode}]"
    return _trim(output, max_chars)


def load_project_context_text(project_path: str, max_chars: int = 12000) -> str:
    root = Path(project_path).expanduser().resolve()
    if not root.exists() or not root.is_dir():
        return f"Project path is not a directory: {root}"

    budget = max(1000, min(max_chars, 100000))
    per_file_budget = max(1000, budget // 8)
    emit_progress(f"loading project context: {root}")

    payload: dict[str, object] = {
        "project_path": str(root),
        "files": {},
        "git": {},
    }
    files: dict[str, str] = {}
    for name in PROJECT_CONTEXT_FILES:
        content = _read_optional_file(root / name, per_file_budget)
        if content:
            files[name] = content
    payload["files"] = files

    if (root / ".git").exists():
        payload["git"] = {
            "branch": _git_output(root, ["branch", "--show-current"], 1000),
            "status": _git_output(root, ["status", "--short"], per_file_budget),
            "recent_commits": _git_output(root, ["log", "--oneline", "-5"], per_file_budget),
            "diff_stat": _git_output(root, ["diff", "--stat"], per_file_budget),
        }

    return _trim(json.dumps(payload, ensure_ascii=False, indent=2), budget)


@tool
def load_project_context(project_path: str, max_chars: int = 12000) -> str:
    """Load README/status/todo/report and lightweight git state for one project."""
    return load_project_context_text(project_path=project_path, max_chars=max_chars)
