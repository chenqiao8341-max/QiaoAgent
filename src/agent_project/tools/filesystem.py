from __future__ import annotations

import os
from pathlib import Path

from langchain_core.tools import tool


def _workspace_root() -> Path:
    configured = os.getenv("AGENT_WORKSPACE_ROOT")
    if configured:
        return Path(configured).expanduser().resolve()
    return Path.cwd().resolve()


def _resolve_path(path: str) -> Path:
    return Path(path).expanduser().resolve()


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


def _ask_human_approval(action: str, path: Path, reason: str) -> bool:
    if not _approval_enabled():
        return False

    print("\n[approval required]")
    print(f"Action: {action}")
    print(f"Path: {path}")
    print(f"Reason: {reason}")
    answer = input("Allow this operation? Type yes to approve: " ).strip().lower()
    return answer == "yes"


def _check_read_allowed(path: Path) -> tuple[bool, str]:
    root = _workspace_root()
    if _is_inside(path, root):
        return True, ""

    reason = f"path is outside AGENT_WORKSPACE_ROOT={root}"
    if _ask_human_approval("read local file", path, reason):
        return True, ""
    return False, f"Read denied: {reason}"


def _check_write_allowed(path: Path) -> tuple[bool, str]:
    root = _workspace_root()
    if _is_inside(path, root):
        reason = "write operations require explicit human approval"
    else:
        reason = f"path is outside AGENT_WORKSPACE_ROOT={root} and write operations require approval"

    if _ask_human_approval("write local file", path, reason):
        return True, ""
    return False, f"Write denied: {reason}"


@tool
def list_local_directory(path: str = ".") -> str:
    """List files and directories at a local path."""
    target = _resolve_path(path)
    allowed, message = _check_read_allowed(target)
    if not allowed:
        return message
    if not target.exists():
        return f"Path does not exist: {target}"
    if not target.is_dir():
        return f"Path is not a directory: {target}"

    entries = []
    for child in sorted(target.iterdir(), key=lambda item: (not item.is_dir(), item.name.lower())):
        suffix = "/" if child.is_dir() else ""
        entries.append(f"{child.name}{suffix}")
    return "\n".join(entries) if entries else "(empty directory)"


@tool
def read_local_file(path: str, max_chars: int = 20000) -> str:
    """Read a UTF-8 local text file, with optional truncation."""
    target = _resolve_path(path)
    allowed, message = _check_read_allowed(target)
    if not allowed:
        return message
    if not target.exists():
        return f"File does not exist: {target}"
    if not target.is_file():
        return f"Path is not a file: {target}"

    try:
        text = target.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return f"Could not decode as UTF-8 text file: {target}"

    if max_chars > 0 and len(text) > max_chars:
        return text[:max_chars] + f"\n\n[truncated after {max_chars} characters]"
    return text


@tool
def write_local_file(path: str, content: str, mode: str = "overwrite") -> str:
    """Write UTF-8 text to a local file after human approval. Mode: overwrite or append."""
    target = _resolve_path(path)
    if mode not in {"overwrite", "append"}:
        return "Invalid mode. Use overwrite or append."

    allowed, message = _check_write_allowed(target)
    if not allowed:
        return message

    target.parent.mkdir(parents=True, exist_ok=True)
    if mode == "append":
        with target.open("a", encoding="utf-8") as handle:
            handle.write(content)
    else:
        target.write_text(content, encoding="utf-8")

    return f"Wrote {len(content)} characters to {target} with mode={mode}."
