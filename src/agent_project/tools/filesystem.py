from __future__ import annotations

import difflib
import os
from pathlib import Path

from langchain_core.tools import tool

from agent_project.tools.progress import emit_progress


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
        reason = (
            f"path is outside AGENT_WORKSPACE_ROOT={root} and write operations require approval"
        )

    if _ask_human_approval("write local file", path, reason):
        return True, ""
    return False, f"Write denied: {reason}"


def _line_delta(old: str, new: str) -> tuple[int, int]:
    added = 0
    removed = 0
    for line in difflib.ndiff(old.splitlines(), new.splitlines()):
        if line.startswith("+ "):
            added += 1
        elif line.startswith("- "):
            removed += 1
    return added, removed


@tool
def list_local_directory(path: str = ".") -> str:
    """List files and directories at a local path."""
    target = _resolve_path(path)
    emit_progress(f"reading directory: {target}")
    allowed, message = _check_read_allowed(target)
    if not allowed:
        emit_progress(f"directory read denied: {target}")
        return message
    if not target.exists():
        emit_progress(f"directory not found: {target}")
        return f"Path does not exist: {target}"
    if not target.is_dir():
        emit_progress(f"path is not a directory: {target}")
        return f"Path is not a directory: {target}"

    entries = []
    for child in sorted(target.iterdir(), key=lambda item: (not item.is_dir(), item.name.lower())):
        suffix = "/" if child.is_dir() else ""
        entries.append(f"{child.name}{suffix}")
    emit_progress(f"directory read complete: {target} ({len(entries)} entries)")
    return "\n".join(entries) if entries else "(empty directory)"


@tool
def read_local_file(path: str, max_chars: int = 20000) -> str:
    """Read a UTF-8 local text file, with optional truncation."""
    target = _resolve_path(path)
    emit_progress(f"reading file: {target}")
    allowed, message = _check_read_allowed(target)
    if not allowed:
        emit_progress(f"file read denied: {target}")
        return message
    if not target.exists():
        emit_progress(f"file not found: {target}")
        return f"File does not exist: {target}"
    if not target.is_file():
        emit_progress(f"path is not a file: {target}")
        return f"Path is not a file: {target}"

    try:
        text = target.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        emit_progress(f"file read failed, not UTF-8: {target}")
        return f"Could not decode as UTF-8 text file: {target}"

    if max_chars > 0 and len(text) > max_chars:
        emit_progress(f"file read complete: {target} ({len(text)} chars, truncated to {max_chars})")
        return text[:max_chars] + f"\n\n[truncated after {max_chars} characters]"
    emit_progress(f"file read complete: {target} ({len(text)} chars)")
    return text


@tool
def write_local_file(path: str, content: str, mode: str = "overwrite") -> str:
    """Write UTF-8 text to a local file after human approval. Mode: overwrite or append."""
    target = _resolve_path(path)
    emit_progress(f"preparing file write: {target} (mode={mode})")
    if mode not in {"overwrite", "append"}:
        emit_progress(f"file write rejected, invalid mode: {mode}")
        return "Invalid mode. Use overwrite or append."

    allowed, message = _check_write_allowed(target)
    if not allowed:
        emit_progress(f"file write denied: {target}")
        return message

    old_text = ""
    if target.exists() and target.is_file():
        try:
            old_text = target.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            old_text = ""

    target.parent.mkdir(parents=True, exist_ok=True)
    if mode == "append":
        with target.open("a", encoding="utf-8") as handle:
            handle.write(content)
        added = len(content.splitlines())
        removed = 0
    else:
        target.write_text(content, encoding="utf-8")
        added, removed = _line_delta(old_text, content)

    emit_progress(
        f"file write complete: {target} ({len(content)} chars, +{added}/-{removed} lines)"
    )
    return f"Wrote {len(content)} characters to {target} with mode={mode}."
