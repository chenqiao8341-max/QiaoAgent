from __future__ import annotations

import os
import re
from datetime import datetime, timezone
from pathlib import Path

from langchain_core.tools import tool

from agent_project.tools.codex_delegate import start_codex_session
from agent_project.tools.progress import emit_progress
from agent_project.tools.storage import connect, state_db_path


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _workspace_root() -> Path:
    configured = os.getenv("AGENT_WORKSPACE_ROOT")
    if configured:
        return Path(configured).expanduser().resolve()
    return Path.cwd().resolve()


def _is_inside(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _state_dir() -> Path:
    return state_db_path().parent


def _packet_dir() -> Path:
    return _state_dir() / "codex_review_packets"


def _slug(text: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", text.strip())[:60].strip("-")
    return slug or "task"


def _split_lines(value: str) -> list[str]:
    return [line.strip() for line in value.splitlines() if line.strip()]


def _read_text_artifact(path_text: str, root: Path, max_chars: int) -> str:
    path = Path(path_text).expanduser()
    if not path.is_absolute():
        path = (root / path).resolve()
    else:
        path = path.resolve()
    if not _is_inside(path, root):
        return f"### {path}\n\n[skipped: outside AGENT_WORKSPACE_ROOT={root}]"
    if not path.exists():
        return f"### {path}\n\n[missing]"
    if not path.is_file():
        return f"### {path}\n\n[skipped: not a file]"
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return f"### {path}\n\n[skipped: not UTF-8 text]"
    if max_chars > 0 and len(text) > max_chars:
        text = text[:max_chars] + f"\n\n[truncated after {max_chars} characters]"
    return f"### {path}\n\n```text\n{text}\n```"


def _section(title: str, body: str) -> str:
    body = body.strip()
    if not body:
        body = "(none provided)"
    return f"## {title}\n\n{body}"


@tool
def record_task_difficulty_judgment(
    task: str,
    difficulty: str,
    rationale: str,
    confidence: float = 0.5,
    planned_attempt: str = "",
    should_request_codex_review: bool = False,
) -> str:
    """Record the agent's own task difficulty judgment before or during a task."""
    normalized = difficulty.strip().lower()
    if normalized not in {"easy", "medium", "hard", "unknown"}:
        normalized = "unknown"
    confidence = max(0.0, min(float(confidence), 1.0))
    created_at = _now()
    with connect() as connection:
        cursor = connection.execute(
            """
            INSERT INTO task_difficulty_judgments(
                task, difficulty, confidence, rationale, planned_attempt,
                should_request_codex_review, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                task,
                normalized,
                confidence,
                rationale,
                planned_attempt,
                1 if should_request_codex_review else 0,
                created_at,
            ),
        )
        judgment_id = cursor.lastrowid

    if normalized == "hard" or should_request_codex_review:
        next_step = (
            "After completing one serious attempt, create a Codex review packet "
            "with create_codex_review_packet and start review with start_codex_review."
        )
    else:
        next_step = "Codex review is optional unless the task becomes risky or uncertain."
    return (
        f"Recorded difficulty judgment {judgment_id}: difficulty={normalized}, "
        f"confidence={confidence:.2f}. {next_step}"
    )


@tool
def create_codex_review_packet(
    original_task: str,
    attempt_summary: str,
    difficulty_judgment: str = "",
    result_paths: str = "",
    source_urls: str = "",
    source_notes: str = "",
    excerpt_paths: str = "",
    key_excerpts: str = "",
    questions_for_codex: str = "",
    cwd: str = "",
    max_file_chars: int = 20000,
) -> str:
    """Create a Markdown packet containing a first attempt and evidence for Codex review."""
    root = Path(cwd).expanduser().resolve() if cwd else _workspace_root()
    created_at = _now()
    packet_dir = _packet_dir()
    packet_dir.mkdir(parents=True, exist_ok=True)
    packet_path = packet_dir / f"{created_at.replace(':', '-')}-{_slug(original_task)}.md"

    result_sections = [
        _read_text_artifact(path, root=root, max_chars=max_file_chars)
        for path in _split_lines(result_paths)
    ]
    excerpt_sections = [
        _read_text_artifact(path, root=root, max_chars=max_file_chars)
        for path in _split_lines(excerpt_paths)
    ]
    source_list = "\n".join(f"- {url}" for url in _split_lines(source_urls))

    packet = "\n\n".join(
        [
            "# Codex Review Packet",
            _section("Original Task", original_task),
            _section("Agent Difficulty Judgment", difficulty_judgment),
            _section("Agent First Attempt Summary", attempt_summary),
            _section("Questions For Codex", questions_for_codex),
            _section("Source URLs", source_list),
            _section("Source Notes", source_notes),
            _section("Key Excerpts", key_excerpts),
            _section("Result Files", "\n\n".join(result_sections)),
            _section("Excerpt Files", "\n\n".join(excerpt_sections)),
            _section(
                "Review Instructions",
                "\n".join(
                    [
                        "Please review the agent's first attempt as an independent judge.",
                        "Check correctness, missing evidence, unsupported claims, and task fit.",
                        "Prefer using the provided artifacts before re-searching or re-reading.",
                        "If changes are needed, give concrete edits or ask a focused question.",
                    ]
                ),
            ),
        ]
    )
    packet_path.write_text(packet, encoding="utf-8")

    with connect() as connection:
        cursor = connection.execute(
            """
            INSERT INTO codex_review_packets(
                original_task, packet_path, cwd, difficulty_judgment, attempt_summary,
                source_urls, result_paths, excerpt_paths, questions_for_codex, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                original_task,
                str(packet_path),
                str(root),
                difficulty_judgment,
                attempt_summary,
                source_urls,
                result_paths,
                excerpt_paths,
                questions_for_codex,
                created_at,
            ),
        )
        packet_id = cursor.lastrowid

    emit_progress(f"created Codex review packet {packet_id}: {packet_path}")
    return (
        f"Created Codex review packet {packet_id}: {packet_path}\n\n"
        f"Use start_codex_review(packet_path='{packet_path}', cwd='{root}') "
        "to request review."
    )


@tool
def start_codex_review(
    packet_path: str,
    cwd: str = "",
    codex_command: str = "",
    model: str = "",
    timeout_seconds: int = 1800,
    max_output_chars: int = 60000,
) -> str:
    """Start a Codex review session from a review packet created after a hard task attempt."""
    root = Path(cwd).expanduser().resolve() if cwd else _workspace_root()
    path = Path(packet_path).expanduser().resolve()
    if not path.exists() or not path.is_file():
        return f"Codex review packet does not exist: {path}"
    packet = path.read_text(encoding="utf-8")
    prompt = "\n\n".join(
        [
            "You are Codex acting as an independent reviewer for another agent's work.",
            (
                "Review the packet below. Do not redo expensive retrieval unless "
                "the packet is insufficient."
            ),
            (
                "Return: verdict, high-priority issues, suggested fixes, and "
                "whether further interaction is needed."
            ),
            packet,
        ]
    )
    emit_progress(f"starting Codex review for packet: {path}")
    result = start_codex_session.invoke(
        {
            "prompt": prompt,
            "cwd": str(root),
            "timeout_seconds": timeout_seconds,
            "codex_command": codex_command,
            "model": model,
            "max_output_chars": max_output_chars,
        }
    )
    session_id = ""
    match = re.search(r"session_id=([^\s]+)", result)
    if match:
        session_id = match.group(1)
    with connect() as connection:
        connection.execute(
            """
            UPDATE codex_review_packets
            SET codex_session_id = ?, codex_status = ?, reviewed_at = ?
            WHERE packet_path = ?
            """,
            (session_id, "started", _now(), str(path)),
        )
    return result


@tool
def list_codex_review_packets(limit: int = 10) -> str:
    """List recent Codex review packets."""
    limit = max(1, min(int(limit), 50))
    with connect() as connection:
        rows = connection.execute(
            """
            SELECT id, created_at, codex_session_id, packet_path, original_task
            FROM codex_review_packets
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    if not rows:
        return "No Codex review packets stored."
    return "\n".join(
        f"{row['id']} {row['created_at']} session={row['codex_session_id'] or '-'} "
        f"{row['original_task'][:100]} -> {row['packet_path']}"
        for row in rows
    )
