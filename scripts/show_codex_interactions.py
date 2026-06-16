#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sqlite3
from pathlib import Path
from typing import Any


CODEX_TOOL_NAMES = {
    "rewrite_task_for_codex",
    "run_codex_task",
    "start_codex_session",
    "continue_codex_session",
    "test_codex_connectivity",
}


def _state_dir() -> Path:
    return Path(__file__).resolve().parents[1] / ".agent_state"


def _redact(text: str) -> str:
    patterns = [
        r"sk-[A-Za-z0-9_-]{12,}",
        r"(?i)(api[_-]?key\s*[=:]\s*)[A-Za-z0-9._-]+",
        r"(?i)(authorization:\s*bearer\s+)[A-Za-z0-9._-]+",
    ]
    redacted = text
    for pattern in patterns:
        redacted = re.sub(
            pattern,
            lambda match: (
                match.group(1) + "***REDACTED***"
                if match.groups()
                else "***REDACTED***"
            ),
            redacted,
        )
    return redacted


def _format_value(value: Any, max_chars: int) -> str:
    if not isinstance(value, str):
        value = json.dumps(value, ensure_ascii=False, indent=2)
    value = _redact(value)
    if max_chars > 0 and len(value) > max_chars:
        return value[:max_chars] + f"\n\n[truncated after {max_chars} characters]"
    return value


def _session_records() -> dict[str, dict[str, Any]]:
    index = _state_dir() / "session_index.jsonl"
    if not index.exists():
        return {}
    records: dict[str, dict[str, Any]] = {}
    for line in index.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        records[str(record.get("session_id", ""))] = record
    return records


def _resolve_session(selector: str, latest: bool) -> Path:
    records = _session_records()
    if latest:
        if not records:
            raise SystemExit("No saved sessions found.")
        record = sorted(
            records.values(),
            key=lambda item: item.get("updated_at", ""),
            reverse=True,
        )[0]
        return Path(record["path"])
    if not selector:
        raise SystemExit("Provide a session id/prefix, path, or --latest.")
    candidate = Path(selector).expanduser()
    if candidate.exists():
        return candidate.resolve()
    matches = [
        record for session_id, record in records.items()
        if session_id == selector or session_id.startswith(selector)
    ]
    if not matches:
        raise SystemExit(f"No session found for: {selector}")
    return Path(matches[0]["path"])


def _latest_messages(session_path: Path) -> list[dict[str, Any]]:
    latest: list[dict[str, Any]] = []
    for line in session_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if record.get("type") == "message_snapshot":
            latest = record.get("messages") or []
    return latest


def _tool_call_name(call: dict[str, Any]) -> str:
    return str(call.get("name") or call.get("function", {}).get("name") or "")


def _tool_call_args(call: dict[str, Any]) -> Any:
    args = call.get("args")
    if args is not None:
        return args
    raw = call.get("function", {}).get("arguments", "")
    try:
        return json.loads(raw)
    except Exception:
        return raw


def show_session(session_path: Path, max_chars: int) -> None:
    print(f"Session: {session_path}")
    messages = _latest_messages(session_path)
    found = 0
    for index, message in enumerate(messages, start=1):
        data = message.get("data", {})
        if message.get("type") == "ai":
            calls = (
                data.get("tool_calls")
                or data.get("additional_kwargs", {}).get("tool_calls")
                or []
            )
            for call in calls:
                name = _tool_call_name(call)
                if name not in CODEX_TOOL_NAMES:
                    continue
                found += 1
                print(f"\n## Message {index}: agent -> {name}")
                print(_format_value(_tool_call_args(call), max_chars=max_chars))
        elif message.get("type") == "tool" and data.get("name") in CODEX_TOOL_NAMES:
            found += 1
            print(f"\n## Message {index}: {data.get('name')} -> agent")
            print(_format_value(data.get("content", ""), max_chars=max_chars))
    if not found:
        print("No Codex tool interactions found in this session.")


def show_codex_task(task_id: int, max_chars: int) -> None:
    db_path = _state_dir() / "agent.sqlite3"
    if not db_path.exists():
        raise SystemExit(f"State DB not found: {db_path}")
    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row
    row = connection.execute("SELECT * FROM codex_tasks WHERE id = ?", (task_id,)).fetchone()
    if row is None:
        raise SystemExit(f"Codex task not found: {task_id}")
    for key in row.keys():
        print(f"\n## {key}")
        print(_format_value(row[key], max_chars=max_chars))


def list_codex_tasks(limit: int) -> None:
    db_path = _state_dir() / "agent.sqlite3"
    if not db_path.exists():
        print(f"State DB not found: {db_path}")
        return
    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row
    rows = connection.execute(
        """
        SELECT id, status, created_at, completed_at, substr(original_task, 1, 140) task
        FROM codex_tasks
        ORDER BY id DESC
        LIMIT ?
        """,
        (max(1, min(limit, 100)),),
    ).fetchall()
    for row in rows:
        print(f"{row['id']} [{row['status']}] {row['created_at']} {row['task']}")


def list_session_interactions(limit: int) -> None:
    db_path = _state_dir() / "agent.sqlite3"
    if not db_path.exists():
        print(f"State DB not found: {db_path}")
        return
    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row
    try:
        rows = connection.execute(
            """
            SELECT id, action, codex_session_id, command_name, status, created_at,
                   substr(prompt, 1, 140) prompt
            FROM codex_session_interactions
            ORDER BY id DESC
            LIMIT ?
            """,
            (max(1, min(limit, 100)),),
        ).fetchall()
    except sqlite3.OperationalError as exc:
        print(f"Could not read codex_session_interactions: {exc}")
        return
    for row in rows:
        print(
            f"{row['id']} [{row['status']}] {row['action']} "
            f"session={row['codex_session_id'] or '-'} command={row['command_name']} "
            f"{row['created_at']} {row['prompt']}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect agent <-> Codex interactions.")
    parser.add_argument("session", nargs="?", help="Session id/prefix or session JSONL path.")
    parser.add_argument("--latest", action="store_true", help="Use the latest saved agent session.")
    parser.add_argument("--task-id", type=int, help="Show one run_codex_task SQLite record.")
    parser.add_argument("--list-tasks", action="store_true", help="List run_codex_task records.")
    parser.add_argument(
        "--list-session-interactions",
        action="store_true",
        help="List start_codex_session/continue_codex_session records.",
    )
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--max-chars", type=int, default=20000)
    args = parser.parse_args()

    if args.list_tasks:
        list_codex_tasks(args.limit)
        return
    if args.list_session_interactions:
        list_session_interactions(args.limit)
        return
    if args.task_id is not None:
        show_codex_task(args.task_id, max_chars=args.max_chars)
        return
    session_path = _resolve_session(args.session or "", latest=args.latest)
    show_session(session_path, max_chars=args.max_chars)


if __name__ == "__main__":
    main()
