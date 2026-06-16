from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from langchain_core.messages import BaseMessage, messages_from_dict, messages_to_dict

from agent_project.tools.storage import state_db_path


@dataclass(frozen=True)
class AgentSession:
    session_id: str
    path: Path
    cwd: str
    created_at: str
    updated_at: str
    title: str = ""
    provider: str = ""
    message_count: int = 0


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _state_dir() -> Path:
    configured = os.getenv("AGENT_SESSION_DIR")
    if configured:
        return Path(configured).expanduser().resolve()
    return state_db_path().parent


def sessions_dir() -> Path:
    return _state_dir() / "sessions"


def session_index_path() -> Path:
    return _state_dir() / "session_index.jsonl"


def _session_path(session_id: str, created_at: str) -> Path:
    created = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
    date_dir = (
        sessions_dir()
        / f"{created.year:04d}"
        / f"{created.month:02d}"
        / f"{created.day:02d}"
    )
    stamp = created.strftime("%Y-%m-%dT%H-%M-%S")
    return date_dir / f"session-{stamp}-{session_id}.jsonl"


def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")


def _message_text(message: BaseMessage) -> str:
    content = message.content
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and "text" in item:
                parts.append(str(item["text"]))
            elif hasattr(item, "text"):
                parts.append(str(item.text))
        return "\n".join(part.strip() for part in parts if part).strip()
    return str(content).strip()


def _title_from_messages(messages: list[BaseMessage], fallback: str = "Untitled session") -> str:
    for message in messages:
        if getattr(message, "type", "") == "human":
            text = _message_text(message)
            if text:
                return text[:80]
    return fallback


def create_session(cwd: str, provider: str = "") -> AgentSession:
    created_at = _now()
    session_id = str(uuid.uuid4())
    path = _session_path(session_id, created_at)
    session = AgentSession(
        session_id=session_id,
        path=path,
        cwd=str(Path(cwd).expanduser().resolve()),
        created_at=created_at,
        updated_at=created_at,
        provider=provider,
    )
    _append_jsonl(
        path,
        {
            "timestamp": created_at,
            "type": "session_meta",
            "session_id": session_id,
            "cwd": session.cwd,
            "provider": provider,
        },
    )
    _write_index(session)
    return session


def _write_index(session: AgentSession) -> None:
    _append_jsonl(
        session_index_path(),
        {
            "timestamp": _now(),
            "session_id": session.session_id,
            "path": str(session.path),
            "cwd": session.cwd,
            "created_at": session.created_at,
            "updated_at": session.updated_at,
            "title": session.title,
            "provider": session.provider,
            "message_count": session.message_count,
        },
    )


def append_turn(session: AgentSession, messages: list[BaseMessage]) -> AgentSession:
    updated_at = _now()
    title = _title_from_messages(messages, fallback=session.title or "Untitled session")
    record = {
        "timestamp": updated_at,
        "type": "message_snapshot",
        "session_id": session.session_id,
        "messages": messages_to_dict(messages),
    }
    _append_jsonl(session.path, record)
    updated = AgentSession(
        session_id=session.session_id,
        path=session.path,
        cwd=session.cwd,
        created_at=session.created_at,
        updated_at=updated_at,
        title=title,
        provider=session.provider,
        message_count=len(messages),
    )
    _write_index(updated)
    return updated


def _session_from_record(record: dict[str, Any]) -> AgentSession | None:
    try:
        return AgentSession(
            session_id=str(record["session_id"]),
            path=Path(str(record["path"])).expanduser().resolve(),
            cwd=str(record.get("cwd", "")),
            created_at=str(record.get("created_at") or record.get("timestamp") or ""),
            updated_at=str(record.get("updated_at") or record.get("timestamp") or ""),
            title=str(record.get("title", "")),
            provider=str(record.get("provider", "")),
            message_count=int(record.get("message_count") or 0),
        )
    except (KeyError, TypeError, ValueError):
        return None


def list_sessions(include_all: bool = False, cwd: str | None = None) -> list[AgentSession]:
    index = session_index_path()
    if not index.exists():
        return []

    current_cwd = str(Path(cwd or Path.cwd()).expanduser().resolve())
    by_id: dict[str, AgentSession] = {}
    with index.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            session = _session_from_record(record)
            if session is None or not session.path.exists():
                continue
            if not include_all and session.cwd != current_cwd:
                continue
            by_id[session.session_id] = session

    return sorted(by_id.values(), key=lambda item: item.updated_at, reverse=True)


def find_session(session_id_or_prefix: str, include_all: bool = True) -> AgentSession | None:
    target = session_id_or_prefix.strip()
    if not target:
        return None

    matches = [
        session
        for session in list_sessions(include_all=include_all)
        if session.session_id == target or session.session_id.startswith(target)
    ]
    if not matches:
        return None
    return sorted(matches, key=lambda item: len(item.session_id))[0]


def load_session_messages(session: AgentSession) -> list[BaseMessage]:
    latest_messages: list[dict[str, Any]] | None = None
    with session.path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("type") == "message_snapshot":
                latest_messages = record.get("messages")

    if latest_messages is None:
        return []
    return list(messages_from_dict(latest_messages))


def format_session_line(session: AgentSession, index: int | None = None) -> str:
    prefix = f"{index}. " if index is not None else ""
    title = session.title or "Untitled session"
    short_id = session.session_id[:8]
    cwd_name = Path(session.cwd).name if session.cwd else ""
    return (
        f"{prefix}{short_id}  {session.updated_at}  "
        f"{session.message_count} messages  {cwd_name}  {title}"
    )
