from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any

from langchain_core.tools import tool

from agent_project.tools.progress import emit_progress
from agent_project.tools.storage import connect, state_db_path


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _json_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return value
        return _json_text(parsed)
    if isinstance(value, dict):
        if "text" in value:
            return str(value["text"])
        if "content" in value:
            return _json_text(value["content"])
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def _sender_name(sender: dict[str, Any]) -> str:
    if not sender:
        return ""
    if "sender_id" in sender:
        sender_id = sender.get("sender_id") or {}
        return (
            sender_id.get("user_id")
            or sender_id.get("open_id")
            or sender_id.get("union_id")
            or ""
        )
    return sender.get("user_id") or sender.get("open_id") or sender.get("union_id") or ""


def verify_feishu_token(payload: dict[str, Any]) -> bool:
    expected = os.getenv("FEISHU_VERIFICATION_TOKEN", "").strip()
    if not expected:
        return True
    return payload.get("token") == expected or payload.get("header", {}).get("token") == expected


def handle_feishu_event(payload: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    if payload.get("type") == "url_verification" and "challenge" in payload:
        if not verify_feishu_token(payload):
            return 403, {"error": "invalid verification token"}
        return 200, {"challenge": payload["challenge"]}

    if not verify_feishu_token(payload):
        return 403, {"error": "invalid verification token"}

    event = payload.get("event") or {}
    message = event.get("message") or {}
    if not message:
        return 200, {"ok": True, "stored": False, "reason": "no message in event"}

    store_feishu_message_event(payload)
    return 200, {"ok": True, "stored": True}


def store_feishu_message_event(payload: dict[str, Any]) -> int | None:
    event = payload.get("event") or {}
    message = event.get("message") or {}
    sender = event.get("sender") or {}

    message_id = message.get("message_id") or payload.get("uuid") or f"local-{_now()}"
    chat_id = message.get("chat_id") or ""
    message_type = message.get("message_type") or ""
    content = _json_text(message.get("content"))
    sender_id = _sender_name(sender)
    sender_name = sender.get("sender_name") or sender_id
    received_at = _now()

    emit_progress(f"received Feishu message from {sender_name or sender_id}: {content[:80]}")
    with connect() as connection:
        cursor = connection.execute(
            """
            INSERT OR IGNORE INTO feishu_messages(
                message_id, chat_id, sender_id, sender_name, message_type,
                content, raw_event, received_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                message_id,
                chat_id,
                sender_id,
                sender_name,
                message_type,
                content,
                json.dumps(payload, ensure_ascii=False),
                received_at,
            ),
        )
        if cursor.rowcount == 0:
            return None
        return cursor.lastrowid


def _unsummarized_messages(limit: int = 50):
    with connect() as connection:
        return connection.execute(
            """
            SELECT * FROM feishu_messages
            WHERE report_id IS NULL
            ORDER BY received_at ASC, id ASC
            LIMIT ?
            """,
            (max(1, min(limit, 200)),),
        ).fetchall()


def _build_report_text(rows) -> str:
    by_sender: dict[str, list[str]] = {}
    for row in rows:
        sender = row["sender_name"] or row["sender_id"] or "unknown"
        by_sender.setdefault(sender, []).append(row["content"])

    lines = [f"本次共收到 {len(rows)} 条飞书消息。"]
    for sender, messages in by_sender.items():
        lines.append(f"\n来自 {sender} 的 {len(messages)} 条消息：")
        for message in messages[:10]:
            lines.append(f"- {message}")
        if len(messages) > 10:
            lines.append(f"- ... 还有 {len(messages) - 10} 条")
    lines.append("\n建议：优先处理包含明确请求、截止时间或阻塞信息的消息。")
    return "\n".join(lines)


@tool
def generate_feishu_report(limit: int = 50) -> str:
    """Summarize unsummarized Feishu messages and store a report in SQLite."""
    rows = _unsummarized_messages(limit=limit)
    emit_progress(f"generating Feishu report ({len(rows)} pending messages)")
    if not rows:
        return "No unsummarized Feishu messages."

    summary = _build_report_text(rows)
    created_at = _now()
    title = f"Feishu message report {created_at}"
    with connect() as connection:
        cursor = connection.execute(
            """
            INSERT INTO feishu_reports(title, summary, message_count, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (title, summary, len(rows), created_at),
        )
        report_id = cursor.lastrowid
        message_ids = [row["id"] for row in rows]
        connection.executemany(
            "UPDATE feishu_messages SET report_id = ? WHERE id = ?",
            [(report_id, message_id) for message_id in message_ids],
        )

    emit_progress(f"Feishu report stored: {report_id} ({state_db_path()})")
    return f"Report {report_id}: {title}\n\n{summary}"


@tool
def list_feishu_reports(limit: int = 10) -> str:
    """List recent Feishu message reports stored in SQLite."""
    limit = max(1, min(limit, 50))
    with connect() as connection:
        rows = connection.execute(
            """
            SELECT * FROM feishu_reports
            ORDER BY created_at DESC, id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()

    if not rows:
        return "No Feishu reports stored."
    return "\n".join(
        f"{row['id']}: {row['title']} ({row['message_count']} messages)" for row in rows
    )


@tool
def get_feishu_report(report_id: int) -> str:
    """Return one Feishu message report by id."""
    with connect() as connection:
        row = connection.execute(
            "SELECT * FROM feishu_reports WHERE id = ?",
            (report_id,),
        ).fetchone()
    if row is None:
        return f"Feishu report not found: {report_id}"
    return f"Report {row['id']}: {row['title']}\n\n{row['summary']}"


@tool
def list_feishu_messages(limit: int = 20, only_unsummarized: bool = True) -> str:
    """List recent Feishu messages captured by the watcher."""
    limit = max(1, min(limit, 100))
    where = "WHERE report_id IS NULL" if only_unsummarized else ""
    with connect() as connection:
        rows = connection.execute(
            f"""
            SELECT * FROM feishu_messages
            {where}
            ORDER BY received_at DESC, id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()

    if not rows:
        return "No Feishu messages found."
    lines = []
    for row in rows:
        sender = row["sender_name"] or row["sender_id"] or "unknown"
        lines.append(f"{row['id']} [{row['received_at']}] {sender}: {row['content']}")
    return "\n".join(lines)
