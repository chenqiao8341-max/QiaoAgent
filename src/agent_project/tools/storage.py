from __future__ import annotations

import os
import sqlite3
from pathlib import Path


def state_db_path() -> Path:
    configured = os.getenv("AGENT_STATE_DB_PATH")
    if configured:
        return Path(configured).expanduser().resolve()

    workspace = os.getenv("AGENT_WORKSPACE_ROOT")
    root = Path(workspace).expanduser().resolve() if workspace else Path.cwd().resolve()
    return root / ".agent_state" / "agent.sqlite3"


def connect() -> sqlite3.Connection:
    path = state_db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    _initialize(connection)
    return connection


def _initialize(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS memories (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            namespace TEXT NOT NULL DEFAULT 'default',
            content TEXT NOT NULL,
            source TEXT NOT NULL DEFAULT '',
            tags TEXT NOT NULL DEFAULT '',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_memories_namespace_created
            ON memories(namespace, created_at);

        CREATE TABLE IF NOT EXISTS task_queues (
            queue_id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS task_steps (
            queue_id TEXT NOT NULL,
            step_index INTEGER NOT NULL,
            description TEXT NOT NULL,
            status TEXT NOT NULL,
            note TEXT NOT NULL DEFAULT '',
            updated_at TEXT NOT NULL,
            PRIMARY KEY (queue_id, step_index),
            FOREIGN KEY (queue_id) REFERENCES task_queues(queue_id) ON DELETE CASCADE
        );

        CREATE INDEX IF NOT EXISTS idx_task_steps_queue
            ON task_steps(queue_id, step_index);

        CREATE TABLE IF NOT EXISTS feishu_messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            message_id TEXT NOT NULL UNIQUE,
            chat_id TEXT NOT NULL DEFAULT '',
            sender_id TEXT NOT NULL DEFAULT '',
            sender_name TEXT NOT NULL DEFAULT '',
            message_type TEXT NOT NULL DEFAULT '',
            content TEXT NOT NULL DEFAULT '',
            raw_event TEXT NOT NULL DEFAULT '',
            received_at TEXT NOT NULL,
            report_id INTEGER
        );

        CREATE INDEX IF NOT EXISTS idx_feishu_messages_report
            ON feishu_messages(report_id, received_at);

        CREATE TABLE IF NOT EXISTS feishu_reports (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            summary TEXT NOT NULL,
            message_count INTEGER NOT NULL,
            created_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS codex_tasks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            original_task TEXT NOT NULL,
            rewritten_task TEXT NOT NULL,
            cwd TEXT NOT NULL DEFAULT '',
            status TEXT NOT NULL,
            exit_code INTEGER,
            stdout TEXT NOT NULL DEFAULT '',
            stderr TEXT NOT NULL DEFAULT '',
            created_at TEXT NOT NULL,
            completed_at TEXT
        );
        """
    )
