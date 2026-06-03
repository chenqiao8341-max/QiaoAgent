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
        """
    )
