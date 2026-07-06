from __future__ import annotations

from datetime import datetime, timezone

from langchain_core.tools import tool

from agent_project.tools.progress import emit_progress
from agent_project.tools.storage import connect, state_db_path


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _clean_namespace(namespace: str) -> str:
    return namespace.strip() or "default"


def _format_memory(row) -> str:
    tags = f" | tags: {row['tags']}" if row["tags"] else ""
    source = f" | source: {row['source']}" if row["source"] else ""
    return (
        f"Memory {row['id']} [{row['namespace']}]"
        f" created={row['created_at']}{source}{tags}\n{row['content']}"
    )


@tool
def remember_memory(
    content: str,
    namespace: str = "default",
    source: str = "",
    tags: str = "",
) -> str:
    """Persist a long-term memory item in SQLite."""
    text = content.strip()
    if not text:
        return "Cannot store an empty memory."

    namespace = _clean_namespace(namespace)
    now = _now()
    emit_progress(f"storing memory in namespace {namespace}: {text[:80]}")
    with connect() as connection:
        cursor = connection.execute(
            """
            INSERT INTO memories(namespace, content, source, tags, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (namespace, text, source.strip(), tags.strip(), now, now),
        )
        memory_id = cursor.lastrowid

    emit_progress(f"memory stored: {memory_id} ({state_db_path()})")
    return f"Stored memory {memory_id} in namespace {namespace}."


@tool
def search_memories(query: str = "", namespace: str = "default", limit: int = 10) -> str:
    """Search long-term memories by substring. Empty query returns recent memories."""
    namespace = _clean_namespace(namespace)
    limit = max(1, min(limit, 50))
    emit_progress(f"searching memories namespace={namespace} query={query!r}")

    with connect() as connection:
        if query.strip():
            pattern = f"%{query.strip()}%"
            rows = connection.execute(
                """
                SELECT * FROM memories
                WHERE namespace = ?
                  AND (content LIKE ? OR source LIKE ? OR tags LIKE ?)
                ORDER BY created_at DESC, id DESC
                LIMIT ?
                """,
                (namespace, pattern, pattern, pattern, limit),
            ).fetchall()
        else:
            rows = connection.execute(
                """
                SELECT * FROM memories
                WHERE namespace = ?
                ORDER BY created_at DESC, id DESC
                LIMIT ?
                """,
                (namespace, limit),
            ).fetchall()

    emit_progress(f"memory search complete: {len(rows)} results")
    if not rows:
        return f"No memories found in namespace {namespace}."
    return "\n\n".join(_format_memory(row) for row in rows)


@tool
def get_memory(memory_id: int) -> str:
    """Return one persisted long-term memory by numeric id."""
    emit_progress(f"reading memory: {memory_id}")
    with connect() as connection:
        row = connection.execute("SELECT * FROM memories WHERE id = ?", (memory_id,)).fetchone()

    if row is None:
        return f"Memory not found: {memory_id}"
    return _format_memory(row)


@tool
def delete_memory(memory_id: int) -> str:
    """Delete one persisted long-term memory by numeric id."""
    emit_progress(f"deleting memory: {memory_id}")
    with connect() as connection:
        cursor = connection.execute("DELETE FROM memories WHERE id = ?", (memory_id,))

    if cursor.rowcount == 0:
        return f"Memory not found: {memory_id}"
    emit_progress(f"memory deleted: {memory_id}")
    return f"Deleted memory {memory_id}."


def store_reflection_memory(
    *,
    task_type: str,
    failure_type: str,
    reflection: str,
    created_from_trace_id: str = "",
) -> int:
    text = reflection.strip()
    if not text:
        return 0
    namespace = "reflections"
    source = created_from_trace_id.strip()
    tags = ",".join(
        part
        for part in [
            f"task_type:{task_type.strip() or 'unknown'}",
            f"failure_type:{failure_type.strip() or 'unknown'}",
        ]
        if part
    )
    now = _now()
    with connect() as connection:
        cursor = connection.execute(
            """
            INSERT INTO memories(namespace, content, source, tags, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (namespace, text, source, tags, now, now),
        )
        return int(cursor.lastrowid or 0)


def reflection_memory_context(query: str, task_type: str = "", limit: int = 3) -> str:
    clean_query = query.strip()
    clean_task_type = task_type.strip()
    limit = max(1, min(limit, 10))
    patterns = [f"%{clean_query}%"] if clean_query else []
    if clean_task_type:
        patterns.append(f"%task_type:{clean_task_type}%")

    with connect() as connection:
        if patterns:
            clauses = " OR ".join("(content LIKE ? OR tags LIKE ?)" for _pattern in patterns)
            params: list[str | int] = []
            for pattern in patterns:
                params.extend([pattern, pattern])
            params.append(limit)
            rows = connection.execute(
                f"""
                SELECT * FROM memories
                WHERE namespace = 'reflections'
                  AND ({clauses})
                ORDER BY created_at DESC, id DESC
                LIMIT ?
                """,
                params,
            ).fetchall()
        else:
            rows = connection.execute(
                """
                SELECT * FROM memories
                WHERE namespace = 'reflections'
                ORDER BY created_at DESC, id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()

    if not rows:
        return ""
    lines = ["Relevant reflection memories:"]
    for row in rows:
        lines.append(f"- [{row['id']}] {row['tags']}: {row['content']}")
    return "\n".join(lines)



def recent_memory_context(namespace: str = "default", limit: int = 5) -> str:
    namespace = _clean_namespace(namespace)
    limit = max(0, min(limit, 20))
    if limit == 0:
        return ""

    with connect() as connection:
        rows = connection.execute(
            """
            SELECT * FROM memories
            WHERE namespace = ?
            ORDER BY created_at DESC, id DESC
            LIMIT ?
            """,
            (namespace, limit),
        ).fetchall()

    if not rows:
        return ""

    lines = ["Recent long-term memories:"]
    for row in rows:
        lines.append(f"- [{row['id']}] {row['content']}")
    return "\n".join(lines)
