from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

from langchain_core.tools import tool

from agent_project.config import load_settings
from agent_project.tools.progress import emit_progress
from agent_project.tools.storage import connect, state_db_path
from agent_project.tools.work_management import WorkRecordItem, parse_work_record, work_record_path


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _item_text(item: WorkRecordItem) -> str:
    parts = [item.title]
    if item.paths:
        parts.append("项目文件: " + " ".join(item.paths))
    if item.progress:
        parts.append("工作进展: " + item.progress)
    if item.raw:
        parts.append(item.raw)
    return "\n".join(parts)


def _content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        return vector
    return [value / norm for value in vector]


def _cosine(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    return sum(a * b for a, b in zip(left, right))


@lru_cache(maxsize=2)
def _load_embedding_model(model_path: str, device: str) -> Any:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError(
            "sentence-transformers is not installed. Run `pip install -e .` first."
        ) from exc

    path = Path(model_path).expanduser().resolve()
    if not path.exists():
        raise RuntimeError(f"Embedding model path does not exist: {path}")
    return SentenceTransformer(str(path), local_files_only=True, device=device)


def _embed_texts(texts: list[str], model_path: str) -> list[list[float]]:
    device = load_settings().agent_embedding_device
    model = _load_embedding_model(model_path, device)
    vectors = model.encode(texts, normalize_embeddings=True)
    return [list(map(float, vector)) for vector in vectors]


def _current_items() -> list[WorkRecordItem]:
    path = work_record_path()
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8", errors="replace")
    return parse_work_record(text)


def index_work_record_items(force: bool = False) -> tuple[int, int]:
    settings = load_settings()
    items = _current_items()
    if not items:
        return 0, 0

    model_path = settings.agent_embedding_model_path
    prepared = [(item, _item_text(item)) for item in items]
    hashes = {item.title: _content_hash(text) for item, text in prepared}

    to_embed: list[tuple[WorkRecordItem, str]] = []
    with connect() as connection:
        for item, text in prepared:
            row = connection.execute(
                """
                SELECT content_hash, model_path FROM work_record_vectors
                WHERE title = ?
                """,
                (item.title,),
            ).fetchone()
            if (
                force
                or row is None
                or row["content_hash"] != hashes[item.title]
                or row["model_path"] != model_path
            ):
                to_embed.append((item, text))

    if not to_embed:
        return len(items), 0

    embeddings = _embed_texts([text for _item, text in to_embed], model_path)
    now = _now()
    with connect() as connection:
        for (item, _text), embedding in zip(to_embed, embeddings):
            connection.execute(
                """
                INSERT INTO work_record_vectors(
                    title, paths, progress, raw, content_hash, embedding_json,
                    model_path, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(title) DO UPDATE SET
                    paths = excluded.paths,
                    progress = excluded.progress,
                    raw = excluded.raw,
                    content_hash = excluded.content_hash,
                    embedding_json = excluded.embedding_json,
                    model_path = excluded.model_path,
                    updated_at = excluded.updated_at
                """,
                (
                    item.title,
                    json.dumps(item.paths, ensure_ascii=False),
                    item.progress,
                    item.raw,
                    hashes[item.title],
                    json.dumps(_normalize(embedding)),
                    model_path,
                    now,
                ),
            )
    return len(items), len(to_embed)


def search_work_record_items(query: str, limit: int = 5) -> list[dict[str, Any]]:
    text = query.strip()
    if not text:
        return []

    settings = load_settings()
    model_path = settings.agent_embedding_model_path
    query_vector = _normalize(_embed_texts([text], model_path)[0])
    with connect() as connection:
        rows = connection.execute(
            """
            SELECT * FROM work_record_vectors
            WHERE model_path = ?
            """,
            (model_path,),
        ).fetchall()

    scored = []
    for row in rows:
        try:
            embedding = json.loads(row["embedding_json"])
            paths = json.loads(row["paths"] or "[]")
        except json.JSONDecodeError:
            continue
        scored.append(
            {
                "title": row["title"],
                "paths": paths,
                "progress": row["progress"],
                "raw": row["raw"],
                "score": _cosine(query_vector, embedding),
                "updated_at": row["updated_at"],
            }
        )
    scored.sort(key=lambda item: item["score"], reverse=True)
    return scored[: max(1, min(limit, 20))]


def best_work_record_match(query: str, threshold: float = 0.25) -> tuple[str, float]:
    try:
        if not search_work_record_items("", limit=1):
            index_work_record_items(force=False)
        results = search_work_record_items(query, limit=1)
    except RuntimeError:
        return "", 0.0
    if not results or results[0]["score"] < threshold:
        return "", results[0]["score"] if results else 0.0
    return str(results[0]["title"]), float(results[0]["score"])


@tool
def index_work_record_vectors(force: bool = False) -> str:
    """Build or refresh the embedding index for the configured work record Markdown file."""
    path = work_record_path()
    emit_progress(f"indexing work record vectors: {path}")
    try:
        total, indexed = index_work_record_items(force=force)
    except RuntimeError as exc:
        return f"Cannot index work record vectors: {exc}"
    if total == 0:
        return f"No work record items found in {path}."
    return (
        f"Work record vector index ready ({state_db_path()}).\n"
        f"items: {total}\n"
        f"embedded_or_updated: {indexed}"
    )


@tool
def search_work_record_vectors(query: str, limit: int = 5) -> str:
    """Search work record items by semantic similarity using the local embedding model."""
    clean_query = query.strip()
    if not clean_query:
        return "Cannot search work record vectors without a query."
    emit_progress(f"searching work record vectors: {clean_query[:80]}")
    try:
        results = search_work_record_items(clean_query, limit=limit)
    except RuntimeError as exc:
        return f"Cannot search work record vectors: {exc}"
    if not results:
        return "No work record vector results found. Run index_work_record_vectors first."

    lines = [f"Work record vector results ({state_db_path()}):"]
    for index, result in enumerate(results, start=1):
        lines.append(f"{index}. {result['title']} score={result['score']:.3f}")
        for path in result["paths"]:
            lines.append(f"   - {path}")
        progress = str(result["progress"]).replace("\n", " ")[:180]
        if progress:
            lines.append(f"   Progress: {progress}")
    return "\n".join(lines)
