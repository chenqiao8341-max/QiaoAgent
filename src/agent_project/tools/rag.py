from __future__ import annotations

import hashlib
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, get_args

from langchain_core.tools import tool

from agent_project.config import load_settings
from agent_project.rag_schemas import KnowledgeChunk, KnowledgeSource, RetrievedChunk, SourceType
from agent_project.tools.progress import emit_progress
from agent_project.tools.storage import connect, state_db_path
from agent_project.tools.work_vectors import _cosine, _embed_texts, _normalize


DOCUMENT_SUFFIXES = {".md", ".markdown", ".txt", ".rst"}
ALLOWED_SOURCE_TYPES = set(get_args(SourceType))


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _stable_id(*parts: str) -> str:
    joined = "\x1f".join(parts)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def _coerce_source_type(source_type: str) -> SourceType:
    clean = source_type.strip() or "docs"
    if clean not in ALLOWED_SOURCE_TYPES:
        return "docs"
    return clean  # type: ignore[return-value]


def _display_path(path: Path) -> str:
    try:
        return path.resolve().as_posix()
    except OSError:
        return path.as_posix()


def _heading_anchor(heading_path: str, fallback: str) -> str:
    heading = heading_path.split(">")[-1].strip().lower()
    slug = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "-", heading).strip("-")
    return slug or fallback


def _split_text_windows(text: str, max_chars: int, overlap: int) -> list[str]:
    clean = text.strip()
    if not clean:
        return []
    max_chars = max(300, min(max_chars, 8000))
    overlap = max(0, min(overlap, max_chars // 4))
    if len(clean) <= max_chars:
        return [clean]

    chunks: list[str] = []
    start = 0
    while start < len(clean):
        end = min(len(clean), start + max_chars)
        if end < len(clean):
            break_at = max(
                clean.rfind("\n\n", start, end),
                clean.rfind("\n", start, end),
                clean.rfind(" ", start, end),
            )
            if break_at > start + max_chars // 2:
                end = break_at
        chunk = clean[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(clean):
            break
        start = max(start + 1, end - overlap)
    return chunks


def chunk_document_text(
    text: str,
    path: str = "",
    max_chars_per_chunk: int = 1200,
    overlap: int = 120,
) -> list[dict[str, str]]:
    headings: list[str] = []
    section_lines: list[str] = []
    sections: list[tuple[str, str]] = []

    def flush() -> None:
        section_text = "\n".join(section_lines).strip()
        if section_text:
            sections.append((" > ".join(headings), section_text))
        section_lines.clear()

    for line in text.splitlines():
        match = re.match(r"^(#{1,6})\s+(.+?)\s*$", line)
        if match:
            flush()
            level = len(match.group(1))
            heading = match.group(2).strip()
            headings[:] = [*headings[: level - 1], heading]
        section_lines.append(line)
    flush()

    if not sections and text.strip():
        sections = [("", text.strip())]

    chunks: list[dict[str, str]] = []
    for heading_path, section_text in sections:
        for chunk_text in _split_text_windows(section_text, max_chars_per_chunk, overlap):
            chunks.append(
                {
                    "path": path,
                    "heading_path": heading_path,
                    "text": chunk_text,
                }
            )
    return chunks


def _iter_document_paths(paths: str) -> list[Path]:
    seen: set[Path] = set()
    results: list[Path] = []
    raw_parts = [part.strip() for part in re.split(r"[\n,]+", paths) if part.strip()]
    for raw in raw_parts:
        candidate = Path(raw).expanduser().resolve()
        if candidate.is_dir():
            files = sorted(
                path
                for path in candidate.rglob("*")
                if path.is_file()
                and path.suffix.lower() in DOCUMENT_SUFFIXES
                and ".git" not in path.parts
            )
        else:
            files = [candidate]
        for path in files:
            if path in seen or not path.exists() or not path.is_file():
                continue
            if path.suffix.lower() not in DOCUMENT_SUFFIXES:
                continue
            seen.add(path)
            results.append(path)
    return results


def index_document_paths(
    paths: str,
    source_type: str = "docs",
    force: bool = False,
    max_chars_per_chunk: int = 1200,
) -> dict[str, Any]:
    clean_source_type = _coerce_source_type(source_type)
    document_paths = _iter_document_paths(paths)
    if not document_paths:
        return {
            "documents": 0,
            "updated_documents": 0,
            "chunks": 0,
            "embedding_model": "",
            "embedding_error": "",
        }

    now = _now()
    prepared_sources: list[KnowledgeSource] = []
    prepared_chunks: list[KnowledgeChunk] = []
    skipped = 0

    with connect() as connection:
        for path in document_paths:
            text = path.read_text(encoding="utf-8", errors="replace")
            content_hash = _content_hash(text)
            display_path = _display_path(path)
            stat = path.stat()
            source_id = _stable_id(clean_source_type, display_path)
            existing = connection.execute(
                """
                SELECT content_hash FROM knowledge_sources
                WHERE source_id = ?
                """,
                (source_id,),
            ).fetchone()
            chunk_count = connection.execute(
                """
                SELECT COUNT(*) AS count FROM knowledge_chunks
                WHERE source_id = ?
                """,
                (source_id,),
            ).fetchone()["count"]
            if not force and existing and existing["content_hash"] == content_hash and chunk_count:
                skipped += 1
                continue

            title = path.stem
            prepared_sources.append(
                KnowledgeSource(
                    source_id=source_id,
                    source_type=clean_source_type,
                    path=display_path,
                    title=title,
                    content_hash=content_hash,
                    mtime=stat.st_mtime,
                    indexed_at=now,
                )
            )
            for index, chunk in enumerate(
                chunk_document_text(
                    text,
                    path=display_path,
                    max_chars_per_chunk=max_chars_per_chunk,
                ),
                start=1,
            ):
                chunk_hash = _content_hash(chunk["text"])
                anchor = _heading_anchor(chunk["heading_path"], f"chunk-{index}")
                citation_id = f"{display_path}#{anchor}-{index}"
                prepared_chunks.append(
                    KnowledgeChunk(
                        chunk_id=_stable_id(source_id, str(index), chunk_hash),
                        source_id=source_id,
                        source_type=clean_source_type,
                        path=display_path,
                        title=title,
                        heading_path=chunk["heading_path"],
                        citation_id=citation_id,
                        content_hash=chunk_hash,
                        mtime=stat.st_mtime,
                        text=chunk["text"],
                        indexed_at=now,
                    )
                )

    embedding_model = ""
    embedding_error = ""
    embeddings: list[list[float]] = [[] for _chunk in prepared_chunks]
    if prepared_chunks:
        model_path = load_settings().agent_embedding_model_path
        try:
            embeddings = [_normalize(vector) for vector in _embed_texts([c.text for c in prepared_chunks], model_path)]
            embedding_model = model_path
        except Exception as exc:
            embedding_error = str(exc)

    with connect() as connection:
        for source in prepared_sources:
            connection.execute(
                """
                INSERT INTO knowledge_sources(
                    source_id, source_type, path, url, title, content_hash,
                    mtime, permissions, indexed_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(source_id) DO UPDATE SET
                    source_type = excluded.source_type,
                    path = excluded.path,
                    url = excluded.url,
                    title = excluded.title,
                    content_hash = excluded.content_hash,
                    mtime = excluded.mtime,
                    permissions = excluded.permissions,
                    indexed_at = excluded.indexed_at
                """,
                (
                    source.source_id,
                    source.source_type,
                    source.path,
                    source.url,
                    source.title,
                    source.content_hash,
                    source.mtime,
                    source.permissions,
                    source.indexed_at,
                ),
            )
            connection.execute("DELETE FROM knowledge_chunks WHERE source_id = ?", (source.source_id,))

        for chunk, embedding in zip(prepared_chunks, embeddings):
            connection.execute(
                """
                INSERT INTO knowledge_chunks(
                    chunk_id, source_id, source_type, path, url, title, heading_path,
                    citation_id, content_hash, mtime, text, embedding_model,
                    embedding_json, permissions, indexed_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    chunk.chunk_id,
                    chunk.source_id,
                    chunk.source_type,
                    chunk.path,
                    chunk.url,
                    chunk.title,
                    chunk.heading_path,
                    chunk.citation_id,
                    chunk.content_hash,
                    chunk.mtime,
                    chunk.text,
                    embedding_model,
                    json.dumps(embedding, ensure_ascii=False),
                    chunk.permissions,
                    chunk.indexed_at,
                ),
            )

    return {
        "documents": len(document_paths),
        "updated_documents": len(prepared_sources),
        "skipped_documents": skipped,
        "chunks": len(prepared_chunks),
        "embedding_model": embedding_model,
        "embedding_error": embedding_error,
    }


def _terms(text: str) -> set[str]:
    return {
        term.lower()
        for term in re.findall(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]", text)
        if term.strip()
    }


def _lexical_score(query: str, text: str) -> float:
    query_terms = _terms(query)
    text_terms = _terms(text)
    if not query_terms or not text_terms:
        return 0.0
    overlap = len(query_terms & text_terms)
    return overlap / math.sqrt(len(query_terms) * len(text_terms))


def _chunk_from_row(row: Any, embedding: list[float]) -> KnowledgeChunk:
    return KnowledgeChunk(
        chunk_id=row["chunk_id"],
        source_id=row["source_id"],
        source_type=row["source_type"],
        path=row["path"],
        url=row["url"],
        title=row["title"],
        heading_path=row["heading_path"],
        citation_id=row["citation_id"],
        content_hash=row["content_hash"],
        mtime=row["mtime"],
        text=row["text"],
        embedding_model=row["embedding_model"],
        embedding=embedding,
        permissions=row["permissions"],
        indexed_at=row["indexed_at"],
    )


def search_knowledge_records(query: str, limit: int = 5) -> list[RetrievedChunk]:
    clean_query = query.strip()
    if not clean_query:
        return []
    limit = max(1, min(limit, 20))
    settings = load_settings()
    model_path = settings.agent_embedding_model_path

    query_vector: list[float] = []
    try:
        query_vector = _normalize(_embed_texts([clean_query], model_path)[0])
    except Exception:
        query_vector = []

    with connect() as connection:
        rows = connection.execute(
            """
            SELECT *
            FROM knowledge_chunks
            ORDER BY indexed_at DESC
            """
        ).fetchall()

    scored: list[tuple[float, str, KnowledgeChunk]] = []
    for row in rows:
        try:
            embedding = json.loads(row["embedding_json"] or "[]")
        except json.JSONDecodeError:
            embedding = []
        lexical = _lexical_score(clean_query, f"{row['title']}\n{row['heading_path']}\n{row['text']}")
        vector_score = (
            _cosine(query_vector, embedding)
            if query_vector and embedding and row["embedding_model"] == model_path
            else 0.0
        )
        if vector_score:
            score = max(vector_score, lexical * 0.75)
            method = "vector"
        else:
            score = lexical
            method = "lexical"
        if score <= 0:
            continue
        scored.append((score, method, _chunk_from_row(row, embedding)))

    scored.sort(key=lambda item: item[0], reverse=True)
    return [
        RetrievedChunk(chunk=chunk, score=score, rank=index, retrieval_method=method)
        for index, (score, method, chunk) in enumerate(scored[:limit], start=1)
    ]


def _snippet(text: str, max_chars: int = 500) -> str:
    clean = re.sub(r"\s+", " ", text).strip()
    if len(clean) <= max_chars:
        return clean
    return clean[:max_chars].rstrip() + "..."


def format_retrieved_chunks(results: list[RetrievedChunk]) -> str:
    if not results:
        return "No knowledge chunks found. Run index_documents first or broaden the query."
    lines = [f"Knowledge results ({state_db_path()}):"]
    for result in results:
        chunk = result.chunk
        heading = f" heading={chunk.heading_path}" if chunk.heading_path else ""
        lines.append(
            f"{result.rank}. score={result.score:.3f} method={result.retrieval_method} "
            f"[{chunk.citation_id}] title={chunk.title}{heading}"
        )
        lines.append(f"   {_snippet(chunk.text)}")
    return "\n".join(lines)


def extract_citation_ids(text: str) -> set[str]:
    return {
        match.strip()
        for match in re.findall(r"\[([^\[\]\n]{1,240})\]", text)
        if match.strip()
    }


def verify_citation_ids(answer: str, allowed_citation_ids: list[str] | set[str]) -> dict[str, Any]:
    used = extract_citation_ids(answer)
    allowed = {citation.strip() for citation in allowed_citation_ids if citation.strip()}
    missing = sorted(citation for citation in used if citation not in allowed)
    return {
        "ok": not missing,
        "used": sorted(used),
        "allowed": sorted(allowed),
        "missing": missing,
    }


@tool
def index_documents(
    paths: str,
    source_type: str = "docs",
    force: bool = False,
    max_chars_per_chunk: int = 1200,
) -> str:
    """Index Markdown/text documents into the local knowledge chunk store."""
    emit_progress(f"indexing documents: {paths[:120]}")
    result = index_document_paths(
        paths=paths,
        source_type=source_type,
        force=force,
        max_chars_per_chunk=max_chars_per_chunk,
    )
    lines = [
        f"Knowledge index ready ({state_db_path()}).",
        f"documents: {result['documents']}",
        f"updated_documents: {result['updated_documents']}",
        f"skipped_documents: {result.get('skipped_documents', 0)}",
        f"chunks: {result['chunks']}",
    ]
    if result.get("embedding_model"):
        lines.append(f"embedding_model: {result['embedding_model']}")
    if result.get("embedding_error"):
        lines.append(f"embedding_fallback: lexical search ({result['embedding_error']})")
    return "\n".join(lines)


@tool
def search_knowledge(query: str, limit: int = 5) -> str:
    """Search indexed knowledge chunks and return citation IDs with snippets."""
    emit_progress(f"searching knowledge: {query[:100]}")
    return format_retrieved_chunks(search_knowledge_records(query, limit=limit))


@tool
def answer_with_citations(query: str, limit: int = 5) -> str:
    """Return grounded evidence snippets and citation IDs for a cited answer."""
    results = search_knowledge_records(query, limit=limit)
    if not results:
        return "No grounded evidence found. Run index_documents first or broaden the query."
    citation_ids = [result.chunk.citation_id for result in results]
    lines = ["Grounded evidence:"]
    for result in results:
        chunk = result.chunk
        lines.append(f"- [{chunk.citation_id}] {_snippet(chunk.text, max_chars=700)}")
    lines.append("")
    lines.append("Allowed citation IDs: " + ", ".join(citation_ids))
    return "\n".join(lines)


@tool
def verify_answer_citations(answer: str, allowed_citation_ids: str) -> str:
    """Verify that bracketed citations in an answer are from the allowed citation IDs."""
    allowed = [part.strip() for part in re.split(r"[\n,]+", allowed_citation_ids) if part.strip()]
    result = verify_citation_ids(answer, allowed)
    if result["ok"]:
        return f"Citations verified. used={result['used']}"
    return f"Citation verification failed. missing={result['missing']} allowed={result['allowed']}"
