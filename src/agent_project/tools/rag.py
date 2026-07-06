from __future__ import annotations

import hashlib
import ast
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, get_args

from langchain_core.tools import tool

from agent_project.config import load_settings
from agent_project.rag_schemas import Citation, KnowledgeChunk, KnowledgeSource, RetrievedChunk, SourceType
from agent_project.tracing import TraceEvent
from agent_project.tools.progress import emit_progress
from agent_project.tools.storage import connect, state_db_path
from agent_project.tools.work_vectors import _cosine, _embed_texts, _normalize


DOCUMENT_SUFFIXES = {".md", ".markdown", ".txt", ".rst", ".py"}
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


def chunk_python_source(
    text: str,
    path: str = "",
    max_chars_per_chunk: int = 1600,
    overlap: int = 120,
) -> list[dict[str, str]]:
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return chunk_document_text(text, path=path, max_chars_per_chunk=max_chars_per_chunk, overlap=overlap)

    lines = text.splitlines()
    chunks: list[dict[str, str]] = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            start = max(getattr(node, "lineno", 1), 1)
            end = max(getattr(node, "end_lineno", start), start)
            symbol_type = "class" if isinstance(node, ast.ClassDef) else "function"
            symbol_text = "\n".join(lines[start - 1 : end]).strip()
            heading_path = f"{Path(path).name} > {symbol_type} {node.name}" if path else f"{symbol_type} {node.name}"
            for chunk_text in _split_text_windows(symbol_text, max_chars_per_chunk, overlap):
                chunks.append({"path": path, "heading_path": heading_path, "text": chunk_text})

    prelude_lines = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            break
        start = max(getattr(node, "lineno", 1), 1)
        end = max(getattr(node, "end_lineno", start), start)
        prelude_lines.extend(lines[start - 1 : end])
    prelude = "\n".join(prelude_lines).strip()
    if prelude:
        heading_path = f"{Path(path).name} > module prelude" if path else "module prelude"
        chunks.insert(0, {"path": path, "heading_path": heading_path, "text": prelude})

    if chunks:
        return chunks
    return chunk_document_text(text, path=path, max_chars_per_chunk=max_chars_per_chunk, overlap=overlap)


def chunk_source_text(
    text: str,
    path: str = "",
    max_chars_per_chunk: int = 1200,
    overlap: int = 120,
) -> list[dict[str, str]]:
    if Path(path).suffix.lower() == ".py":
        return chunk_python_source(
            text,
            path=path,
            max_chars_per_chunk=max_chars_per_chunk,
            overlap=overlap,
        )
    return chunk_document_text(
        text,
        path=path,
        max_chars_per_chunk=max_chars_per_chunk,
        overlap=overlap,
    )


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
                chunk_source_text(
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


def _rerank_score(query: str, chunk: KnowledgeChunk, lexical: float, vector_score: float) -> float:
    query_terms = _terms(query)
    heading_terms = _terms(f"{chunk.title} {chunk.heading_path}")
    heading_boost = 0.15 if query_terms and query_terms & heading_terms else 0.0
    citation_boost = 0.05 if chunk.citation_id else 0.0
    hybrid = (0.55 * vector_score if vector_score else 0.0) + (0.45 * lexical)
    if not vector_score:
        hybrid = lexical
    return min(1.0, hybrid + heading_boost + citation_boost)


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


def search_knowledge_records(
    query: str,
    limit: int = 5,
    source_types: list[str] | None = None,
) -> list[RetrievedChunk]:
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

    allowed_source_types = {
        _coerce_source_type(source_type)
        for source_type in (source_types or [])
        if source_type.strip()
    }

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
        if allowed_source_types and row["source_type"] not in allowed_source_types:
            continue
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
        chunk = _chunk_from_row(row, embedding)
        score = _rerank_score(clean_query, chunk, lexical, vector_score)
        method = "hybrid_rerank" if vector_score else "lexical_rerank"
        if score <= 0:
            continue
        scored.append((score, method, chunk))

    scored.sort(key=lambda item: item[0], reverse=True)
    return [
        RetrievedChunk(chunk=chunk, score=score, rank=index, retrieval_method=method)
        for index, (score, method, chunk) in enumerate(scored[:limit], start=1)
    ]


def retrieved_chunks_payload(results: list[RetrievedChunk]) -> list[dict[str, Any]]:
    payload: list[dict[str, Any]] = []
    for result in results:
        chunk = result.chunk
        payload.append(
            {
                "rank": result.rank,
                "query_score": round(result.score, 6),
                "score": round(result.score, 6),
                "retrieval_method": result.retrieval_method,
                "source": chunk.path or chunk.url or chunk.title,
                "source_id": chunk.source_id,
                "source_type": chunk.source_type,
                "chunk_id": chunk.chunk_id,
                "citation_id": chunk.citation_id,
                "title": chunk.title,
                "heading_path": chunk.heading_path,
                "snippet": _snippet(chunk.text),
            }
        )
    return payload


def citations_for_results(results: list[RetrievedChunk]) -> list[Citation]:
    citations: list[Citation] = []
    for result in results:
        chunk = result.chunk
        citations.append(
            Citation(
                citation_id=chunk.citation_id,
                source_id=chunk.source_id,
                chunk_id=chunk.chunk_id,
                label=chunk.title or chunk.path or chunk.citation_id,
                path=chunk.path,
                url=chunk.url,
                heading_path=chunk.heading_path,
            )
        )
    return citations


def search_knowledge_payload(query: str, limit: int = 5) -> dict[str, Any]:
    results = search_knowledge_records(query, limit=limit)
    return {
        "query": query.strip(),
        "top_k": max(1, min(limit, 20)),
        "results": retrieved_chunks_payload(results),
        "citations": [citation.model_dump() for citation in citations_for_results(results)],
    }


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


def format_answer_evidence(query: str, results: list[RetrievedChunk]) -> str:
    if not results:
        return "No grounded evidence found. Run index_documents first or broaden the query."
    citation_ids = [result.chunk.citation_id for result in results]
    lines = [f"Grounded evidence for: {query.strip()}"]
    for result in results:
        chunk = result.chunk
        label = chunk.title
        if chunk.heading_path:
            label = f"{label} / {chunk.heading_path}" if label else chunk.heading_path
        lines.append(
            f"- [{chunk.citation_id}] source={chunk.path or chunk.url or label} "
            f"score={result.score:.3f}"
        )
        lines.append(f"  {_snippet(chunk.text, max_chars=700)}")
    lines.append("")
    lines.append("Allowed citation IDs: " + ", ".join(citation_ids))
    lines.append(
        "Draft answer rule: every factual claim must include one of the allowed bracketed citations."
    )
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


def _answer_claim_terms(answer: str, citation_ids: set[str]) -> set[str]:
    clean = answer
    for citation_id in citation_ids:
        clean = clean.replace(f"[{citation_id}]", " ")
    stopwords = {
        "the",
        "and",
        "for",
        "with",
        "that",
        "this",
        "from",
        "into",
        "using",
        "use",
        "are",
        "was",
        "were",
        "根据",
        "引用",
        "来源",
        "当前",
        "可以",
    }
    return {term for term in _terms(clean) if len(term) > 1 and term not in stopwords}


def verify_answer_against_retrieved_chunks(
    answer: str,
    retrieved_chunks: list[RetrievedChunk],
) -> dict[str, Any]:
    id_result = verify_citation_ids(
        answer,
        [result.chunk.citation_id for result in retrieved_chunks],
    )
    if not id_result["ok"]:
        return {**id_result, "supported": False, "unsupported": id_result["used"]}

    used = set(id_result["used"])
    if not used:
        return {**id_result, "ok": False, "supported": False, "unsupported": [], "reason": "no citations used"}

    answer_terms = _answer_claim_terms(answer, used)
    chunks_by_citation = {result.chunk.citation_id: result.chunk for result in retrieved_chunks}
    unsupported: list[str] = []
    support: dict[str, dict[str, Any]] = {}
    for citation_id in sorted(used):
        chunk = chunks_by_citation[citation_id]
        evidence_terms = _terms(f"{chunk.title}\n{chunk.heading_path}\n{chunk.text}")
        overlap = sorted(answer_terms & evidence_terms)
        support[citation_id] = {
            "overlap_terms": overlap[:20],
            "overlap_count": len(overlap),
        }
        if answer_terms and not overlap:
            unsupported.append(citation_id)

    return {
        **id_result,
        "ok": not unsupported,
        "supported": not unsupported,
        "unsupported": unsupported,
        "support": support,
    }


def rag_search_trace_event(query: str, limit: int, results: list[RetrievedChunk]) -> TraceEvent:
    return TraceEvent(
        event_type="rag_retrieval",
        tool="search_knowledge",
        content=f"query={query.strip()} top_k={limit} results={len(results)}",
        args={
            "query": query.strip(),
            "top_k": max(1, min(limit, 20)),
            "results": retrieved_chunks_payload(results),
        },
        ok=True,
    )


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
    return format_answer_evidence(query, results)


@tool
def verify_answer_citations(answer: str, allowed_citation_ids: str) -> str:
    """Verify that bracketed citations in an answer are from the allowed citation IDs."""
    allowed = [part.strip() for part in re.split(r"[\n,]+", allowed_citation_ids) if part.strip()]
    result = verify_citation_ids(answer, allowed)
    if result["ok"]:
        return f"Citations verified. used={result['used']}"
    return f"Citation verification failed. missing={result['missing']} allowed={result['allowed']}"
