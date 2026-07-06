from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


SourceType = Literal[
    "docs",
    "feishu_report",
    "work_record",
    "project_readme",
    "code_structure",
    "note",
    "paper_summary",
    "sop",
    "project_context",
]


class KnowledgeSource(BaseModel):
    source_id: str
    source_type: SourceType
    path: str = ""
    url: str = ""
    title: str = ""
    content_hash: str = ""
    mtime: float | None = None
    permissions: str = "default"
    indexed_at: str = ""


class KnowledgeChunk(BaseModel):
    chunk_id: str
    source_id: str
    source_type: SourceType
    path: str = ""
    url: str = ""
    title: str = ""
    heading_path: str = ""
    citation_id: str
    content_hash: str
    mtime: float | None = None
    text: str
    embedding_model: str = ""
    embedding: list[float] = Field(default_factory=list)
    permissions: str = "default"
    indexed_at: str = ""


class RetrievedChunk(BaseModel):
    chunk: KnowledgeChunk
    score: float
    rank: int
    retrieval_method: str


class Citation(BaseModel):
    citation_id: str
    source_id: str
    chunk_id: str
    label: str
    path: str = ""
    url: str = ""
    heading_path: str = ""
