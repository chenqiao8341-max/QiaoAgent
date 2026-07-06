from __future__ import annotations

import json
import subprocess

import pytest

from agent_project.rag_schemas import Citation, KnowledgeChunk, KnowledgeSource
from agent_project.tools.git_safety import ensure_work_branch_state
from agent_project.tools.human_gate import (
    approve_human_gate,
    approve_human_gate_record,
    build_human_gate_resume_state,
    create_human_gate_request,
    list_human_gates,
)
from agent_project.tools.project_context import load_project_context_text
from agent_project.tools.rag import (
    citations_for_results,
    chunk_source_text,
    index_document_paths,
    rag_search_trace_event,
    search_knowledge_payload,
    search_knowledge_records,
    verify_answer_against_retrieved_chunks,
    verify_citation_ids,
)
from agent_project.tools.storage import connect


def test_rag_schema_defines_chunk_and_citation_contract() -> None:
    source = KnowledgeSource(
        source_id="source-1",
        source_type="docs",
        path="/tmp/docs/readme.md",
        title="README",
        indexed_at="2026-01-01T00:00:00+00:00",
    )
    chunk = KnowledgeChunk(
        chunk_id="chunk-1",
        source_id=source.source_id,
        source_type=source.source_type,
        path=source.path,
        title=source.title,
        heading_path="README > Intro",
        citation_id="docs/readme.md#intro",
        content_hash="abc",
        text="Project introduction",
        indexed_at=source.indexed_at,
    )
    citation = Citation(
        citation_id=chunk.citation_id,
        source_id=source.source_id,
        chunk_id=chunk.chunk_id,
        label="README intro",
        path=chunk.path,
        heading_path=chunk.heading_path,
    )

    assert citation.citation_id == "docs/readme.md#intro"
    assert chunk.source_type == "docs"


def test_storage_initializes_rag_and_human_gate_tables(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("AGENT_STATE_DB_PATH", str(tmp_path / "agent.sqlite3"))
    with connect() as connection:
        tables = {
            row["name"]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }

    assert "human_gate_requests" in tables
    assert "knowledge_sources" in tables
    assert "knowledge_chunks" in tables


def test_human_gate_request_round_trip(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("AGENT_STATE_DB_PATH", str(tmp_path / "agent.sqlite3"))

    gate_id = create_human_gate_request(
        user_input="dangerous change",
        reason="high risk",
        state={"status": "need_user", "plan": [{"step": 1}]},
    )

    assert gate_id in list_human_gates.invoke({"status": "pending"})
    assert "Approved human gate" in approve_human_gate.invoke({"gate_id": gate_id})
    with connect() as connection:
        row = connection.execute(
            "SELECT status, state_json FROM human_gate_requests WHERE gate_id = ?",
            (gate_id,),
        ).fetchone()

    assert row["status"] == "approved"
    assert json.loads(row["state_json"])["status"] == "need_user"


def test_human_gate_resume_state_clears_gate_and_preserves_approval(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("AGENT_STATE_DB_PATH", str(tmp_path / "agent.sqlite3"))

    gate_id = create_human_gate_request(
        user_input="delete production data",
        reason="high risk",
        state={
            "status": "need_user",
            "needs_human": True,
            "human_gate_reason": "high risk",
            "task_type": "code_task",
            "risk": "high",
            "plan": [{"step": 1, "action": "ask_user", "description": "confirm"}],
        },
    )

    ok, message = approve_human_gate_record(gate_id, response="approved by operator")
    resume_state = build_human_gate_resume_state(gate_id, response="approved by operator")

    assert ok is True
    assert "Approved human gate" in message
    assert resume_state["status"] == "running"
    assert resume_state["needs_human"] is False
    assert resume_state["human_gate_reason"] == ""
    assert resume_state["approved_human_gate_id"] == gate_id
    assert resume_state["approved_human_gate_reason"] == "high risk"
    assert resume_state["human_gate_response"] == "approved by operator"


def test_load_project_context_reads_status_and_git(tmp_path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    (project / "README.md").write_text("# Demo\n\nReadme body", encoding="utf-8")
    (project / "todo.md").write_text("- next task", encoding="utf-8")

    context = load_project_context_text(str(project))
    payload = json.loads(context)

    assert payload["project_path"] == str(project)
    assert "README.md" in payload["files"]
    assert "todo.md" in payload["files"]


def test_rag_indexes_searches_and_verifies_citations(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("AGENT_STATE_DB_PATH", str(tmp_path / "agent.sqlite3"))
    monkeypatch.setenv("AGENT_EMBEDDING_MODEL_PATH", str(tmp_path / "missing-embedding-model"))
    doc = tmp_path / "agent-notes.md"
    doc.write_text(
        "# Human Gate\n\nHuman gate approvals pause risky workflow execution.\n"
        "Operators approve a saved state before resume.\n",
        encoding="utf-8",
    )

    result = index_document_paths(str(doc), force=True)
    matches = search_knowledge_records("human gate saved state approval", limit=1)

    assert result["documents"] == 1
    assert result["chunks"] >= 1
    assert matches
    citation_id = matches[0].chunk.citation_id
    assert "agent-notes.md" in citation_id
    assert verify_citation_ids(f"Use approval before resume [{citation_id}]", [citation_id])["ok"]
    assert not verify_citation_ids("Unsupported [missing-citation]", [citation_id])["ok"]


def test_rag_chunks_python_source_by_function_and_class() -> None:
    chunks = chunk_source_text(
        "class Router:\n    pass\n\n\ndef build_plan():\n    return []\n",
        path="/repo/workflow.py",
    )
    headings = {chunk["heading_path"] for chunk in chunks}

    assert "workflow.py > class Router" in headings
    assert "workflow.py > function build_plan" in headings


def test_rag_payload_citations_and_trace_include_auditable_fields(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("AGENT_STATE_DB_PATH", str(tmp_path / "agent.sqlite3"))
    monkeypatch.setenv("AGENT_EMBEDDING_MODEL_PATH", str(tmp_path / "missing-embedding-model"))
    doc = tmp_path / "sop.md"
    doc.write_text(
        "# Release SOP\n\nRun live eval before a release and keep cited evidence in the trace.\n",
        encoding="utf-8",
    )

    index_document_paths(str(doc), source_type="sop", force=True)
    results = search_knowledge_records("release live eval cited evidence trace", limit=2)
    payload = search_knowledge_payload("release live eval cited evidence trace", limit=2)
    citations = citations_for_results(results)
    event = rag_search_trace_event("release live eval cited evidence trace", 2, results)

    assert payload["query"] == "release live eval cited evidence trace"
    assert payload["top_k"] == 2
    assert payload["results"][0]["source"]
    assert payload["results"][0]["chunk_id"]
    assert payload["results"][0]["citation_id"]
    assert payload["results"][0]["retrieval_method"].endswith("rerank")
    assert citations[0].chunk_id == results[0].chunk.chunk_id
    assert event.event_type == "rag_retrieval"
    assert event.args["query"] == "release live eval cited evidence trace"
    assert event.args["top_k"] == 2
    assert event.args["results"][0]["score"] >= 0


def test_rag_verifies_answer_support_against_retrieved_chunks(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("AGENT_STATE_DB_PATH", str(tmp_path / "agent.sqlite3"))
    monkeypatch.setenv("AGENT_EMBEDDING_MODEL_PATH", str(tmp_path / "missing-embedding-model"))
    doc = tmp_path / "rag.md"
    doc.write_text(
        "# Citation Verifier\n\nCitation verifier checks that answer citations come from retrieved chunks.\n",
        encoding="utf-8",
    )

    index_document_paths(str(doc), force=True)
    results = search_knowledge_records("citation verifier retrieved chunks", limit=1)
    citation_id = results[0].chunk.citation_id

    supported = verify_answer_against_retrieved_chunks(
        f"Citation verifier checks retrieved chunks [{citation_id}].",
        results,
    )
    unsupported = verify_answer_against_retrieved_chunks(
        f"Database migrations are automatically deployed [{citation_id}].",
        results,
    )

    assert supported["ok"] is True
    assert supported["supported"] is True
    assert unsupported["ok"] is False
    assert unsupported["unsupported"] == [citation_id]


def test_git_safety_creates_isolated_branch(tmp_path) -> None:
    if subprocess.run(["git", "--version"], capture_output=True, text=True).returncode != 0:
        pytest.skip("git is not available")

    repo = tmp_path / "repo"
    repo.mkdir()
    init = subprocess.run(["git", "init", "-b", "main"], cwd=repo, capture_output=True, text=True)
    if init.returncode != 0:
        subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True, text=True)
        subprocess.run(
            ["git", "checkout", "-b", "main"],
            cwd=repo,
            check=True,
            capture_output=True,
            text=True,
        )
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=repo, check=True)
    (repo / "README.md").write_text("# Repo\n", encoding="utf-8")
    subprocess.run(["git", "add", "README.md"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=repo, check=True, capture_output=True, text=True)

    dry_run = ensure_work_branch_state("Phase 3.5 Readiness", cwd=repo, dry_run=True)
    created = ensure_work_branch_state("Phase 3.5 Readiness", cwd=repo)
    current = subprocess.run(
        ["git", "branch", "--show-current"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    assert dry_run["action"] == "would_create_branch"
    assert dry_run["do_not_auto_merge"] is True
    assert created["action"] == "created_branch"
    assert created["current_branch"].startswith("agent/")
    assert current == created["current_branch"]
