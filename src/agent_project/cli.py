from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage
from langchain_core.messages import AIMessage
from langgraph.errors import GraphRecursionError

from agent_project.agent import build_agent, invoke_agent, message_content_to_text, resume_human_gate
from agent_project.config import load_settings
from agent_project.sessions import (
    AgentSession,
    append_turn,
    create_session,
    find_session,
    format_session_line,
    list_sessions,
    load_session_messages,
)
from agent_project.tools.human_gate import approve_human_gate, list_human_gates
from agent_project.tools.rag import (
    format_answer_evidence,
    format_retrieved_chunks,
    index_document_paths,
    search_knowledge_records,
    verify_answer_against_retrieved_chunks,
)
from agent_project.tracing import TraceEvent, TraceStore


def _stream_agent_response(
    agent: Any,
    messages: list,
    recursion_limit: int,
    trace: TraceStore | None = None,
) -> list:
    """Stream the latest agent response and return the updated message history."""
    started_answer = False
    latest_messages = messages
    try:
        for stream_mode, chunk in agent.stream(
            {"messages": messages, "trace_id": trace.trace_id if trace else ""},
            config={"recursion_limit": recursion_limit},
            stream_mode=["messages", "values"],
        ):
            if stream_mode == "messages":
                message_chunk, _metadata = chunk
                if getattr(message_chunk, "type", None) != "AIMessageChunk":
                    continue

                text = message_content_to_text(message_chunk.content)
                if text:
                    if not started_answer:
                        print("\nAgent: ", end="", flush=True)
                        started_answer = True
                    print(text, end="", flush=True)

            elif stream_mode == "values":
                if "messages" in chunk:
                    latest_messages = chunk["messages"]
                if chunk.get("final_answer"):
                    latest_messages = [
                        *latest_messages,
                        AIMessage(content=chunk["final_answer"]),
                    ]
    except GraphRecursionError:
        if started_answer:
            print()
        answer = (
            "\nAgent stopped because it reached AGENT_RECURSION_LIMIT="
            f"{recursion_limit}. The partial session was saved; run "
            "`agent-chat resume --last` to continue, or raise AGENT_RECURSION_LIMIT "
            "for long research tasks."
        )
        print(answer)
        if trace is not None:
            trace.finish(answer, success=False, error_type="GraphRecursionError")
        return latest_messages

    if started_answer:
        print()
    elif latest_messages and getattr(latest_messages[-1], "type", "") == "ai":
        print(f"\nAgent: {message_content_to_text(latest_messages[-1].content)}")
    if trace is not None and latest_messages:
        answer = message_content_to_text(latest_messages[-1].content)
        trace.finish(answer, success=True)
    return latest_messages


def _choose_session(include_all: bool) -> AgentSession | None:
    sessions = list_sessions(include_all=include_all)
    if not sessions:
        scope = "all workspaces" if include_all else "this workspace"
        print(f"No saved sessions found for {scope}.")
        return None

    print("Saved sessions:")
    for index, session in enumerate(sessions[:20], start=1):
        print(format_session_line(session, index=index))

    choice = input("\nResume session number or ID: ").strip()
    if not choice:
        return None
    if choice.isdigit():
        selected_index = int(choice)
        if 1 <= selected_index <= min(len(sessions), 20):
            return sessions[selected_index - 1]
        print("Invalid session number.")
        return None

    session = find_session(choice, include_all=True)
    if session is None:
        print(f"No saved session found with ID prefix: {choice}")
    return session


def _resolve_resume_session(session_id: str, include_all: bool, last: bool) -> AgentSession | None:
    if session_id:
        session = find_session(session_id, include_all=True)
        if session is None:
            print(f"No saved session found with ID prefix: {session_id}")
        return session

    sessions = list_sessions(include_all=include_all)
    if last:
        if not sessions:
            scope = "all workspaces" if include_all else "this workspace"
            print(f"No saved sessions found for {scope}.")
            return None
        return sessions[0]

    return _choose_session(include_all=include_all)


def run_chat(session: AgentSession | None = None) -> None:
    settings = load_settings()
    agent = build_agent(settings)
    if session is None:
        session = create_session(cwd=str(Path.cwd()), provider=settings.model_provider)
        messages = []
        print(
            f"Agent ready. Provider: {settings.model_provider}. "
            f"Session: {session.session_id[:8]}. Type 'exit' or 'quit' to stop."
        )
    else:
        messages = load_session_messages(session)
        print(
            f"Agent ready. Provider: {settings.model_provider}. "
            f"Resumed session: {session.session_id[:8]} ({len(messages)} messages). "
            "Type 'exit' or 'quit' to stop."
        )

    while True:
        try:
            user_text = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            return

        if user_text.lower() in {"exit", "quit"}:
            print("Bye.")
            return
        if not user_text:
            continue

        messages.append(HumanMessage(content=user_text))
        trace = TraceStore()
        trace.start(user_input=user_text, model=_model_name(settings))
        try:
            messages = _stream_agent_response(
                agent,
                messages,
                settings.agent_recursion_limit,
                trace=trace,
            )
        except Exception as exc:
            trace.add_event(TraceEvent(event_type="error", content=str(exc), ok=False))
            trace.finish("", success=False, error_type=type(exc).__name__)
            raise
        session = append_turn(session, messages)


def _model_name(settings: Any) -> str:
    if settings.model_provider == "local-vllm":
        return settings.local_vllm_model
    if settings.model_provider == "openai-compatible":
        return settings.openai_compatible_model
    if settings.model_provider == "google":
        return settings.google_model
    if settings.model_provider == "anthropic":
        return settings.anthropic_model
    return settings.openai_model


def list_saved_sessions(include_all: bool = False) -> None:
    sessions = list_sessions(include_all=include_all)
    if not sessions:
        scope = "all workspaces" if include_all else "this workspace"
        print(f"No saved sessions found for {scope}.")
        return
    for index, session in enumerate(sessions, start=1):
        print(format_session_line(session, index=index))


def main() -> None:
    argv = sys.argv[1:]
    if argv and argv[0] == "knowledge":
        parser = argparse.ArgumentParser(description="Manage the local private knowledge base.")
        subparsers = parser.add_subparsers(dest="command", required=True)

        index_parser = subparsers.add_parser("index", help="Index Markdown/text files or folders.")
        index_parser.add_argument("paths", nargs="+", help="Document paths or directories.")
        index_parser.add_argument("--source-type", default="docs", help="Knowledge source type.")
        index_parser.add_argument("--force", action="store_true", help="Re-index unchanged documents.")
        index_parser.add_argument("--max-chars", type=int, default=1200, help="Maximum chunk size.")

        search_parser = subparsers.add_parser("search", help="Search indexed knowledge.")
        search_parser.add_argument("query", help="Search query.")
        search_parser.add_argument("--limit", type=int, default=5, help="Maximum chunks to return.")

        answer_parser = subparsers.add_parser(
            "answer",
            help="Return grounded evidence and allowed citation IDs for a query.",
        )
        answer_parser.add_argument("query", help="Question to answer from private knowledge.")
        answer_parser.add_argument("--limit", type=int, default=5, help="Maximum chunks to return.")

        verify_parser = subparsers.add_parser(
            "verify",
            help="Verify an answer against retrieved chunks for a query.",
        )
        verify_parser.add_argument("query", help="Original retrieval query.")
        verify_parser.add_argument("answer", help="Answer text containing bracketed citation IDs.")
        verify_parser.add_argument("--limit", type=int, default=5, help="Maximum chunks to retrieve.")

        args = parser.parse_args(argv[1:])
        load_settings()
        if args.command == "index":
            result = index_document_paths(
                "\n".join(args.paths),
                source_type=args.source_type,
                force=args.force,
                max_chars_per_chunk=args.max_chars,
            )
            print(
                "\n".join(
                    [
                        "Knowledge index ready.",
                        f"documents: {result['documents']}",
                        f"updated_documents: {result['updated_documents']}",
                        f"skipped_documents: {result.get('skipped_documents', 0)}",
                        f"chunks: {result['chunks']}",
                    ]
                )
            )
            if result.get("embedding_error"):
                print(f"embedding_fallback: lexical search ({result['embedding_error']})")
            return
        if args.command == "search":
            print(format_retrieved_chunks(search_knowledge_records(args.query, limit=args.limit)))
            return
        if args.command == "answer":
            print(format_answer_evidence(args.query, search_knowledge_records(args.query, limit=args.limit)))
            return
        if args.command == "verify":
            results = search_knowledge_records(args.query, limit=args.limit)
            print(verify_answer_against_retrieved_chunks(args.answer, results))
            return

    if argv and argv[0] == "human-gates":
        parser = argparse.ArgumentParser(description="List saved human gate approval requests.")
        parser.add_argument("--status", default="pending", help="Gate status to list.")
        parser.add_argument("--limit", type=int, default=20, help="Maximum rows to show.")
        args = parser.parse_args(argv[1:])
        load_settings()
        print(list_human_gates.invoke({"status": args.status, "limit": args.limit}))
        return

    if argv and argv[0] == "approve":
        parser = argparse.ArgumentParser(description="Approve and resume a saved human gate.")
        parser.add_argument("gate_id", help="Human gate ID.")
        parser.add_argument("--response", default="approved", help="Approval note to pass to the agent.")
        parser.add_argument(
            "--no-resume",
            action="store_true",
            help="Only approve the gate; do not resume the saved workflow.",
        )
        args = parser.parse_args(argv[1:])
        settings = load_settings()
        if args.no_resume:
            print(approve_human_gate.invoke({"gate_id": args.gate_id, "response": args.response}))
        else:
            print(resume_human_gate(args.gate_id, response=args.response, settings=settings))
        return

    if argv and argv[0] == "resume":
        parser = argparse.ArgumentParser(description="Resume a saved interactive session.")
        parser.add_argument("session_id", nargs="?", help="Session ID or unique prefix.")
        parser.add_argument(
            "--all",
            action="store_true",
            help="Include sessions outside the current working directory.",
        )
        parser.add_argument(
            "--last",
            action="store_true",
            help="Resume the most recent session without showing the picker.",
        )
        args = parser.parse_args(argv[1:])
        load_settings()
        session = _resolve_resume_session(
            session_id=args.session_id or "",
            include_all=args.all,
            last=args.last,
        )
        if session is not None:
            run_chat(session=session)
        return

    if argv and argv[0] == "sessions":
        parser = argparse.ArgumentParser(description="List saved sessions.")
        parser.add_argument(
            "--all",
            action="store_true",
            help="Include sessions outside the current working directory.",
        )
        args = parser.parse_args(argv[1:])
        load_settings()
        list_saved_sessions(include_all=args.all)
        return

    parser = argparse.ArgumentParser(
        description="Run the LangGraph agent.",
        epilog=(
            "Session commands: agent-chat resume [--last|--all] [session_id], "
            "agent-chat sessions [--all], agent-chat human-gates [--status pending], "
            "agent-chat approve <gate_id>"
        ),
    )
    parser.add_argument("message", nargs="*", help="Optional one-shot message.")
    args = parser.parse_args(argv)

    if args.message:
        print(invoke_agent(" ".join(args.message)))
        return

    run_chat()


if __name__ == "__main__":
    main()
