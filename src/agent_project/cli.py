from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage

from agent_project.agent import build_agent, invoke_agent, message_content_to_text
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


def _stream_agent_response(agent: Any, messages: list, recursion_limit: int) -> list:
    """Stream the latest agent response and return the updated message history."""
    started_answer = False
    latest_messages = messages
    for stream_mode, chunk in agent.stream(
        {"messages": messages},
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
            latest_messages = chunk["messages"]

    if started_answer:
        print()
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
        messages = _stream_agent_response(agent, messages, settings.agent_recursion_limit)
        session = append_turn(session, messages)


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
            "agent-chat sessions [--all]"
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
