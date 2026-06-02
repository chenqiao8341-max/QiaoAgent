from __future__ import annotations

import argparse
from typing import Any

from langchain_core.messages import HumanMessage

from agent_project.agent import build_agent, invoke_agent, message_content_to_text
from agent_project.config import load_settings


def _stream_agent_response(agent: Any, messages: list) -> list:
    """Stream the latest agent response and return the updated message history."""
    started_answer = False
    latest_messages = messages
    for stream_mode, chunk in agent.stream(
        {"messages": messages},
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


def run_chat() -> None:
    settings = load_settings()
    agent = build_agent(settings)
    messages = []

    print(f"Agent ready. Provider: {settings.model_provider}. Type 'exit' or 'quit' to stop.")
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
        messages = _stream_agent_response(agent, messages)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the LangGraph agent.")
    parser.add_argument("message", nargs="*", help="Optional one-shot message.")
    args = parser.parse_args()

    if args.message:
        print(invoke_agent(" ".join(args.message)))
        return

    run_chat()


if __name__ == "__main__":
    main()
