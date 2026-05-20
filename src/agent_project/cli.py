from __future__ import annotations

import argparse

from langchain_core.messages import HumanMessage

from agent_project.agent import build_agent, invoke_agent
from agent_project.config import load_settings


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
        result = agent.invoke({"messages": messages})
        messages = result["messages"]
        print(f"\nAgent: {messages[-1].content}")


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
