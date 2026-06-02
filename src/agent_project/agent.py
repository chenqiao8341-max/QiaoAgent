from __future__ import annotations

from typing import Any

from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent

from agent_project.config import Settings, load_settings
from agent_project.llms import build_chat_model
from agent_project.tools import get_tools


DEFAULT_SYSTEM_PROMPT = """You are a practical AI agent.
Use tools when they help answer accurately.
Use local file tools when the user asks you to inspect or edit files.
Use shell command tools when the user asks you to run commands or verify project behavior.
Use web search and browser tools when current external information or page inspection is needed.
For explicit search requests, call web_search before answering; only say search is unavailable if the tool returns an error.
Use task queue tools to plan and track multi-step work.
When a tool asks for human approval, wait for the user decision and respect it.
Keep answers concise, but explain important assumptions."""


def message_content_to_text(content: Any) -> str:
    """Convert LangChain message content or content blocks into displayable text."""
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and "text" in item:
                parts.append(str(item["text"]))
            elif hasattr(item, "text"):
                parts.append(str(item.text))
        return "\n".join(part for part in parts if part)

    if isinstance(content, dict) and "text" in content:
        return str(content["text"])

    return str(content)


def build_agent(settings: Settings | None = None, system_prompt: str = DEFAULT_SYSTEM_PROMPT):
    settings = settings or load_settings()
    model = build_chat_model(settings)
    tools = get_tools()
    try:
        return create_react_agent(model, tools, prompt=system_prompt)
    except TypeError:
        return create_react_agent(model, tools, state_modifier=system_prompt)


def invoke_agent(user_input: str, settings: Settings | None = None) -> str:
    agent = build_agent(settings)
    result = agent.invoke({"messages": [HumanMessage(content=user_input)]})
    return message_content_to_text(result["messages"][-1].content)
