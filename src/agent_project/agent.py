from __future__ import annotations

from typing import Any

from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent

from agent_project.config import Settings, load_settings
from agent_project.llms import build_chat_model
from agent_project.tools import get_tools
from agent_project.tools.memory import recent_memory_context
from agent_project.tools.skills import skill_catalog_text


DEFAULT_SYSTEM_PROMPT = """You are a practical AI agent.
Use tools when they help answer accurately.
Use local file tools when the user asks you to inspect or edit files.
Use shell command tools when the user asks you to run commands or verify project behavior.
Use web search and browser tools when current external information or page inspection is needed.
For explicit search requests, call web_search before answering.
Only say search is unavailable if the tool returns an error.
Use memory tools to store and retrieve durable user preferences, project facts,
and reusable context.
Use Feishu tools to inspect captured messages and generate message reports.
Use skills when a listed skill matches the user request: first call read_skill for that skill,
then follow its instructions and read referenced skill files only when needed.
Use Codex delegation tools only when the user asks to hand a coding task to Codex;
rewrite the task before delegating it.
Use task queue tools to plan and track multi-step work; task queues persist across sessions.
Use work management tools when the user provides manually organized Feishu messages, work updates, or asks to manage work: capture messages, match them to the work record, create work tasks, and choose self/codex/ask_user/defer routes.
When routing to Codex, create a work task first, then rewrite the task before delegation.
When a tool asks for human approval, the prompt appears in the terminal, not a popup dialog.
If a file or shell tool returns denied, do not repeat the same tool call; tell the user what setting or terminal approval is needed.
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


def _system_prompt_with_context(system_prompt: str, settings: Settings) -> str:
    parts = [system_prompt]
    if settings.agent_enable_skills:
        skill_catalog = skill_catalog_text(limit=settings.agent_skill_catalog_limit)
        if skill_catalog:
            parts.append(skill_catalog)

    memory_context = recent_memory_context(limit=settings.agent_memory_context_limit)
    if memory_context:
        parts.append(memory_context)
    return "\n\n".join(parts)


def build_agent(settings: Settings | None = None, system_prompt: str = DEFAULT_SYSTEM_PROMPT):
    settings = settings or load_settings()
    system_prompt = _system_prompt_with_context(system_prompt, settings)
    model = build_chat_model(settings)
    tools = get_tools()
    try:
        return create_react_agent(model, tools, prompt=system_prompt)
    except TypeError:
        return create_react_agent(model, tools, state_modifier=system_prompt)


def invoke_agent(user_input: str, settings: Settings | None = None) -> str:
    settings = settings or load_settings()
    agent = build_agent(settings)
    result = agent.invoke(
        {"messages": [HumanMessage(content=user_input)]},
        config={"recursion_limit": settings.agent_recursion_limit},
    )
    return message_content_to_text(result["messages"][-1].content)
