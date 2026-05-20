from __future__ import annotations

from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent

from agent_project.config import Settings, load_settings
from agent_project.llms import build_chat_model
from agent_project.tools import get_tools


DEFAULT_SYSTEM_PROMPT = """You are a practical AI agent.
Use tools when they help answer accurately.
Keep answers concise, but explain important assumptions."""


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
    result = agent.invoke(
        {
            "messages": [HumanMessage(content=user_input)]
        }
    )
    return result["messages"][-1].content
