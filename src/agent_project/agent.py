from __future__ import annotations

from typing import Any

from langchain_core.messages import HumanMessage
from langgraph.errors import GraphRecursionError
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
For research-and-report tasks, make a brief source plan, diversify searches,
avoid opening the same URL repeatedly, and keep a small working list of sources already read.
PDF pages can be opened with open_web_page; use their extracted text when available.
For broad overview reports, 6-10 high-quality, diverse sources are usually enough unless
the user explicitly asks for exhaustive coverage.
Once you have enough sources for a useful answer, write the requested file before doing
more optional searching. If a research tool says a budget or hard limit was reached,
stop opening new sources for that task and synthesize, write, or ask the user instead.
Use memory tools to store and retrieve durable user preferences, project facts,
and reusable context.
Use Feishu tools to inspect captured messages and generate message reports.
Use skills when a listed skill matches the user request: first call read_skill for that skill,
then follow its instructions and read referenced skill files only when needed.
Use Codex delegation tools only when the user asks to hand a task to Codex.
For one-shot tasks use run_codex_task.
For interactive Codex work, call start_codex_session, read the returned output and session_id,
then call continue_codex_session with that exact session_id as many times as needed.
After every Codex response, decide whether the user goal is complete before sending another prompt.
For substantial tasks, judge difficulty yourself with record_task_difficulty_judgment.
Use your own task understanding rather than fixed keyword rules.
If you judge a task hard, first make one serious attempt yourself.
Then create a Codex review packet with create_codex_review_packet and start review with
start_codex_review. Include the task, your attempt summary, produced files, source URLs,
important excerpts or excerpt files, unresolved uncertainties, and focused questions.
For research reports, pass the Markdown report path, source URLs, source notes, and key
text extracted from important PDFs/pages so Codex does not need to redo retrieval.
If the user names codex-proxy-* or another Codex wrapper, pass it as codex_command, never as model.
Do not inspect, print, curl, or otherwise expose Codex API keys.
Use test_codex_connectivity or the Codex wrapper command instead.
If a Codex session tool fails, retry that Codex tool at most once or report the failure.
Do not switch to shell commands to inspect Codex configuration, credentials, or APIs.
Rewrite the task before one-shot delegation when useful.
Use task queue tools to plan and track multi-step work; task queues persist across sessions.
Use work management tools when the user provides manually organized Feishu messages,
work updates, or asks to manage work: capture messages, match them to the work record,
create work tasks, and choose self/codex/ask_user/defer routes.
When routing to Codex, create a work task first, then rewrite the task before delegation.
When a tool asks for human approval, the prompt appears in the terminal, not a popup dialog.
If a file or shell tool returns denied, do not repeat the same tool call.
Tell the user what setting or terminal approval is needed.
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
    try:
        result = agent.invoke(
            {"messages": [HumanMessage(content=user_input)]},
            config={"recursion_limit": settings.agent_recursion_limit},
        )
    except GraphRecursionError:
        return (
            "Agent stopped because it reached AGENT_RECURSION_LIMIT="
            f"{settings.agent_recursion_limit}. Raise AGENT_RECURSION_LIMIT for long "
            "research tasks, or use interactive chat sessions so the work can be resumed."
        )
    return message_content_to_text(result["messages"][-1].content)
