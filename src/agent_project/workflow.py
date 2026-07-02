from __future__ import annotations

import json
import re
from typing import Any, Literal, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import create_react_agent

from agent_project.config import Settings
from agent_project.llms import build_chat_model
from agent_project.tools import get_tools
from agent_project.tracing import TraceEvent


TaskType = Literal["chat", "work_message", "code_task", "research", "file_task", "rag_qa"]
WorkflowStatus = Literal["running", "need_user", "done", "failed"]
WorkflowRoute = Literal["self", "codex", "ask_user", "defer"]
WorkflowRisk = Literal["low", "medium", "high"]
WorkflowDifficulty = Literal["low", "high"]


class AgentState(TypedDict, total=False):
    messages: list[Any]
    user_input: str
    task_type: TaskType
    route: WorkflowRoute
    risk: WorkflowRisk
    difficulty: WorkflowDifficulty
    plan: list[dict[str, Any]]
    current_step: int
    tool_results: list[dict[str, Any]]
    reflections: list[str]
    final_answer: str
    status: WorkflowStatus
    error_type: str
    trace_events: list[TraceEvent]


ROUTER_PROMPT = """You are the Router node.
Classify the user request as chat, work_message, code_task, research, file_task, or rag_qa.
Prefer work_message for Feishu/work updates and project routing.
Prefer code_task for implementation, debugging, deployment, tests, or shell-heavy work.
Prefer file_task for local file inspection or document edits.
Prefer research for web/current-information reports.
Prefer rag_qa for questions grounded in local docs or work records."""

PLANNER_PROMPT = """You are the Planner node.
Create a short structured plan with at most five steps.
The plan should be executable by the available tools or by Codex delegation."""

EXECUTOR_PROMPT = """You are the Executor node.
Execute the plan using available tools. For work messages, capture and route the message first.
For code/deployment/test-heavy tasks, create a work task and prepare Codex delegation when useful.
Respect approval settings and stop if user confirmation is required."""

VERIFIER_PROMPT = """You are the Verifier node.
Check whether the result satisfies the user request and whether more tool work or user input is needed."""

REFLECTOR_PROMPT = """You are the Reflector node.
If the workflow failed, summarize the failure cause and what should be done differently next time."""

FINALIZER_PROMPT = """You are the Finalizer node.
Produce a concise user-facing answer with the result, important assumptions, and verification status."""


def build_workflow_agent(settings: Settings, system_prompt: str):
    model = build_chat_model(settings)
    tools = get_tools()
    executor = _build_react_executor(model, tools, f"{system_prompt}\n\n{EXECUTOR_PROMPT}")

    graph = StateGraph(AgentState)
    graph.add_node("router", _router_node(model))
    graph.add_node("planner", _planner_node(model))
    graph.add_node("executor", _executor_node(executor, settings))
    graph.add_node("verifier", _verifier_node(model))
    graph.add_node("reflector", _reflector_node(model))
    graph.add_node("finalizer", _finalizer_node(model))

    graph.set_entry_point("router")
    graph.add_edge("router", "planner")
    graph.add_conditional_edges(
        "planner",
        _after_planner,
        {
            "need_user": "finalizer",
            "execute": "executor",
        },
    )
    graph.add_edge("executor", "verifier")
    graph.add_conditional_edges(
        "verifier",
        _after_verifier,
        {
            "reflect": "reflector",
            "finalize": "finalizer",
        },
    )
    graph.add_edge("reflector", "finalizer")
    graph.add_edge("finalizer", END)
    return graph.compile()


def _build_react_executor(model: Any, tools: list[Any], prompt: str):
    try:
        return create_react_agent(model, tools, prompt=prompt)
    except TypeError:
        return create_react_agent(model, tools, state_modifier=prompt)


def _router_node(model: Any):
    def run(state: AgentState) -> AgentState:
        user_input = _state_user_input(state)
        fallback_task_type = classify_task_type(user_input)
        fallback_route, fallback_risk, fallback_difficulty = classify_route_risk_difficulty(
            user_input,
            fallback_task_type,
        )
        payload = _invoke_json_node(
            model=model,
            system_prompt=ROUTER_PROMPT
            + """
Return only JSON:
{
  "task_type": "chat|work_message|code_task|research|file_task|rag_qa",
  "route": "self|codex|ask_user|defer",
  "risk": "low|medium|high",
  "difficulty": "low|high",
  "rationale": "short reason"
}
Use ask_user for high-risk or underspecified requests. Use codex for complex code/deploy/test work.
""",
            user_prompt=f"Classify this request:\n{user_input}",
            fallback={
                "task_type": fallback_task_type,
                "route": fallback_route,
                "risk": fallback_risk,
                "difficulty": fallback_difficulty,
                "rationale": "deterministic fallback",
            },
        )
        task_type = _coerce_literal(
            payload.get("task_type"),
            {"chat", "work_message", "code_task", "research", "file_task", "rag_qa"},
            fallback_task_type,
        )
        route = _coerce_literal(
            payload.get("route"),
            {"self", "codex", "ask_user", "defer"},
            fallback_route,
        )
        risk = _coerce_literal(payload.get("risk"), {"low", "medium", "high"}, fallback_risk)
        difficulty = _coerce_literal(payload.get("difficulty"), {"low", "high"}, fallback_difficulty)
        return {
            **state,
            "user_input": user_input,
            "task_type": task_type,
            "route": route,
            "risk": risk,
            "difficulty": difficulty,
            "status": "running",
            "trace_events": _append_event(
                state,
                TraceEvent(
                    event_type="router",
                    content=(
                        f"task_type={task_type} route={route} risk={risk} "
                        f"difficulty={difficulty}; {payload.get('rationale', '')}"
                    ),
                    ok=True,
                ),
            ),
        }

    return run


def _planner_node(model: Any):
    def run(state: AgentState) -> AgentState:
        fallback_plan = build_plan(
            user_input=state.get("user_input", ""),
            task_type=state.get("task_type", "chat"),
            route=state.get("route", "self"),
            risk=state.get("risk", "low"),
            difficulty=state.get("difficulty", "low"),
        )
        payload = _invoke_json_node(
            model=model,
            system_prompt=PLANNER_PROMPT
            + """
Return only JSON:
{
  "status": "running|need_user",
  "plan": [
    {"step": 1, "action": "route_work|retrieve_context|execute|delegate|verify|ask_user", "description": "..."}
  ],
  "rationale": "short reason"
}
Rules:
- Use at most five steps.
- For high risk, status must be need_user and plan must ask for confirmation.
- For "belongs to which project" or work-record matching, include search_work_record_vectors or capture_work_message in the plan description.
- For route=codex, include create_work_task and rewrite_task_for_codex before delegation.
""",
            user_prompt=_planner_user_prompt(state, fallback_plan),
            fallback={"status": "need_user" if state.get("risk") == "high" else "running", "plan": fallback_plan},
        )
        plan = _coerce_plan(payload.get("plan"), fallback_plan)
        status: WorkflowStatus = _coerce_literal(
            payload.get("status"),
            {"running", "need_user"},
            "need_user" if state.get("risk") == "high" else "running",
        )
        if state.get("risk") == "high":
            status = "need_user"
        return {
            **state,
            "plan": plan,
            "current_step": 0,
            "status": status,
            "trace_events": _append_event(
                state,
                TraceEvent(
                    event_type="planner",
                    content=f"{_format_plan(plan)}\nrationale: {payload.get('rationale', '')}",
                    ok=True,
                ),
            ),
        }

    return run


def _executor_node(executor: Any, settings: Settings):
    def run(state: AgentState) -> AgentState:
        if state.get("status") == "need_user":
            return state

        prompt = _executor_prompt(state)
        try:
            result = executor.invoke(
                {"messages": [HumanMessage(content=prompt)]},
                config={"recursion_limit": settings.agent_recursion_limit},
            )
        except Exception as exc:
            return {
                **state,
                "status": "failed",
                "error_type": type(exc).__name__,
                "tool_results": [
                    *state.get("tool_results", []),
                    {"ok": False, "content": str(exc)},
                ],
                "trace_events": _append_event(
                    state,
                    TraceEvent(event_type="executor", content=str(exc), ok=False),
                ),
            }

        answer = _message_content_to_text(result["messages"][-1].content)
        return {
            **state,
            "tool_results": [
                *state.get("tool_results", []),
                {"ok": True, "content": answer},
            ],
            "final_answer": answer,
            "status": "running",
            "trace_events": _append_event(
                state,
                TraceEvent(event_type="executor", content=answer, ok=True),
            ),
        }

    return run


def _verifier_node(model: Any):
    def run(state: AgentState) -> AgentState:
        fallback_ok = state.get("status") != "failed" and bool(state.get("final_answer", "").strip())
        fallback_status: WorkflowStatus = "done" if fallback_ok else "failed"
        fallback_reason = (
            "final answer present"
            if fallback_ok
            else f"execution failed: {state.get('error_type', '') or 'missing_final_answer'}"
        )
        payload = _invoke_json_node(
            model=model,
            system_prompt=VERIFIER_PROMPT
            + """
Return only JSON:
{
  "status": "done|failed|need_user",
  "ok": true,
  "reason": "short verification reason",
  "missing": ["optional missing item"]
}
Check whether the answer satisfies the plan and whether user input is required.
""",
            user_prompt=_verifier_user_prompt(state),
            fallback={"status": fallback_status, "ok": fallback_ok, "reason": fallback_reason},
        )
        status: WorkflowStatus = _coerce_literal(
            payload.get("status"),
            {"done", "failed", "need_user"},
            fallback_status,
        )
        ok = bool(payload.get("ok", status == "done"))
        if state.get("status") == "failed":
            status = "failed"
            ok = False
        return {
            **state,
            "status": status,
            "trace_events": _append_event(
                state,
                TraceEvent(
                    event_type="verifier",
                    content=str(payload.get("reason", fallback_reason)),
                    ok=ok,
                ),
            ),
        }

    return run


def _reflector_node(model: Any):
    def run(state: AgentState) -> AgentState:
        fallback_reflection = (
            f"Workflow failed for task_type={state.get('task_type', '')}; "
            f"error_type={state.get('error_type', '') or 'missing_final_answer'}."
        )
        payload = _invoke_json_node(
            model=model,
            system_prompt=REFLECTOR_PROMPT
            + """
Return only JSON:
{
  "reflection": "one actionable lesson for future similar tasks",
  "failure_type": "wrong_tool|bad_tool_args|missing_context|hallucinated_fact|unsafe_action|over_delegation|under_delegation|rag_miss|planning_loop|unknown"
}
""",
            user_prompt=_reflection_user_prompt(state),
            fallback={"reflection": fallback_reflection, "failure_type": "unknown"},
        )
        reflection = str(payload.get("reflection") or fallback_reflection)
        return {
            **state,
            "reflections": [*state.get("reflections", []), reflection],
            "trace_events": _append_event(
                state,
                TraceEvent(
                    event_type="reflector",
                    content=f"{payload.get('failure_type', 'unknown')}: {reflection}",
                    ok=False,
                ),
            ),
        }

    return run


def _finalizer_node(model: Any):
    def run(state: AgentState) -> AgentState:
        if state.get("status") == "need_user":
            fallback_answer = _confirmation_answer(state)
        elif state.get("status") == "failed":
            fallback_answer = state.get("final_answer") or (
                "Workflow failed before completing the task. "
                f"error_type: {state.get('error_type', '') or 'unknown'}"
            )
        else:
            fallback_answer = state.get("final_answer", "")
        payload = _invoke_json_node(
            model=model,
            system_prompt=FINALIZER_PROMPT
            + """
Return only JSON:
{
  "final_answer": "concise user-facing answer",
  "status": "done|failed|need_user"
}
Do not invent tool results. Preserve required confirmation prompts for high-risk tasks.
""",
            user_prompt=_finalizer_user_prompt(state, fallback_answer),
            fallback={"final_answer": fallback_answer, "status": state.get("status", "done")},
        )
        answer = str(payload.get("final_answer") or fallback_answer)
        return {
            **state,
            "final_answer": answer,
            "trace_events": _append_event(
                state,
                TraceEvent(event_type="finalizer", content=answer, ok=state.get("status") != "failed"),
            ),
        }

    return run


def classify_task_type(user_input: str) -> TaskType:
    lowered = user_input.lower()
    work_markers = ["飞书", "工作消息", "工作收件箱", "会议纪要", "aaa-work", "归属项目"]
    code_markers = [
        "实现",
        "修改",
        "bug",
        "报错",
        "部署",
        "测试",
        "脚本",
        "接口",
        "nginx",
        "worker",
        "代码",
        "branch",
    ]
    file_markers = ["读取", "查看", "文件", "目录", "markdown", "readme", ".py", ".md"]
    research_markers = ["搜索", "调研", "最新", "论文", "网页", "报告", "综述"]
    rag_markers = ["根据", "文档", "知识库", "引用", "来源", "工作记录"]

    if any(marker in lowered for marker in ["飞书", "工作消息", "工作收件箱", "会议纪要"]):
        return "work_message"
    if any(marker in lowered for marker in code_markers):
        return "code_task"
    if any(marker in lowered for marker in work_markers):
        return "work_message"
    if any(marker in lowered for marker in research_markers):
        return "research"
    if any(marker in lowered for marker in rag_markers):
        return "rag_qa"
    if any(marker in lowered for marker in file_markers):
        return "file_task"
    return "chat"


def classify_route_risk_difficulty(
    user_input: str,
    task_type: TaskType,
) -> tuple[WorkflowRoute, WorkflowRisk, WorkflowDifficulty]:
    lowered = user_input.lower()
    high_risk_markers = ["删除", "drop table", "生产", "线上", "付费", "密钥", "credential", "reset --hard"]
    medium_risk_markers = ["修改", "实现", "部署", "迁移", "写入", "覆盖", "commit", "分支"]
    high_difficulty_markers = ["重构", "全自动化", "多节点", "复杂", "部署", "训练", "评测", "pipeline"]

    risk: WorkflowRisk = "low"
    if any(marker in lowered for marker in high_risk_markers):
        risk = "high"
    elif task_type in {"code_task", "file_task"} or any(marker in lowered for marker in medium_risk_markers):
        risk = "medium"

    difficulty: WorkflowDifficulty = (
        "high" if any(marker in lowered for marker in high_difficulty_markers) else "low"
    )

    if risk == "high":
        route: WorkflowRoute = "ask_user"
    elif task_type == "code_task" and difficulty == "high":
        route = "codex"
    elif "之后" in lowered or "有空" in lowered or "备忘" in lowered:
        route = "defer"
    else:
        route = "self"
    return route, risk, difficulty


def build_plan(
    user_input: str,
    task_type: TaskType,
    route: WorkflowRoute,
    risk: WorkflowRisk,
    difficulty: WorkflowDifficulty,
) -> list[dict[str, Any]]:
    if risk == "high":
        return [
            {
                "step": 1,
                "action": "ask_user",
                "description": "Request confirmation before high-risk work.",
            }
        ]

    steps: list[str]
    if task_type == "work_message":
        steps = [
            "Capture and classify the work message.",
            "Match it to an existing work item using vector search when available.",
            "Create or update a work task with the selected route.",
            "Report the matched work, route, priority, and next action.",
        ]
    elif route == "codex":
        steps = [
            "Inspect the request and relevant local context.",
            "Create a structured work task.",
            "Rewrite the task for Codex delegation.",
            "Delegate or prepare the Codex handoff.",
            "Summarize status and verification needs.",
        ]
    elif task_type == "research":
        steps = [
            "Identify source needs.",
            "Search or open relevant sources.",
            "Extract grounded notes.",
            "Synthesize the answer with source context.",
        ]
    elif task_type in {"file_task", "rag_qa"}:
        steps = [
            "Identify relevant local files or knowledge records.",
            "Read or search the grounded context.",
            "Answer using cited local evidence.",
        ]
    else:
        steps = ["Answer directly unless tools are needed.", "Use tools for verifiable facts."]

    if risk == "medium":
        steps.insert(0, "Use an isolated branch or approval-safe workflow before repo changes.")
    if difficulty == "high" and route != "codex":
        steps.append("Run verifier checks and record failures for reflection if needed.")
    return [{"step": index, "action": _step_action(text), "description": text} for index, text in enumerate(steps, start=1)][:5]


def _step_action(description: str) -> str:
    lowered = description.lower()
    if "codex" in lowered or "delegate" in lowered:
        return "delegate"
    if "search" in lowered or "read" in lowered or "context" in lowered:
        return "retrieve_context"
    if "capture" in lowered or "classify" in lowered:
        return "route_work"
    if "verifier" in lowered or "checks" in lowered:
        return "verify"
    return "execute"


def _after_planner(state: AgentState) -> str:
    return "need_user" if state.get("status") == "need_user" else "execute"


def _after_verifier(state: AgentState) -> str:
    return "reflect" if state.get("status") == "failed" else "finalize"


def _state_user_input(state: AgentState) -> str:
    if state.get("user_input"):
        return state["user_input"]
    messages = state.get("messages", [])
    if not messages:
        return ""
    return _message_content_to_text(messages[-1].content)


def _executor_prompt(state: AgentState) -> str:
    lines = [
        f"User request:\n{state.get('user_input', '')}",
        "",
        f"Router: task_type={state.get('task_type')} route={state.get('route')} "
        f"risk={state.get('risk')} difficulty={state.get('difficulty')}",
        "",
        "Plan:",
        _format_plan(state.get("plan", [])),
        "",
        "Execute this plan and return the final user-facing answer.",
    ]
    return "\n".join(lines)


def _planner_user_prompt(state: AgentState, fallback_plan: list[dict[str, Any]]) -> str:
    return "\n".join(
        [
            f"User request:\n{state.get('user_input', '')}",
            "",
            f"Router output: task_type={state.get('task_type')} route={state.get('route')} "
            f"risk={state.get('risk')} difficulty={state.get('difficulty')}",
            "",
            "Fallback plan for reference:",
            _format_plan(fallback_plan),
        ]
    )


def _verifier_user_prompt(state: AgentState) -> str:
    return "\n".join(
        [
            f"User request:\n{state.get('user_input', '')}",
            "",
            f"Router: task_type={state.get('task_type')} route={state.get('route')} "
            f"risk={state.get('risk')} difficulty={state.get('difficulty')}",
            "",
            "Plan:",
            _format_plan(state.get("plan", [])),
            "",
            f"Executor final answer:\n{state.get('final_answer', '')}",
            "",
            f"Executor status: {state.get('status')}",
            f"Error type: {state.get('error_type', '')}",
        ]
    )


def _reflection_user_prompt(state: AgentState) -> str:
    return "\n".join(
        [
            f"User request:\n{state.get('user_input', '')}",
            "",
            "Plan:",
            _format_plan(state.get("plan", [])),
            "",
            f"Status: {state.get('status')}",
            f"Error type: {state.get('error_type', '')}",
            f"Tool results: {state.get('tool_results', [])}",
        ]
    )


def _finalizer_user_prompt(state: AgentState, fallback_answer: str) -> str:
    return "\n".join(
        [
            f"User request:\n{state.get('user_input', '')}",
            "",
            f"Workflow status: {state.get('status')}",
            f"Task type: {state.get('task_type')}",
            f"Route: {state.get('route')}",
            "",
            "Plan:",
            _format_plan(state.get("plan", [])),
            "",
            f"Executor answer:\n{state.get('final_answer', '')}",
            "",
            f"Fallback final answer:\n{fallback_answer}",
        ]
    )


def _confirmation_answer(state: AgentState) -> str:
    return (
        "This request was classified as high risk and needs confirmation before execution.\n"
        f"task_type: {state.get('task_type')}\n"
        f"risk: {state.get('risk')}\n"
        "Planned steps:\n"
        f"{_format_plan(state.get('plan', []))}"
    )


def _format_plan(plan: list[dict[str, Any]]) -> str:
    return "\n".join(
        f"{step.get('step', index)}. [{step.get('action', 'execute')}] {step.get('description', '')}"
        for index, step in enumerate(plan, start=1)
    )


def _append_event(state: AgentState, event: TraceEvent) -> list[TraceEvent]:
    return [*state.get("trace_events", []), event]


def _invoke_json_node(
    model: Any,
    system_prompt: str,
    user_prompt: str,
    fallback: dict[str, Any],
) -> dict[str, Any]:
    try:
        result = model.invoke(
            [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt),
            ]
        )
        text = _message_content_to_text(result.content)
        parsed = _parse_json_object(text)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass
    return fallback


def _parse_json_object(text: str) -> dict[str, Any] | None:
    stripped = text.strip()
    if not stripped:
        return None
    try:
        parsed = json.loads(stripped)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        pass

    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", stripped, flags=re.DOTALL)
    if fenced:
        try:
            parsed = json.loads(fenced.group(1))
            return parsed if isinstance(parsed, dict) else None
        except json.JSONDecodeError:
            pass

    start = stripped.find("{")
    end = stripped.rfind("}")
    if start >= 0 and end > start:
        try:
            parsed = json.loads(stripped[start : end + 1])
            return parsed if isinstance(parsed, dict) else None
        except json.JSONDecodeError:
            return None
    return None


def _coerce_literal(value: Any, allowed: set[str], fallback: Any) -> Any:
    if isinstance(value, str) and value in allowed:
        return value
    return fallback


def _coerce_plan(value: Any, fallback: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return fallback
    plan: list[dict[str, Any]] = []
    for index, item in enumerate(value[:5], start=1):
        if not isinstance(item, dict):
            continue
        description = str(item.get("description") or "").strip()
        if not description:
            continue
        action = str(item.get("action") or _step_action(description)).strip()
        plan.append(
            {
                "step": int(item.get("step") or index),
                "action": action,
                "description": description,
            }
        )
    return plan or fallback


def _message_content_to_text(content: Any) -> str:
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
