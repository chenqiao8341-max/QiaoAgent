from __future__ import annotations

import json
import re
import time
from typing import Any, Literal, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import create_react_agent

from agent_project.config import Settings
from agent_project.execution_policy import policy_for_task, task_policy_violation
from agent_project.llms import build_chat_model
from agent_project.tools import get_tools_for_task
from agent_project.tools.human_gate import create_human_gate_request
from agent_project.tools.memory import reflection_memory_context, store_reflection_memory
from agent_project.tools.work_management import capture_work_message
from agent_project.tools.rag import extract_citation_ids, verify_answer_against_retrieved_payload
from agent_project.tracing import TraceEvent


TaskType = Literal["chat", "work_message", "code_task", "research", "file_task", "rag_qa"]
WorkflowStatus = Literal["running", "need_user", "done", "failed"]
WorkflowRoute = Literal["self", "codex", "ask_user", "defer"]
WorkflowRisk = Literal["low", "medium", "high"]
WorkflowDifficulty = Literal["low", "high"]
RetryTarget = Literal["planner", "executor", "finalizer"]
CODEX_COMMAND_PRIORITY = ("codex-proxy-anyrouter", "codex-proxy-cccx", "codex")


class AgentState(TypedDict, total=False):
    messages: list[Any]
    user_input: str
    task_type: TaskType
    route: WorkflowRoute
    risk: WorkflowRisk
    difficulty: WorkflowDifficulty
    project_confidence: float
    plan: list[dict[str, Any]]
    current_step: int
    step_results: list[dict[str, Any]]
    tool_results: list[dict[str, Any]]
    reflections: list[str]
    final_answer: str
    status: WorkflowStatus
    error_type: str
    node_reports: list[dict[str, Any]]
    needs_human: bool
    human_gate_reason: str
    human_gate_id: str
    approved_human_gate_id: str
    approved_human_gate_reason: str
    human_gate_response: str
    verifier_failures: int
    retry_target: RetryTarget
    trace_id: str
    trace_events: list[TraceEvent]
    retrieval_needed: bool
    retrieval_scope: str


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
For medium-risk repo work, call ensure_work_branch before edits, use an isolated branch, and do not merge it back to main automatically.
Before starting Codex delegation, call test_codex_connectivity with command_names="codex-proxy-anyrouter,codex-proxy-cccx,codex".
Choose the first usable Codex command in this exact priority order: codex-proxy-anyrouter, codex-proxy-cccx, codex.
Pass that chosen command with codex_command when calling run_codex_task, start_codex_session, or continue_codex_session.
Respect approval settings and stop if user confirmation is required."""

VERIFIER_PROMPT = """You are the Verifier node.
Check whether the result satisfies the user request and whether more tool work or user input is needed."""

REFLECTOR_PROMPT = """You are the Reflector node.
If the workflow failed, summarize the failure cause and decide whether the planner or executor should retry."""

FINALIZER_PROMPT = """You are the Finalizer node.
Produce a concise user-facing answer with the result, important assumptions, and verification status."""


def build_workflow_agent(settings: Settings, system_prompt: str):
    model = build_chat_model(settings)

    graph = StateGraph(AgentState)
    graph.add_node("router", _router_node(model))
    graph.add_node("planner", _planner_node(model))
    graph.add_node("executor", _executor_node(model, settings, system_prompt))
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
    graph.add_conditional_edges(
        "reflector",
        _after_reflector,
        {
            "planner": "planner",
            "executor": "executor",
            "finalize": "finalizer",
        },
    )
    graph.add_edge("finalizer", END)
    return graph.compile(checkpointer=MemorySaver())


def _build_react_executor(model: Any, tools: list[Any], prompt: str):
    try:
        return create_react_agent(model, tools, prompt=prompt)
    except TypeError:
        return create_react_agent(model, tools, state_modifier=prompt)


def _router_node(model: Any):
    def run(state: AgentState) -> AgentState:
        user_input = _state_user_input(state)
        if _is_human_gate_resume(state) and state.get("task_type"):
            report = _node_report(
                "router",
                "Router resumed an approved human gate using saved route state.",
                status="running",
                needs_human=False,
                data={"human_gate_id": state.get("approved_human_gate_id", "")},
            )
            return {
                **state,
                "user_input": user_input,
                "status": "running",
                "needs_human": False,
                "human_gate_reason": "",
                "node_reports": _append_node_report(state, report),
                "trace_events": _append_event(
                    state,
                    TraceEvent(event_type="router", content=report["summary"], ok=True),
                ),
            }

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
  "project_confidence": 0.0,
  "retrieval_needed": true,
  "retrieval_scope": "docs|work_records|project|all|none",
  "rationale": "short reason"
}
Use ask_user for high-risk or underspecified requests. Use codex for complex code/deploy/test work.
For work_message requests that need project ownership, set project_confidence from 0 to 1.
For rag_qa, local document questions, explicit citation requests, or work-record questions, set retrieval_needed=true.
""",
            user_prompt=f"Classify this request:\n{user_input}",
            fallback={
                "task_type": fallback_task_type,
                "route": fallback_route,
                "risk": fallback_risk,
                "difficulty": fallback_difficulty,
                "project_confidence": 0.8,
                "retrieval_needed": fallback_task_type in {"rag_qa", "work_message"},
                "retrieval_scope": _fallback_retrieval_scope(user_input, fallback_task_type),
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
        project_confidence = _coerce_float(payload.get("project_confidence"), 0.8)
        retrieval_needed = _coerce_bool(
            payload.get("retrieval_needed"),
            task_type in {"rag_qa", "work_message"},
        )
        retrieval_scope = _coerce_literal(
            payload.get("retrieval_scope"),
            {"docs", "work_records", "project", "all", "none"},
            _fallback_retrieval_scope(user_input, task_type),
        )
        human_gate_reason = _router_human_gate_reason(task_type, project_confidence)
        status: WorkflowStatus = "need_user" if human_gate_reason else "running"
        report = _node_report(
            "router",
            (
                f"task_type={task_type} route={route} risk={risk} "
                f"difficulty={difficulty} project_confidence={project_confidence:.2f} "
                f"retrieval_needed={retrieval_needed} retrieval_scope={retrieval_scope}"
            ),
            status=status,
            needs_human=bool(human_gate_reason),
            human_gate_reason=human_gate_reason,
            data={"rationale": payload.get("rationale", "")},
        )
        return {
            **state,
            "user_input": user_input,
            "task_type": task_type,
            "route": route,
            "risk": risk,
            "difficulty": difficulty,
            "project_confidence": project_confidence,
            "retrieval_needed": retrieval_needed,
            "retrieval_scope": retrieval_scope,
            "status": status,
            "node_reports": _append_node_report(state, report),
            "needs_human": bool(human_gate_reason),
            "human_gate_reason": human_gate_reason,
            "trace_events": _append_event(
                state,
                TraceEvent(
                    event_type="router",
                    content=f"{report['summary']}; {payload.get('rationale', '')}",
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
        if _is_human_gate_resume(state) and state.get("plan"):
            report = _node_report(
                "planner",
                "Planner resumed an approved human gate using the saved plan.",
                status="running",
                needs_human=False,
                data={"human_gate_id": state.get("approved_human_gate_id", "")},
            )
            return {
                **state,
                "current_step": state.get("current_step", 0),
                "status": "running",
                "needs_human": False,
                "human_gate_reason": "",
                "node_reports": _append_node_report(state, report),
                "trace_events": _append_event(
                    state,
                    TraceEvent(event_type="planner", content=report["summary"], ok=True),
                ),
            }

        if state.get("needs_human"):
            report = _node_report(
                "planner",
                "Planner paused because an earlier human gate is pending.",
                status="need_user",
                needs_human=True,
                human_gate_reason=state.get("human_gate_reason", ""),
            )
            return {
                **state,
                "plan": fallback_plan,
                "current_step": 0,
                "status": "need_user",
                "node_reports": _append_node_report(state, report),
                "trace_events": _append_event(
                    state,
                    TraceEvent(event_type="planner", content=report["summary"], ok=True),
                ),
            }
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
- For medium-risk repo changes, plan ensure_work_branch, use an isolated branch, and explicitly do not merge it to main automatically.
- For "belongs to which project" or work-record matching, include search_work_record_vectors or capture_work_message in the plan description.
- For rag_qa or retrieval_needed=true, include search_knowledge or answer_with_citations before final synthesis.
- For citation answers, include verify_answer_citations after drafting the answer.
- For route=codex, include create_work_task, rewrite_task_for_codex, test_codex_connectivity, and then delegation with the first usable command in priority order: codex-proxy-anyrouter, codex-proxy-cccx, codex.
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
        if status == "need_user" and state.get("risk") != "high":
            status = "running"
        if state.get("risk") == "high":
            status = "need_user"
        human_gate_reason = (
            "Planner classified this request as high risk and requires human approval."
            if state.get("risk") == "high"
            else ""
        )
        report = _node_report(
            "planner",
            _format_plan(plan),
            status=status,
            needs_human=bool(human_gate_reason),
            human_gate_reason=human_gate_reason,
            data={"rationale": payload.get("rationale", "")},
        )
        return {
            **state,
            "plan": plan,
            "current_step": 0,
            "status": status,
            "node_reports": _append_node_report(state, report),
            "needs_human": bool(human_gate_reason),
            "human_gate_reason": human_gate_reason,
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


def _executor_node(model: Any, settings: Settings, system_prompt: str):
    def run(state: AgentState) -> AgentState:
        if state.get("status") == "need_user":
            return state
        if state.get("task_type") == "work_message":
            bounded = _execute_bounded_work_message(state)
            report = _node_report(
                "executor",
                bounded["answer"],
                status="running" if bounded["ok"] else "failed",
                needs_human=False,
                data={"step_results": bounded["step_results"]},
            )
            return {
                **state,
                "current_step": 1,
                "step_results": [*state.get("step_results", []), *bounded["step_results"]],
                "tool_results": [
                    *state.get("tool_results", []),
                    {"ok": bounded["ok"], "content": bounded["answer"]},
                ],
                "final_answer": bounded["answer"],
                "status": "running" if bounded["ok"] else "failed",
                "error_type": "" if bounded["ok"] else "WorkMessageExecutionError",
                "node_reports": _append_node_report(state, report),
                "trace_events": _append_event(
                    {**state, "trace_events": [*state.get("trace_events", []), *bounded["tool_events"]]},
                    TraceEvent(event_type="executor", content=bounded["answer"], ok=bounded["ok"]),
                ),
            }

        started = time.monotonic()
        tool_trace_events: list[TraceEvent] = []
        step_results: list[dict[str, Any]] = []
        answer = ""
        status: WorkflowStatus = "running"
        error_type = ""

        all_task_tools = get_tools_for_task(state.get("task_type", "chat"), state.get("route", "self"))
        plan = state.get("plan", []) or [{"step": 1, "action": "execute", "description": "Answer directly."}]

        for index, step in enumerate(plan):
            violation = _policy_violation_from_trace_events(
                state,
                tool_trace_events,
                elapsed_seconds=time.monotonic() - started,
            )
            if violation:
                status = "failed"
                error_type = "TaskExecutionPolicy"
                answer = violation
                break

            prompt = _executor_step_prompt(state, step, step_index=index)
            step_tool_names = _step_tool_names(state, step)
            step_tools = [tool for tool in all_task_tools if tool.name in step_tool_names]
            executor = _build_react_executor(model, step_tools, f"{system_prompt}\n\n{EXECUTOR_PROMPT}")
            try:
                result = executor.invoke(
                    {"messages": [HumanMessage(content=prompt)]},
                    config={"recursion_limit": min(settings.agent_recursion_limit, 8)},
                )
            except Exception as exc:
                status = "failed"
                error_type = type(exc).__name__
                answer = str(exc)
                step_results.append(
                    {
                        "step": step.get("step", index + 1),
                        "ok": False,
                        "content": answer,
                        "error_type": error_type,
                    }
                )
                break

            step_answer = _message_content_to_text(result["messages"][-1].content)
            answer = step_answer
            step_events = _trace_events_from_executor_messages(result.get("messages", []))
            if state.get("retrieval_needed") or state.get("task_type") == "rag_qa":
                step_events.extend(_rag_trace_events_from_tool_results(state, result.get("messages", [])))
            tool_trace_events.extend(step_events)
            step_results.append(
                {
                    "step": step.get("step", index + 1),
                    "action": step.get("action", "execute"),
                    "ok": True,
                    "content": step_answer,
                }
            )

        if status == "failed":
            report = _node_report(
                "executor",
                answer,
                status="failed",
                needs_human=False,
                data={"error_type": error_type, "step_results": step_results},
            )
            return {
                **state,
                "status": "failed",
                "error_type": error_type,
                "current_step": len(step_results),
                "step_results": [*state.get("step_results", []), *step_results],
                "tool_results": [
                    *state.get("tool_results", []),
                    {"ok": False, "content": answer},
                ],
                "node_reports": _append_node_report(state, report),
                "trace_events": _append_event(
                    {**state, "trace_events": [*state.get("trace_events", []), *tool_trace_events]},
                    TraceEvent(event_type="executor", content=answer, ok=False),
                ),
            }

        human_gate_reason = _executor_human_gate_reason(answer)
        status = "need_user" if human_gate_reason else "running"
        report = _node_report(
            "executor",
            answer,
            status=status,
            needs_human=bool(human_gate_reason),
            human_gate_reason=human_gate_reason,
            data={"step_results": step_results},
        )
        return {
            **state,
            "current_step": len(step_results),
            "step_results": [*state.get("step_results", []), *step_results],
            "tool_results": [
                *state.get("tool_results", []),
                {"ok": True, "content": answer},
            ],
            "final_answer": answer,
            "status": status,
            "node_reports": _append_node_report(state, report),
            "needs_human": bool(human_gate_reason),
            "human_gate_reason": human_gate_reason,
            "trace_events": _append_event(
                {**state, "trace_events": [*state.get("trace_events", []), *tool_trace_events]},
                TraceEvent(event_type="executor", content=answer, ok=True),
            ),
        }

    return run


def _execute_bounded_work_message(state: AgentState | dict[str, Any]) -> dict[str, Any]:
    user_input = str(state.get("user_input", ""))
    tool_args = {"content": user_input, "source": "agent", "sender": ""}
    call_event = TraceEvent(
        event_type="tool_call",
        tool="capture_work_message",
        args=tool_args,
        ok=None,
        node="executor",
        tool_call_id="bounded-work-message",
    )
    try:
        answer = capture_work_message.invoke(tool_args)
    except Exception as exc:
        answer = str(exc)
        result_event = TraceEvent(
            event_type="tool_result",
            tool="capture_work_message",
            content=answer,
            ok=False,
            node="executor",
            raw_error=type(exc).__name__,
            tool_call_id="bounded-work-message",
        )
        ok = False
    else:
        result_event = TraceEvent(
            event_type="tool_result",
            tool="capture_work_message",
            content=answer,
            ok=True,
            node="executor",
            tool_call_id="bounded-work-message",
        )
        ok = True
    return {
        "ok": ok,
        "answer": answer,
        "tool_events": [call_event, result_event],
        "step_results": [
            {
                "step": 1,
                "action": "route_work",
                "ok": ok,
                "content": answer,
            }
        ],
    }


def _verifier_node(model: Any):
    def run(state: AgentState) -> AgentState:
        if state.get("needs_human"):
            report = _node_report(
                "verifier",
                "Verifier paused because human approval is pending.",
                status="need_user",
                needs_human=True,
                human_gate_reason=state.get("human_gate_reason", ""),
            )
            return {
                **state,
                "status": "need_user",
                "node_reports": _append_node_report(state, report),
                "trace_events": _append_event(
                    state,
                    TraceEvent(event_type="verifier", content=report["summary"], ok=True),
                ),
            }
        work_message_check = _deterministic_work_message_verification(state)
        if work_message_check is not None:
            report = _node_report(
                "verifier",
                work_message_check["reason"],
                status="done" if work_message_check["ok"] else "failed",
                needs_human=False,
                data={"ok": work_message_check["ok"], "verifier_failures": state.get("verifier_failures", 0)},
            )
            return {
                **state,
                "status": "done" if work_message_check["ok"] else "failed",
                "node_reports": _append_node_report(state, report),
                "trace_events": _append_event(
                    state,
                    TraceEvent(
                        event_type="verifier",
                        content=work_message_check["reason"],
                        ok=work_message_check["ok"],
                    ),
                ),
            }
        rag_check = _deterministic_rag_verification(state)
        if rag_check is not None and not rag_check["ok"]:
            verifier_failures = state.get("verifier_failures", 0) + 1
            report = _node_report(
                "verifier",
                "RAG citation verification failed.",
                status="failed",
                needs_human=False,
                data={"ok": False, "verifier_failures": verifier_failures, "rag_check": rag_check},
            )
            return {
                **state,
                "status": "failed",
                "verifier_failures": verifier_failures,
                "node_reports": _append_node_report(state, report),
                "trace_events": _append_event(
                    state,
                    TraceEvent(
                        event_type="verifier",
                        content="RAG citation verification failed.",
                        args={"rag_check": rag_check},
                        ok=False,
                    ),
                ),
            }
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
        if status == "failed":
            ok = False
        if not ok and status == "done":
            status = "failed"
        verifier_failures = state.get("verifier_failures", 0)
        if not ok or status in {"failed", "need_user"}:
            verifier_failures += 1
        human_gate_reason = (
            "Verifier failed 3 times and requires human review."
            if verifier_failures >= 3
            else ""
        )
        if human_gate_reason:
            status = "need_user"
        elif status == "need_user":
            status = "failed"
        report = _node_report(
            "verifier",
            str(payload.get("reason", fallback_reason)),
            status=status,
            needs_human=bool(human_gate_reason),
            human_gate_reason=human_gate_reason,
            data={"ok": ok, "verifier_failures": verifier_failures},
        )
        return {
            **state,
            "status": status,
            "verifier_failures": verifier_failures,
            "node_reports": _append_node_report(state, report),
            "needs_human": bool(human_gate_reason),
            "human_gate_reason": human_gate_reason,
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
  "failure_type": "wrong_tool|bad_tool_args|missing_context|hallucinated_fact|unsafe_action|over_delegation|under_delegation|rag_miss|planning_loop|unknown",
  "retry_target": "planner|executor|finalizer"
}
Use retry_target="planner" when the plan, context, route, scope, or decomposition is wrong.
Use retry_target="executor" when the plan is usable but tool choice, tool args, delegation, or execution failed.
Use retry_target="finalizer" only when retrying would not help.
""",
            user_prompt=_reflection_user_prompt(state),
            fallback={"reflection": fallback_reflection, "failure_type": "unknown"},
        )
        reflection = str(payload.get("reflection") or fallback_reflection)
        failure_type = _coerce_literal(
            payload.get("failure_type"),
            {
                "wrong_tool",
                "bad_tool_args",
                "missing_context",
                "hallucinated_fact",
                "unsafe_action",
                "over_delegation",
                "under_delegation",
                "rag_miss",
                "planning_loop",
                "unknown",
            },
            "unknown",
        )
        retry_target: RetryTarget = _coerce_literal(
            payload.get("retry_target"),
            {"planner", "executor", "finalizer"},
            _retry_target_for_failure(failure_type, reflection),
        )
        if state.get("needs_human"):
            retry_target = "finalizer"
        retry_status: WorkflowStatus = "failed" if retry_target == "finalizer" else "running"
        memory_id = store_reflection_memory(
            task_type=state.get("task_type", ""),
            failure_type=failure_type,
            reflection=reflection,
            created_from_trace_id=str(state.get("trace_id", "")),
        )
        report = _node_report(
            "reflector",
            reflection,
            status=retry_status,
            needs_human=False,
            data={
                "failure_type": failure_type,
                "retry_target": retry_target,
                "reflection_memory_id": memory_id,
            },
        )
        return {
            **state,
            "reflections": [*state.get("reflections", []), reflection],
            "retry_target": retry_target,
            "status": retry_status,
            "node_reports": _append_node_report(state, report),
            "trace_events": _append_event(
                state,
                TraceEvent(
                    event_type="reflector",
                    content=f"{failure_type} -> {retry_target}: {reflection}",
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
            fallback_answer = _rag_failure_answer(state) or state.get("final_answer") or (
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
Do not invent tool results. Preserve required human gate prompts when needs_human=true.
""",
            user_prompt=_finalizer_user_prompt(state, fallback_answer),
            fallback={"final_answer": fallback_answer, "status": state.get("status", "done")},
        )
        answer = str(payload.get("final_answer") or fallback_answer)
        if state.get("status") == "failed" and _rag_failure_answer(state):
            answer = fallback_answer
        human_gate_id = state.get("human_gate_id", "")
        if state.get("needs_human") and not human_gate_id:
            human_gate_id = create_human_gate_request(
                user_input=state.get("user_input", ""),
                reason=state.get("human_gate_reason", ""),
                state={**state, "final_answer": answer},
            )
            answer = f"{answer}\n\nhuman_gate_id: {human_gate_id}"
        report = _node_report(
            "finalizer",
            answer,
            status=state.get("status", "done"),
            needs_human=state.get("needs_human", False),
            human_gate_reason=state.get("human_gate_reason", ""),
            data={"human_gate_id": human_gate_id} if human_gate_id else None,
        )
        return {
            **state,
            "final_answer": answer,
            "human_gate_id": human_gate_id,
            "node_reports": _append_node_report(state, report),
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
            "Report the matched work, route, priority, and next action from the capture result.",
        ]
    elif route == "codex":
        steps = [
            "Inspect the request and relevant local context.",
            "Create a structured work task and rewrite it for Codex delegation.",
            _codex_connectivity_step(),
            "Delegate to Codex with the first usable command from codex-proxy-anyrouter, codex-proxy-cccx, codex.",
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
            "Identify relevant local files or indexed private knowledge records.",
            "Call search_knowledge or answer_with_citations to retrieve grounded chunks and allowed citation IDs.",
            "Draft the answer using bracketed citations from retrieved chunks.",
            "Call verify_answer_citations before finalizing cited claims.",
        ]
    else:
        steps = ["Answer directly unless tools are needed.", "Use tools for verifiable facts."]

    if risk == "medium":
        steps.insert(
            0,
            "Call ensure_work_branch before repo changes, use an isolated branch, and do not merge it to main automatically.",
        )
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
    if _has_rag_verification_failure(state):
        return "finalize"
    return "reflect" if state.get("status") == "failed" else "finalize"


def _after_reflector(state: AgentState) -> str:
    if state.get("needs_human") or state.get("status") == "need_user":
        return "finalize"
    retry_target = state.get("retry_target", "finalizer")
    if retry_target == "planner":
        return "planner"
    if retry_target == "executor":
        return "executor"
    return "finalize"


def _has_rag_verification_failure(state: AgentState | dict[str, Any]) -> bool:
    if state.get("task_type") != "rag_qa" and not state.get("retrieval_needed"):
        return False
    for report in state.get("node_reports", []):
        if not isinstance(report, dict):
            continue
        data = report.get("data", {})
        if isinstance(data, dict) and isinstance(data.get("rag_check"), dict):
            return True
        if report.get("summary") == "RAG citation verification failed.":
            return True
    return False


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
        f"Retrieval: needed={state.get('retrieval_needed', False)} scope={state.get('retrieval_scope', 'none')}",
        "",
        "Plan:",
        _format_plan(state.get("plan", [])),
        "",
        "Reflections from prior failed attempts:",
        _format_reflections(state.get("reflections", [])),
        "",
        _persistent_reflection_context(state),
        "",
        "Human gate policy:",
        "- Default to no human approval unless needs_human is justified.",
        "- Do not merge an isolated branch back to main automatically.",
        "- For medium-risk repo work, call ensure_work_branch before changing files.",
        "",
        "Private knowledge RAG policy:",
        "- For rag_qa or retrieval_needed=true, call search_knowledge or answer_with_citations before answering.",
        "- Use only retrieved citation IDs in bracketed citations.",
        "- Call verify_answer_citations before finalizing answers with citations.",
        "- If no chunks are retrieved, say the local knowledge base lacks grounded evidence.",
        "",
        "Codex delegation policy:",
        _codex_delegation_policy_text(),
        "",
        "Execute this plan and return the final user-facing answer.",
    ]
    if _is_human_gate_resume(state):
        lines.insert(
            1,
            "\n".join(
                [
                    "",
                    "Approved human gate:",
                    f"- gate_id: {state.get('approved_human_gate_id', '')}",
                    f"- reason: {state.get('approved_human_gate_reason', '')}",
                    f"- response: {state.get('human_gate_response', '')}",
                ]
            ),
        )
    return "\n".join(lines)


def _executor_step_prompt(state: AgentState, step: dict[str, Any], step_index: int) -> str:
    plan = state.get("plan", [])
    policy = policy_for_task(state.get("task_type", "chat"), state.get("route", "self"))
    allowed_tools = _step_tool_names(state, step)
    prior_results = state.get("step_results", [])
    return "\n".join(
        [
            f"User request:\n{state.get('user_input', '')}",
            "",
            f"Router: task_type={state.get('task_type')} route={state.get('route')} "
            f"risk={state.get('risk')} difficulty={state.get('difficulty')}",
            f"Execution policy: max_tool_calls={policy.max_tool_calls} max_seconds={policy.max_seconds}",
            "Allowed tools: " + ", ".join(sorted(allowed_tools)),
            "",
            f"Current step {step_index + 1}/{max(len(plan), 1)}:",
            f"{step.get('step', step_index + 1)}. [{step.get('action', 'execute')}] {step.get('description', '')}",
            "",
            "Prior step results:",
            _format_step_results(prior_results),
            "",
            "Execute only the current step. Do not perform later plan steps.",
            "Use at most one tool call unless the current step explicitly requires more.",
            "Return the current step result and any user-facing conclusion if this is the final step.",
        ]
    )


def _step_tool_names(state: AgentState | dict[str, Any], step: dict[str, Any]) -> set[str]:
    task_type = str(state.get("task_type", "chat"))
    route = str(state.get("route", "self"))
    action = str(step.get("action", "execute"))
    description = str(step.get("description", "")).lower()
    policy = policy_for_task(task_type, route)

    if task_type == "work_message":
        if action == "route_work" or "capture" in description or "classify" in description:
            return {"capture_work_message"}
        if "vector" in description or "match" in description:
            return {"search_work_record_vectors"}
        if "task" in description:
            return {"create_work_task", "update_work_task"}
        return {"list_work_inbox", "list_work_tasks"}
    if task_type == "rag_qa":
        if action == "retrieve_context":
            return {"search_knowledge", "answer_with_citations"}
        if action == "verify":
            return {"verify_answer_citations", "verify_answer_against_retrieved_chunks_tool"}
    if task_type == "chat":
        return set(policy.allowed_tools)
    return set(policy.allowed_tools)


def _planner_user_prompt(state: AgentState, fallback_plan: list[dict[str, Any]]) -> str:
    return "\n".join(
        [
            f"User request:\n{state.get('user_input', '')}",
            "",
            f"Router output: task_type={state.get('task_type')} route={state.get('route')} "
            f"risk={state.get('risk')} difficulty={state.get('difficulty')}",
            f"Retrieval: needed={state.get('retrieval_needed', False)} scope={state.get('retrieval_scope', 'none')}",
            "",
            "Reflections from prior failed attempts:",
            _format_reflections(state.get("reflections", [])),
            "",
            _persistent_reflection_context(state),
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
            f"Retrieval: needed={state.get('retrieval_needed', False)} scope={state.get('retrieval_scope', 'none')}",
            "",
            "Plan:",
            _format_plan(state.get("plan", [])),
            "",
            _persistent_reflection_context(state),
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
            f"Verifier failures: {state.get('verifier_failures', 0)}",
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
            f"Needs human: {state.get('needs_human', False)}",
            f"Human gate reason: {state.get('human_gate_reason', '')}",
            "",
            "Plan:",
            _format_plan(state.get("plan", [])),
            "",
            "Node reports:",
            _format_node_reports(state.get("node_reports", [])),
            "",
            f"Executor answer:\n{state.get('final_answer', '')}",
            "",
            f"Fallback final answer:\n{fallback_answer}",
        ]
    )


def _confirmation_answer(state: AgentState) -> str:
    reason = state.get("human_gate_reason", "") or "Human approval is required before execution."
    return (
        "This workflow needs human approval before continuing.\n"
        f"reason: {reason}\n"
        f"task_type: {state.get('task_type')}\n"
        f"risk: {state.get('risk')}\n"
        "Planned steps:\n"
        f"{_format_plan(state.get('plan', []))}"
    )


def _rag_failure_answer(state: AgentState) -> str:
    if state.get("task_type") != "rag_qa" and not state.get("retrieval_needed"):
        return ""
    reports = state.get("node_reports", [])
    rag_reasons = []
    for report in reports:
        data = report.get("data", {}) if isinstance(report, dict) else {}
        rag_check = data.get("rag_check") if isinstance(data, dict) else None
        if isinstance(rag_check, dict):
            reason = str(rag_check.get("reason") or rag_check.get("missing") or rag_check.get("unsupported") or "")
            if reason:
                rag_reasons.append(reason)
    reason_text = "; ".join(rag_reasons) or "citation verification failed"
    return (
        "I could not verify the cited answer against the retrieved knowledge chunks. "
        f"reason: {reason_text}"
    )


def _format_plan(plan: list[dict[str, Any]]) -> str:
    return "\n".join(
        f"{step.get('step', index)}. [{step.get('action', 'execute')}] {step.get('description', '')}"
        for index, step in enumerate(plan, start=1)
    )


def _format_reflections(reflections: list[str]) -> str:
    if not reflections:
        return "(none)"
    return "\n".join(f"- {reflection}" for reflection in reflections[-3:])


def _format_step_results(results: list[dict[str, Any]]) -> str:
    if not results:
        return "(none)"
    lines = []
    for result in results[-5:]:
        content = str(result.get("content", "")).replace("\n", " ")
        if len(content) > 500:
            content = content[:500].rstrip() + "..."
        lines.append(
            f"- step={result.get('step', '')} ok={result.get('ok', False)} "
            f"action={result.get('action', '')} result={content}"
        )
    return "\n".join(lines)


def _persistent_reflection_context(state: AgentState) -> str:
    try:
        context = reflection_memory_context(
            query=state.get("user_input", ""),
            task_type=state.get("task_type", ""),
            limit=3,
        )
    except Exception:
        return "Relevant reflection memories: (unavailable)"
    return context or "Relevant reflection memories: (none)"


def _is_human_gate_resume(state: AgentState) -> bool:
    return bool(state.get("approved_human_gate_id"))


def _codex_command_names() -> str:
    return ",".join(CODEX_COMMAND_PRIORITY)


def _codex_connectivity_step() -> str:
    return (
        "Call test_codex_connectivity with "
        f"command_names=\"{_codex_command_names()}\" before starting Codex."
    )


def _codex_delegation_policy_text() -> str:
    return "\n".join(
        [
            _codex_connectivity_step(),
            "Read the connectivity result and choose the first usable command in this order: "
            + ", ".join(CODEX_COMMAND_PRIORITY)
            + ".",
            (
                "Pass the chosen command via codex_command to run_codex_task, "
                "start_codex_session, or continue_codex_session."
            ),
            "If none of those commands are usable, do not start Codex; report the connectivity blocker.",
        ]
    )


def _format_node_reports(reports: list[dict[str, Any]]) -> str:
    if not reports:
        return "(none)"
    return "\n".join(
        (
            f"- {report.get('node', 'unknown')}: status={report.get('status', '')} "
            f"needs_human={report.get('needs_human', False)} "
            f"summary={report.get('summary', '')}"
        )
        for report in reports
    )


def _append_event(state: AgentState, event: TraceEvent) -> list[TraceEvent]:
    return [*state.get("trace_events", []), event]


def _policy_violation_from_trace_events(
    state: AgentState,
    events: list[TraceEvent],
    elapsed_seconds: float,
) -> str:
    policy = policy_for_task(state.get("task_type", "chat"), state.get("route", "self"))
    tool_call_count = sum(1 for event in events if event.event_type == "tool_call")
    repeated_file_failure = _repeated_file_tool_failure(events)
    if repeated_file_failure:
        return repeated_file_failure
    return task_policy_violation(
        policy,
        tool_call_count=tool_call_count,
        elapsed_seconds=elapsed_seconds,
    )


def _repeated_file_tool_failure(events: list[Any]) -> str:
    file_failures = 0
    for event in events:
        if getattr(event, "event_type", "") != "tool_result":
            continue
        if getattr(event, "tool", "") not in {"read_local_file", "list_local_directory"}:
            continue
        content = str(getattr(event, "content", "")).lower()
        if any(marker in content for marker in ["not found", "no such file", "denied", "permission"]):
            file_failures += 1
    if file_failures >= 2:
        return "Repeated file tool failures; stop retrying similar paths and summarize known context."
    return ""


def _trace_events_from_executor_messages(messages: list[Any]) -> list[TraceEvent]:
    events: list[TraceEvent] = []
    for message in messages:
        for call in getattr(message, "tool_calls", None) or []:
            if not isinstance(call, dict):
                continue
            name = str(call.get("name") or "")
            args = call.get("args") if isinstance(call.get("args"), dict) else {}
            events.append(
                TraceEvent(
                    event_type="tool_call",
                    tool=name,
                    args=args,
                    tool_call_id=str(call.get("id") or ""),
                    ok=None,
                )
            )
        if getattr(message, "type", "") == "tool":
            name = str(getattr(message, "name", "") or getattr(message, "tool_call_id", ""))
            events.append(
                TraceEvent(
                    event_type="tool_result",
                    tool=name,
                    content=_message_content_to_text(getattr(message, "content", "")),
                    tool_call_id=str(getattr(message, "tool_call_id", "")),
                    ok=True,
                )
            )
    return events


def _rag_trace_events_from_tool_results(state: AgentState, messages: list[Any]) -> list[TraceEvent]:
    events: list[TraceEvent] = []
    call_args_by_id: dict[str, dict[str, Any]] = {}
    for message in messages:
        for call in getattr(message, "tool_calls", None) or []:
            if isinstance(call, dict) and call.get("name") in {"search_knowledge", "answer_with_citations"}:
                args = call.get("args") if isinstance(call.get("args"), dict) else {}
                call_args_by_id[str(call.get("id") or "")] = args
    for message in messages:
        if getattr(message, "type", "") != "tool":
            continue
        name = str(getattr(message, "name", "") or getattr(message, "tool_call_id", ""))
        if name not in {"search_knowledge", "answer_with_citations"}:
            continue
        content = _message_content_to_text(getattr(message, "content", ""))
        payload = _rag_payload_from_tool_result(content)
        call_args = call_args_by_id.get(str(getattr(message, "tool_call_id", "")), {})
        payload["query"] = str(call_args.get("query") or payload.get("query") or "")
        payload["top_k"] = int(call_args.get("limit") or call_args.get("top_k") or payload.get("top_k") or 0)
        if content.startswith("No knowledge chunks found") or content.startswith("No grounded evidence found"):
            events.append(
                TraceEvent(
                    event_type="rag_retrieval",
                    tool=name,
                    content="query= top_k=0 results=0",
                    args={**payload, "results": []},
                    ok=True,
                )
            )
            continue
        events.append(
            TraceEvent(
                event_type="rag_retrieval",
                tool=name,
                content=f"query={payload['query']} top_k={payload['top_k']} results={len(payload['results'])}",
                args=payload,
                ok=True,
            )
        )
    return events


def _rag_payload_from_tool_result(content: str) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    for index, line in enumerate(content.splitlines(), start=1):
        match = re.search(
            r"^\s*(?:[-*]|\d+\.)\s+"
            r"(?:score=(?P<score>[0-9.]+)\s+)?"
            r"(?:method=(?P<method>[a-zA-Z0-9_]+)\s+)?"
            r"\[(?P<citation>[^\]]+)\]"
            r"(?:\s+source=(?P<source>\S+))?"
            r"(?:\s+title=(?P<title>.*?))?"
            r"(?:\s+heading=(?P<heading>.*))?$",
            line,
        )
        if not match:
            continue
        try:
            score = float(match.group("score") or 0.0)
        except ValueError:
            score = 0.0
        results.append(
            {
                "rank": len(results) + 1,
                "score": score,
                "retrieval_method": match.group("method") or "tool_result",
                "source": (match.group("source") or "").strip(),
                "title": (match.group("title") or "").strip(),
                "heading_path": (match.group("heading") or "").strip(),
                "chunk_id": "",
                "citation_id": match.group("citation").strip(),
                "snippet": _following_snippet(content.splitlines(), index),
            }
        )
    return {"query": "", "top_k": len(results), "results": results}


def _following_snippet(lines: list[str], one_based_index: int) -> str:
    if one_based_index >= len(lines):
        return ""
    candidate = lines[one_based_index].strip()
    return candidate[:500]


def _deterministic_rag_verification(state: AgentState) -> dict[str, Any] | None:
    if not (state.get("retrieval_needed") or state.get("task_type") == "rag_qa"):
        return None
    answer = state.get("final_answer", "")
    if not extract_citation_ids(answer):
        return {"ok": False, "reason": "no citations used", "used": [], "allowed": []}
    retrieval_events = [
        event for event in state.get("trace_events", []) if event.event_type == "rag_retrieval"
    ]
    payload = {"results": []}
    for event in retrieval_events:
        payload["results"].extend(event.args.get("results", []))
    if not payload["results"]:
        return {"ok": False, "reason": "no retrieved chunks available", "used": [], "allowed": []}
    return verify_answer_against_retrieved_payload(answer, payload)


def _deterministic_work_message_verification(state: AgentState) -> dict[str, Any] | None:
    if state.get("task_type") != "work_message":
        return None
    answer = state.get("final_answer", "")
    ok = "Inbox message" in answer and "route:" in answer and "priority:" in answer
    return {
        "ok": ok,
        "reason": "work message captured with route and priority" if ok else "missing work route evidence",
    }


def _node_report(
    node: str,
    summary: str,
    *,
    status: str,
    needs_human: bool,
    human_gate_reason: str = "",
    data: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "node": node,
        "summary": summary,
        "status": status,
        "needs_human": needs_human,
        "human_gate_reason": human_gate_reason,
        "data": data or {},
    }


def _append_node_report(state: AgentState, report: dict[str, Any]) -> list[dict[str, Any]]:
    return [*state.get("node_reports", []), report]


def _router_human_gate_reason(task_type: TaskType, project_confidence: float) -> str:
    if task_type == "work_message" and project_confidence < 0.35:
        return "Router project ownership confidence is low; human project assignment is required."
    return ""


def _fallback_retrieval_scope(user_input: str, task_type: TaskType) -> str:
    lowered = user_input.lower()
    if task_type == "work_message" or any(marker in lowered for marker in ["工作记录", "aaa-work"]):
        return "work_records"
    if any(marker in lowered for marker in ["项目", "readme", "代码结构"]):
        return "project"
    if task_type == "rag_qa":
        return "all"
    return "none"


def _executor_human_gate_reason(answer: str) -> str:
    lowered = answer.lower()
    merge_markers = [
        "merge request",
        "pull request",
        "open a pr",
        "opened pr",
        "merge to main",
        "merge into main",
        "合并请求",
        "发起pr",
        "发起 pr",
        "合并到main",
        "合并到 main",
    ]
    if any(marker in lowered for marker in merge_markers):
        return "Executor initiated or requested a branch merge; human approval is required."
    return ""


def _retry_target_for_failure(failure_type: str, reflection: str = "") -> RetryTarget:
    planner_failures = {
        "missing_context",
        "rag_miss",
        "planning_loop",
        "under_delegation",
        "over_delegation",
        "unsafe_action",
    }
    executor_failures = {"wrong_tool", "bad_tool_args", "hallucinated_fact"}
    if failure_type in planner_failures:
        return "planner"
    if failure_type in executor_failures:
        return "executor"

    lowered = reflection.lower()
    planner_markers = ["plan", "planning", "context", "scope", "route", "decomposition", "missing context"]
    executor_markers = ["tool", "argument", "args", "execute", "execution", "codex", "delegation"]
    if any(marker in lowered for marker in planner_markers):
        return "planner"
    if any(marker in lowered for marker in executor_markers):
        return "executor"
    return "executor"


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


def _coerce_float(value: Any, fallback: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return fallback
    return max(0.0, min(number, 1.0))


def _coerce_bool(value: Any, fallback: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "yes", "1"}:
            return True
        if lowered in {"false", "no", "0"}:
            return False
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
