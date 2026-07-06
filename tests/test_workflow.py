from __future__ import annotations

from agent_project.workflow import (
    _after_reflector,
    _after_verifier,
    _coerce_plan,
    _codex_delegation_policy_text,
    _executor_human_gate_reason,
    _executor_prompt,
    _executor_step_prompt,
    _policy_violation_from_trace_events,
    _planner_node,
    _parse_json_object,
    _rag_trace_events_from_tool_results,
    _reflector_node,
    _retry_target_for_failure,
    _router_node,
    _router_human_gate_reason,
    _trace_events_from_executor_messages,
    _verifier_node,
    build_plan,
    classify_route_risk_difficulty,
    classify_task_type,
)
from langchain_core.messages import AIMessage, ToolMessage


class _FakeJsonModel:
    def __init__(self, content: str) -> None:
        self.content = content

    def invoke(self, _messages):
        return type("FakeMessage", (), {"content": self.content})()


def test_workflow_router_classifies_work_message() -> None:
    task_type = classify_task_type("飞书消息：医疗翻译服务 nginx timeout，判断归属项目。")
    route, risk, difficulty = classify_route_risk_difficulty(
        "飞书消息：医疗翻译服务 nginx timeout，判断归属项目。",
        task_type,
    )

    assert task_type == "work_message"
    assert route == "self"
    assert risk == "low"
    assert difficulty == "low"


def test_workflow_router_marks_rag_questions_for_retrieval() -> None:
    state = _router_node(
        _FakeJsonModel(
            '{"task_type": "rag_qa", "route": "self", "risk": "low", '
            '"difficulty": "low", "project_confidence": 0.9, '
            '"retrieval_needed": true, "retrieval_scope": "all", '
            '"rationale": "needs local citations"}'
        )
    )({"messages": [type("Message", (), {"content": "根据知识库回答并给引用"})()]})

    assert state["task_type"] == "rag_qa"
    assert state["retrieval_needed"] is True
    assert state["retrieval_scope"] == "all"


def test_workflow_router_classifies_high_difficulty_code_task() -> None:
    task_type = classify_task_type("实现全自动化多节点工作流并补测试。")
    route, risk, difficulty = classify_route_risk_difficulty(
        "实现全自动化多节点工作流并补测试。",
        task_type,
    )

    assert task_type == "code_task"
    assert route == "codex"
    assert risk == "medium"
    assert difficulty == "high"


def test_workflow_medium_risk_plan_forbids_auto_merge() -> None:
    plan = build_plan(
        user_input="修改接口并补测试。",
        task_type="code_task",
        route="self",
        risk="medium",
        difficulty="low",
    )

    assert "isolated branch" in plan[0]["description"]
    assert "do not merge it to main automatically" in plan[0]["description"]


def test_workflow_rag_plan_requires_search_and_citation_verification() -> None:
    plan = build_plan(
        user_input="根据知识库说明 human gate，并给引用。",
        task_type="rag_qa",
        route="self",
        risk="low",
        difficulty="low",
    )
    descriptions = "\n".join(step["description"] for step in plan)

    assert "search_knowledge or answer_with_citations" in descriptions
    assert "verify_answer_citations" in descriptions


def test_workflow_codex_plan_tests_connectivity_before_delegation() -> None:
    plan = build_plan(
        user_input="实现全自动化多节点工作流并补测试。",
        task_type="code_task",
        route="codex",
        risk="medium",
        difficulty="high",
    )
    descriptions = "\n".join(step["description"] for step in plan)

    assert "test_codex_connectivity" in descriptions
    assert "codex-proxy-anyrouter,codex-proxy-cccx,codex" in descriptions
    assert descriptions.index("test_codex_connectivity") < descriptions.index("Delegate to Codex")


def test_executor_prompt_includes_codex_connectivity_priority() -> None:
    prompt = _executor_prompt(
        {
            "user_input": "实现复杂代码任务",
            "task_type": "code_task",
            "route": "codex",
            "risk": "medium",
            "difficulty": "high",
            "plan": build_plan(
                user_input="实现复杂代码任务",
                task_type="code_task",
                route="codex",
                risk="medium",
                difficulty="high",
            ),
        }
    )

    assert "test_codex_connectivity" in prompt
    assert "codex-proxy-anyrouter" in prompt
    assert "codex-proxy-cccx" in prompt
    assert "codex_command" in prompt


def test_executor_step_prompt_limits_context_to_one_plan_step() -> None:
    state = {
        "user_input": "处理工作消息",
        "task_type": "work_message",
        "route": "self",
        "risk": "low",
        "difficulty": "low",
        "plan": [
            {"step": 1, "action": "route_work", "description": "Capture message."},
            {"step": 2, "action": "execute", "description": "Summarize result."},
        ],
    }

    prompt = _executor_step_prompt(state, state["plan"][0], step_index=0)

    assert "Current step 1/2" in prompt
    assert "Capture message." in prompt
    assert "Summarize result." not in prompt
    assert "Allowed tools:" in prompt
    assert "capture_work_message" in prompt
    assert "execute_shell_command" not in prompt


def test_policy_violation_from_trace_events_counts_tool_calls() -> None:
    state = {"task_type": "work_message", "route": "self"}
    events = [
        type("Event", (), {"event_type": "tool_call"})()
        for _ in range(7)
    ]

    violation = _policy_violation_from_trace_events(state, events, elapsed_seconds=1.0)

    assert "tool call budget" in violation


def test_policy_violation_backs_off_repeated_file_failures() -> None:
    state = {"task_type": "file_task", "route": "self"}
    events = [
        type("Event", (), {"event_type": "tool_result", "tool": "read_local_file", "content": "not found"})(),
        type("Event", (), {"event_type": "tool_result", "tool": "read_local_file", "content": "denied"})(),
    ]

    violation = _policy_violation_from_trace_events(state, events, elapsed_seconds=1.0)

    assert "Repeated file tool failures" in violation


def test_executor_prompt_includes_approved_human_gate_context() -> None:
    prompt = _executor_prompt(
        {
            "user_input": "继续高风险任务",
            "task_type": "code_task",
            "route": "ask_user",
            "risk": "high",
            "difficulty": "high",
            "plan": [{"step": 1, "action": "ask_user", "description": "confirm"}],
            "approved_human_gate_id": "gate-1",
            "approved_human_gate_reason": "high risk",
            "human_gate_response": "approved by operator",
        }
    )

    assert "Approved human gate" in prompt
    assert "gate-1" in prompt
    assert "approved by operator" in prompt


def test_codex_delegation_policy_uses_expected_priority() -> None:
    policy = _codex_delegation_policy_text()

    assert 'command_names="codex-proxy-anyrouter,codex-proxy-cccx,codex"' in policy
    assert policy.index("codex-proxy-anyrouter") < policy.index("codex-proxy-cccx")
    assert policy.index("codex-proxy-cccx") < policy.index("codex.")


def test_workflow_plan_stops_high_risk_requests_for_confirmation() -> None:
    plan = build_plan(
        user_input="删除线上数据库并 reset --hard。",
        task_type="code_task",
        route="ask_user",
        risk="high",
        difficulty="high",
    )

    assert plan == [
        {
            "step": 1,
            "action": "ask_user",
            "description": "Request confirmation before high-risk work.",
        }
    ]


def test_human_gate_resume_reuses_saved_route_and_plan() -> None:
    state = {
        "user_input": "删除线上数据库前已获得审批。",
        "task_type": "code_task",
        "route": "ask_user",
        "risk": "high",
        "difficulty": "high",
        "plan": [{"step": 1, "action": "ask_user", "description": "confirm high risk"}],
        "status": "need_user",
        "needs_human": True,
        "human_gate_reason": "high risk",
        "approved_human_gate_id": "gate-1",
        "approved_human_gate_reason": "high risk",
        "human_gate_response": "approved",
    }

    routed = _router_node(_FakeJsonModel("{}"))(state)
    planned = _planner_node(_FakeJsonModel("{}"))(routed)

    assert routed["status"] == "running"
    assert routed["needs_human"] is False
    assert planned["status"] == "running"
    assert planned["needs_human"] is False
    assert planned["plan"] == state["plan"]


def test_parse_json_object_accepts_fenced_json() -> None:
    parsed = _parse_json_object('```json\n{"status": "done", "ok": true}\n```')
    assert parsed == {"status": "done", "ok": True}


def test_router_low_project_confidence_requires_human_gate() -> None:
    reason = _router_human_gate_reason("work_message", 0.2)

    assert "project ownership confidence is low" in reason
    assert _router_human_gate_reason("work_message", 0.8) == ""
    assert _router_human_gate_reason("chat", 0.1) == ""


def test_executor_merge_request_requires_human_gate() -> None:
    reason = _executor_human_gate_reason("Opened a pull request to merge into main.")

    assert "human approval" in reason
    assert _executor_human_gate_reason("Created an isolated branch and stopped before merge.") == ""


def test_verifier_does_not_request_human_before_failure_threshold() -> None:
    verifier = _verifier_node(_FakeJsonModel('{"status": "need_user", "ok": false, "reason": "missing"}'))

    state = verifier({"final_answer": "partial", "status": "running", "verifier_failures": 1})

    assert state["status"] == "failed"
    assert state["verifier_failures"] == 2
    assert state["needs_human"] is False


def test_verifier_treats_done_with_not_ok_as_failure() -> None:
    verifier = _verifier_node(_FakeJsonModel('{"status": "done", "ok": false, "reason": "not enough"}'))

    state = verifier({"final_answer": "partial", "status": "running", "verifier_failures": 0})

    assert state["status"] == "failed"
    assert state["verifier_failures"] == 1
    assert _after_verifier(state) == "reflect"


def test_verifier_requests_human_on_third_failure_and_skips_reflector() -> None:
    verifier = _verifier_node(_FakeJsonModel('{"status": "failed", "ok": false, "reason": "still wrong"}'))

    state = verifier({"final_answer": "partial", "status": "running", "verifier_failures": 2})

    assert state["status"] == "need_user"
    assert state["verifier_failures"] == 3
    assert state["needs_human"] is True
    assert "Verifier failed 3 times" in state["human_gate_reason"]
    assert _after_verifier(state) == "finalize"


def test_reflector_routes_planner_failures_back_to_planner() -> None:
    reflector = _reflector_node(
        _FakeJsonModel(
            '{"reflection": "Need better context before execution.", "failure_type": "missing_context"}'
        )
    )

    state = reflector({"status": "failed", "tool_results": []})

    assert state["retry_target"] == "planner"
    assert state["status"] == "running"
    assert _after_reflector(state) == "planner"


def test_reflector_routes_executor_failures_back_to_executor() -> None:
    reflector = _reflector_node(
        _FakeJsonModel(
            '{"reflection": "The selected tool arguments were wrong.", "failure_type": "bad_tool_args"}'
        )
    )

    state = reflector({"status": "failed", "tool_results": []})

    assert state["retry_target"] == "executor"
    assert state["status"] == "running"
    assert _after_reflector(state) == "executor"


def test_reflector_persists_reflection_memory(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("AGENT_STATE_DB_PATH", str(tmp_path / "agent.sqlite3"))
    reflector = _reflector_node(
        _FakeJsonModel(
            '{"reflection": "Use the file search tool before answering.", '
            '"failure_type": "missing_context"}'
        )
    )

    state = reflector({"status": "failed", "task_type": "file_task", "tool_results": []})

    report = state["node_reports"][-1]
    assert report["data"]["reflection_memory_id"] > 0


def test_reflector_honors_explicit_finalizer_retry_target() -> None:
    reflector = _reflector_node(
        _FakeJsonModel(
            '{"reflection": "Retrying cannot help without external input.", '
            '"failure_type": "unknown", "retry_target": "finalizer"}'
        )
    )

    state = reflector({"status": "failed", "tool_results": []})

    assert state["retry_target"] == "finalizer"
    assert state["status"] == "failed"
    assert _after_reflector(state) == "finalize"


def test_retry_target_for_failure_defaults_to_reasonable_node() -> None:
    assert _retry_target_for_failure("planning_loop") == "planner"
    assert _retry_target_for_failure("wrong_tool") == "executor"
    assert _retry_target_for_failure("unknown", "route and scope were wrong") == "planner"
    assert _retry_target_for_failure("unknown", "tool args failed") == "executor"


def test_trace_events_from_executor_messages_extracts_tool_calls() -> None:
    messages = [
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "search_knowledge",
                    "args": {"query": "agent tracing"},
                    "id": "call-1",
                    "type": "tool_call",
                }
            ],
        ),
        ToolMessage(content="result text", name="search_knowledge", tool_call_id="call-1"),
    ]

    events = _trace_events_from_executor_messages(messages)

    assert [event.event_type for event in events] == ["tool_call", "tool_result"]
    assert events[0].tool == "search_knowledge"
    assert events[0].args == {"query": "agent tracing"}
    assert events[1].content == "result text"


def test_rag_trace_uses_actual_tool_result_payload() -> None:
    content = "\n".join(
        [
            "Knowledge results (/tmp/agent.sqlite3):",
            "1. score=0.900 method=lexical_rerank [docs/a.md#intro] title=a heading=Intro",
            "   chunk text",
        ]
    )
    messages = [ToolMessage(content=content, name="search_knowledge", tool_call_id="call-1")]

    events = _rag_trace_events_from_tool_results({"user_input": "unrelated query"}, messages)

    assert len(events) == 1
    assert events[0].args["query"] == ""
    assert events[0].args["results"][0]["citation_id"] == "docs/a.md#intro"
    assert events[0].args["results"][0]["score"] == 0.9


def test_coerce_plan_limits_and_normalizes_steps() -> None:
    fallback = [{"step": 1, "action": "execute", "description": "fallback"}]
    plan = _coerce_plan(
        [
            {"description": "Read context"},
            {"step": 2, "action": "verify", "description": "Check result"},
            {"step": 3},
        ],
        fallback,
    )

    assert plan == [
        {"step": 1, "action": "retrieve_context", "description": "Read context"},
        {"step": 2, "action": "verify", "description": "Check result"},
    ]
