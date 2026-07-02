from __future__ import annotations

from agent_project.workflow import build_plan, classify_route_risk_difficulty, classify_task_type


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
