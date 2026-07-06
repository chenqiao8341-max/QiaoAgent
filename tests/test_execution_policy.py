from __future__ import annotations

from agent_project.execution_policy import (
    policy_for_task,
    select_tool_names,
    task_policy_violation,
)
from agent_project.tools import get_tools_for_task


def test_policy_selects_minimal_chat_tools() -> None:
    policy = policy_for_task("chat", "self")

    assert policy.max_tool_calls <= 2
    assert select_tool_names("chat", "self") == {
        "calculator",
        "current_time",
        "configured_provider",
    }


def test_dynamic_tool_subset_excludes_shell_for_work_messages() -> None:
    tool_names = {tool.name for tool in get_tools_for_task("work_message", "self")}

    assert "capture_work_message" in tool_names
    assert "search_work_record_vectors" in tool_names
    assert "execute_shell_command" not in tool_names
    assert "run_codex_task" not in tool_names


def test_policy_reports_budget_violations() -> None:
    policy = policy_for_task("work_message", "self")

    assert task_policy_violation(policy, tool_call_count=policy.max_tool_calls, elapsed_seconds=1.0) == ""
    assert "tool call budget" in task_policy_violation(
        policy,
        tool_call_count=policy.max_tool_calls + 1,
        elapsed_seconds=1.0,
    )
    assert "time budget" in task_policy_violation(
        policy,
        tool_call_count=0,
        elapsed_seconds=policy.max_seconds + 0.1,
    )
