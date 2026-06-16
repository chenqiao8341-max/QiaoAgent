from __future__ import annotations

import ast
import datetime as dt
import operator
import os

from langchain_core.tools import tool

from agent_project.tools.codex_delegate import (
    continue_codex_session,
    get_codex_task,
    list_codex_tasks,
    rewrite_task_for_codex,
    start_codex_session,
    run_codex_task,
)
from agent_project.tools.codex_connectivity import test_codex_connectivity
from agent_project.tools.filesystem import list_local_directory, read_local_file, write_local_file
from agent_project.tools.feishu import (
    generate_feishu_report,
    get_feishu_report,
    list_feishu_messages,
    list_feishu_reports,
)
from agent_project.tools.memory import delete_memory, get_memory, remember_memory, search_memories
from agent_project.tools.shell import execute_shell_command
from agent_project.tools.skills import list_skills, read_skill, read_skill_file
from agent_project.tools.web import web_search
from agent_project.tools.browser import list_web_page_links, open_web_page
from agent_project.tools.tasks import (
    create_task_queue,
    get_task_queue,
    list_task_queues,
    update_task_step,
)
from agent_project.tools.work_management import (
    add_work_record_item,
    capture_work_message,
    create_work_task,
    list_work_inbox,
    list_work_record_items,
    list_work_tasks,
    update_work_task,
)


_BINARY_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def _eval_arithmetic(node: ast.AST) -> int | float:
    if isinstance(node, ast.Expression):
        return _eval_arithmetic(node.body)

    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value

    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY_OPS:
        left = _eval_arithmetic(node.left)
        right = _eval_arithmetic(node.right)
        return _BINARY_OPS[type(node.op)](left, right)

    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        value = _eval_arithmetic(node.operand)
        return _UNARY_OPS[type(node.op)](value)

    raise ValueError("Only numeric arithmetic expressions are allowed.")


@tool
def calculator(expression: str) -> str:
    """Evaluate a numeric arithmetic expression."""
    try:
        parsed = ast.parse(expression, mode="eval")
        result = _eval_arithmetic(parsed)
    except Exception as exc:
        return f"Calculation error: {exc}"

    return str(result)


@tool
def current_time() -> str:
    """Return the current local time in ISO 8601 format."""
    return dt.datetime.now().astimezone().isoformat(timespec="seconds")


@tool
def configured_provider() -> str:
    """Return the currently configured model provider name."""
    return os.getenv("MODEL_PROVIDER", "openai")


def get_tools():
    return [
        calculator,
        current_time,
        configured_provider,
        list_local_directory,
        read_local_file,
        write_local_file,
        execute_shell_command,
        web_search,
        open_web_page,
        list_web_page_links,
        generate_feishu_report,
        list_feishu_reports,
        get_feishu_report,
        list_feishu_messages,
        rewrite_task_for_codex,
        start_codex_session,
        continue_codex_session,
        run_codex_task,
        list_codex_tasks,
        get_codex_task,
        test_codex_connectivity,
        remember_memory,
        search_memories,
        get_memory,
        delete_memory,
        list_skills,
        read_skill,
        read_skill_file,
        list_work_record_items,
        add_work_record_item,
        capture_work_message,
        create_work_task,
        list_work_inbox,
        list_work_tasks,
        update_work_task,
        create_task_queue,
        update_task_step,
        get_task_queue,
        list_task_queues,
    ]
