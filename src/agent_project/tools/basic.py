from __future__ import annotations

import ast
import datetime as dt
import operator
import os

from langchain_core.tools import tool

from agent_project.tools.filesystem import list_local_directory, read_local_file, write_local_file
from agent_project.tools.memory import delete_memory, get_memory, remember_memory, search_memories
from agent_project.tools.shell import execute_shell_command
from agent_project.tools.web import web_search
from agent_project.tools.browser import list_web_page_links, open_web_page
from agent_project.tools.tasks import (
    create_task_queue,
    get_task_queue,
    list_task_queues,
    update_task_step,
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
        remember_memory,
        search_memories,
        get_memory,
        delete_memory,
        create_task_queue,
        update_task_step,
        get_task_queue,
        list_task_queues,
    ]
