from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class TaskExecutionPolicy:
    task_type: str
    allowed_tools: set[str]
    max_tool_calls: int
    max_seconds: float
    required_artifacts: set[str] = field(default_factory=set)


CHAT_TOOLS = {"calculator", "current_time", "configured_provider"}

WORK_MESSAGE_TOOLS = {
    "capture_work_message",
    "create_work_task",
    "list_work_inbox",
    "list_work_record_items",
    "list_work_tasks",
    "update_work_task",
    "preload_work_record_embedding_model",
    "index_work_record_vectors",
    "search_work_record_vectors",
}

RAG_TOOLS = {
    "index_documents",
    "search_knowledge",
    "answer_with_citations",
    "verify_answer_citations",
    "verify_answer_against_retrieved_chunks_tool",
}

RESEARCH_TOOLS = {
    "web_search",
    "open_web_page",
    "list_web_page_links",
}

FILE_TOOLS = {
    "list_local_directory",
    "read_local_file",
    "load_project_context",
    "search_knowledge",
    "answer_with_citations",
}

CODE_TOOLS = {
    "list_local_directory",
    "read_local_file",
    "write_local_file",
    "execute_shell_command",
    "ensure_work_branch",
    "rewrite_task_for_codex",
    "start_codex_session",
    "continue_codex_session",
    "run_codex_task",
    "list_codex_tasks",
    "get_codex_task",
    "test_codex_connectivity",
    "create_codex_review_packet",
    "start_codex_review",
    "list_codex_review_packets",
    "record_task_difficulty_judgment",
}

TASK_POLICIES: dict[str, TaskExecutionPolicy] = {
    "chat": TaskExecutionPolicy("chat", CHAT_TOOLS, max_tool_calls=2, max_seconds=20),
    "work_message": TaskExecutionPolicy(
        "work_message",
        WORK_MESSAGE_TOOLS,
        max_tool_calls=6,
        max_seconds=45,
        required_artifacts={"work_route"},
    ),
    "rag_qa": TaskExecutionPolicy(
        "rag_qa",
        RAG_TOOLS,
        max_tool_calls=5,
        max_seconds=60,
        required_artifacts={"citations"},
    ),
    "research": TaskExecutionPolicy("research", RESEARCH_TOOLS, max_tool_calls=10, max_seconds=120),
    "file_task": TaskExecutionPolicy("file_task", FILE_TOOLS, max_tool_calls=6, max_seconds=60),
    "code_task": TaskExecutionPolicy("code_task", CODE_TOOLS, max_tool_calls=12, max_seconds=180),
}


def policy_for_task(task_type: str, route: str = "self") -> TaskExecutionPolicy:
    policy = TASK_POLICIES.get(task_type, TASK_POLICIES["chat"])
    if route == "codex" and task_type != "code_task":
        allowed = set(policy.allowed_tools) | {
            "create_work_task",
            "rewrite_task_for_codex",
            "test_codex_connectivity",
            "run_codex_task",
            "start_codex_session",
            "continue_codex_session",
        }
        return TaskExecutionPolicy(
            task_type=policy.task_type,
            allowed_tools=allowed,
            max_tool_calls=max(policy.max_tool_calls, 8),
            max_seconds=max(policy.max_seconds, 120),
            required_artifacts=policy.required_artifacts,
        )
    return policy


def select_tool_names(task_type: str, route: str = "self") -> set[str]:
    return set(policy_for_task(task_type, route).allowed_tools)


def task_policy_violation(
    policy: TaskExecutionPolicy,
    *,
    tool_call_count: int,
    elapsed_seconds: float,
) -> str:
    if tool_call_count > policy.max_tool_calls:
        return f"Exceeded tool call budget: {tool_call_count}/{policy.max_tool_calls}."
    if elapsed_seconds > policy.max_seconds:
        return f"Exceeded time budget: {elapsed_seconds:.1f}s/{policy.max_seconds:.1f}s."
    return ""
