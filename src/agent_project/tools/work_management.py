from __future__ import annotations

import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from langchain_core.tools import tool

from agent_project.tools.progress import emit_progress
from agent_project.tools.storage import connect, state_db_path


ROUTES = {"self", "codex", "ask_user", "defer"}
PRIORITIES = {"low", "normal", "high", "urgent"}
TASK_STATUSES = {"pending", "in_progress", "completed", "blocked", "cancelled"}


@dataclass(frozen=True)
class WorkRecordItem:
    title: str
    paths: list[str]
    progress: str
    raw: str


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def work_record_path() -> Path:
    configured = os.getenv("AGENT_WORK_RECORD_PATH", "/home/qiao/work/aaa-work.md")
    return Path(configured).expanduser().resolve()


def _read_record() -> str:
    path = work_record_path()
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def parse_work_record(text: str) -> list[WorkRecordItem]:
    container_mode = _has_work_record_container(text)
    sections: list[tuple[str, list[str]]] = []
    current_title = ""
    current_lines: list[str] = []
    for line in text.splitlines():
        match = re.match(r"^(#+)\s+(.+?)\s*$", line)
        if match and _is_work_item_heading(match.group(1), match.group(2), container_mode):
            if current_title:
                sections.append((current_title, current_lines))
            current_title = match.group(2).strip()
            current_lines = []
        else:
            current_lines.append(line)
    if current_title:
        sections.append((current_title, current_lines))

    items: list[WorkRecordItem] = []
    for title, lines in sections:
        paths: list[str] = []
        progress_lines: list[str] = []
        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue
            path_matches = re.findall(r"`(/[^`]+)`|(?<![\w`])(/[^\s`，。；,;]+)", stripped)
            for backticked, plain in path_matches:
                candidate = (backticked or plain).strip()
                if candidate and candidate not in paths:
                    paths.append(candidate)
            if any(
                marker in stripped
                for marker in ("状态", "最近进展", "工作进展", "下一步", "主要不足", "更新时间")
            ):
                progress_lines.append(stripped)
        items.append(
            WorkRecordItem(
                title=title,
                paths=paths,
                progress="\n".join(progress_lines),
                raw="\n".join(lines).strip(),
            )
        )
    return items


def _has_work_record_container(text: str) -> bool:
    lines = text.splitlines()
    top_level_titles = [
        match.group(1).strip()
        for line in lines
        if (match := re.match(r"^#\s+(.+?)\s*$", line))
    ]
    return bool(top_level_titles and top_level_titles[0] in {"工作记录", "Work Record"})


def _is_work_item_heading(marker: str, title: str, container_mode: bool) -> bool:
    clean_title = title.strip()
    if clean_title in {"工作记录", "Work Record"}:
        return False
    if len(marker) == 1:
        return True
    if container_mode and len(marker) == 2:
        return clean_title not in {"项目文件", "项目路径", "工作进展", "最近进展", "进展"}
    return False


def _keywords(text: str) -> set[str]:
    lowered = text.lower()
    words = set(re.findall(r"[a-zA-Z0-9_+-]{3,}|[\u4e00-\u9fff]{2,}", lowered))
    aliases = {
        "nlu": {"nlu", "ner", "作者", "机构", "测评", "bert", "llamafactory"},
        "vllm": {"vllm", "模型", "部署", "worker", "gpu", "qwen", "nginx"},
        "医疗": {"医疗", "翻译", "nhtai", "术语", "comet", "nginx", "worker"},
        "测评": {"测评", "评测", "healthbench", "medhallu", "medxpertqa", "knows"},
        "数学": {"数学", "论文", "weyl", "witt", "lifting", "lauritzen", "thomsen"},
    }
    for key, values in aliases.items():
        if key in lowered:
            words.update(values)
    return words


def _classify_against_record(content: str, items: list[WorkRecordItem]) -> tuple[str, str]:
    try:
        from agent_project.tools.work_vectors import best_work_record_match

        matched_title, score = best_work_record_match(content)
        if matched_title:
            return matched_title, f"existing_work_vector_score_{score:.3f}"
    except Exception:
        pass

    content_words = _keywords(content)
    best_title = ""
    best_score = 0
    for item in items:
        haystack = " ".join([item.title, item.progress, " ".join(item.paths)])
        score = len(content_words & _keywords(haystack))
        for path in item.paths:
            name = Path(path).name.lower()
            if name and name in content.lower():
                score += 3
        if score > best_score:
            best_title = item.title
            best_score = score
    if best_score <= 0:
        return "", "new_work_or_unclassified"
    return best_title, "existing_work"


def _route_message(content: str, matched_work: str) -> tuple[str, str, str]:
    lowered = content.lower()
    urgent_markers = ["紧急", "马上", "今天", "阻塞", "故障", "挂了", "报错", "timeout", "崩", "失败"]
    codex_markers = [
        "代码",
        "修改",
        "实现",
        "bug",
        "报错",
        "部署",
        "nginx",
        "docker",
        "服务",
        "测试",
        "脚本",
        "接口",
        "pipeline",
    ]
    ask_markers = ["是否", "吗", "？", "?", "什么时候", "确认", "问一下"]
    defer_markers = ["记录", "备忘", "之后", "有空", "暂时"]

    priority = "high" if any(marker in lowered for marker in urgent_markers) else "normal"
    if any(marker in lowered for marker in codex_markers):
        return "codex", priority, "包含代码、部署、服务或测试信号，适合改写任务后交给 Codex。"
    if any(marker in lowered for marker in ask_markers) and not matched_work:
        return "ask_user", priority, "消息像是需要进一步确认，且没有匹配到已有工作。"
    if any(marker in lowered for marker in defer_markers):
        return "defer", "low" if priority == "normal" else priority, "消息更像备忘或低紧急度记录。"
    return "self", priority, "可由 agent 自己整理、更新工作记录或生成下一步。"


def _format_item(item: WorkRecordItem) -> str:
    lines = [f"# {item.title}"]
    if item.paths:
        lines.append("项目文件：")
        lines.extend(f"- {path}" for path in item.paths)
    if item.progress:
        lines.append("工作进展：")
        lines.append(item.progress)
    return "\n".join(lines)


@tool
def list_work_record_items() -> str:
    """List work items parsed from the configured work record Markdown file."""
    path = work_record_path()
    emit_progress(f"reading work record: {path}")
    items = parse_work_record(_read_record())
    if not items:
        return f"No work items found in {path}."
    lines = [f"Work record: {path}"]
    for index, item in enumerate(items, start=1):
        lines.append(f"\n{index}. {item.title}")
        if item.paths:
            lines.extend(f"   - {path}" for path in item.paths)
        if item.progress:
            preview = item.progress.replace("\n", " ")[:180]
            lines.append(f"   Progress: {preview}")
    return "\n".join(lines)


@tool
def add_work_record_item(title: str, paths: list[str], progress: str = "", status: str = "待启动") -> str:
    """Append a new work item section to the work record Markdown file."""
    clean_title = title.strip()
    clean_paths = [path.strip() for path in paths if path.strip()]
    if not clean_title:
        return "Cannot add work item without a title."
    if not clean_paths:
        return "Cannot add work item without at least one project path."

    record_path = work_record_path()
    record_path.parent.mkdir(parents=True, exist_ok=True)
    existing = _read_record().rstrip()
    now = datetime.now().date().isoformat()
    section_lines = [
        f"# {clean_title}",
        "## 项目文件",
        *clean_paths,
        "## 工作进展",
        progress.strip() or f"状态：{status.strip() or '待启动'}",
        f"更新时间：{now}",
    ]
    new_text = (existing + "\n\n" if existing else "") + "\n".join(section_lines) + "\n"
    record_path.write_text(new_text, encoding="utf-8")
    emit_progress(f"work record item added: {clean_title} ({record_path})")
    return f"Added work item '{clean_title}' to {record_path}."


@tool
def capture_work_message(content: str, source: str = "manual", sender: str = "") -> str:
    """Capture a manually provided work/Feishu message, classify it, route it, and store it."""
    clean_content = content.strip()
    if not clean_content:
        return "Cannot capture an empty work message."

    items = parse_work_record(_read_record())
    matched_work, classification = _classify_against_record(clean_content, items)
    route, priority, rationale = _route_message(clean_content, matched_work)
    now = _now()
    with connect() as connection:
        cursor = connection.execute(
            """
            INSERT INTO work_inbox_messages(
                source, sender, content, matched_work, classification, route,
                priority, status, rationale, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                source.strip() or "manual",
                sender.strip(),
                clean_content,
                matched_work,
                classification,
                route,
                priority,
                "triaged",
                rationale,
                now,
                now,
            ),
        )
        message_id = cursor.lastrowid

    emit_progress(f"work message captured: {message_id} -> {matched_work or classification} / {route}")
    lines = [
        f"Inbox message {message_id} captured.",
        f"classification: {classification}",
        f"matched_work: {matched_work or '(none)'}",
        f"route: {route}",
        f"priority: {priority}",
        f"rationale: {rationale}",
    ]
    if classification == "new_work_or_unclassified":
        lines.append("next: ask the user whether this is a new work item, or call add_work_record_item if title/path are known.")
    elif route == "codex":
        lines.append("next: create a work task and rewrite it for Codex before delegating.")
    elif route == "self":
        lines.append("next: handle directly or update the work record.")
    return "\n".join(lines)


@tool
def create_work_task(
    title: str,
    description: str,
    work_title: str = "",
    route: str = "self",
    priority: str = "normal",
    inbox_message_id: int | None = None,
) -> str:
    """Create a structured work task from an inbox message or direct user request."""
    clean_title = title.strip()
    if not clean_title:
        return "Cannot create a work task without a title."
    normalized_route = route.strip().lower() or "self"
    normalized_priority = priority.strip().lower() or "normal"
    if normalized_route not in ROUTES:
        return f"Invalid route. Use one of: {', '.join(sorted(ROUTES))}."
    if normalized_priority not in PRIORITIES:
        return f"Invalid priority. Use one of: {', '.join(sorted(PRIORITIES))}."

    now = _now()
    with connect() as connection:
        cursor = connection.execute(
            """
            INSERT INTO work_tasks(
                work_title, title, description, route, priority, status,
                inbox_message_id, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                work_title.strip(),
                clean_title,
                description.strip(),
                normalized_route,
                normalized_priority,
                "pending",
                inbox_message_id,
                now,
                now,
            ),
        )
        task_id = cursor.lastrowid
        if inbox_message_id:
            connection.execute(
                "UPDATE work_inbox_messages SET status = ?, updated_at = ? WHERE id = ?",
                ("task_created", now, inbox_message_id),
            )

    emit_progress(f"work task created: {task_id} ({normalized_route}, {normalized_priority})")
    return (
        f"Work task {task_id} created.\n"
        f"work_title: {work_title or '(none)'}\n"
        f"title: {clean_title}\n"
        f"route: {normalized_route}\n"
        f"priority: {normalized_priority}\n"
        f"status: pending"
    )


@tool
def list_work_inbox(limit: int = 20, status: str = "") -> str:
    """List captured work inbox messages from SQLite."""
    limit = max(1, min(limit, 100))
    where = ""
    params: list[object] = []
    if status.strip():
        where = "WHERE status = ?"
        params.append(status.strip())
    params.append(limit)
    with connect() as connection:
        rows = connection.execute(
            f"""
            SELECT * FROM work_inbox_messages
            {where}
            ORDER BY created_at DESC, id DESC
            LIMIT ?
            """,
            params,
        ).fetchall()
    if not rows:
        return "No work inbox messages found."
    lines = [f"Work inbox ({state_db_path()}):"]
    for row in rows:
        preview = row["content"].replace("\n", " ")[:120]
        lines.append(
            f"{row['id']}. [{row['status']}] {row['priority']} {row['route']} "
            f"work={row['matched_work'] or row['classification']} :: {preview}"
        )
    return "\n".join(lines)


@tool
def list_work_tasks(limit: int = 20, status: str = "") -> str:
    """List structured work tasks from SQLite."""
    limit = max(1, min(limit, 100))
    where = ""
    params: list[object] = []
    if status.strip():
        where = "WHERE status = ?"
        params.append(status.strip())
    params.append(limit)
    with connect() as connection:
        rows = connection.execute(
            f"""
            SELECT * FROM work_tasks
            {where}
            ORDER BY updated_at DESC, id DESC
            LIMIT ?
            """,
            params,
        ).fetchall()
    if not rows:
        return "No work tasks found."
    lines = [f"Work tasks ({state_db_path()}):"]
    for row in rows:
        lines.append(
            f"{row['id']}. [{row['status']}] {row['priority']} {row['route']} "
            f"{row['work_title'] or '(no work)'} :: {row['title']}"
        )
    return "\n".join(lines)


@tool
def update_work_task(task_id: int, status: str, note: str = "") -> str:
    """Update a work task status. Status: pending, in_progress, completed, blocked, cancelled."""
    normalized = status.strip().lower()
    if normalized not in TASK_STATUSES:
        return f"Invalid status. Use one of: {', '.join(sorted(TASK_STATUSES))}."
    now = _now()
    completed_at = now if normalized == "completed" else None
    with connect() as connection:
        row = connection.execute("SELECT * FROM work_tasks WHERE id = ?", (task_id,)).fetchone()
        if row is None:
            return f"Work task not found: {task_id}"
        description = row["description"]
        if note.strip():
            description = f"{description}\n\nUpdate {now}: {note.strip()}".strip()
        connection.execute(
            """
            UPDATE work_tasks
            SET status = ?, description = ?, updated_at = ?, completed_at = ?
            WHERE id = ?
            """,
            (normalized, description, now, completed_at, task_id),
        )
    emit_progress(f"work task updated: {task_id} -> {normalized}")
    return f"Work task {task_id} updated to {normalized}."
