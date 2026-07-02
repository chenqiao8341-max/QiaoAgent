from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from uuid import uuid4

from langchain_core.tools import tool

from agent_project.tools.browser import open_web_page
from agent_project.tools.codex_delegate import start_codex_session
from agent_project.tools.progress import emit_progress
from agent_project.tools.storage import connect, state_db_path
from agent_project.tools.web import web_search


GOAL_STATUSES = {"active", "completed", "blocked", "cancelled"}
GOAL_PHASES = {
    "created",
    "researching",
    "gap_analysis",
    "prioritizing",
    "delegating",
    "testing",
    "iterating",
    "completed",
    "blocked",
    "cancelled",
}
PRIORITIES = {"low", "normal", "high", "urgent"}


@dataclass(frozen=True)
class AgentGoal:
    goal_id: str
    title: str
    objective: str
    status: str
    phase: str
    priority: str
    iteration: int
    success_criteria: str
    evidence: str
    codex_session_id: str
    created_at: str
    updated_at: str
    completed_at: str | None = None


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _normalize_status(status: str) -> str | None:
    normalized = status.strip().lower()
    return normalized if normalized in GOAL_STATUSES else None


def _normalize_phase(phase: str) -> str | None:
    normalized = phase.strip().lower()
    return normalized if normalized in GOAL_PHASES else None


def _clean_priority(priority: str) -> str:
    normalized = priority.strip().lower() or "normal"
    return normalized if normalized in PRIORITIES else "normal"


def _row_to_goal(row) -> AgentGoal:
    return AgentGoal(
        goal_id=row["goal_id"],
        title=row["title"],
        objective=row["objective"],
        status=row["status"],
        phase=row["phase"],
        priority=row["priority"],
        iteration=int(row["iteration"]),
        success_criteria=row["success_criteria"],
        evidence=row["evidence"],
        codex_session_id=row["codex_session_id"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        completed_at=row["completed_at"],
    )


def _load_goal(goal_id: str) -> AgentGoal | None:
    with connect() as connection:
        row = connection.execute(
            "SELECT * FROM agent_goals WHERE goal_id = ?",
            (goal_id.strip(),),
        ).fetchone()
    if row is None:
        return None
    return _row_to_goal(row)


def _append_event(
    goal_id: str,
    event_type: str,
    summary: str,
    details: str = "",
    phase: str = "",
) -> None:
    with connect() as connection:
        connection.execute(
            """
            INSERT INTO agent_goal_events(goal_id, event_type, phase, summary, details, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                goal_id,
                event_type.strip() or "note",
                phase.strip(),
                summary.strip(),
                details.strip(),
                _now(),
            ),
        )


def _format_goal(goal: AgentGoal, include_evidence: bool = True) -> str:
    lines = [
        f"Goal {goal.goal_id}: {goal.title}",
        f"status: {goal.status}",
        f"phase: {goal.phase}",
        f"priority: {goal.priority}",
        f"iteration: {goal.iteration}",
        f"created_at: {goal.created_at}",
        f"updated_at: {goal.updated_at}",
        f"completed_at: {goal.completed_at or ''}",
        f"codex_session_id: {goal.codex_session_id or ''}",
        "",
        "objective:",
        goal.objective,
    ]
    if goal.success_criteria:
        lines.extend(["", "success_criteria:", goal.success_criteria])
    if include_evidence and goal.evidence:
        lines.extend(["", "evidence:", goal.evidence])
    return "\n".join(lines)


def _recent_goal_events(goal_id: str, limit: int = 20) -> list[str]:
    with connect() as connection:
        rows = connection.execute(
            """
            SELECT event_type, phase, summary, details, created_at
            FROM agent_goal_events
            WHERE goal_id = ?
            ORDER BY created_at DESC, id DESC
            LIMIT ?
            """,
            (goal_id, max(1, min(int(limit), 100))),
        ).fetchall()
    events = []
    for row in rows:
        details = f" | {row['details']}" if row["details"] else ""
        events.append(
            f"- {row['created_at']} [{row['event_type']}/{row['phase']}] "
            f"{row['summary']}{details}"
        )
    return events


def _set_goal_state(
    goal: AgentGoal,
    *,
    status: str | None = None,
    phase: str | None = None,
    iteration: int | None = None,
    evidence: str | None = None,
    codex_session_id: str | None = None,
) -> None:
    now = _now()
    new_status = status if status is not None else goal.status
    new_phase = phase if phase is not None else goal.phase
    completed_at = (
        now if new_status in {"completed", "cancelled"} and not goal.completed_at else goal.completed_at
    )
    with connect() as connection:
        connection.execute(
            """
            UPDATE agent_goals
            SET status = ?, phase = ?, iteration = ?, evidence = ?,
                codex_session_id = ?, updated_at = ?, completed_at = ?
            WHERE goal_id = ?
            """,
            (
                new_status,
                new_phase,
                goal.iteration if iteration is None else max(0, int(iteration)),
                goal.evidence if evidence is None else evidence,
                goal.codex_session_id if codex_session_id is None else codex_session_id,
                now,
                completed_at,
                goal.goal_id,
            ),
        )


def _first_urls(search_output: str, limit: int) -> list[str]:
    urls: list[str] = []
    for line in search_output.splitlines():
        stripped = line.strip()
        if stripped.startswith("http://") or stripped.startswith("https://"):
            if stripped not in urls:
                urls.append(stripped)
        if len(urls) >= limit:
            break
    return urls


def _compact(text: str, max_chars: int) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[:max_chars] + f"\n\n[truncated after {max_chars} characters]"


def _self_improvement_cycle_instruction(goal: AgentGoal) -> str:
    return "\n".join(
        [
            f"继续 agent self-improvement goal {goal.goal_id}: {goal.title}",
            "请读取 get_agent_goal 的历史事件，检查 Codex 修改后的项目状态，运行必要验证。",
            "如果改进合理，调用 update_agent_goal 进入 iterating 或 completed；",
            "如果还不够，继续调研/差距分析，选择下一项最高优先级改进并再次委托 Codex。",
        ]
    )


@tool
def create_agent_goal(
    title: str,
    objective: str,
    success_criteria: str = "",
    priority: str = "normal",
) -> str:
    """Create a persistent high-level goal for autonomous multi-turn work."""
    clean_title = title.strip()
    clean_objective = objective.strip()
    if not clean_title:
        return "Cannot create an agent goal without a title."
    if not clean_objective:
        return "Cannot create an agent goal without an objective."

    goal_id = uuid4().hex[:8]
    now = _now()
    normalized_priority = _clean_priority(priority)
    with connect() as connection:
        connection.execute(
            """
            INSERT INTO agent_goals(
                goal_id, title, objective, status, phase, priority, iteration,
                success_criteria, evidence, codex_session_id, created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                goal_id,
                clean_title,
                clean_objective,
                "active",
                "created",
                normalized_priority,
                0,
                success_criteria.strip(),
                "",
                "",
                now,
                now,
            ),
        )
        connection.execute(
            """
            INSERT INTO agent_goal_events(goal_id, event_type, phase, summary, details, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                goal_id,
                "created",
                "created",
                clean_title,
                clean_objective,
                now,
            ),
        )

    emit_progress(f"agent goal created: {goal_id} ({clean_title})")
    goal = _load_goal(goal_id)
    return _format_goal(goal) if goal else f"Goal created: {goal_id}"


@tool
def update_agent_goal(
    goal_id: str,
    status: str = "",
    phase: str = "",
    priority: str = "",
    iteration_delta: int = 0,
    evidence: str = "",
    codex_session_id: str = "",
    event_summary: str = "",
    event_details: str = "",
) -> str:
    """Update a persistent goal's status, phase, iteration, evidence, or Codex session."""
    goal = _load_goal(goal_id)
    if goal is None:
        return f"Agent goal not found: {goal_id}"

    if status.strip():
        normalized_status = _normalize_status(status)
        if normalized_status is None:
            return f"Invalid status. Use one of: {', '.join(sorted(GOAL_STATUSES))}."
        new_status = normalized_status
    else:
        new_status = goal.status

    if phase.strip():
        normalized_phase = _normalize_phase(phase)
        if normalized_phase is None:
            return f"Invalid phase. Use one of: {', '.join(sorted(GOAL_PHASES))}."
        new_phase = normalized_phase
    else:
        new_phase = goal.phase

    new_priority = _clean_priority(priority) if priority.strip() else goal.priority
    new_iteration = max(0, goal.iteration + int(iteration_delta))
    new_evidence = evidence.strip() if evidence.strip() else goal.evidence
    new_codex_session_id = (
        codex_session_id.strip() if codex_session_id.strip() else goal.codex_session_id
    )
    completed_at = (
        _now() if new_status in {"completed", "cancelled"} and not goal.completed_at else goal.completed_at
    )
    if new_status == "blocked":
        new_phase = "blocked"
    elif new_status == "completed":
        new_phase = "completed"
    elif new_status == "cancelled":
        new_phase = "cancelled"

    now = _now()
    with connect() as connection:
        connection.execute(
            """
            UPDATE agent_goals
            SET status = ?, phase = ?, priority = ?, iteration = ?, evidence = ?,
                codex_session_id = ?, updated_at = ?, completed_at = ?
            WHERE goal_id = ?
            """,
            (
                new_status,
                new_phase,
                new_priority,
                new_iteration,
                new_evidence,
                new_codex_session_id,
                now,
                completed_at,
                goal.goal_id,
            ),
        )
        if event_summary.strip() or event_details.strip():
            connection.execute(
                """
                INSERT INTO agent_goal_events(
                    goal_id, event_type, phase, summary, details, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    goal.goal_id,
                    "update",
                    new_phase,
                    event_summary.strip() or "goal updated",
                    event_details.strip(),
                    now,
                ),
            )

    emit_progress(f"agent goal updated: {goal.goal_id} -> {new_status}/{new_phase}")
    updated = _load_goal(goal.goal_id)
    return _format_goal(updated) if updated else f"Goal updated: {goal.goal_id}"


@tool
def record_agent_goal_event(
    goal_id: str,
    event_type: str,
    summary: str,
    details: str = "",
    phase: str = "",
) -> str:
    """Append an event, finding, decision, test result, or blocker to an agent goal."""
    goal = _load_goal(goal_id)
    if goal is None:
        return f"Agent goal not found: {goal_id}"
    if not summary.strip():
        return "Cannot record a goal event without a summary."
    if phase.strip():
        normalized_phase = _normalize_phase(phase)
        if normalized_phase is None:
            return f"Invalid phase. Use one of: {', '.join(sorted(GOAL_PHASES))}."
        event_phase = normalized_phase
    else:
        event_phase = goal.phase
    _append_event(
        goal.goal_id,
        event_type=event_type,
        phase=event_phase,
        summary=summary,
        details=details,
    )
    emit_progress(f"agent goal event recorded: {goal.goal_id} {event_type}")
    return f"Recorded event for goal {goal.goal_id}: {summary.strip()}"


@tool
def get_agent_goal(goal_id: str, include_events: bool = True, event_limit: int = 20) -> str:
    """Read one persistent agent goal and recent events."""
    goal = _load_goal(goal_id)
    if goal is None:
        return f"Agent goal not found: {goal_id}"

    lines = [_format_goal(goal)]
    if include_events:
        limit = max(1, min(int(event_limit), 100))
        with connect() as connection:
            rows = connection.execute(
                """
                SELECT event_type, phase, summary, details, created_at
                FROM agent_goal_events
                WHERE goal_id = ?
                ORDER BY created_at DESC, id DESC
                LIMIT ?
                """,
                (goal.goal_id, limit),
            ).fetchall()
        if rows:
            lines.append("\nrecent_events:")
            for row in rows:
                detail = f" | {row['details'][:300]}" if row["details"] else ""
                lines.append(
                    f"- {row['created_at']} [{row['event_type']}/{row['phase']}] "
                    f"{row['summary']}{detail}"
                )
    return "\n".join(lines)


@tool
def list_agent_goals(status: str = "", limit: int = 20) -> str:
    """List persistent high-level goals, optionally filtered by status."""
    normalized_status = status.strip().lower()
    limit = max(1, min(int(limit), 100))
    query = """
        SELECT goal_id, title, status, phase, priority, iteration, updated_at
        FROM agent_goals
    """
    params: tuple[object, ...] = ()
    if normalized_status:
        if normalized_status not in GOAL_STATUSES:
            return f"Invalid status. Use one of: {', '.join(sorted(GOAL_STATUSES))}."
        query += " WHERE status = ?"
        params = (normalized_status,)
    query += " ORDER BY updated_at DESC, created_at DESC LIMIT ?"
    params = (*params, limit)

    emit_progress(f"listing agent goals from {state_db_path()}")
    with connect() as connection:
        rows = connection.execute(query, params).fetchall()

    if not rows:
        return "No agent goals stored."
    return "\n".join(
        f"{row['goal_id']} [{row['status']}/{row['phase']}] "
        f"priority={row['priority']} iteration={row['iteration']} "
        f"updated={row['updated_at']} {row['title']}"
        for row in rows
    )


@tool
def create_self_improvement_codex_prompt(
    goal_id: str,
    gap_analysis: str,
    prioritized_improvement: str,
    test_plan: str,
    cwd: str = "/home/qiao/work/agent_project",
) -> str:
    """Create a Codex prompt for one self-improvement iteration tied to a goal."""
    goal = _load_goal(goal_id)
    if goal is None:
        return f"Agent goal not found: {goal_id}"
    if not gap_analysis.strip():
        return "gap_analysis is required."
    if not prioritized_improvement.strip():
        return "prioritized_improvement is required."

    next_iteration = goal.iteration + 1
    prompt = "\n".join(
        [
            "You are Codex working on Qiao's local agent project.",
            "",
            f"Repository: {cwd}",
            f"Agent goal id: {goal.goal_id}",
            f"Agent goal title: {goal.title}",
            "",
            "Workflow background:",
            (
                "The local agent is trying to gain Codex-like persistent goal behavior. "
                "It should be able to research mature coding agents such as Codex and "
                "Claude Code, assess its own capability gaps, choose the highest-priority "
                "improvement, delegate implementation to Codex, restart/test itself, and "
                "iterate with the same goal context."
            ),
            "",
            "Current objective:",
            goal.objective,
            "",
            "Current success criteria:",
            goal.success_criteria or "(not specified)",
            "",
            "Gap analysis gathered by the agent:",
            gap_analysis.strip(),
            "",
            f"Highest-priority improvement for iteration {next_iteration}:",
            prioritized_improvement.strip(),
            "",
            "Requested Codex work:",
            "- Inspect the repository before editing.",
            "- Implement the prioritized improvement with scoped changes.",
            "- Prefer codex-proxy-1 if you need to start or continue Codex from inside the project.",
            "- Add or update tests/docs when useful.",
            "- Run focused verification when possible.",
            "- After implementation, tell the agent exactly how to restart and test itself.",
            "- Include a follow-up prompt the agent can send after restart to continue this goal.",
            "",
            "Agent-side test plan to preserve or improve:",
            test_plan.strip() or "(Codex should propose focused verification.)",
            "",
            "Return changed files, verification results, blockers, and the next agent prompt.",
        ]
    )
    _append_event(
        goal.goal_id,
        event_type="codex_prompt_prepared",
        phase="delegating",
        summary=f"Prepared Codex prompt for iteration {next_iteration}",
        details=prioritized_improvement.strip(),
    )
    return prompt


@tool
def run_agent_goal_cycle(
    goal_id: str,
    mode: str = "auto",
    research_queries: list[str] | None = None,
    gap_analysis: str = "",
    prioritized_improvement: str = "",
    test_plan: str = "",
    cwd: str = "/home/qiao/work/agent_project",
    codex_command: str = "codex-proxy-1",
    start_codex: bool = False,
    max_sources_per_query: int = 2,
    timeout_seconds: int = 1800,
) -> str:
    """Run one controlled step of a goal loop: research, delegate to Codex, or record testing."""
    goal = _load_goal(goal_id)
    if goal is None:
        return f"Agent goal not found: {goal_id}"

    normalized_mode = mode.strip().lower() or "auto"
    if normalized_mode == "auto":
        if goal.phase in {"created", "researching"}:
            normalized_mode = "research"
        elif goal.phase in {"gap_analysis", "prioritizing", "delegating"}:
            normalized_mode = "delegate"
        else:
            normalized_mode = "test"
    if normalized_mode not in {"research", "delegate", "test"}:
        return "Invalid mode. Use one of: auto, research, delegate, test."

    if normalized_mode == "research":
        emit_progress(f"running goal research cycle: {goal.goal_id}")
        _set_goal_state(goal, phase="researching")
        queries = [query.strip() for query in (research_queries or []) if query.strip()]
        if not queries:
            queries = [
                "OpenAI Codex coding agent capabilities",
                "Claude Code agent capabilities",
                "coding agent persistent goals task planning tool use",
            ]

        outputs: list[str] = []
        source_urls: list[str] = []
        for query in queries[:5]:
            search_result = web_search.invoke(
                {"query": query, "max_results": 5, "timeout_seconds": 10}
            )
            outputs.append(f"## Search: {query}\n\n{search_result}")
            for url in _first_urls(search_result, max_sources_per_query):
                source_urls.append(url)
                page = open_web_page.invoke(
                    {"url": url, "max_chars": 6000, "timeout_seconds": 15}
                )
                outputs.append(f"## Source: {url}\n\n{page}")

        evidence = _compact("\n\n".join(outputs), 40000)
        _append_event(
            goal.goal_id,
            event_type="research",
            phase="researching",
            summary=f"Ran {len(queries[:5])} research queries and opened {len(source_urls)} sources",
            details="\n".join(source_urls),
        )
        refreshed = _load_goal(goal.goal_id) or goal
        _set_goal_state(refreshed, phase="gap_analysis", evidence=evidence)
        return "\n".join(
            [
                f"Goal {goal.goal_id} research cycle complete.",
                "Next: synthesize the evidence into gap_analysis and prioritized_improvement,",
                "then call run_agent_goal_cycle(mode='delegate', ...).",
                "",
                "Sources opened:",
                "\n".join(f"- {url}" for url in source_urls) or "(none)",
                "",
                "Evidence excerpt:",
                _compact(evidence, 8000),
            ]
        )

    if normalized_mode == "delegate":
        emit_progress(f"running goal delegation cycle: {goal.goal_id}")
        if not gap_analysis.strip():
            return (
                "gap_analysis is required for delegation. Use get_agent_goal to read "
                "research evidence, synthesize gaps, then retry."
            )
        if not prioritized_improvement.strip():
            return "prioritized_improvement is required for delegation."

        refreshed = _load_goal(goal.goal_id) or goal
        prompt = create_self_improvement_codex_prompt.invoke(
            {
                "goal_id": refreshed.goal_id,
                "gap_analysis": gap_analysis,
                "prioritized_improvement": prioritized_improvement,
                "test_plan": test_plan,
                "cwd": cwd,
            }
        )
        _set_goal_state(
            refreshed,
            phase="delegating",
            iteration=refreshed.iteration + 1,
        )

        codex_result = ""
        session_id = ""
        if start_codex:
            codex_result = start_codex_session.invoke(
                {
                    "prompt": prompt,
                    "cwd": cwd,
                    "timeout_seconds": timeout_seconds,
                    "codex_command": codex_command or "codex-proxy-1",
                    "max_output_chars": 60000,
                }
            )
            for token in codex_result.replace("\n", " ").split():
                if token.startswith("session_id="):
                    session_id = token.split("=", 1)[1].strip()
                    if session_id == "(unknown)":
                        session_id = ""
                    break
            latest = _load_goal(goal.goal_id) or refreshed
            _set_goal_state(latest, phase="testing", codex_session_id=session_id)
            _append_event(
                goal.goal_id,
                event_type="codex_delegated",
                phase="delegating",
                summary="Started Codex session for self-improvement iteration",
                details=codex_result[:12000],
            )
        else:
            latest = _load_goal(goal.goal_id) or refreshed
            _set_goal_state(latest, phase="delegating")
            _append_event(
                goal.goal_id,
                event_type="codex_prompt_ready",
                phase="delegating",
                summary="Prepared Codex prompt but did not start Codex",
                details=prompt[:12000],
            )

        next_prompt = _self_improvement_cycle_instruction(_load_goal(goal.goal_id) or refreshed)
        parts = [
            f"Goal {goal.goal_id} delegation cycle prepared.",
            f"start_codex: {start_codex}",
            f"codex_command: {codex_command or 'codex-proxy-1'}",
            "",
            "Next agent prompt after restart:",
            next_prompt,
            "",
            "Codex prompt:",
            prompt,
        ]
        if codex_result:
            parts.extend(["", "Codex result:", codex_result])
        return "\n".join(parts)

    emit_progress(f"running goal testing cycle: {goal.goal_id}")
    summary = test_plan.strip() or "Testing cycle requested; inspect project and run focused checks."
    _append_event(
        goal.goal_id,
        event_type="test_cycle",
        phase="testing",
        summary=summary,
        details=(
            "Use shell/file tools to run the verification that Codex recommended. "
            "Record results with record_agent_goal_event or update_agent_goal."
        ),
    )
    refreshed = _load_goal(goal.goal_id) or goal
    _set_goal_state(refreshed, phase="testing")
    return "\n".join(
        [
            f"Goal {goal.goal_id} moved to testing.",
            "Run the project's focused verification now. After that:",
            "- call update_agent_goal(status='completed') if success criteria are met;",
            "- call update_agent_goal(phase='iterating') if another improvement is needed;",
            "- call update_agent_goal(status='blocked') if the same blocker repeats.",
            "",
            "Recent events:",
            "\n".join(_recent_goal_events(goal.goal_id, limit=10)),
        ]
    )
