# Agent Self-Upgrade Request: Codex-Assisted Self-Improvement Workflow

## Background

This is Qiao's local agent project (`/home/qiao/work/agent_project`), built on LangChain + LangGraph with `create_react_agent`. It supports multi-provider models (OpenAI, Google, Anthropic, vLLM, DashScope), tool calling, Codex-style skills, SQLite memory/task queues/goals, work inbox routing, and Codex delegation.

## Research Findings: Claude Code & Codex Capabilities

### Claude Code Core Capabilities (v2.1.x)

1. **Sub-Agents**: Specialized AI assistants with independent context windows, custom system prompts, tool restrictions, and model selection. Can be defined as Markdown files with YAML frontmatter or programmatically. Support nested subagents (up to 5 levels), automatic invocation via description matching, and explicit @-mention invocation.

2. **Hooks**: Programmable automation at 8+ lifecycle events (UserPromptSubmit, PreToolUse, PostToolUse, SessionStart/End, SubAgentResponse, Notification, ConfigChange, etc.). Can run shell commands, block operations, auto-format code, inject context after compaction, auto-approve permissions, and send notifications. Supports command hooks, prompt-based hooks (LLM-evaluated conditions), agent-based hooks, and HTTP hooks.

3. **Agent Teams**: Multiple independent Claude Code sessions coordinated as a team. One session acts as lead, teammates work independently with shared task lists, direct inter-agent messaging, plan approval workflows, and file-locking task claiming. Supports in-process and split-pane (tmux/iTerm2) modes.

4. **Todo Lists**: Structured task management within sessions. Auto-updated by Claude, supports check/uncheck, reordering, and status tracking.

5. **Checkpointing/rewind**: File change checkpointing allows rewinding to previous states.

6. **Skills**: Plugin-like markdown instructions loaded at session start, with references and scripts.

7. **Plugins**: Packaged extensions distributable across projects.

8. **Headless Mode**: CI/CD integration with `--output-format stream-json`.

9. **Persistent Memory**: Per-agent memory directories (`~/.claude/agent-memory/`).

### OpenAI Codex Capabilities

1. **Self-Improvement Loop**: Research → Gap Analysis → Prioritized Improvement → Codex Delegation → Testing → Iteration.
2. **Review Packets**: Handoff mechanism for hard tasks with evidence, excerpts, and focused questions.
3. **Difficulty Judgment**: Agent self-assesses task difficulty before attempting.
4. **Session Resume**: Local JSONL transcript with resume capability.

## Gap Analysis: Current Agent vs. Mature Agents

### Gap 1: Hooks / Deterministic Automation (HIGH PRIORITY)
**Current**: Agent relies entirely on LLM to decide when to run tools. No deterministic hooks.
**Claude Code**: Hooks fire at 8+ lifecycle events, can auto-format, block operations, inject context, send notifications, auto-approve permissions.
**Impact**: Without hooks, the agent cannot enforce project rules, auto-format, or automate repetitive tasks deterministically.

### Gap 2: Structured Todo Lists (MEDIUM-HIGH PRIORITY)
**Current**: Task queues exist but are manual state tracking tools, not integrated into the agent loop.
**Claude Code**: Todo Lists are auto-updated by Claude during execution, visible in the UI, support check/uncheck/reorder.
**Impact**: Agent has no built-in task tracking that Claude auto-manages during execution.

### Gap 3: Agent Teams / Multi-Agent Collaboration (MEDIUM PRIORITY)
**Current**: Single ReAct agent only. No parallel agent coordination.
**Claude Code**: Agent Teams with shared task lists, inter-agent messaging, plan approval, file-locking task claiming.
**Impact**: Cannot parallelize independent subtasks or run competing hypothesis debugging.

### Gap 4: Checkpointing / File Rewind (MEDIUM PRIORITY)
**Current**: No file change checkpointing. Write operations require human approval but no rollback.
**Claude Code**: Checkpointing allows rewinding file changes to previous states.
**Impact**: No safe rollback mechanism for agent file modifications.

### Gap 5: Parallel Sub-Agents with Context Isolation (MEDIUM PRIORITY)
**Current**: Single agent loop with tool calls. No sub-agent spawning.
**Claude Code**: Sub-agents run in isolated context windows, can run in parallel, report summaries back.
**Impact**: Cannot parallelize independent research or keep exploration separate from main context.

### Gap 6: System Prompt Engineering & Context Management (LOW-MEDIUM PRIORITY)
**Current**: Single monolithic system prompt. Memory injection is simple (recent N items).
**Claude Code**: CLAUDE.md for project conventions, SessionStart hooks for context re-injection after compaction, skill catalog injection.
**Impact**: Context management is basic; no automatic context recovery after compaction.

## Prioritized Improvement Plan

### Priority 1: Hooks System (Highest Impact)
Implement a hooks system that allows deterministic automation at key agent lifecycle events. This is the most impactful gap because:
- Hooks provide deterministic control that LLM-only approaches cannot match
- Auto-formatting, pre-write validation, and post-tool notifications are immediately useful
- Can be built as a LangGraph hook/callback system

### Priority 2: Todo Lists Integration
Integrate structured todo lists that the agent can auto-update during execution, providing visible task tracking.

### Priority 3: Parallel Sub-Agent Support
Add the ability to spawn sub-agents for parallel independent tasks with context isolation.

## Implementation Request for Codex

Please implement the following changes to `/home/qiao/work/agent_project`:

### Task 1: Hooks System
Create `src/agent_project/tools/hooks.py` with:
- `register_hook(name, event_type, command, matcher)` - Register a hook
- `list_hooks()` - List all registered hooks
- `remove_hook(name)` - Remove a hook
- Hook events: `pre_tool_use`, `post_tool_use`, `session_start`, `notification`
- Hook execution via subprocess (like shell tool but deterministic)
- SQLite persistence in `agent_hooks` and `agent_hook_events` tables
- PreToolUse hooks can block operations by returning a special "BLOCK" response
- PostToolUse hooks can run formatting/validation commands

### Task 2: Todo Lists
Create `src/agent_project/tools/todo_lists.py` with:
- `create_todo_list(title, items)` - Create a todo list with items
- `update_todo_item(list_id, item_index, status, note)` - Update item status (pending/in_progress/completed)
- `get_todo_list(list_id)` - Get a todo list with all items
- `list_todo_lists()` - List all todo lists
- SQLite persistence in `todo_lists` and `todo_items` tables
- Auto-append to system prompt when relevant

### Task 3: Update agent.py
- Integrate hooks into the agent build process (register hooks before creating agent)
- Add todo list context to system prompt when active
- Update DEFAULT_SYSTEM_PROMPT to mention hooks and todo lists

### Task 4: Update basic.py
- Register new tools: `register_hook`, `list_hooks`, `remove_hook`, `create_todo_list`, `update_todo_item`, `get_todo_list`, `list_todo_lists`

### Task 5: Update storage.py
- Add tables: `agent_hooks`, `agent_hook_events`, `todo_lists`, `todo_items`

### Task 6: Update README.md and agentlearn.md
- Document the new hooks and todo list capabilities

## Verification Plan

After implementation:
1. Run `python3 -m pytest` or manual tests to verify hooks register and fire
2. Verify todo lists create, update, and persist correctly
3. Verify agent can use hooks and todo lists in a conversation
4. Check that existing tools still work (no regression)

## Constraints

- Keep changes scoped to the requested improvements
- Preserve all existing tools and functionality
- Use SQLite for persistence (consistent with existing pattern)
- Follow existing code style (type hints, docstrings, @tool decorator)
- Prefer codex-proxy-1 for any Codex delegation from within

## Agent Goal Context

This work is tied to agent goal `ca9715d0`: "Codex辅助下的自我升级工作流"
The goal tracks the full self-improvement cycle: research → gap analysis → prioritization → Codex delegation → testing → iteration.
