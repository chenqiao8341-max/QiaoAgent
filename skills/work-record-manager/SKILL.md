---
name: work-record-manager
description: Use when the user asks to manage Qiao's work record, organize work progress, update /home/qiao/work/aaa-work.md, insert a new work item with project paths, summarize progress across project folders, or improve Markdown structure for project management notes.
---

# Work Record Manager

This skill helps maintain Qiao's personal work record:

```text
/home/qiao/work/aaa-work.md
```

Use this skill when the user says things like:

- 整理工作记录
- 更新工作进展
- 工作记录中插入新工作，路径是 ...
- 总结我各个项目当前进度
- 美化/规范化工作记录 Markdown

## Document Style

Keep the work record as concise Markdown. Prefer this structure:

```markdown
# 工作记录

## <工作名称>

- 状态：进行中 / 基本完成 / 暂停 / 待启动
- 项目文件：
  - `/absolute/path/one`
  - `/absolute/path/two`
- 最近进展：...
- 下一步：...
- 更新时间：YYYY-MM-DD
```

Rules:

- Use absolute paths in backticks.
- Keep each work item scannable; avoid long essays.
- Preserve user-written facts unless newer project evidence clearly updates them.
- If a project path is missing or unreadable, record that as a note instead of inventing progress.
- When adding a new work item, insert it as its own `##` section.


## Work Inbox and Routing

When the user pastes manually organized Feishu messages or informal work notes:

1. Call `capture_work_message` for each coherent message or task-like chunk.
2. Use the returned `matched_work`, `classification`, `route`, and `priority` to decide the next action.
3. If the route is `self`, handle the task directly when it is small, then update the work record if useful.
4. If the route is `codex`, call `create_work_task` first, then use `rewrite_task_for_codex` before any `run_codex_task` delegation.
5. If the route is `ask_user`, ask one short clarifying question instead of guessing.
6. If the route is `defer`, keep it in the inbox/task list and mention that it was deferred.

Use `list_work_inbox` and `list_work_tasks` when the user asks what is pending.

Routing meanings:

- `self`: the agent can summarize, update Markdown, inspect a small number of files, or create a simple note.
- `codex`: code changes, deployment debugging, script/test execution, or multi-file engineering work.
- `ask_user`: missing project path, ambiguous ownership, or unclear desired action.
- `defer`: low urgency notes or reminders.

## Updating Progress

When asked to organize or update the work record:

1. Read `/home/qiao/work/aaa-work.md` first.
2. Extract each work item and its project paths.
3. For each path, inspect only high-signal files first:
   - `README.md`
   - `agentlearn.md`
   - `docs/`
   - recent reports under `reports/`, `results/`, or deployment notes
   - git status/log if shell access is enabled and useful
4. Update `最近进展`, `下一步`, and `更新时间` from observed evidence.
5. Keep uncertain conclusions phrased as uncertainty.

## Markdown Writing

When rewriting the file:

- Normalize headings and spacing.
- Use bullet lists for project paths and next steps.
- Do not include hidden reasoning or tool logs.
- Do not remove a work item unless the user explicitly asks.
