# Agent Project

一个基础 LangChain + LangGraph agent 项目，支持：

- OpenAI / Google Gemini / Anthropic / 本地 vLLM provider 切换
- OpenAI-compatible API 接入，例如 DeepSeek、Kimi、DashScope/Qwen、本地 vLLM
- LangGraph `create_react_agent`
- Tool calling
- Codex-style skills：基于 `SKILL.md` 的可插拔任务知识
- 本地文件读取、目录列表、写入
- 可配置的人类审批：可对写入、越权读取和 shell 命令进行终端确认
- shell 命令执行，用于项目检查和自动化验证
- 自动联网搜索，基于 DuckDuckGo HTML 搜索页
- 轻量浏览器操作：打开网页、抽取正文、列出链接
- SQLite 长期记忆、工作收件箱、工作任务和持久化任务队列，用于跨会话保存偏好、项目事实和任务状态
- 工具执行过程可见化：读取文件、检索网页、执行命令、更新任务时输出进度
- 命令行交互
- 简单可扩展的项目结构

## 目录

```text
agent_project/
  src/agent_project/
    agent.py          # LangGraph agent 构建
    cli.py            # 命令行入口
    config.py         # 环境变量配置
    llms.py           # 多 provider 模型工厂
    tools/basic.py    # tool 注册入口和基础 tools
    tools/filesystem.py # 本地文件 tools
    tools/shell.py     # shell 命令 tool
    tools/web.py       # 联网搜索 tool
    tools/browser.py   # 轻量浏览器 tools
    tools/tasks.py     # 多步骤任务队列 tools
    tools/skills.py    # Codex-style skill 发现和读取 tools
  skills/
    work-record-manager/SKILL.md # 工作记录管理示例 skill
  examples/
    run_once.py       # 单次调用示例
```

## 安装

```bash
cd /home/qiao/work/agent_project
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
cp .env.example .env
```

编辑 `.env`，填入对应 provider 的 API key，然后运行：

```bash
agent-chat
```

## Session 恢复

交互式 `agent-chat` 会自动把每轮对话保存到本地 JSONL session 文件，默认位置是当前工作目录下的 `.agent_state/sessions/`，索引文件是 `.agent_state/session_index.jsonl`。这和 Codex 的思路类似：保留完整本地 transcript，再通过 resume 命令恢复。

常用命令：

```bash
agent-chat resume          # 从当前工作目录的历史 session 中选择
agent-chat resume --last   # 直接恢复当前工作目录最近一次 session
agent-chat resume --all    # 显示所有工作目录的 session
agent-chat resume <id>     # 用 session id 或唯一前缀恢复
agent-chat sessions        # 列出当前工作目录的 session
```

如果要把 session 存到自定义目录，可以设置：

```env
AGENT_SESSION_DIR=/home/qiao/.agent_project
```

也可以单次运行示例：

```bash
python examples/run_once.py "帮我计算 12 * 7，再告诉我当前配置的 provider"
```

## Provider 切换

在 `.env` 中设置：

```env
MODEL_PROVIDER=openai
```

可选值：

- `openai`
- `google`
- `anthropic`
- `openai-compatible`
- `local-vllm` / `vllm` / `local`
- `dashscope` / `bailian` / `aliyun`

对应模型名可通过 `OPENAI_MODEL`、`GOOGLE_MODEL`、`ANTHROPIC_MODEL` 配置。

接入 OpenAI-compatible 服务时：

```env
MODEL_PROVIDER=openai-compatible
OPENAI_COMPATIBLE_API_KEY=your-key
OPENAI_COMPATIBLE_BASE_URL=https://api.deepseek.com/v1
OPENAI_COMPATIBLE_MODEL=deepseek-chat
```

接入本地 vLLM 服务时，先用 `/home/qiao/work/llm_deploy` 启动模型，例如：

```bash
python3 /home/qiao/work/llm_deploy/llmctl.py start qwen-3.6-35b-a3b --sudo
python3 /home/qiao/work/llm_deploy/llmctl.py health --base-url http://127.0.0.1:8000
```

然后在 `.env` 中设置：

```env
MODEL_PROVIDER=local-vllm
LOCAL_VLLM_BASE_URL=http://127.0.0.1:8000/v1
LOCAL_VLLM_MODEL=qwen-3.6-35b-a3b
LOCAL_VLLM_API_KEY=local-vllm
```

`LOCAL_VLLM_MODEL` 要和 `work/llm_deploy/model_registry.json` 里的 `served_model_name` 一致。`local-vllm` provider 会自动把 `127.0.0.1` / `localhost` 加入 `NO_PROXY`，避免本地请求被系统代理转发。

## Shell、联网搜索、浏览器和任务队列

新增能力都通过 `.env` 开关控制：

```env
AGENT_ENABLE_SHELL_COMMANDS=true
AGENT_ENABLE_WEB_SEARCH=true
AGENT_ENABLE_BROWSER_TOOLS=true
AGENT_RECURSION_LIMIT=30
```

可用工具包括：

- `execute_shell_command`：经过终端确认后执行 shell 命令，并返回 exit code、stdout、stderr。
- `web_search`：联网搜索并返回搜索结果标题和 URL。
- `open_web_page`：打开网页并返回标题和可读正文。
- `list_web_page_links`：列出网页中的链接。
- `remember_memory` / `search_memories` / `get_memory` / `delete_memory`：保存、检索、读取和删除 SQLite 长期记忆。
- `create_task_queue` / `update_task_step` / `get_task_queue` / `list_task_queues`：创建和维护 SQLite 持久化任务队列。

浏览器工具是轻量文本浏览器，不能运行 JavaScript、登录、点击动态按钮或截图；需要真实浏览器自动化时可再接 Playwright。任务队列保存在 SQLite 中，重启后仍可继续读取和更新。`AGENT_RECURSION_LIMIT` 用来限制一次 agent 调用中的最大推理/工具循环步数，避免异常重试长期占用 CPU。

## Skills

agent 支持 Codex-style skills。每个 skill 是一个目录，至少包含 `SKILL.md`：

```text
skill-name/
  SKILL.md
  references/   # 可选
  scripts/      # 可选
  assets/       # 可选
```

`SKILL.md` 需要 YAML frontmatter：

```markdown
---
name: work-record-manager
description: Use when the user asks to manage Qiao's work record.
---

# Work Record Manager
...
```

启动 agent 时，系统提示只注入 skill 的 `name` 和 `description`。当用户请求匹配某个 skill 时，agent 应先调用 `read_skill` 读取正文，再按需调用 `read_skill_file` 读取 `references/` 或 `scripts/` 中的资源。

相关配置：

```env
AGENT_ENABLE_SKILLS=true
AGENT_SKILLS_DIRS=/home/qiao/work/agent_project/skills:/home/qiao/.codex/skills
AGENT_SKILL_CATALOG_LIMIT=25
```

内置示例 `skills/work-record-manager/SKILL.md` 会指导 agent 维护 `/home/qiao/work/aaa-work.md`。

## 工作收件箱和任务路由

面向“半替代我工作”的主流程是：你手动粘贴飞书消息、会议纪要或临时想法，agent 先收进工作收件箱，再判断归属工作和处理路线。

可用工具：

- `list_work_record_items`：读取并解析 `/home/qiao/work/aaa-work.md`。
- `capture_work_message`：保存一条手动整理的工作/飞书消息，自动匹配已有工作，生成 `self` / `codex` / `ask_user` / `defer` 路由。
- `create_work_task`：把 inbox 消息或用户请求转成结构化工作任务。
- `list_work_inbox` / `list_work_tasks`：查看待处理消息和任务。
- `update_work_task`：更新任务状态。
- `add_work_record_item`：向工作记录插入新工作。

推荐输入方式：

```text
这是我整理的飞书消息：
1. 医疗翻译服务 nginx timeout，可能需要检查4卡worker和2卡worker部署。
2. NLU 小模型测评报告需要今天发一下。
请收进工作收件箱，判断分别属于哪项工作，并给出处理路线。
```

路由含义：

- `self`：agent 自己整理、读少量文件、更新 Markdown。
- `codex`：代码修改、部署排查、脚本执行、测试验证，适合改写任务后交给 Codex。
- `ask_user`：信息不足，先问你一个短问题。
- `defer`：低优先级备忘，先存起来。

## Feishu 监视和 Codex 代理

飞书监视采用事件回调模式。启动 watcher：

```bash
cd /home/qiao/work/agent_project
.venv/bin/agent-feishu-watch --host 0.0.0.0 --port 8787
```

然后在飞书开放平台把事件回调 URL 指向这个服务，例如 `https://你的域名/` 或内网穿透后的 URL。飞书 URL verification 请求会返回 `challenge`。可选环境变量：

```env
FEISHU_WATCH_HOST=0.0.0.0
FEISHU_WATCH_PORT=8787
FEISHU_VERIFICATION_TOKEN=
FEISHU_REPORT_INTERVAL_SECONDS=300
FEISHU_REPORT_BATCH_SIZE=10
```

收到飞书消息后，watcher 会写入 SQLite，并按间隔或批量阈值生成报告。agent 内可用工具：

- `list_feishu_messages`
- `generate_feishu_report`
- `list_feishu_reports`
- `get_feishu_report`

Codex 代理默认关闭。确认要允许 agent 唤起 Codex 后，在 `.env` 中设置：

```env
AGENT_ENABLE_CODEX_DELEGATION=true
AGENT_CODEX_COMMAND=codex-proxy-anyrouter
```

`run_codex_task` 使用 `codex exec` 非交互执行任务，不会进入终端 TUI；如果你平时在终端使用 `codex-proxy-*`，这里应把 `AGENT_CODEX_COMMAND` 设为同一个 wrapper，或者调用工具时传入 `codex_command`。

可用工具：

- `rewrite_task_for_codex`：把你的任务改写成更适合 Codex 执行的说明。
- `run_codex_task`：调用指定的 Codex 命令执行一次性 `exec` 任务，并把 stdout/stderr/status 写入 SQLite。
- `start_codex_session`：启动一次 Codex `exec` 会话，返回 `session_id` 和输出，供 agent 判断下一步。
- `continue_codex_session`：用指定 `session_id` 继续同一个 Codex 会话，agent 每轮读取返回后再决定是否继续或结束。
- `list_codex_tasks` / `get_codex_task`：查看历史 Codex 任务。
- `test_codex_connectivity`：扫描 `PATH` 中的 `codex` 和 `codex-proxy-*`，分别发送 `你好`，30 秒内成功返回且 exit code 为 0 的配置会进入可用名单。

`run_codex_task` 默认使用 `AGENT_CODEX_COMMAND` 指定的命令执行 `exec -s workspace-write --color never`，工作目录默认是 `AGENT_WORKSPACE_ROOT`。当前 Codex CLI 不再支持旧的 `exec -a/--approval-policy` 参数，工具会保留 `approval_policy` 入参但不传给 `exec`。
`test_codex_connectivity` 默认使用 `codex exec -s read-only --color never`，不会写入任务数据库。

## SQLite 长期记忆和任务队列

agent 使用 SQLite 保存长期记忆和任务队列。默认路径由 `.env` 控制：

```env
AGENT_STATE_DB_PATH=/home/qiao/work/agent_project/.agent_state/agent.sqlite3
AGENT_MEMORY_CONTEXT_LIMIT=5
```

如果没有设置 `AGENT_STATE_DB_PATH`，默认会使用 `AGENT_WORKSPACE_ROOT/.agent_state/agent.sqlite3`；如果也没有设置 workspace，则使用当前工作目录下的 `.agent_state/agent.sqlite3`。

长期记忆工具：

- `remember_memory`：保存一条长期记忆，可带 namespace、source、tags。
- `search_memories`：按关键词检索记忆；空关键词返回最近记忆。
- `get_memory`：按 ID 读取一条记忆。
- `delete_memory`：按 ID 删除一条记忆。

任务队列工具保持原来的名字和参数，但现在写入 SQLite，重启 `agent-chat` 后仍可通过 `list_task_queues` 和 `get_task_queue` 找回。

构建 agent 时会自动把最近 `AGENT_MEMORY_CONTEXT_LIMIT` 条 default namespace 记忆追加到系统提示词里；设为 `0` 可以关闭自动注入，仍然保留手动 memory tools。

## 执行过程可见化

工具执行时会在终端输出简短进度行，例如：

```text
[agent] searching web: weyl algebra automorphism
[agent] search result: Example Domain -> https://example.com/
[agent] reading file: /home/qiao/work/agent_project/README.md
[agent] file write complete: /path/to/file.py (1200 chars, +8/-2 lines)
```

这个功能默认开启，可以通过 `.env` 关闭：

```env
AGENT_SHOW_TOOL_PROGRESS=false
```

进度行只说明工具正在做什么和结果规模，不会替代工具返回值，也不会改变 agent 的推理流程。

## 添加 Tool

在 `src/agent_project/tools/basic.py` 中添加：

```python
from langchain_core.tools import tool

@tool
def your_tool(input_text: str) -> str:
    """说明这个 tool 的用途。"""
    return input_text
```

然后把它加入 `get_tools()` 返回列表即可。


## 本地文件访问

当前 agent 内置三个文件工具：

- `list_local_directory`：列出本地目录。
- `read_local_file`：读取 UTF-8 文本文件。
- `write_local_file`：写入或追加 UTF-8 文本文件。

默认通过 `.env` 控制文件工具边界：

```env
AGENT_WORKSPACE_ROOT=/home/qiao/work/agent_project
AGENT_ENABLE_HUMAN_APPROVAL=true
```

权限策略：

- 读取 `AGENT_WORKSPACE_ROOT` 内的文件或目录：直接允许。
- 读取 `AGENT_WORKSPACE_ROOT` 外的路径：终端请求人类确认。
- 写入任何路径：终端请求人类确认。

当需要确认时，终端会显示类似：

```text
[approval required]
Action: write local file
Path: /path/to/file.txt
Reason: write operations require explicit human approval
Allow this operation? Type yes to approve:
```

只有输入完整的 `yes` 才会执行；其他输入都会拒绝。




## 阿里云百炼配置

百炼按量计费的 OpenAI 兼容接口需要配置正确的地域 Base URL 和模型名。中国内地北京地域通常使用：

```env
MODEL_PROVIDER=openai-compatible
OPENAI_COMPATIBLE_API_KEY=你的百炼APIKey
OPENAI_COMPATIBLE_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
OPENAI_COMPATIBLE_MODEL=qwen3.6-plus
```

也可以使用 provider 别名：

```env
MODEL_PROVIDER=dashscope
DASHSCOPE_API_KEY=你的百炼APIKey
OPENAI_COMPATIBLE_MODEL=qwen-plus
```

注意模型名是 `qwen3.6-plus`，不是 `qwen-3.6-plus`。不同地域的 API Key 和 Base URL 不通用。

## 安全确认机制

`AGENT_ENABLE_HUMAN_APPROVAL=true` 时，涉及风险的操作会在终端请求确认：

```text
[approval required]
Action: write local file
Path: /path/to/file
Reason: write operations require explicit terminal approval
Allow this operation? Type yes to approve:
```

这里没有图形弹窗；需要在运行 `agent-chat` 的终端里输入 `yes`。其他输入会拒绝。

`AGENT_ENABLE_HUMAN_APPROVAL=false` 时，`AGENT_WORKSPACE_ROOT` 内的写文件和 shell 命令会自动执行；工作区外操作会直接拒绝，不会反复询问。建议个人自用时把 `AGENT_WORKSPACE_ROOT` 设置为 `/home/qiao/work`。
