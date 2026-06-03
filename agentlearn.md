# Agent Project Learning Notes

这份文档记录当前项目的架构、模块职责、能力来源和核心原理，方便学习和后续扩展。

## 1. 项目定位

这个项目是一个基础的 LangChain + LangGraph agent。它把大语言模型、工具调用、本地文件访问、shell 命令执行、联网搜索、轻量浏览器操作、多步骤任务队列和命令行交互组合在一起。

当前 agent 的核心特点：

- 支持多种模型 provider。
- 使用 LangGraph `create_react_agent` 构建 ReAct agent。
- 支持工具调用。
- 支持本地文件读取、目录列表、写入。
- 支持 shell 命令执行。
- 支持自动联网搜索。
- 支持轻量浏览器操作：打开网页、抽取正文、列出链接。
- 支持进程内多步骤任务队列。
- 本地写入、越权读取和 shell 命令执行都会请求终端人工确认。
- 支持交互式 CLI 和单次命令调用。

## 2. 目录结构

```text
agent_project/
  README.md
  agentlearn.md
  pyproject.toml
  .env.example
  examples/
    run_once.py
  src/agent_project/
    __init__.py
    agent.py
    cli.py
    config.py
    llms.py
    tools/
      __init__.py
      basic.py
      filesystem.py
      shell.py
      web.py
      browser.py
      tasks.py
```

## 3. 主调用链

交互模式：

```text
agent-chat
  -> cli.py main()
  -> run_chat()
  -> load_settings()
  -> build_agent()
  -> build_chat_model()
  -> get_tools()
  -> create_react_agent()
  -> agent.stream(...)
```

单次模式：

```text
agent-chat "问题"
  -> cli.py main()
  -> invoke_agent("问题")
  -> build_agent()
  -> agent.invoke(...)
  -> message_content_to_text(...)
```

## 4. 核心模块

### 4.1 `config.py`

`config.py` 负责读取环境变量并生成 `Settings` 配置对象。它不创建模型，也不执行 agent，只负责把 `.env` 中的配置整理成结构化数据。

主要配置：

- `MODEL_PROVIDER`：选择模型供应商。
- `MODEL_TEMPERATURE`：控制模型输出随机性。
- `OPENAI_MODEL`：OpenAI 模型名。
- `GOOGLE_MODEL`：Gemini 模型名。
- `ANTHROPIC_MODEL`：Anthropic 模型名。
- `OPENAI_COMPATIBLE_API_KEY`：OpenAI-compatible API key。
- `OPENAI_COMPATIBLE_BASE_URL`：OpenAI-compatible API 地址。
- `OPENAI_COMPATIBLE_MODEL`：OpenAI-compatible 模型名。
- `LOCAL_VLLM_BASE_URL`：本地 vLLM OpenAI-compatible API 地址，默认 `http://127.0.0.1:8000/v1`。
- `LOCAL_VLLM_MODEL`：本地 vLLM 暴露的模型名，默认 `qwen-3.6-35b-a3b`。
- `LOCAL_VLLM_API_KEY`：本地 vLLM 占位 API key，默认 `local-vllm`。
- `AGENT_STATE_DB_PATH`：SQLite 长期记忆和持久任务队列数据库路径。
- `AGENT_MEMORY_CONTEXT_LIMIT`：构建 agent 时自动注入最近多少条 default namespace 记忆，设为 `0` 关闭。
- `AGENT_WORKSPACE_ROOT`：本地文件和 shell 工具的默认工作边界。
- `AGENT_ENABLE_HUMAN_APPROVAL`：是否启用人工审批。
- `AGENT_ENABLE_SHELL_COMMANDS`：是否启用 shell 命令工具。
- `AGENT_ENABLE_WEB_SEARCH`：是否启用联网搜索工具。
- `AGENT_ENABLE_BROWSER_TOOLS`：是否启用轻量浏览器工具。
- `AGENT_SHOW_TOOL_PROGRESS`：是否在终端显示工具执行过程。

### 4.2 `llms.py`

`llms.py` 负责根据配置创建聊天模型。核心函数是：

```python
build_chat_model(settings)
```

它根据 `settings.model_provider` 选择不同的 LangChain chat model：

- `openai` -> `ChatOpenAI`
- `google` -> `ChatGoogleGenerativeAI`
- `anthropic` -> `ChatAnthropic`
- `openai-compatible` -> 带 `base_url` 的 `ChatOpenAI`
- `local-vllm` / `vllm` / `local` -> 默认连接本机 vLLM `http://127.0.0.1:8000/v1`
- `dashscope` / `bailian` / `aliyun` -> 默认使用阿里云 DashScope OpenAI 兼容地址

这个模块的原理是“模型工厂”：上层代码不关心具体 provider，只调用 `build_chat_model()`，由它返回一个统一接口的 chat model。

本地 vLLM provider 也是通过 `ChatOpenAI` 接入，因为 vLLM 暴露的是 OpenAI-compatible API。`llms.py` 会在使用 `local-vllm` 时把 `127.0.0.1` 或 `localhost` 加入 `NO_PROXY/no_proxy`，避免本地请求被系统代理转发。

`LOCAL_VLLM_MODEL` 应该和 `/home/qiao/work/llm_deploy/model_registry.json` 中对应模型的 `served_model_name` 一致。例如当前 `qwen-3.6-35b-a3b`。

### 4.3 `agent.py`

`agent.py` 是 agent 的组装中心。

它做四件事：

1. 定义系统提示词 `DEFAULT_SYSTEM_PROMPT`。
2. 调用 `build_chat_model()` 创建模型。
3. 调用 `get_tools()` 获取工具列表。
4. 调用 `create_react_agent()` 创建 LangGraph ReAct agent。

系统提示词会告诉 agent：

- 文件相关问题可以用本地文件工具。
- 命令执行和项目验证可以用 shell 工具。
- 当前外部信息或网页检查可以用 web search 和 browser 工具。
- 长期偏好、项目事实和可复用上下文可以用 memory 工具保存和检索。
- 多步骤工作可以用持久化 task queue 工具跟踪。
- 工具要求人工确认时必须等待并尊重用户决定。

`invoke_agent()` 用于单次调用，适合 CLI 单次模式、脚本或未来 API 服务复用。

`message_content_to_text()` 用来把不同 provider 返回的 message content 转成纯文本。因为有些 provider 返回字符串，有些返回 content blocks，所以需要统一格式。

### 4.4 `cli.py`

`cli.py` 是命令行入口。

它支持两种模式：

- 交互模式：直接运行 `agent-chat`。
- 单次模式：运行 `agent-chat "你的问题"`。

交互模式会保存 `messages` 历史，因此 agent 能看到当前会话前文。它使用：

```python
agent.stream({"messages": messages}, stream_mode=["messages", "values"])
```

其中：

- `messages` 流用于实时打印模型输出。
- `values` 流用于拿到最终 LangGraph 状态并更新对话历史。

### 4.5 `tools/basic.py`

`basic.py` 定义和注册基础工具。当前工具列表由 `get_tools()` 返回：

- `calculator`
- `current_time`
- `configured_provider`
- `list_local_directory`
- `read_local_file`
- `write_local_file`
- `execute_shell_command`
- `web_search`
- `open_web_page`
- `list_web_page_links`
- `remember_memory`
- `search_memories`
- `get_memory`
- `delete_memory`
- `create_task_queue`
- `update_task_step`
- `get_task_queue`
- `list_task_queues`

LangChain 的 `@tool` 装饰器会把普通 Python 函数包装成模型可调用的 tool。`get_tools()` 是 agent 能力的集中注册表。

### 4.6 `tools/filesystem.py`

`filesystem.py` 提供本地文件能力。

当前三个工具：

- `list_local_directory(path=".")`
- `read_local_file(path, max_chars=20000)`
- `write_local_file(path, content, mode="overwrite")`

权限策略：

- 读取 `AGENT_WORKSPACE_ROOT` 内路径：直接允许。
- 读取 `AGENT_WORKSPACE_ROOT` 外路径：需要终端人工确认。
- 写入任何路径：需要终端人工确认。

这个模块的关键原理是：agent 不直接拥有无限制文件权限，而是通过受控 tool 访问本机文件系统。

### 4.7 `tools/shell.py`

`shell.py` 提供 shell 命令执行能力。

当前工具：

```python
execute_shell_command(command, cwd=".", timeout_seconds=30, max_chars=20000)
```

执行策略：

- 如果 `AGENT_ENABLE_SHELL_COMMANDS=false`，直接拒绝执行。
- 如果工作目录不存在或不是目录，直接返回错误。
- 每次 shell 命令执行都需要终端人工确认。
- 命令通过 `subprocess.run(..., shell=True, capture_output=True)` 执行。
- 返回内容包含 exit code、stdout、stderr。
- 超时会返回超时提示和已捕获输出。
- 输出过长会被截断。

这个模块的关键原理是：shell 命令能力很强，因此必须放在明确的审批闸门后面，并增加超时和输出限制。

### 4.8 `tools/web.py`

`web.py` 提供自动联网搜索能力。

当前工具：

```python
web_search(query, max_results=5, timeout_seconds=10)
```

实现原理：

1. 用标准库 `urllib.request` 请求 DuckDuckGo HTML 搜索页。
2. 用 `HTMLParser` 解析搜索结果链接。
3. 清理 DuckDuckGo 跳转链接中的 `uddg` 参数。
4. 返回标题和 URL。

控制开关：

```env
AGENT_ENABLE_WEB_SEARCH=true
```

当前实现不依赖第三方搜索 API，也不需要额外 key。限制是搜索结果质量依赖 DuckDuckGo HTML 页面结构，且只能返回标题和 URL，不做深度网页阅读。深度阅读由 browser 工具完成。

### 4.9 `tools/browser.py`

`browser.py` 提供轻量浏览器操作能力。

当前工具：

```python
open_web_page(url, max_chars=20000, timeout_seconds=15)
list_web_page_links(url, max_links=30, timeout_seconds=15)
```

实现原理：

1. 用 `urllib.request` 获取网页 HTML。
2. 用 `HTMLParser` 跳过 `script/style/noscript`。
3. 抽取页面标题、正文文本和链接。
4. 返回可读文本或链接列表。

控制开关：

```env
AGENT_ENABLE_BROWSER_TOOLS=true
```

当前是“文本浏览器”，不是完整浏览器自动化。它不能运行 JavaScript，不能点击动态按钮，不能登录，也不能截图。后续如果需要真实浏览器控制，可以接入 Playwright。

### 4.10 `tools/storage.py`

`storage.py` 是 SQLite 存储基础模块。它负责：

- 解析 `AGENT_STATE_DB_PATH`。
- 默认使用 `AGENT_WORKSPACE_ROOT/.agent_state/agent.sqlite3`。
- 初始化 `memories`、`task_queues`、`task_steps` 三张表。

### 4.11 `tools/memory.py`

`memory.py` 提供长期记忆能力。当前工具：

```python
remember_memory(content, namespace="default", source="", tags="")
search_memories(query="", namespace="default", limit=10)
get_memory(memory_id)
delete_memory(memory_id)
```

长期记忆适合保存用户偏好、项目事实、反复会用到的上下文。它不是向量数据库，目前用 SQLite `LIKE` 做关键词检索。

### 4.12 `tools/tasks.py`

`tasks.py` 提供 SQLite 持久化多步骤任务队列能力。

当前工具：

```python
create_task_queue(title, steps)
update_task_step(queue_id, step_index, status, note="")
get_task_queue(queue_id)
list_task_queues()
```

任务状态支持：

- `pending`
- `in_progress`
- `completed`
- `blocked`

实现原理：

- 用 `TaskQueue` 保存一个任务队列。
- 用 `TaskStep` 保存每一步的描述、状态和备注。
- 用模块级字典 `_TASK_QUEUES` 存储当前进程内的队列。
- 创建队列时生成短 ID。
- 更新任务时通过 queue ID 和 step index 定位具体步骤。

当前任务队列写入 SQLite，重启程序后仍然保留。

## 5. 当前能力和对应代码

| 能力 | 对应代码 | 原理 |
| --- | --- | --- |
| 多 provider 模型接入 | `config.py`, `llms.py` | 从 `.env` 读取配置，再创建对应 LangChain chat model |
| ReAct agent | `agent.py` | 使用 LangGraph `create_react_agent` 组合模型和工具 |
| 单次调用 | `agent.py`, `cli.py` | `invoke_agent()` 调用 `agent.invoke()` |
| 交互式聊天 | `cli.py` | 循环读取用户输入，并维护 `messages` 历史 |
| 流式输出 | `cli.py` | 使用 `agent.stream()` 读取 message chunk |
| 数学计算 | `tools/basic.py` | 用 Python AST 安全解析简单数值表达式 |
| 当前时间 | `tools/basic.py` | 调用 `datetime.now().astimezone()` |
| provider 查询 | `tools/basic.py` | 读取 `MODEL_PROVIDER` 环境变量 |
| 本地目录列表 | `tools/filesystem.py` | 通过 `Path.iterdir()` 列目录 |
| 本地文件读取 | `tools/filesystem.py` | 读取 UTF-8 文本并截断过长输出 |
| 本地文件写入 | `tools/filesystem.py` | 人工确认后覆盖或追加写入 |
| shell 命令执行 | `tools/shell.py` | 人工确认后调用 `subprocess.run()` |
| 自动联网搜索 | `tools/web.py` | 请求搜索页并解析标题和 URL |
| 轻量浏览器操作 | `tools/browser.py` | 请求网页 HTML，抽取正文和链接 |
| 长期记忆 | `tools/memory.py`, `tools/storage.py` | 用 SQLite 保存和检索持久记忆 |
| 多步骤任务队列 | `tools/tasks.py`, `tools/storage.py` | 用 SQLite 保存任务队列和步骤状态 |
| 工具执行过程可见化 | `tools/progress.py`, 各 tool 模块, `cli.py` | 工具执行时向 stderr 打印简短进度，CLI 等回答文本出现后再打印 `Agent:` |

## 6. 工具执行过程可见化

当前项目通过 `tools/progress.py` 暴露 agent 的执行过程。核心函数是：

```python
emit_progress(message)
```

文件、shell、web、browser、task 工具会在开始、完成、失败或拒绝时调用它。终端会看到类似：

```text
[agent] searching web: weyl algebra automorphism
[agent] search result: Example Domain -> https://example.com/
[agent] reading file: /home/qiao/work/agent_project/README.md
[agent] file write complete: /path/to/file.py (1200 chars, +8/-2 lines)
```

这个机制的关键点是：进度来自真实工具执行，不依赖模型自己描述流程。它默认开启，可以通过 `.env` 设置 `AGENT_SHOW_TOOL_PROGRESS=false` 关闭。

CLI 的 `_stream_agent_response()` 会等收到模型回答文本时才打印 `Agent:` 前缀，避免工具进度行被挤到最终回答前缀后面。

## 7. 人工审批机制

当前项目的人工审批通过终端输入实现。

当工具需要审批时，会打印：

```text
[approval required]
Action: ...
Path/CWD: ...
Reason: ...
Allow this operation? Type yes to approve:
```

只有输入完整的 `yes` 才允许执行。

目前需要审批的操作：

- workspace 外读取文件或目录。
- 写入任何文件。
- 执行任何 shell 命令。

联网搜索和浏览器读取默认不走人工审批，但可以通过环境变量关闭。

## 8. ReAct Agent 原理

ReAct 是 Reasoning + Acting 的缩写。

在这个项目中，LangGraph ReAct agent 大致流程是：

1. 用户输入问题。
2. 模型判断是否需要工具。
3. 如果需要工具，模型生成 tool call。
4. LangGraph 调用对应 Python tool。
5. 工具返回结果。
6. 模型根据工具结果继续推理。
7. 模型输出最终答案。

所以 agent 的能力不是只来自模型本身，还来自 `get_tools()` 注册进去的工具。

## 9. 如何新增一个能力

新增能力通常按这个流程：

1. 在 `src/agent_project/tools/` 中新增或修改工具模块。
2. 使用 `@tool` 装饰器包装函数。
3. 在 `tools/basic.py` 的 `get_tools()` 中注册新工具。
4. 如果需要配置，在 `config.py` 和 `.env.example` 中增加环境变量。
5. 如果是高风险能力，加入人工审批、边界检查、超时或输出限制。
6. 更新 README 或 `agentlearn.md`。

例如新增数据库查询能力，可以创建：

```text
src/agent_project/tools/database.py
```

然后在 `basic.py` 中注册：

```python
from agent_project.tools.database import query_database

def get_tools():
    return [
        ...
        query_database,
    ]
```

## 10. 当前限制

当前项目仍然不具备这些能力：

- 真实浏览器自动化：不能运行 JavaScript、不能登录、不能截图、不能点击动态按钮。
- 对话历史持久化：当前长期记忆和任务队列已持久化，但完整对话 transcript 还没有自动写入数据库。
- 结构化 planner：任务队列只是状态跟踪工具，不会强制 agent 按计划执行。
- 多 agent 协作。
- Web API 服务。
- Shell 命令白名单或细粒度权限策略。
- 搜索结果引用规范化：当前搜索工具只返回标题和 URL，没有自动生成引用格式。

后续可以继续扩展：

- 用 Playwright 实现真实浏览器操作。
- 给长期记忆增加向量检索或全文检索。
- 增加 planner/executor 分层。
- 增加 web API 服务。
- 给 shell 工具增加命令白名单、黑名单和审计日志。

## 11. 学习建议

建议按这个顺序阅读源码：

1. `README.md`
2. `.env.example`
3. `src/agent_project/config.py`
4. `src/agent_project/llms.py`
5. `src/agent_project/tools/basic.py`
6. `src/agent_project/tools/filesystem.py`
7. `src/agent_project/tools/shell.py`
8. `src/agent_project/tools/web.py`
9. `src/agent_project/tools/browser.py`
10. `src/agent_project/tools/tasks.py`
11. `src/agent_project/agent.py`
12. `src/agent_project/cli.py`

理解重点是：

- 配置如何进入程序。
- 模型如何被创建。
- 工具如何被注册。
- LangGraph 如何把模型和工具组合成 agent。
- 高风险能力如何通过人工审批控制。
- 网络类工具如何把外部信息转成模型可读文本。
- 任务队列如何辅助 agent 做多步骤工作。


## 12. Feishu 监视和 Codex 代理

飞书能力由 `agent_project.feishu_watch` 和 `tools/feishu.py` 提供。它使用飞书事件回调模式：常驻 `agent-feishu-watch` HTTP 服务，接收消息事件，写入 SQLite 的 `feishu_messages`，再生成 `feishu_reports`。

Codex 代理由 `tools/codex_delegate.py` 提供。核心流程是：先用 `rewrite_task_for_codex` 把用户任务改写成更清晰的 Codex 指令，再用 `run_codex_task` 调用本机 `codex exec`，并把任务状态和输出写入 `codex_tasks` 表。该能力默认由 `AGENT_ENABLE_CODEX_DELEGATION=false` 关闭，需要显式打开。
