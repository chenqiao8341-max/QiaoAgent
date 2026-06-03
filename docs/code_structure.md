# 项目代码结构说明

这份文档按文件解释当前 agent 项目的职责。建议阅读顺序是：`README.md` -> `config.py` -> `llms.py` -> `tools/basic.py` -> `agent.py` -> `cli.py`。

## 根目录文件

### `README.md`

项目的使用说明。它面向使用者，回答怎么安装、怎么配置 `.env`、怎么运行 agent、怎么切换不同模型 provider。

### `.env.example`

环境变量模板。真正运行时你应该复制一份为 `.env`，然后填入自己的 API key 和模型名。

关键配置包括：

- `MODEL_PROVIDER`：选择模型供应商，例如 `openai`、`google`、`anthropic`、`openai-compatible`、`local-vllm`。
- `OPENAI_MODEL` / `GOOGLE_MODEL` / `ANTHROPIC_MODEL`：各 provider 的模型名。
- `OPENAI_COMPATIBLE_BASE_URL`：用于 DeepSeek、Kimi、DashScope/Qwen 这类兼容 OpenAI API 的云端服务。
- `LOCAL_VLLM_BASE_URL` / `LOCAL_VLLM_MODEL`：用于本机 vLLM OpenAI-compatible 服务。
- `AGENT_STATE_DB_PATH`：SQLite 长期记忆和持久任务队列数据库路径。
- `AGENT_MEMORY_CONTEXT_LIMIT`：自动注入到系统提示词里的最近记忆数量。
- `MODEL_TEMPERATURE`：控制模型输出随机性。

### `.env`

本地真实配置文件，通常不提交到 Git。程序通过 `python-dotenv` 自动读取它。

### `pyproject.toml`

Python 项目的包配置文件。它定义了：

- 项目名、版本、Python 版本要求。
- LangChain、LangGraph、各 provider adapter 等依赖。
- 命令行入口 `agent-chat = "agent_project.cli:main"`。

安装 `pip install -e .` 后，系统会生成 `agent-chat` 命令。

### `.gitignore`

告诉 Git 忽略哪些本地文件，例如 `.env`、`.venv/`、`__pycache__/`。

## `src/agent_project/` 包

这里是项目主体代码。因为项目使用标准的 `src` 布局，真正的 Python 包名是 `agent_project`。

### `src/agent_project/config.py`

负责读取配置。

核心对象是 `Settings`，它把环境变量整理成一个结构化配置对象。`load_settings()` 会调用 `load_dotenv()` 读取 `.env`，然后返回 `Settings`。

这个文件不直接创建模型，也不运行 agent；它只负责配置。

### `src/agent_project/llms.py`

负责根据配置创建聊天模型。

核心函数是 `build_chat_model(settings)`。它读取 `settings.model_provider`，然后选择对应 LangChain 模型类：

- `openai` -> `ChatOpenAI`
- `google` -> `ChatGoogleGenerativeAI`
- `anthropic` -> `ChatAnthropic`
- `openai-compatible` -> 带 `base_url` 的 `ChatOpenAI`
- `local-vllm` / `vllm` / `local` -> 默认指向 `http://127.0.0.1:8000/v1` 的 `ChatOpenAI`

如果以后要接新 provider，通常优先改这个文件。

本地 vLLM 服务由 `/home/qiao/work/llm_deploy` 启动，模型名应使用 `model_registry.json` 里的 `served_model_name`。`local-vllm` provider 会自动设置 `NO_PROXY/no_proxy`，避免 `127.0.0.1` 请求经过代理。

### `src/agent_project/tools/basic.py`

定义 agent 可以调用的工具。

当前工具包括：

- `calculator`：计算简单数值表达式。
- `current_time`：返回本地当前时间。
- `configured_provider`：返回当前配置的 provider。
- `list_local_directory`：列出本地目录。
- `read_local_file`：读取本地文本文件。
- `write_local_file`：经过人类确认后写入本地文本文件。
- `execute_shell_command`：经过人类确认后执行 shell 命令。
- `web_search`：联网搜索并返回标题和 URL。
- `open_web_page`：打开网页并返回可读正文。
- `list_web_page_links`：列出网页链接。
- `remember_memory` / `search_memories` / `get_memory` / `delete_memory`：维护 SQLite 长期记忆。
- `create_task_queue` / `update_task_step` / `get_task_queue` / `list_task_queues`：维护 SQLite 持久化多步骤任务队列。

LangChain 通过 `@tool` 装饰器把普通 Python 函数包装成可被模型 tool call 的工具。`get_tools()` 返回工具列表，`agent.py` 会读取它。文件工具是在 `filesystem.py` 中定义，再由 `basic.py` 统一注册。

如果你要新增工具，通常新建独立 tool 模块，然后在这里导入并加入 `get_tools()`。

### `src/agent_project/tools/filesystem.py`

本地文件访问工具模块。它让 agent 可以通过 tool call 读取目录、读取文本文件、写入文本文件。

当前暴露三个工具：

- `list_local_directory(path=".")`：列出本地目录内容。
- `read_local_file(path, max_chars=20000)`：读取 UTF-8 文本文件，超过长度会截断返回。
- `write_local_file(path, content, mode="overwrite")`：写入或追加 UTF-8 文本文件。

它的权限边界由 `.env` 控制：

```env
AGENT_WORKSPACE_ROOT=/home/qiao/work/agent_project
AGENT_ENABLE_HUMAN_APPROVAL=true
```

策略是：工作目录内读取直接允许；工作目录外读取需要人类确认；任何写入都需要人类确认。确认流程由 `_ask_human_approval()` 完成，它会在终端打印操作、路径、原因，并要求输入完整的 `yes`。

这个模块相当于给 agent 加了一层“本机文件能力 + 人类审批闸门”。它不是操作系统级权限系统，而是 agent 工具层面的许可流程。

### `src/agent_project/tools/shell.py`

shell 命令执行工具。当前暴露 `execute_shell_command(command, cwd=".", timeout_seconds=30, max_chars=20000)`。

它会检查 `AGENT_ENABLE_SHELL_COMMANDS`，确认工作目录存在，然后通过终端人工审批再调用 `subprocess.run()`。输出包含 exit code、stdout 和 stderr，并受超时和最大字符数限制。

### `src/agent_project/tools/web.py`

自动联网搜索工具。当前暴露 `web_search(query, max_results=5, timeout_seconds=10)`。

它通过标准库请求 DuckDuckGo HTML 搜索页，解析结果链接，清理 DuckDuckGo 跳转 URL 后返回标题和 URL。开关是 `AGENT_ENABLE_WEB_SEARCH`。

### `src/agent_project/tools/browser.py`

轻量文本浏览器工具。当前暴露 `open_web_page(url, max_chars=20000, timeout_seconds=15)` 和 `list_web_page_links(url, max_links=30, timeout_seconds=15)`。

它请求网页 HTML，跳过 `script`、`style`、`noscript`，抽取标题、正文和链接。开关是 `AGENT_ENABLE_BROWSER_TOOLS`。它不是完整浏览器自动化，不能运行 JavaScript、登录、点击动态按钮或截图。

### `src/agent_project/tools/storage.py`

SQLite 存储基础模块。它负责解析 `AGENT_STATE_DB_PATH`，创建 `.agent_state/agent.sqlite3`，并初始化 memories、task_queues、task_steps 三张表。

### `src/agent_project/tools/memory.py`

长期记忆工具模块。它提供 `remember_memory`、`search_memories`、`get_memory` 和 `delete_memory`，把用户偏好、项目事实和可复用上下文保存到 SQLite。

### `src/agent_project/tools/tasks.py`

SQLite 持久化多步骤任务队列工具。当前暴露 `create_task_queue`、`update_task_step`、`get_task_queue` 和 `list_task_queues`。

任务步骤支持 `pending`、`in_progress`、`completed`、`blocked` 四种状态。队列保存在 SQLite 里，重启程序后仍然可以读取。

### `src/agent_project/tools/progress.py`

工具进度输出模块。它提供 `emit_progress(message)`，默认把简短执行过程写到 stderr，例如正在读取哪个文件、搜索什么关键词、打开哪个 URL、写入多少字符和行数变化。

开关由 `.env` 中的 `AGENT_SHOW_TOOL_PROGRESS` 控制，默认开启。关闭后工具仍正常执行，只是不再打印进度行。

### `src/agent_project/tools/__init__.py`

工具模块的导出口。它把 `get_tools` 暴露给外部代码，方便 `agent.py` 用 `from agent_project.tools import get_tools` 导入。

### `src/agent_project/agent.py`

agent 的核心组装文件。

它做四件事：

1. 定义 `DEFAULT_SYSTEM_PROMPT`，即 agent 的系统提示词。
2. 用 `build_chat_model()` 创建模型。
3. 用 `get_tools()` 获取工具。
4. 用 LangGraph 的 `create_react_agent()` 创建 ReAct agent。

`invoke_agent(user_input)` 是一个单次调用入口，适合脚本或 API 层复用。

`message_content_to_text(content)` 用来把模型返回内容转成纯文本。有些模型会返回 content blocks，例如：

```python
[{"type": "text", "text": "你好", "extras": {...}}]
```

这个函数会只抽取其中的 `text` 字段，避免 CLI 打印出完整列表和 `extras`。

### `src/agent_project/cli.py`

命令行入口。

它支持两种模式：

- 交互模式：直接运行 `agent-chat`，进入循环对话。
- 单次模式：运行 `agent-chat 你的问题`，只调用一次然后退出。

交互模式里会维护 `messages` 历史，所以 agent 可以看到前面的对话。打印回答时，它会走 `_stream_agent_response()`，内部调用 `agent.stream(..., stream_mode=["messages", "values"])`：`messages` 流负责实时打印模型文本，`values` 流负责拿到最终图状态并更新历史。CLI 会等真正收到模型回答文本时才打印 `Agent:` 前缀，这样工具进度行可以先独立显示。

### `src/agent_project/__init__.py`

包初始化文件。当前只保存版本号。它的存在也表示 `agent_project` 是一个 Python 包。

## `examples/`

### `examples/run_once.py`

单次调用示例脚本。它手动把 `src` 加入 `sys.path`，这样即使没有安装包，也能从源码直接运行。

示例：

```bash
python examples/run_once.py "帮我计算 12 * 7"
```

## 当前调用链

运行 `agent-chat` 时，大致流程是：

```text
cli.py main()
  -> run_chat()
    -> load_settings()
    -> build_agent()
      -> build_chat_model()
      -> get_tools()
      -> create_react_agent()
    -> _stream_agent_response(agent, messages)
      -> agent.stream({"messages": messages}, stream_mode=["messages", "values"])
      -> messages 流: message_content_to_text(message_chunk.content)
      -> messages 流: print(text, end="", flush=True)
      -> values 流: latest_messages = chunk["messages"]
```

运行 `agent-chat "问题"` 时，大致流程是：

```text
cli.py main()
  -> invoke_agent("问题")
    -> build_agent()
    -> agent.invoke(...)
    -> message_content_to_text(...)
```

## 修改输出格式应该看哪里

你这次遇到的输出问题来自 `messages[-1].content`。不同 provider 的 content 格式不完全一致：有的直接返回字符串，有的返回列表形式的 content blocks。

现在统一处理位置在：

- `agent.py` 的 `message_content_to_text()`：定义如何抽取文本。
- `cli.py` 的 `_stream_agent_response()`：流式接收 LangGraph chunk，打印 AI message chunk，并保存最终消息历史。

以后如果你想保留 tool call 信息、token usage、provider metadata，可以另外写一个 formatter；如果只想给用户看自然语言回答，就继续走 `message_content_to_text()`。


## 流式输出在哪里改

当前交互模式的流式输出在 `src/agent_project/cli.py` 的 `_stream_agent_response()` 中。它调用：

```python
agent.stream({"messages": messages}, stream_mode=["messages", "values"])
```

这里同时开启两个流：`messages` 返回模型生成过程中的 message chunk，用来实时打印；`values` 返回 LangGraph 当前状态，用来在流结束后保存最新 `messages` 历史。代码只打印 `AIMessageChunk`，避免把用户输入、工具返回值或中间状态误打印成最终回答。

如果你想调整流式表现，优先改这里：

- 想改前缀：修改 `print("\nAgent: ", end="", flush=True)`。
- 想禁用流式：把 `run_chat()` 里的 `_stream_agent_response(agent, messages)` 换回 `agent.invoke(...)`。
- 想给 API 复用：可以把 `_stream_agent_response()` 的核心逻辑抽到 `agent.py`，让 CLI 和 Web/API 层共享。

注意：单次模式 `agent-chat "问题"` 现在仍然使用 `invoke_agent()` 一次性返回；只有交互模式 `agent-chat` 是流式输出。


## 本地文件读写和人类许可

文件能力由 `src/agent_project/tools/filesystem.py` 提供。agent 并不是直接拥有无限制的本机权限，而是通过 LangChain tool call 请求这些操作。

读取规则：

- 路径在 `AGENT_WORKSPACE_ROOT` 内：允许读取。
- 路径在 `AGENT_WORKSPACE_ROOT` 外：调用 `_ask_human_approval()`，需要终端输入 `yes`。

写入规则：

- 任何写入都调用 `_ask_human_approval()`。
- 只有输入完整的 `yes` 才执行写入。
- 写入模式支持 `overwrite` 和 `append`。

这个机制对应“提权申请人类许可”：当 agent 想越过默认读取范围，或者想修改文件时，它会在终端列出操作、路径、原因，然后等待人类批准。

如果未来要加数据库写入、真实浏览器自动化等高风险能力，建议也沿用同样结构：独立 tool 模块 + 明确边界 + `_ask_human_approval()` 或更强的审批器。
