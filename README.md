# Agent Project

一个基础 LangChain + LangGraph agent 项目，支持：

- OpenAI / Google Gemini / Anthropic provider 切换
- OpenAI-compatible API 接入，例如 DeepSeek、Kimi、DashScope/Qwen、local vLLM
- LangGraph `create_react_agent`
- Tool calling
- 本地文件读取、目录列表、写入
- 写入和越权读取时向人类申请终端确认
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
    tools/basic.py    # 示例 tools
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
- `dashscope` / `bailian` / `aliyun`

对应模型名可通过 `OPENAI_MODEL`、`GOOGLE_MODEL`、`ANTHROPIC_MODEL` 配置。

接入 OpenAI-compatible 服务时：

```env
MODEL_PROVIDER=openai-compatible
OPENAI_COMPATIBLE_API_KEY=your-key
OPENAI_COMPATIBLE_BASE_URL=https://api.deepseek.com/v1
OPENAI_COMPATIBLE_MODEL=deepseek-chat
```

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
