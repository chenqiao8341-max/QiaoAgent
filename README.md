# Agent Project

一个基础 LangChain + LangGraph agent 项目，支持：

- OpenAI / Google Gemini / Anthropic provider 切换
- OpenAI-compatible API 接入，例如 DeepSeek、Kimi、DashScope/Qwen、local vLLM
- LangGraph `create_react_agent`
- Tool calling
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
