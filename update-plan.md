## 背景
- embedding模型见/home/qiao/models/embedding，建议选用models/embedding/Qwen__Qwen3-Embedding-0.6B
- agent主脑模型建议选用Qwen3.6-35b-a3b，模型文件在models/Qwen/Qwen3.6-35B-A3B，目前通过vllm部署，已接入agent，具体部署方式可以见work/llm_deploy（后续如果要额外部署embedding模型的话也许要调低这个较大模型部署时的gpu_memory_utilization）

## agent项目当前简介
`QiaoAgent`现在已经有不错的骨架：

- Agent 主体在 [agent.py](C:/Users/qiao/Desktop/work/QiaoAgent/src/agent_project/agent.py:1)，用 LangGraph `create_react_agent`
- 配置在 [config.py](C:/Users/qiao/Desktop/work/QiaoAgent/src/agent_project/config.py:1)，支持 OpenAI-compatible、本地 vLLM、Qwen
- 工具注册在 [basic.py](C:/Users/qiao/Desktop/work/QiaoAgent/src/agent_project/tools/basic.py:1)，已有文件、shell、web、browser、memory、tasks、skills、飞书、Codex delegation
- 状态库在 [storage.py](C:/Users/qiao/Desktop/work/QiaoAgent/src/agent_project/tools/storage.py:1)，已有 SQLite 表
- Codex 协作在 [codex_delegate.py](C:/Users/qiao/Desktop/work/QiaoAgent/src/agent_project/tools/codex_delegate.py:1)
- 困难任务审查包在 [codex_judge.py](C:/Users/qiao/Desktop/work/QiaoAgent/src/agent_project/tools/codex_judge.py:1)
- 工作消息路由在 [work_management.py](C:/Users/qiao/Desktop/work/QiaoAgent/src/agent_project/tools/work_management.py:1)

它目前的问题也很清楚：还是 ReAct 单体 Agent 为主；没有系统化评测；没有标准化轨迹日志；没有多 Agent 编排；没有 RAG；没有基于失败案例的自我改进训练闭环。

## 建议项目架构
最终搭建起一个“全自动化工作流agent”：

消息入口
  ↓
规则 + embedding 检索召回候选项目
  ↓
Qwen3.6 Router：判断归属哪个项目
  ↓
读取该项目状态（举例如下）：
  - README
  - current_status.md
  - todo.md
  - last_report.json
  - 最近 git diff
  - 最近任务记录
  ↓
Qwen3.6 Planner：生成计划
  ↓
风险&难度分级（绝大多数任务都应该是中低风险）：
  - 低风险低难度：自动执行
  - 低风险高难度：拉起codex --yolo会话执行
  - 中风险低难度：开新分支，然后自动执行
  - 中风险高难度：开新分支，然后拉起codex --yolo执行
  - 高风险低难度：请求确认计划
  - 高风险高难度：请求确认计划
  ↓
Qwen3.6 Executor：按照planner的计划分级，调用工具执行或者拉起codex执行
  ↓
确定性检查：
  - test
  - lint
  - typecheck
  - 编译
  - LaTeX build
  - schema 校验
  ↓
Qwen3.6 Verifier：根据 checklist 审查
  ↓
不合格：返工 ——> Qwen3.6 Reflector 反思
合格：Qwen3.6 Finalizer写入结构化日志 + 输出报告


## 具体实施路线
- 第一阶段：将aaa-work.md构建成可以做检索的向量数据库，方便embedding模型判定“当前消息应当属于aaa-work中哪个项目，或者应该属于新项目”
- 第二阶段：补齐轨迹日志与评测底座

新增模块（仅供参考）：

```text
src/agent_project/evals/
  schemas.py
  dataset.py
  runner.py
  metrics.py
  reports.py

src/agent_project/tracing/
  trace_store.py
  event.py
```

你要记录每次 Agent 执行的完整轨迹：

```json
{
  "task_id": "work_001",
  "user_input": "...",
  "model": "qwen-3.6-35b-a3b",
  "steps": [
    {"type": "llm", "content": "..."},
    {"type": "tool_call", "tool": "search_memories", "args": {}, "ok": true},
    {"type": "tool_result", "content": "..."}
  ],
  "final_answer": "...",
  "latency_ms": 12345,
  "success": true,
  "error_type": ""
}
```

评测集先做 80 条，不要贪多：

- 20 条消息分类任务
- 20 条工作任务拆解任务
- 15 条本地文件检索/修改建议任务
- 15 条 RAG 问答任务
- 10 条 Codex delegation 路由任务

指标：

- `route_accuracy`：self/codex/ask_user/defer 是否正确
- `tool_call_accuracy`：工具是否选对、参数是否合理
- `task_success_rate`：最终任务是否完成
- `groundedness`：回答是否有依据
- `latency_seconds`
- `tokens_or_prompt_chars`
- `human_intervention_count`

- 第三阶段：把 ReAct 单体升级为显式多节点图

不要继续只用 `create_react_agent` 包到底。改成 LangGraph `StateGraph`：

```text
User Input
  -> Router
  -> Planner
  -> Executor
  -> Verifier
  -> Reflector
  -> Finalizer
```

各节点职责：

- `Router`：判断任务类型：chat / work_message / code_task / research / file_task / rag_qa
- `Planner`：生成结构化计划，最多 5 步
- `Executor`：执行工具
- `Verifier`：检查是否满足目标、是否需要补工具、是否需要问用户
- `Reflector`：失败时写入反思记忆
- `Finalizer`：生成面向用户的答案

六个节点对应本地部署的Qwen3.5-35b-a3b模型的六种调用模式：
也就是同一个模型，六种system prompt

状态结构：

```python
class AgentState(TypedDict):
    messages: list
    task_type: str
    plan: list[dict]
    current_step: int
    tool_results: list[dict]
    reflections: list[str]
    final_answer: str
    status: Literal["running", "need_user", "done", "failed"]
```

这一步是项目从“用了 LangGraph”变成“懂 Agent 编排”的关键。

- 第四阶段：做私有知识库 RAG

知识库来源：

```text
docs/
飞书消息报告
aaa-work.md
项目 README
代码结构文档
你自己的学习笔记
Agent 论文摘要
常用工作 SOP
```

技术选择：

- embedding：`bge-m3` 或 `Qwen3-Embedding`
- reranker：`bge-reranker-v2-m3` 或 Qwen reranker
- 向量库：先用 Chroma/FAISS，后续可换 Milvus
- chunk：Markdown 按标题切，代码按函数/类切
- retrieval：hybrid，关键词 + 向量 + rerank

新增工具：

```text
tools/rag.py
  index_documents()
  search_knowledge()
  answer_with_citations()
```

RAG 回答必须带引用：

```text
根据 docs/code_structure.md 的项目结构说明，当前工具注册入口是 ...
来源：
1. docs/code_structure.md#tools/basic.py
2. src/agent_project/tools/basic.py
```

- 第五阶段：做反思记忆与自我改进

参考 Reflexion：失败后不训练权重，先写“语言反思”进记忆，下次检索使用。([arxiv.org](https://arxiv.org/abs/2303.11366))

失败类型设计：

```text
wrong_tool
bad_tool_args
missing_context
hallucinated_fact
unsafe_action
over_delegation
under_delegation
rag_miss
planning_loop
```

失败后写入：

```json
{
  "namespace": "reflections",
  "task_type": "work_message",
  "failure_type": "wrong_tool",
  "reflection": "遇到飞书消息分类时，应先调用 capture_work_message，而不是直接回答。",
  "created_from_trace_id": "..."
}
```

Verifier 节点每次执行前检索相关反思：

```text
search_memories(namespace="reflections", query=task_type + user_input)
```

做消融实验：

| 设置 | route accuracy | task success | avg steps |
| --- | --- | --- | --- |
| baseline ReAct | 例如 62% | 48% | 7.8 |
| + planner | 70% | 55% | 6.1 |
| + verifier | 76% | 63% | 6.5 |
| + reflection memory | 82% | 70% | 5.9 |

- 第六阶段：做本地模型能力增强

最后阶段从底模下手

实验组：

```text
Qwen3.6-35b-a3b baseline
Qwen3.6-35b-a3b + structured graph
Qwen3.6-35b-a3b + RAG
Qwen3.6-35b-a3b + reflection memory
Qwen3.6-35b-a3b + trajectory SFT/LoRA
强闭源模型 judge，只做评测，不做主模型
```

训练数据来自你自己的轨迹：

```json
{
  "instruction": "请处理这条飞书工作消息...",
  "input": "...",
  "expected_plan": [...],
  "expected_tool_calls": [...],
  "expected_answer": "..."
}
```

LoRA 训练目标不要一上来训“聊天能力”，要训这三类：

- 路由分类
- 工具选择
- 计划生成


- 第七阶段：做一个可展示的 Demo 场景

推荐场景：

> 消息进入系统后，Agent 自动识别所属项目，查工作记录和知识库，生成处理计划；简单任务自己完成，代码任务交给 Codex，本地文件任务请求权限，困难任务生成审查包；最后产出任务状态、执行轨迹、评测分数和反思记录。

展示脚本：

```text
1. 输入一批模拟消息
2. 系统自动分类和路由
3. 对一个 bug 类任务生成 Codex delegation
4. 对一个知识问答任务走 RAG 并给引用
5. 对一次失败任务写入 reflection
6. 重新跑同类任务，展示成功率提升
7. 打开 eval report，展示指标表格
```

