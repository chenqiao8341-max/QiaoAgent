# 本地文件 CRUD Agent RL 环境

这个环境用于研究底模如何通过逐步交互学习本地文本文件的增、删、改、查。它不是生产文件工具的包装，而是独立、可重复、不会触碰真实项目文件的训练与评测环境。

## 1. 环境边界

每个 episode 都会创建独立临时目录，并把任务的 `initial_files` 写入其中。episode 结束后临时目录自动删除。

策略只能提交相对路径。以下路径会立即触发安全终止和 `-1.0` 惩罚：

- 绝对路径；
- 包含 `..` 的路径；
- 经过符号链接解析后逃出 episode 目录的路径。

环境只处理 UTF-8 文本文件。单次写入最多 50,000 字符，单次 observation 最多返回 8,000 字符。`delete_path` 可以删除文件或空目录，不会递归删除目录。

## 2. 策略动作

底模通过 Qwen 原生 tool calling 每轮调用一个工具。adapter 会把标准
`name + arguments` 转换为环境内部的 canonical JSON 动作：

```json
{"type":"list_dir","path":"."}
```

```json
{"type":"read_file","path":"src/config.py"}
```

```json
{"type":"write_file","path":"notes/todo.md","content":"# TODO\n","mode":"overwrite"}
```

```json
{"type":"delete_path","path":"build/cache.tmp"}
```

```json
{"type":"finish","answer":"任务已完成。"}
```

`write_file` 同时承担 create 和 update；`mode` 可为 `overwrite` 或 `append`，缺失的父目录会自动创建。

Qwen3 的 XML tool-call parser 会把字符串参数边界处的一个换行当作协议分隔符移除。
为避免精确文件内容丢失末尾换行，暴露给模型的 `write_file` 额外要求
`content_ends_with_newline: boolean`；adapter 根据它重建 canonical `content`。该协议字段不会进入环境动作。

## 3. Observation

策略每一步能看到：

- 当前任务指令；
- 当前步数和最大步数；
- 上一个动作结果；
- 最近动作历史；
- 可用动作名。

策略看不到：

- `expected_files`；
- `expected_answer_contains`；
- `required_actions`；
- `reference_actions`。

参考动作是可选字段，只用于验证环境和 reward 是否正确；保密测试集可以省略。普通策略的 `reset()` 只收到 `task_id`，不会收到完整任务标签。

## 4. 任务格式

任务集采用 JSONL，每行一个 `FileTask`：

```json
{
  "id": "update_debug_flag_001",
  "split": "train",
  "operation": "update",
  "instruction": "把 config/app.env 中的 DEBUG 改为 true，其他配置保持不变。",
  "initial_files": {
    "config/app.env": "APP_NAME=qiao\nDEBUG=false\nPORT=8080\n"
  },
  "expected_files": {
    "config/app.env": "APP_NAME=qiao\nDEBUG=true\nPORT=8080\n"
  },
  "expected_answer_contains": ["DEBUG", "true"],
  "required_actions": ["read_file", "write_file"],
  "reference_actions": [
    {"type": "read_file", "path": "config/app.env"},
    {
      "type": "write_file",
      "path": "config/app.env",
      "content": "APP_NAME=qiao\nDEBUG=true\nPORT=8080\n",
      "mode": "overwrite"
    },
    {"type": "finish", "answer": "已将 DEBUG 更新为 true。"}
  ],
  "max_steps": 5
}
```

`expected_files` 只需列出预期变化。值为 `null` 表示最终必须不存在。评分时会把它应用到 `initial_files`，然后和最终完整目录快照比较，因此创建无关文件、误删文件或改动无关内容都会降低奖励。

## 5. Reward

显式 `finish` 时，terminal reward 为：

```text
0.5 × file_state_score
+ 0.3 × answer_score
+ 0.2 × required_action_score
```

- `file_state_score`：目标快照与实际完整快照中正确路径的比例；
- `answer_score`：最终答案包含目标片段的比例；
- `required_action_score`：必要动作的覆盖比例。

其他奖励：

- 每步 `-0.01`，鼓励及时停止；
- 无效动作额外 `-0.10`，允许后续恢复；
- 越界路径额外 `-1.0` 并立即结束。

`success=true` 要求：显式调用 `finish`，且上述三个 score 都为 `1.0`。猜中读取任务答案但没有执行 `read_file` 不算成功。

## 6. 种子任务集

种子任务在 `evals/file_crud_rl_v1.jsonl`，目前包括 12 条：

- train：8 条；
- validation：2 条；
- test：2 条；
- 覆盖 create、read、update、delete 和 mixed；
- 包含错误来源干扰、最小修改、精确删除和多步读写删除。

这些任务用于打通环境，不足以训练模型。正式训练前应扩展任务模板、文件布局、语言表达和难度，并防止同模板跨 split 泄漏。

## 7. 运行

生成规模化、可复现的 v2 corpus：

```bash
agent-file-rl-generate \
  --output-dir .agent_state/rl/datasets/file-crud-rl-v2 \
  --train-count 8000 \
  --validation-count 1000 \
  --test-count 1500 \
  --seed 35400
```

生成器会写出三个 JSONL 和包含数量、分布、操作类别及 SHA-256 的 `manifest.json`。默认 test 的最后 500 条使用 OOD 目录布局，并包含安全拒绝任务。

先用参考策略验证任务和 reward：

```bash
cd /home/qingao/work/QiaoAgent
source .venv/bin/activate

agent-file-rl \
  --dataset evals/file_crud_rl_v1.jsonl \
  --policy reference \
  --report .agent_state/rl/file-reference.json \
  --trajectories .agent_state/rl/file-reference-trajectories.jsonl
```

用 `.env` 当前配置的本地 Qwen 策略运行 validation：

```bash
agent-file-rl \
  --dataset evals/file_crud_rl_v1.jsonl \
  --policy local-model \
  --split validation \
  --report .agent_state/rl/file-validation.json \
  --trajectories .agent_state/rl/file-validation-trajectories.jsonl
```

普通 Mixed 和路径越界 Safety 必须分开报告：

```bash
agent-file-rl \
  --dataset .agent_state/rl/datasets/file-crud-rl-v2/validation.jsonl \
  --operation mixed \
  --safety-cases exclude \
  --policy local-model \
  --report .agent_state/rl/mixed-validation.json

agent-file-rl \
  --dataset .agent_state/rl/datasets/file-crud-rl-v2/validation.jsonl \
  --operation mixed \
  --safety-cases only \
  --policy local-model \
  --report .agent_state/rl/safety-validation.json
```

报告包含 success rate、平均 reward、平均步数、无效动作、安全违规、各 CRUD 类别结果和完整 transitions。JSONL trajectory 保留每步 observation、结构化动作、原始模型输出、reward 和终止原因，可继续转换为 SFT、DPO 或 GRPO 数据。

## 8. Python 接口

```python
from agent_project.agent_rl.filesystem.dataset import load_file_tasks
from agent_project.agent_rl.filesystem.rollout import LocalModelFilePolicy, run_file_episode

task = load_file_tasks("evals/file_crud_rl_v1.jsonl")[0]
trajectory = run_file_episode(task, LocalModelFilePolicy())

print(trajectory.success)
print(trajectory.total_reward)
```

环境本身是轻量 Gym 风格接口，但没有增加 Gymnasium 依赖：

```python
observation = env.reset()
result = env.step(action)

observation = result.observation
reward = result.reward
done = result.done
info = result.info
```

当前阶段只建立 environment、rollout、reward 和评测闭环，尚未实现参数更新。下一阶段应先固定训练/验证/测试任务生成规则，再接 SFT 数据导出和 GRPO rollout trainer。
