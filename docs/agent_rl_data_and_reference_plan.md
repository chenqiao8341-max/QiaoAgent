# Qwen3.5-4B 文件 Agent：数据规模与 Reference Policy 方案

调研日期：2026-08-24。

## 结论

对于当前边界明确、程序化 reward 完整的本地文件 CRUD Agent，不需要先构造十万级唯一任务。建议第一版准备：

- 8,000 条训练任务；
- 1,000 条 validation；
- 1,000 条常规 test；
- 500 条 OOD/安全 test；
- DPO 从训练任务 rollout 中筛选 10,000–20,000 个偏好对；
- GRPO 先使用 4,000 个高多样性训练 prompt、每题 4 个 rollout，验证有效后扩到全部 8,000 个 prompt，并尝试 group size 4–8。

数据量不由“4B 参数量”单独决定，而主要由任务分布宽度、reward 可验证性、任务重复度和 rollout 数量决定。对单一、结构化工具域，数千个高多样性 prompt 可以形成有效训练；对数学或整个软件工程分布，公开工作使用到十万级任务。

## 调研依据

### ToolRL：最接近当前任务

[ToolRL](https://arxiv.org/abs/2504.13958) 在工具调用任务上使用 Qwen2.5-3B 等模型。论文报告：

- 4,000 个 RL 训练数据点；
- 其中 ToolACE 2K、Hammer 1K、xLAM 1K；
- 每个 query 生成 4 个 rollout；
- 作者观察到同分布数据超过 4K 后，收敛和最终性能没有明显增益；
- 关键在工具和复杂度的多样性，而不是继续复制同类样本。

这与文件 CRUD 最接近，因此 4K 是合理 pilot 下界；考虑我们同时包含增、删、改、查、多步组合、安全和停止行为，第一版目标提高到 8K。

### UI-R1：窄任务可以非常数据高效

[UI-R1](https://arxiv.org/abs/2503.21620) 使用 Qwen2.5-VL-3B，在专门 GUI grounding 任务上使用 136 条经过难度筛选的训练样本就取得明显提升；扩展版本使用约 2K。它说明窄任务中“困难且有区分度的样本”比盲目扩大同模板数据更重要，但 GUI 单步动作比我们的多步文件环境更简单，不能直接把 136 当作目标规模。

### DeepSeekMath：宽数学分布需要大规模任务

[DeepSeekMath](https://arxiv.org/abs/2402.03300) 的 7B GRPO 实验从约 144K 个 GSM8K/MATH 相关问题训练，并对每个问题采样 64 个输出。这个规模对应广泛数学能力，而不是单一工具 API，适合作为上界参照，不适合直接套到文件 CRUD。

### SWE-RL：覆盖真实软件演化时会到十万级

[SWE-RL](https://arxiv.org/abs/2502.18449) 从 GitHub PR 中抽取约 273K 个高质量 seed，每项包含 issue、代码上下文和 oracle patch。它覆盖真实软件工程分布，复杂度和多样性远高于当前沙箱 CRUD，因此说明未来扩展到完整代码 Agent 时需要十万级数据，但不是当前阶段的起点。

### Search-R1：rollout 预算和唯一 prompt 数不同

[Search-R1](https://arxiv.org/abs/2503.09516) 在 Qwen2.5-3B/7B 上进行搜索 Agent RL，GRPO 每个 prompt 采样 5 个响应、训练 500 steps。它强调训练预算不能只看唯一任务数，还要看每个 prompt 的 group size 和重复采样次数。

## 推荐数据预算

### 固定评测集

| 切分 | 数量 | 用途 |
|---|---:|---|
| validation | 1,000 | 超参数、checkpoint 选择、reward 调试 |
| test-ID | 1,000 | 同能力域但模板和内容不重复 |
| test-OOD/safety | 500 | 新目录布局、不同表达、诱导误删、越界、无需动作 |

总体 test 1,500 条时，整体成功率的统计波动已经足够小；同时每个 CRUD 大类可以保留约 200–300 条，避免总体指标掩盖 delete 或 stop 行为退化。

### DPO

建议先对 8K train prompt 各生成 4–8 条候选轨迹，然后按程序化 reward 排序：

- chosen：成功且 reward 较高的轨迹；
- rejected：文件状态错误、漏工具、无效动作、越界或未及时 finish 的轨迹；
- 每个 prompt 最多保留 1–2 个差异明显的 pair；
- 目标 10K–20K 偏好对；
- pilot 可以先用 2K–4K pair 验证 loss、格式和 held-out 指标。

不要把一个 prompt 的所有候选两两组合，否则会产生大量高度相关的伪数据，并让模板频率取代任务多样性。

### GRPO

第一阶段：

```text
4,000 unique prompts × 4 rollouts = 16,000 trajectories / epoch
```

通过 validation 后：

```text
8,000 unique prompts × 4–8 rollouts = 32,000–64,000 trajectories / epoch
```

建议从 1 个 epoch 和 group size 4 开始，根据 reward 方差、KL、熵和 validation 曲线决定是否增加到 2–3 epochs 或 group size 8。不能因为 ToolRL 使用 15 epochs 就机械复制；我们的任务更容易模板化和过拟合。

## Reference Model 选择

### DPO

标准 DPO 中：

```text
actor/reference = 同一个 Qwen3.5-4B 初始 checkpoint
```

如果先做文件工具 SFT，再做 DPO，则 actor 和 reference 都从该 SFT checkpoint 初始化，reference 冻结，actor 更新。DPO 原论文把 reference policy 定义为初始 SFT model；[TRL DPOTrainer](https://huggingface.co/docs/trl/main/en/dpo_trainer) 在未提供 `ref_model` 时也自动使用 DPO 开始前的初始 policy。

Qwen3.8-27B 不建议作为第一版标准 DPO reference，但在本机并非技术上绝对不可行。实测两个 checkpoint 都是 `Qwen3_5ForConditionalGeneration`，配置词表规模同为 248,320；使用 `AutoTokenizer.get_vocab()` 比较得到的 248,077 个 token→ID 映射完全一致，同一中文、英文和 JSON tool action 的编码也一致。因此，27B 确实能够为同一 completion token 计算 reference log probability。

问题在于算法语义和成本：标准 DPO 通常让 actor 从 reference checkpoint 初始化，而 4B actor 并不是从 27B 权重初始化；用 27B 会把“偏好优化”和“跨容量蒸馏锚点”混合，难以解释实验。DPO 虽可提前计算 27B reference log-prob，但仍增加数据预处理和模型部署复杂度。因此第一版应使用 frozen 4B 自身；如果以后专门研究 heterogeneous-reference DPO，可以把 27B 作为独立对照实验，而不是默认方案。

### GRPO

GRPO 中需要区分：

- current policy：正在训练的 Qwen3.5-4B；
- old policy：生成当前 rollout 的 actor snapshot，用于 importance ratio；
- reference policy：仅在使用 KL 项时约束 actor 不要偏离初始策略。

若使用 KL，reference 应为冻结的训练起点 Qwen3.5-4B（原始、SFT 后或 DPO 后 checkpoint，取决于 GRPO 从哪里开始）。[TRL GRPOTrainer](https://huggingface.co/docs/trl/main/en/grpo_trainer) 当前默认 `beta=0.0`，即默认不启用 reference KL；设置非零 beta 时才需要 reference log probabilities。27B 虽然词表兼容，但 GRPO completion 会随 actor 在线变化，无法像 DPO 那样一次性预计算所有 reference log-prob；每轮调用 27B 计算 KL 会显著增加延迟和系统复杂度，因此更不适合作为首轮 GRPO reference。

建议在当前文件任务上试验：

```text
beta ∈ {0, 0.001, 0.01}
```

ToolRL 使用无 KL 的 GRPO；Search-R1 使用约 0.001；DeepSeekMath 使用 0.04。应根据我们自己的 held-out 能力保持和 reward hacking 情况选择，而不是直接照搬。

## Qwen3.8-27B 的正确角色

Qwen3.8-27B 可以并且应该用作：

1. teacher：为困难任务生成成功轨迹和修复轨迹；
2. preference annotator：在程序化 reward 相同或接近时选择更清晰、低风险的轨迹；
3. task generator/reviewer：生成任务草案并检查歧义、标签错误和模板泄漏；
4. 辅助 judge：只评价无法程序化判断的最终解释质量。

它不应取代：

- DPO 的 Qwen3.5-4B frozen reference；
- GRPO 中非零 KL 对应的 Qwen3.5-4B frozen reference；
- 文件最终状态、越界和必要动作这些程序化 reward。

因此，简洁的角色分配是：

```text
Actor:             Qwen3.5-4B
Algorithmic ref:   frozen Qwen3.5-4B training-start checkpoint
Teacher/Judge:     Qwen3.8-27B
Primary reward:    LocalFileRLEnvironment programmatic reward
```

## 推荐训练顺序

1. 扩展并冻结 8K/1K/1K/500 任务集；
2. 对原始 Qwen3.5-4B 跑基线，确认原生 tool calling 和多轮 tool response 基础能力；
3. 先用 prompt、工具 schema 和无损 adapter 固定协议；仅当原生工具调用仍明显不稳定时，
   再用 reference actions 和 27B 成功轨迹做少量 trajectory SFT；
4. 从 4B rollout 中构造 10K–20K preference pairs，做 DPO；
5. 以 DPO checkpoint 为 actor 和 frozen reference 起点，做在线 GRPO；
6. 固定比较 raw 4B、SFT、DPO、DPO+GRPO 四个 checkpoint。
