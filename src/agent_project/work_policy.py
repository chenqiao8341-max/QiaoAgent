from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class WorkMessagePolicyDecision:
    route: str
    priority: str
    rationale: str


URGENT_MARKERS = ("紧急", "马上", "今天", "阻塞", "故障", "挂了", "报错", "timeout", "崩", "失败")
CODEX_MARKERS = (
    "代码",
    "修改",
    "实现",
    "bug",
    "报错",
    "部署",
    "nginx",
    "docker",
    "服务",
    "测试",
    "脚本",
    "接口",
    "pipeline",
)
ASK_MARKERS = ("是否", "吗", "？", "?", "什么时候", "确认", "问一下")
DEFER_MARKERS = ("记录", "备忘", "之后", "有空", "暂时")


def classify_work_message_policy(content: str, matched_work: str = "") -> WorkMessagePolicyDecision:
    lowered = content.lower()
    priority = "high" if any(marker in lowered for marker in URGENT_MARKERS) else "normal"
    if any(marker in lowered for marker in CODEX_MARKERS):
        return WorkMessagePolicyDecision(
            route="codex",
            priority=priority,
            rationale="包含代码、部署、服务或测试信号，适合改写任务后交给 Codex。",
        )
    if any(marker in lowered for marker in ASK_MARKERS) and not matched_work:
        return WorkMessagePolicyDecision(
            route="ask_user",
            priority=priority,
            rationale="消息像是需要进一步确认，且没有匹配到已有工作。",
        )
    if any(marker in lowered for marker in DEFER_MARKERS):
        return WorkMessagePolicyDecision(
            route="defer",
            priority="low" if priority == "normal" else priority,
            rationale="消息更像备忘或低紧急度记录。",
        )
    return WorkMessagePolicyDecision(
        route="self",
        priority=priority,
        rationale="可由 agent 自己整理、更新工作记录或生成下一步。",
    )
