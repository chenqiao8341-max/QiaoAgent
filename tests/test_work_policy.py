from __future__ import annotations

from agent_project.work_policy import classify_work_message_policy
from agent_project.tools.work_vectors import preload_work_record_embedding_model


def test_work_policy_marks_today_as_high_priority() -> None:
    decision = classify_work_message_policy("今天处理一下医疗翻译服务问题。", matched_work="")

    assert decision.priority == "high"


def test_work_policy_routes_service_failures_to_codex() -> None:
    decision = classify_work_message_policy("nginx timeout 部署报错，需要看代码。", matched_work="医疗翻译")

    assert decision.route == "codex"
    assert "Codex" in decision.rationale


def test_work_policy_defers_later_messages() -> None:
    decision = classify_work_message_policy("这个之后有空记录一下。", matched_work="")

    assert decision.route == "defer"
    assert decision.priority == "low"


def test_embedding_preload_reports_missing_model(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("AGENT_EMBEDDING_MODEL_PATH", str(tmp_path / "missing-model"))

    result = preload_work_record_embedding_model.invoke({})

    assert "Cannot preload embedding model" in result
