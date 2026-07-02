from __future__ import annotations

from agent_project.tools.work_management import capture_work_message, parse_work_record
from agent_project.tools.work_vectors import _cosine, _normalize


def test_parse_work_record_supports_hash_sections() -> None:
    text = """# 工作 A
## 项目文件
/tmp/work-a
## 工作进展
状态：进行中

# 工作 B
## 项目路径
/tmp/work-b
## 最近进展
状态：完成
"""
    items = parse_work_record(text)
    assert [item.title for item in items] == ["工作 A", "工作 B"]
    assert items[0].paths == ["/tmp/work-a"]
    assert "进行中" in items[0].progress


def test_parse_work_record_supports_container_heading_format() -> None:
    text = """# 工作记录

## 基于本地模型的医疗翻译工作流

- 状态：进行中
- 项目文件：
  - `/home/qiao/work/nhtai-service-translation-offline`
- 最近进展：nginx worker timeout
- 下一步：优化稳定性。
"""
    items = parse_work_record(text)
    assert len(items) == 1
    assert items[0].title == "基于本地模型的医疗翻译工作流"
    assert items[0].paths == ["/home/qiao/work/nhtai-service-translation-offline"]
    assert "nginx" in items[0].progress


def test_vector_math_uses_cosine_similarity() -> None:
    left = _normalize([3.0, 4.0])
    right = _normalize([6.0, 8.0])
    assert round(_cosine(left, right), 6) == 1.0


def test_capture_work_message_falls_back_to_keyword_matching(tmp_path, monkeypatch) -> None:
    db_path = tmp_path / "agent.sqlite3"
    record_path = tmp_path / "aaa-work.md"
    record_path.write_text(
        """# 基于本地模型的医疗翻译工作流
## 项目文件
/home/qiao/work/nhtai-service-translation-offline
## 工作进展
nginx worker timeout 服务稳定性
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("AGENT_STATE_DB_PATH", str(db_path))
    monkeypatch.setenv("AGENT_WORK_RECORD_PATH", str(record_path))
    monkeypatch.setenv("AGENT_EMBEDDING_MODEL_PATH", str(tmp_path / "missing-model"))

    result = capture_work_message.invoke(
        {"content": "医疗翻译服务 nginx timeout，需要排查 worker。"}
    )

    assert "matched_work: 基于本地模型的医疗翻译工作流" in result
    assert "route: codex" in result
