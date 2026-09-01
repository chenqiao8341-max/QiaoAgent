from __future__ import annotations

from pathlib import Path

from agent_project.tools.codex_connectivity import _find_git_worktree, _resolve_probe_cwd


def test_find_git_worktree_walks_up_from_nested_path(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    nested = repo / "a" / "b"
    nested.mkdir(parents=True)
    git_dir = repo / ".git"
    git_dir.mkdir()
    (git_dir / "HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")

    assert _find_git_worktree(nested) == repo


def test_resolve_probe_cwd_prefers_configured_directory(tmp_path: Path, monkeypatch) -> None:
    configured = tmp_path / "probe"
    configured.mkdir()
    monkeypatch.setenv("AGENT_CODEX_PROBE_CWD", str(configured))
    monkeypatch.chdir(tmp_path)

    assert _resolve_probe_cwd() == configured.resolve()


def test_resolve_probe_cwd_falls_back_to_package_repo_for_non_repo_cwd(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("AGENT_CODEX_PROBE_CWD", raising=False)
    monkeypatch.delenv("AGENT_WORKSPACE_ROOT", raising=False)
    monkeypatch.chdir(tmp_path)

    resolved = _resolve_probe_cwd()
    assert resolved == Path(__file__).resolve().parents[1]
    assert (resolved / ".git").exists()
