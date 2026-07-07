from __future__ import annotations

import os
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from langchain_core.tools import tool

from agent_project.tools.progress import emit_progress


DEFAULT_PROMPT = "你好"
DEFAULT_TIMEOUT_SECONDS = 30
DEFAULT_SANDBOX = "read-only"
MAX_TIMEOUT_SECONDS = 120
MAX_OUTPUT_CHARS = 4000


@dataclass(frozen=True)
class CodexCandidate:
    name: str
    path: str


@dataclass(frozen=True)
class CodexProbeResult:
    name: str
    path: str
    ok: bool
    status: str
    elapsed_seconds: float
    exit_code: int | None
    output: str
    cwd: str


def _path_entries() -> list[Path]:
    entries: list[Path] = []
    seen: set[Path] = set()
    for raw_entry in os.getenv("PATH", "").split(os.pathsep):
        if not raw_entry:
            continue
        entry = Path(raw_entry).expanduser()
        try:
            resolved = entry.resolve()
        except OSError:
            continue
        if resolved in seen:
            continue
        seen.add(resolved)
        entries.append(resolved)
    return entries


def _find_git_worktree(path: Path) -> Path | None:
    candidates = [path, *path.parents]
    for candidate in candidates:
        git_marker = candidate / ".git"
        if git_marker.exists():
            return candidate
    return None


def _resolve_probe_cwd() -> Path:
    configured = os.getenv("AGENT_CODEX_PROBE_CWD", "").strip()
    if configured:
        path = Path(configured).expanduser()
        if path.is_dir():
            return path.resolve()

    current = Path.cwd().resolve()
    current_worktree = _find_git_worktree(current)
    if current_worktree is not None:
        return current_worktree

    workspace_root = os.getenv("AGENT_WORKSPACE_ROOT", "").strip()
    if workspace_root:
        workspace = Path(workspace_root).expanduser()
        workspace_worktree = _find_git_worktree(workspace.resolve()) if workspace.is_dir() else None
        if workspace_worktree is not None:
            return workspace_worktree

    package_repo = Path(__file__).resolve().parents[3]
    package_worktree = _find_git_worktree(package_repo)
    if package_worktree is not None:
        return package_worktree

    return current


def discover_codex_commands(include_official: bool = True) -> list[CodexCandidate]:
    """Return executable codex/codex-proxy-* commands visible on PATH."""
    candidates: dict[str, str] = {}

    if include_official:
        official = shutil.which("codex")
        if official:
            candidates["codex"] = official

    for directory in _path_entries():
        if not directory.is_dir():
            continue
        try:
            children = list(directory.iterdir())
        except OSError:
            continue
        for child in children:
            name = child.name
            if not name.startswith("codex-proxy-"):
                continue
            if child.is_file() and os.access(child, os.X_OK):
                candidates.setdefault(name, str(child))

    return [
        CodexCandidate(name=name, path=path)
        for name, path in sorted(candidates.items(), key=lambda item: item[0])
    ]


def _coerce_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _trim_output(stdout: str, stderr: str) -> str:
    parts: list[str] = []
    stdout = stdout.strip()
    stderr = stderr.strip()
    if stdout:
        parts.append(f"stdout:\n{stdout}")
    if stderr:
        parts.append(f"stderr:\n{stderr}")
    output = "\n\n".join(parts) or "(no output)"
    if len(output) > MAX_OUTPUT_CHARS:
        return output[:MAX_OUTPUT_CHARS] + f"\n\n[truncated after {MAX_OUTPUT_CHARS} characters]"
    return output


def probe_codex_command(
    candidate: CodexCandidate,
    prompt: str = DEFAULT_PROMPT,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
    sandbox: str = DEFAULT_SANDBOX,
) -> CodexProbeResult:
    timeout = max(1, min(int(timeout_seconds), MAX_TIMEOUT_SECONDS))
    command = [
        candidate.path,
        "exec",
        "-s",
        sandbox,
        "--color",
        "never",
        prompt,
    ]
    probe_cwd = _resolve_probe_cwd()
    start = time.monotonic()
    try:
        completed = subprocess.run(
            command,
            input="",
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
            cwd=str(probe_cwd),
        )
        elapsed = time.monotonic() - start
        output = _trim_output(completed.stdout or "", completed.stderr or "")
        return CodexProbeResult(
            name=candidate.name,
            path=candidate.path,
            ok=completed.returncode == 0,
            status="ok" if completed.returncode == 0 else "failed",
            elapsed_seconds=elapsed,
            exit_code=completed.returncode,
            output=output,
            cwd=str(probe_cwd),
        )
    except subprocess.TimeoutExpired as exc:
        elapsed = time.monotonic() - start
        output = _trim_output(_coerce_text(exc.stdout), _coerce_text(exc.stderr))
        return CodexProbeResult(
            name=candidate.name,
            path=candidate.path,
            ok=False,
            status="timeout",
            elapsed_seconds=elapsed,
            exit_code=None,
            output=output,
            cwd=str(probe_cwd),
        )
    except OSError as exc:
        elapsed = time.monotonic() - start
        return CodexProbeResult(
            name=candidate.name,
            path=candidate.path,
            ok=False,
            status="error",
            elapsed_seconds=elapsed,
            exit_code=None,
            output=str(exc),
            cwd=str(probe_cwd),
        )


def _format_results(results: list[CodexProbeResult]) -> str:
    usable = sorted(result.name for result in results if result.ok)
    lines = [
        "Available Codex configurations:",
        ", ".join(usable) if usable else "(none)",
        "",
        "Details:",
    ]
    for result in sorted(results, key=lambda item: item.name):
        lines.append(
            f"- {result.name}: {result.status}, "
            f"elapsed={result.elapsed_seconds:.1f}s, "
            f"exit_code={result.exit_code}, path={result.path}, cwd={result.cwd}"
        )
        if not result.ok:
            reason = next(
                (
                    line
                    for line in result.output.splitlines()
                    if line and line not in {"stderr:", "stdout:"}
                ),
                "",
            )
            if reason:
                lines.append(f"  reason: {reason[:300]}")
    return "\n".join(lines)


@tool
def test_codex_connectivity(
    prompt: str = DEFAULT_PROMPT,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
    include_official: bool = True,
    command_names: str = "",
) -> str:
    """Test local codex and codex-proxy-* CLI configurations visible on PATH.

    Leave command_names empty to scan every discovered configuration. Set command_names
    only when the caller intentionally wants to restrict the test to a comma-separated subset.
    """
    emit_progress("discovering Codex configurations")
    candidates = discover_codex_commands(include_official=include_official)
    if command_names.strip():
        wanted = {name.strip() for name in command_names.split(",") if name.strip()}
        candidates = [candidate for candidate in candidates if candidate.name in wanted]

    if not candidates:
        return "No Codex configurations found on PATH."

    emit_progress(
        "testing Codex configurations: " + ", ".join(candidate.name for candidate in candidates)
    )
    timeout = max(1, min(int(timeout_seconds), MAX_TIMEOUT_SECONDS))
    worker_count = min(len(candidates), 8)
    results: list[CodexProbeResult] = []
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = {
            executor.submit(
                probe_codex_command,
                candidate,
                prompt=prompt or DEFAULT_PROMPT,
                timeout_seconds=timeout,
            ): candidate
            for candidate in candidates
        }
        for future in as_completed(futures):
            result = future.result()
            emit_progress(f"Codex configuration {result.name}: {result.status}")
            results.append(result)

    return _format_results(results)
