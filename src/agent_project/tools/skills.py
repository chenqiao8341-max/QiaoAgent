from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from langchain_core.tools import tool

from agent_project.tools.progress import emit_progress


@dataclass(frozen=True)
class SkillInfo:
    name: str
    description: str
    root: Path
    skill_md: Path
    metadata: dict[str, str]


def _default_skill_dirs() -> list[Path]:
    project_root = Path(__file__).resolve().parents[3]
    return [project_root / "skills", Path.home() / ".codex" / "skills"]


def _configured_skill_dirs() -> list[Path]:
    configured = os.getenv("AGENT_SKILLS_DIRS", "").strip()
    if configured:
        raw_paths = [part for part in configured.split(os.pathsep) if part.strip()]
        return [Path(part).expanduser().resolve() for part in raw_paths]
    return [path.expanduser().resolve() for path in _default_skill_dirs()]


def skills_enabled() -> bool:
    return os.getenv("AGENT_ENABLE_SKILLS", "true").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _parse_frontmatter(text: str) -> tuple[dict[str, str], str]:
    if not text.startswith("---\n"):
        return {}, text

    end = text.find("\n---", 4)
    if end == -1:
        return {}, text

    raw = text[4:end].strip()
    body_start = end + len("\n---")
    if body_start < len(text) and text[body_start] == "\n":
        body_start += 1

    metadata: dict[str, str] = {}
    current_key: str | None = None
    for line in raw.splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if line.startswith(" ") and current_key:
            metadata[current_key] = f"{metadata[current_key]} {line.strip()}".strip()
            continue
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        current_key = key.strip()
        value = value.strip().strip('"').strip("'")
        metadata[current_key] = value
    return metadata, text[body_start:]


def _safe_skill_name(name: str) -> str:
    return name.strip().lower()


def discover_skills(limit: int | None = None) -> list[SkillInfo]:
    if not skills_enabled():
        return []

    skills: dict[str, SkillInfo] = {}
    for directory in _configured_skill_dirs():
        if not directory.exists():
            continue
        for skill_md in directory.rglob("SKILL.md"):
            try:
                text = skill_md.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                text = skill_md.read_text(errors="replace")
            metadata, _body = _parse_frontmatter(text)
            name = metadata.get("name") or skill_md.parent.name
            description = metadata.get("description", "").strip()
            if not name or not description:
                continue
            key = _safe_skill_name(name)
            skills.setdefault(
                key,
                SkillInfo(
                    name=name,
                    description=description,
                    root=skill_md.parent.resolve(),
                    skill_md=skill_md.resolve(),
                    metadata=metadata,
                ),
            )
            if limit and len(skills) >= limit:
                break
        if limit and len(skills) >= limit:
            break
    return sorted(skills.values(), key=lambda item: item.name)


def skill_catalog_text(limit: int | None = None) -> str:
    skills = discover_skills(limit=limit)
    if not skills:
        return ""

    lines = ["Available skills (metadata only; call read_skill before using details):"]
    for skill in skills:
        lines.append(f"- {skill.name}: {skill.description}")
    return "\n".join(lines)


def _find_skill(name: str) -> SkillInfo | None:
    wanted = _safe_skill_name(name)
    for skill in discover_skills():
        if _safe_skill_name(skill.name) == wanted or skill.root.name.lower() == wanted:
            return skill
    return None


def _safe_child(root: Path, relative_path: str) -> Path | None:
    candidate = (root / relative_path).expanduser().resolve()
    try:
        candidate.relative_to(root)
    except ValueError:
        return None
    if not candidate.is_file():
        return None
    return candidate


@tool
def list_skills() -> str:
    """List available skills by name and description."""
    emit_progress("listing skills")
    catalog = skill_catalog_text()
    return catalog or "No skills are available."


@tool
def read_skill(name: str) -> str:
    """Read the SKILL.md instructions for a named skill after it becomes relevant."""
    skill = _find_skill(name)
    if skill is None:
        return f"Skill not found: {name}"

    emit_progress(f"reading skill: {skill.name}")
    text = skill.skill_md.read_text(encoding="utf-8", errors="replace")
    return f"# Skill: {skill.name}\nPath: {skill.skill_md}\n\n{text}"


@tool
def read_skill_file(name: str, relative_path: str) -> str:
    """Read a bundled skill resource file, such as references/foo.md or scripts/foo.py."""
    skill = _find_skill(name)
    if skill is None:
        return f"Skill not found: {name}"

    target = _safe_child(skill.root, relative_path)
    if target is None:
        return f"Skill file not found or outside skill directory: {relative_path}"

    emit_progress(f"reading skill file: {skill.name}/{relative_path}")
    text = target.read_text(encoding="utf-8", errors="replace")
    return f"# Skill file: {skill.name}/{relative_path}\nPath: {target}\n\n{text}"
