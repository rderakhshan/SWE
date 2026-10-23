"""Discover and load optional workspace skills lazily.

Input: workspace path (str | Path). Output: mapping of skill names to paths.
Example: ``discover_skills("/tmp/project")`` returns ``{"review": Path(...)}"
when ``skills/review/SKILL.md`` exists. ``load_skill`` returns its text.
Skill files are untrusted instructions and should not override security policy.
"""
from pathlib import Path

def discover_skills(workdir: str | Path) -> dict[str, Path]:
    root = Path(workdir) / "skills"
    return {f.parent.name: f for f in sorted(root.glob("*/SKILL.md")) if f.is_file()}

def load_skill(workdir: str | Path, name: str) -> str:
    skills = discover_skills(workdir)
    if name not in skills: raise KeyError(name)
    return skills[name].read_text(encoding="utf-8")[:40_000]
