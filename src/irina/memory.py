"""Workspace-scoped persistent project memory.

Inputs: ``workdir`` (str | Path). Output: project memory (str).
Example: ``read_memory("/tmp/project")`` returns the text of ``IRINA.md``
if present, otherwise ``""``. Legacy ``ODYSSEUS.md`` is read only when the
new convention is absent. Never copy memory between independent rollouts.
"""
from pathlib import Path

def read_memory(workdir: str | Path) -> str:
    for name in ("IRINA.md", "ODYSSEUS.md"):
        file = Path(workdir) / name
        if file.is_file(): return file.read_text(encoding="utf-8")[:40_000]
    return ""
