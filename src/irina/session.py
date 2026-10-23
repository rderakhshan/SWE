"""Append-only JSONL event sessions for agent resume and audit.

Input: ``workdir`` (str | Path), ``session_id`` (optional str), and JSON events.
Output: a Session whose ``append`` persists one record and whose ``read``
returns list[dict]. Example: ``s=Session('/tmp/p'); s.append({'type':'x'})``;
``s.read()`` returns ``[{'type':'x'}]``. Session files are workspace-local.
"""
import json, uuid
from pathlib import Path

class Session:
    def __init__(self, workdir: str | Path, session_id: str | None = None):
        root = Path(workdir) / '.irina' / 'sessions'
        root.mkdir(parents=True, exist_ok=True)
        self.session_id = session_id or uuid.uuid4().hex
        if not self.session_id.isalnum(): raise ValueError('invalid session id')
        self.path = root / f'{self.session_id}.jsonl'
    def append(self, event: dict) -> None:
        with self.path.open('a', encoding='utf-8') as out:
            out.write(json.dumps(event, default=str) + '\n')
            out.flush()
    def read(self) -> list[dict]:
        if not self.path.exists(): return []
        return [json.loads(line) for line in self.path.read_text().splitlines() if line.strip()]
    @staticmethod
    def latest(workdir: str | Path):
        root = Path(workdir) / '.irina' / 'sessions'
        matches = sorted(root.glob('*.jsonl'), key=lambda f: f.stat().st_mtime, reverse=True)
        return Session(workdir, matches[0].stem) if matches else None
