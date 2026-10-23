"""Permission checks and symlink-safe workspace path confinement.

Input: Policy(mode, approver) and tool name/argument mapping. Output:
``(allowed: bool, reason: str)``. Example:
``Policy('read-only').check('write_file', {})`` returns ``(False,'read-only')``.
Deny rules override all modes. This policy is not an OS sandbox.
"""
from dataclasses import dataclass
from pathlib import Path

DENIED_FRAGMENTS = ('rm -rf /', 'mkfs', 'dd if=/dev/', ':(){:|:&};:')
WRITES = {'write_file','edit_file','bash'}
BUILTINS = {'read_file','write_file','edit_file','bash','list_files','grep'}

@dataclass
class Policy:
    mode: str = 'safe'
    approver: object = None
    def __post_init__(self):
        if self.mode not in ('safe','yolo','read-only'): raise ValueError('invalid mode')
    def check(self, name: str, arguments: dict) -> tuple[bool,str]:
        if name not in BUILTINS: return False,'unknown tool'
        if name == 'bash' and any(s in str(arguments.get('command','')) for s in DENIED_FRAGMENTS):
            return False,'denied dangerous command'
        if name in WRITES:
            if self.mode == 'read-only': return False,'read-only'
            if self.mode == 'safe' and not (self.approver and self.approver(name, arguments)):
                return False,'approval denied'
        return True,'allowed'

def jailed_path(root: str | Path, relative: str | Path) -> Path:
    base = Path(root).resolve()
    path = (base / relative).resolve()
    if not path.is_relative_to(base): raise PermissionError('path escapes workspace')
    return path
