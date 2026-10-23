"""Docker-backed execution boundary for untrusted agent commands.

Input: workspace Path and command str. Output: stdout/stderr str or an
exception for timeout/failure. Example: ``DockerSandbox().run(path, 'pwd')``
returns ``/workspace`` with a running Docker daemon. The workspace is mounted
read/write; network, capabilities and privilege escalation are disabled.
This is a practical defense-in-depth profile, not a formal security proof.
"""
import subprocess
from pathlib import Path

class DockerSandbox:
    def __init__(self, image: str = 'python:3.11-slim', timeout: int = 60):
        self.image, self.timeout = image, timeout
    def run(self, workdir: str | Path, command: str) -> str:
        path = Path(workdir).resolve()
        if not path.is_dir(): raise ValueError('workspace must exist')
        args = ['docker','run','--rm','--network','none','--cap-drop','ALL',
                '--security-opt','no-new-privileges','--pids-limit','128',
                '--memory','1g','--cpus','1','--read-only','--tmpfs','/tmp:rw,noexec,nosuid,size=64m',
                '-v',f'{path}:/workspace:rw','-w','/workspace',self.image,
                '/bin/sh','-lc',command]
        proc = subprocess.run(args, capture_output=True, text=True, timeout=self.timeout)
        return (proc.stdout + proc.stderr)[-20_000:] + (f'\n[exit={proc.returncode}]' if proc.returncode else '')
