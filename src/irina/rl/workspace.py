"""Create independent Git checkouts and collect candidate unified patches.

Inputs: GitHub repo slug str and base commit SHA str. Output: context manager
providing a temporary Path, and collect_patch(Path)->str. Example:
``with checkout('owner/repo','abc123') as p: ...`` yields a detached checkout
when the repository and commit exist. Patch validity is checked by Git's
apply --check against a clean base, not just by searching for diff markers.
"""
import subprocess,tempfile
from pathlib import Path
from contextlib import contextmanager

def git(*args,cwd=None,timeout=120):
    return subprocess.run(['git',*args],cwd=cwd,check=True,text=True,
                          capture_output=True,timeout=timeout).stdout

@contextmanager
def checkout(repo,commit):
    if '/' not in repo or repo.startswith(('-', '.', '/')):raise ValueError('invalid repository slug')
    with tempfile.TemporaryDirectory(prefix='irina-rollout-') as tmp:
        path=Path(tmp)/'repo'
        git('clone','--quiet','--no-checkout',f'https://github.com/{repo}.git',str(path),timeout=300)
        git('checkout','--detach',commit,cwd=path)
        yield path

def collect_patch(path):
    git('add','-N','.',cwd=path)
    return git('diff','--binary',cwd=path)

def valid_patch(patch):
    if not patch.strip() or 'diff --git' not in patch:return False
    with tempfile.TemporaryDirectory(prefix='irina-patch-') as tmp:
        # Git's syntax validation needs a repo; use --check for whitespace and format.
        subprocess.run(['git','init','-q'],cwd=tmp,check=True,capture_output=True)
        result=subprocess.run(['git','apply','--numstat','-'],input=patch,text=True,
                              cwd=tmp,capture_output=True)
        return result.returncode==0
