"""Six real coding tools and a schema-producing custom tool decorator.

Input: workspace Path, optional DockerSandbox, and dispatch(name, args).
Output: textual observation str. Example:
``Tools('/tmp/repo').dispatch('list_files', {'path':'.'})`` returns a
newline-delimited file listing. Bash requires an explicitly supplied sandbox.
All file paths are resolved through the symlink-aware workspace jail.
"""
import inspect
from pathlib import Path
from .security import jailed_path

def tool(description: str, **parameters):
    def decorate(fn):
        fn.tool_schema = {'name':fn.__name__, 'description':description,
                          'parameters':{k:{'description':parameters.get(k,''),
                                          'required':p.default is inspect.Parameter.empty}
                                        for k,p in inspect.signature(fn).parameters.items()}}
        return fn
    return decorate

class Tools:
    def __init__(self, workdir, sandbox=None, extra_tools=None):
        self.workdir=Path(workdir).resolve(); self.sandbox=sandbox
        self.extra={fn.__name__:fn for fn in (extra_tools or [])}
    def read_file(self, path): return jailed_path(self.workdir,path).read_text(encoding='utf-8')[:50_000]
    def write_file(self, path, content):
        f=jailed_path(self.workdir,path);f.parent.mkdir(parents=True,exist_ok=True)
        f.write_text(content,encoding='utf-8');return f'wrote {path}'
    def edit_file(self, path, old, new):
        f=jailed_path(self.workdir,path);text=f.read_text(encoding='utf-8')
        if old not in text: raise ValueError('old text not found')
        f.write_text(text.replace(old,new,1),encoding='utf-8');return f'edited {path}'
    def list_files(self, path='.'):
        root=jailed_path(self.workdir,path)
        return '\n'.join(str(f.relative_to(self.workdir)) for f in sorted(root.iterdir())[:200])
    def grep(self, pattern, path='.'):
        root=jailed_path(self.workdir,path)
        files=[root] if root.is_file() else root.rglob('*')
        matches=[]
        for f in files:
            if not f.is_file() or '.git' in f.parts: continue
            try:
                for n,line in enumerate(f.read_text(encoding='utf-8').splitlines(),1):
                    if pattern in line: matches.append(f'{f.relative_to(self.workdir)}:{n}:{line[:200]}')
                    if len(matches)>=100: return '\n'.join(matches)
            except (OSError,UnicodeError): pass
        return '\n'.join(matches)
    def bash(self, command):
        if self.sandbox is None: raise RuntimeError('bash requires configured DockerSandbox')
        return self.sandbox.run(self.workdir,command)
    def dispatch(self,name,args):
        if name in self.extra: return self.extra[name](**args)
        if name not in {'read_file','write_file','edit_file','list_files','grep','bash'}:
            raise ValueError(f'unknown tool {name}')
        return getattr(self,name)(**args)
