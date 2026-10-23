"""Compose the Irina provider, agent loop, tools, policy, memory and sessions.

Input: ``Harness(workdir, provider, policy, sandbox, ...)`` and task str.
Output: ``run(task, max_turns)`` dict with messages/events/reason.
Example: ``Harness('/tmp/repo',provider=p,policy=Policy('read-only')).run('inspect')``
returns a real provider-generated trajectory. ``resume=True`` continues the
last completed persisted conversation; interrupted tool execution is not
replayed automatically. A session is disabled by default in GRPO rollouts.
"""
from pathlib import Path
from .provider import RemoteProvider
from .security import Policy
from .tools import Tools
from .loop import agent_loop
from .memory import read_memory
from .skills import discover_skills
from .session import Session
from .context import compact

SYSTEM = """You are Irina, a coding agent. Inspect and edit repository files to resolve the task.
To invoke a tool emit exactly <tool_call>{"name":"read_file","arguments":{"path":"README.md"}}</tool_call>.
Tools: read_file(path), write_file(path,content), edit_file(path,old,new), list_files(path), grep(pattern,path), bash(command).
Use one tool call per response. When finished, answer without tool tags."""

class Harness:
    def __init__(self,workdir='.',provider=None,policy=None,on_event=None,system_extra='',
                 extra_tools=None,sandbox=None,session_enabled=True,max_context_chars=100_000):
        self.workdir=Path(workdir).resolve();self.provider=provider or RemoteProvider()
        self.policy=policy or Policy('safe');self.sandbox=sandbox
        self.tools=Tools(self.workdir,sandbox=sandbox,extra_tools=extra_tools)
        self.on_event=on_event;self.system_extra=system_extra
        self.session_enabled=session_enabled;self.max_context_chars=max_context_chars
    def run(self,task,max_turns=4,resume=False):
        memory=read_memory(self.workdir)
        skills=discover_skills(self.workdir)
        system=SYSTEM+'\n'+self.system_extra
        if memory: system+='\nProject memory:\n'+memory
        if skills: system+='\nAvailable skills: '+', '.join(skills)
        session=Session.latest(self.workdir) if resume and self.session_enabled else None
        if session:
            past=session.read(); snapshots=[e['messages'] for e in past if e.get('type')=='snapshot']
            messages=snapshots[-1] if snapshots else [{'role':'system','content':system}]
            if task: messages.append({'role':'user','content':task})
        else:
            messages=[{'role':'system','content':system},{'role':'user','content':task}]
            session=Session(self.workdir) if self.session_enabled else None
        messages=compact(messages,self.max_context_chars)
        def emit(event):
            if session: session.append(event)
            if self.on_event: self.on_event(event)
        result=agent_loop(self.provider,self.tools,self.policy,messages,max_turns,emit)
        if session: session.append({'type':'snapshot','messages':result['messages']})
        return result
