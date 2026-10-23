"""Tests for Irina's session, policy, tools, memory and grouped RL mathematics."""
from pathlib import Path
from irina.security import Policy, jailed_path
from irina.tools import Tools
from irina.memory import read_memory
from irina.skills import discover_skills
from irina.session import Session
from irina.context import compact
from irina.harness import Harness
from irina.fleet import run_fleet

class Provider:
    def generate(self, messages): return {'text':'done','tokens':[1]}

def test_session(tmp_path):
    s=Session(tmp_path);s.append({'type':'hello'});assert s.read()==[{'type':'hello'}]
    assert Session.latest(tmp_path).session_id==s.session_id

def test_memory_skills(tmp_path):
    (tmp_path/'IRINA.md').write_text('instructions')
    (tmp_path/'skills'/'test').mkdir(parents=True)
    (tmp_path/'skills'/'test'/'SKILL.md').write_text('skill')
    assert read_memory(tmp_path)=='instructions'
    assert 'test' in discover_skills(tmp_path)

def test_tool_and_jail(tmp_path):
    t=Tools(tmp_path);t.write_file('hello.txt','hello')
    assert t.read_file('hello.txt')=='hello'
    try:jailed_path(tmp_path,'../outside')
    except PermissionError:pass
    else:assert False

def test_policy(tmp_path):
    assert not Policy('read-only').check('bash',{'command':'ls'})[0]
    assert not Policy('yolo').check('bash',{'command':'rm -rf /'})[0]
    assert Policy('yolo').check('edit_file',{})[0]

def test_harness(tmp_path):
    result=Harness(tmp_path,provider=Provider(),policy=Policy('read-only')).run('hello')
    assert result['reason']=='final'
    assert result['events'][0]['tokens']==[1]

def test_fleet(tmp_path):
    jobs=[{'name':'a','workdir':str(tmp_path),'task':'hello'}]
    result=list(run_fleet(jobs,lambda path:Harness(path,provider=Provider(),session_enabled=False)))
    assert result[0]['ok']

def test_compact():
    messages=[{'role':'system','content':'x'},{'role':'user','content':'long'*20},
              {'role':'assistant','content':'ok'}]
    result=compact(messages,10)
    assert result[0]['role']=='system' and result[-1]['content']=='ok'
