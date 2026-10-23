from irina.security import Policy, jailed_path
from irina.loop import parse_action
from irina.rl.rewards import synthetic_reward
from irina.rl.training import advantages
import pytest

def test_mode():
    with pytest.raises(ValueError): Policy("other")
def test_jail(tmp_path):
    with pytest.raises(PermissionError): jailed_path(tmp_path, "../outside")
def test_parse():
    assert parse_action('<tool_call>{"name":"list_files","arguments":{"path":"."}}</tool_call>')["name"] == "list_files"
def test_synthetic():
    assert synthetic_reward("patch", 42, 0) == synthetic_reward("patch", 42, 0)
def test_zero_variance():
    assert advantages([1,1,1]).tolist() == [0,0,0]
