"""Irina public API for real coding-agent harness execution.

Input: a workspace path and a task prompt. Output: Harness.run returns a dict
with messages/events/reason. Example: ``Harness('/tmp/repo').run('inspect README')``
returns a result after a configured provider generates text. A provider key or
local Qwen model is required for actual inference.
"""
from .harness import Harness
from .security import Policy
from .tools import tool
from .fleet import run_fleet
__all__ = ['Harness','Policy','tool','run_fleet']
