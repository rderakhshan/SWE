"""Delegated Irina task execution with inherited workspace and policy.

Input: parent Harness and task str. Output: agent result dict containing
``messages``, ``events``, and ``reason``. Example:
``delegate(parent, 'inspect tests', max_turns=2)`` runs a child with the same
provider and policy, no independent persistent session. Delegation is not an
isolation boundary and must never relax the parent's permissions.
"""
def delegate(parent, task: str, max_turns: int = 2) -> dict:
    from .harness import Harness
    child = Harness(parent.workdir, provider=parent.provider, policy=parent.policy,
                    system_extra=parent.system_extra, sandbox=parent.sandbox,
                    session_enabled=False)
    return child.run(task, max_turns=max_turns)
