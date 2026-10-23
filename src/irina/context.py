"""Context budgeting and compaction for Irina conversations.

Inputs: ``messages`` (list[dict[str,str]]) and ``max_chars`` (int).
Outputs: a shortened list of messages retaining system and recent dialogue.
Example: ``compact([{"role":"system","content":"rules"},{"role":"user","content":"hi"}], 100)``
returns the original two messages. This is deterministic text compaction, not a
model-generated summary; older tool observations may be dropped.
"""

def compact(messages: list[dict], max_chars: int = 100_000) -> list[dict]:
    if max_chars < 1: raise ValueError("max_chars must be positive")
    if sum(len(m.get("content", "")) for m in messages) <= max_chars: return list(messages)
    head = [messages[0]] if messages and messages[0].get("role") == "system" else []
    budget = max(0, max_chars - sum(len(m.get("content", "")) for m in head))
    tail = []
    for m in reversed(messages[len(head):]):
        n = len(m.get("content", ""))
        if n > budget: break
        tail.append(m); budget -= n
    return head + list(reversed(tail))
