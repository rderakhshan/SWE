"""Irina's single provider-independent tool-use agent loop.

Input: provider.generate(messages)->dict, Tools, Policy, message list, turn cap.
Output: dict containing messages/events/reason. Example: with a provider
returning ``{'text':'done'}``, agent_loop returns reason ``'final'``.
Events are optionally streamed to on_event; only tool outputs, not user text,
are appended as observations. A malformed tool call produces a parse error.
"""
import json,re

def parse_action(text: str):
    m=re.search(r'<tool_call>\s*(\{.*?\})\s*</tool_call>',text,re.S)
    if not m: return None
    obj=json.loads(m.group(1))
    if not isinstance(obj,dict) or not isinstance(obj.get('name'),str) or not isinstance(obj.get('arguments'),dict):
        raise ValueError('invalid tool call')
    return obj

def agent_loop(provider,tools,policy,messages,max_turns=4,on_event=None):
    events=[]
    def emit(e):
        events.append(e)
        if on_event: on_event(e)
    for turn in range(max_turns):
        result=provider.generate(messages);text=result['text']
        emit({'type':'generation','turn':turn,'text':text,'tokens':result.get('tokens',[])})
        messages.append({'role':'assistant','content':text})
        try: action=parse_action(text)
        except (ValueError,json.JSONDecodeError) as exc:
            emit({'type':'parse_error','message':str(exc)})
            messages.append({'role':'user','content':'Invalid tool call; provide valid JSON.'});continue
        if action is None: return {'messages':messages,'events':events,'reason':'final'}
        permitted,why=policy.check(action['name'],action['arguments'])
        if not permitted: observation='BLOCKED: '+why
        else:
            try: observation=str(tools.dispatch(action['name'],action['arguments']))
            except Exception as exc: observation=f'TOOL_ERROR: {type(exc).__name__}: {exc}'
        emit({'type':'tool','name':action['name'],'arguments':action['arguments'],'observation':observation[:20000]})
        messages.append({'role':'user','content':'Tool observation:\n'+observation[:20000]})
    return {'messages':messages,'events':events,'reason':'turn_limit'}
