"""Interactive and headless Irina coding-agent command-line interface.

Inputs: ``-d`` workdir, ``-p`` prompt, ``--mode`` permission mode,
``--resume`` session continuation. Output: final agent response to stdout.
Example: ``irina -p 'inspect README' -d . --mode read-only`` prints the
provider-generated answer; a configured model provider is required.
"""
import argparse
from .harness import Harness
from .security import Policy
from .sandbox import DockerSandbox

def main():
    p=argparse.ArgumentParser(description='Irina coding agent')
    p.add_argument('-d','--workdir',default='.')
    p.add_argument('-p','--prompt')
    p.add_argument('--mode',choices=['safe','yolo','read-only'],default='safe')
    p.add_argument('--max-turns',type=int,default=8)
    p.add_argument('--resume',action='store_true')
    p.add_argument('--docker',action='store_true',help='enable sandboxed bash execution')
    args=p.parse_args()
    def approve(name,arguments): return input(f'Approve {name} {arguments}? [y/N] ').lower().strip()=='y'
    harness=Harness(args.workdir,policy=Policy(args.mode,approve),
                    sandbox=DockerSandbox() if args.docker else None)
    if args.prompt or args.resume:
        result=harness.run(args.prompt or '',max_turns=args.max_turns,resume=args.resume)
        print(result['messages'][-1]['content']);return
    while True:
        try: task=input('irina> ')
        except (EOFError,KeyboardInterrupt): break
        if task.strip(): print(harness.run(task,args.max_turns)['messages'][-1]['content'])
