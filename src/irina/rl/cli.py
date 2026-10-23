"""CLI for Irina SWE-bench tasks, diagnostics, training, and inspection.

Inputs: ``irina-grpo doctor|tasks|run|inspect`` with typed options.
Output: JSON status or training artifacts to stdout. Example:
``irina-grpo doctor`` prints dependency and Docker availability without
loading model weights. ``run`` requires dataset, Docker and model downloads.
"""
import argparse,json
from .config import Config

def main():
    p=argparse.ArgumentParser(prog='irina-grpo')
    sub=p.add_subparsers(dest='command',required=True)
    sub.add_parser('doctor')
    tasks=sub.add_parser('tasks');tasks.add_argument('action',choices=['list','show']);tasks.add_argument('instance_id',nargs='?')
    run=sub.add_parser('run')
    run.add_argument('--mode',choices=['educational','realistic'],default='educational')
    run.add_argument('--group-size',type=int,default=4)
    run.add_argument('--max-agent-turns',type=int,default=4)
    run.add_argument('--seed',type=int,default=42)
    run.add_argument('--task-index',type=int,default=0)
    run.add_argument('--instance-id')
    run.add_argument('--model',default='Qwen/Qwen2.5-Coder-0.5B-Instruct')
    run.add_argument('--output-dir',default='runs')
    inspect=sub.add_parser('inspect');inspect.add_argument('run_id');inspect.add_argument('--output-dir',default='runs')
    args=p.parse_args()
    if args.command=='doctor':
        import importlib.util,shutil
        print(json.dumps({k:bool(shutil.which(k)) for k in ('git','docker','uv')}|
                         {k:bool(importlib.util.find_spec(k)) for k in ('torch','transformers','datasets','peft','swebench')},indent=2))
    elif args.command=='tasks':
        if args.action=='show' and args.instance_id:
            from .tasks import load_task
            print(load_task(instance_id=args.instance_id))
        elif args.action=='list':
            from datasets import load_dataset
            for row in load_dataset('princeton-nlp/SWE-bench_Verified',split='test'):print(row['instance_id'])
        else:p.error('tasks show requires instance_id')
    elif args.command=='run':
        from .pipeline import run_pipeline
        config=Config(mode=args.mode,group_size=args.group_size,max_turns=args.max_agent_turns,
                      seed=args.seed,task_index=args.task_index,instance_id=args.instance_id,
                      model_id=args.model,output_dir=args.output_dir)
        print(json.dumps(run_pipeline(config),indent=2))
    else:
        from pathlib import Path
        print((Path(args.output_dir)/args.run_id/'summary.json').read_text())

if __name__ == '__main__':
    main()
