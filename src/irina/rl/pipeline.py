"""End-to-end Irina SWE-bench grouped rollout and GRPO training pipeline.

Input: Config dataclass, default educational. Output: dict with run metadata,
rewards, advantages and optimizer metrics, also written as JSON artifacts.
Example: ``run_pipeline(Config(group_size=4))`` loads a real SWE-bench task,
runs four independent Qwen/Irina attempts and performs a LoRA update, provided
Docker, model weights, and dataset are available. No fake tool execution is used.
"""
import json,time,uuid
from dataclasses import asdict
from pathlib import Path
from .config import Config
from .tasks import load_task
from .workspace import checkout,collect_patch,valid_patch
from .rewards import synthetic_reward,realistic_reward,RewardResult
from .training import train_group,advantages
from irina import Harness,Policy
from irina.sandbox import DockerSandbox
from irina.providers.qwen import QwenProvider

def run_pipeline(config=None):
    config=config or Config()
    # Fail closed before downloading a model or cloning a repository.
    import shutil
    if not shutil.which('docker'):raise RuntimeError('Docker required for isolated agent command execution')
    provider=QwenProvider(config.model_id)
    task=load_task(config.task_index,config.instance_id)
    run_id=f'{int(time.time())}-{uuid.uuid4().hex[:8]}'
    output=Path(config.output_dir)/run_id;output.mkdir(parents=True,exist_ok=False)
    (output/'config.json').write_text(json.dumps(asdict(config),indent=2))
    all_records=[];rewards=[];details=[]
    for i in range(config.group_size):
        with checkout(task.repo,task.base_commit) as workdir:
            provider.generations.clear();events=[]
            agent=Harness(workdir,provider=provider,policy=Policy('yolo'),
                          sandbox=DockerSandbox(),on_event=events.append,session_enabled=False)
            result=agent.run(task.problem_statement,max_turns=config.max_turns)
            patch=collect_patch(workdir)
            if not valid_patch(patch):reward=RewardResult(0.0,'invalid_patch',config.mode)
            elif config.mode=='educational':reward=synthetic_reward(patch,config.seed,i)
            else:reward=realistic_reward(task,patch,f'{run_id}-{i}')
            (output/f'patch-{i}.diff').write_text(patch)
            (output/f'trajectory-{i}.json').write_text(json.dumps({'events':events,'messages':result['messages'],
                 'reason':result['reason'],'reward':asdict(reward)},indent=2))
            all_records.append(list(provider.generations))
            rewards.append(reward.value);details.append(asdict(reward))
    if any(r is None for r in rewards):raise RuntimeError('Cannot train on missing verifier rewards')
    training=train_group(provider,all_records,rewards)
    checkpoint=output/'lora_checkpoint';provider.model.save_pretrained(checkpoint)
    summary={'run_id':run_id,'mode':config.mode,'task':task.instance_id,
             'rewards':rewards,'reward_details':details,'advantages':advantages(rewards).tolist(),
             'training':training,'checkpoint':str(checkpoint)}
    (output/'summary.json').write_text(json.dumps(summary,indent=2))
    return summary
