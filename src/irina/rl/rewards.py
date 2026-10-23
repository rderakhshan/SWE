"""Educational synthetic and official SWE-bench verifier reward backends.

Inputs: candidate unified diff str, task metadata, seed/rollout ID.
Outputs: RewardResult(value:float|None,status:str,provenance:str,details:str).
Example: ``synthetic_reward('diff --git ...',42,0)`` produces a reproducible
number in [0,1). Realistic mode invokes the official swebench harness using
Docker and parses its report; verifier errors never become synthetic rewards.
"""
import hashlib,json,random,tempfile
from dataclasses import dataclass
from pathlib import Path

@dataclass
class RewardResult:
    value:float|None
    status:str
    provenance:str
    details:str=''

def synthetic_reward(patch,seed,rollout_id):
    if not patch.strip():return RewardResult(0.0,'invalid_patch','synthetic')
    digest=hashlib.sha256(f'{seed}:{rollout_id}'.encode()).digest()
    return RewardResult(random.Random(int.from_bytes(digest[:8],'big')).random(),'ok','synthetic')

def realistic_reward(task,patch,run_id):
    if not patch.strip():return RewardResult(0.0,'invalid_patch','swebench')
    try:
        from swebench.harness.run_evaluation import main as evaluate
    except ImportError as exc:
        raise RuntimeError('Install optional realistic dependencies: uv sync --extra realistic') from exc
    import shutil
    if not shutil.which('docker'):raise RuntimeError('Docker is required for official SWE-bench evaluation')
    with tempfile.TemporaryDirectory(prefix='irina-eval-') as tmp:
        root=Path(tmp)
        predictions=root/'predictions.jsonl'
        predictions.write_text(json.dumps({'instance_id':task.instance_id,'model_name_or_path':'irina-qwen',
                                          'model_patch':patch})+'\n')
        # The official harness owns clean checkout, patch application and test evaluation.
        try:
            evaluate(dataset_name='princeton-nlp/SWE-bench_Verified',split='test',
                     instance_ids=[task.instance_id],predictions_path=str(predictions),
                     max_workers=1,run_id=run_id,timeout=1800,cache_level='env',
                     clean=False,force_rebuild=False)
        except Exception as exc:
            raise RuntimeError(f'SWE-bench verifier failed: {exc}') from exc
        # Report location varies with swebench version; find only this run's report.
        reports=list(Path('logs').rglob(f'*{run_id}*/*.json')) if Path('logs').exists() else []
        reports += list(Path('logs').rglob('report.json')) if Path('logs').exists() else []
        for report in reports:
            try: data=json.loads(report.read_text())
            except (ValueError,OSError):continue
            resolved=data.get('resolved_ids') or data.get('resolved') or []
            if isinstance(resolved,list):
                if task.instance_id in resolved:return RewardResult(1.0,'resolved','swebench',str(report))
                # Only a report explicitly listing this instance as evaluated is conclusive.
                evaluated=set(resolved)|set(data.get('unresolved_ids',[]))|set(data.get('error_ids',[]))
                if task.instance_id in data.get('unresolved_ids',[]):return RewardResult(0.0,'unresolved','swebench',str(report))
        raise RuntimeError('SWE-bench evaluation finished but a trustworthy instance result was not found; inspect official logs')
