"""Load official SWE-bench Verified task metadata without reference patches.

Inputs: dataset index int or instance_id str. Output: SWETask dataclass
containing repository, base commit, issue, test patch and test identifiers.
Example: ``load_task(index=0).instance_id`` returns a benchmark task ID after
Hugging Face dataset access. The reference solution patch is never exposed.
"""
import json
from dataclasses import dataclass
@dataclass
class SWETask:
    instance_id:str
    repo:str
    base_commit:str
    problem_statement:str
    test_patch:str
    fail_to_pass:list
    pass_to_pass:list

def load_task(index=0,instance_id=None):
    from datasets import load_dataset
    ds=load_dataset('princeton-nlp/SWE-bench_Verified',split='test')
    if instance_id:
        rows=(row for row in ds if row['instance_id']==instance_id)
        row=next(rows,None)
        if row is None:raise KeyError(instance_id)
    else:row=ds[index]
    def parse(v):return json.loads(v) if isinstance(v,str) else list(v)
    return SWETask(row['instance_id'],row['repo'],row['base_commit'],row['problem_statement'],
                   row['test_patch'],parse(row['FAIL_TO_PASS']),parse(row['PASS_TO_PASS']))
