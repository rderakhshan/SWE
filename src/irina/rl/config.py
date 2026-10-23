"""Typed configuration for the Irina GRPO experiment.

Inputs: mode str, model_id str, group_size int, max_turns int, seed int,
task_index int, optional instance_id str, output_dir str. Output: validated
Config. Example: ``Config(mode='educational').group_size`` returns ``4``.
"""
from dataclasses import dataclass
@dataclass
class Config:
    mode:str='educational'
    model_id:str='Qwen/Qwen2.5-Coder-0.5B-Instruct'
    group_size:int=4
    max_turns:int=4
    seed:int=42
    task_index:int=0
    instance_id:str|None=None
    output_dir:str='runs'
    def __post_init__(self):
        if self.mode not in ('educational','realistic'):raise ValueError('invalid mode')
        if self.group_size<2:raise ValueError('group_size must be >= 2')
        if self.max_turns<1:raise ValueError('max_turns must be positive')
