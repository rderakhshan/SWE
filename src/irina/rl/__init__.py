"""Public entry point for SWE-bench and GRPO experiments.

Input: Config or keyword-free defaults. Output: run_pipeline returns a summary
dict and writes artifacts to runs/. Example: ``run_pipeline(Config())`` runs
educational mode with a real Qwen policy, Docker sandbox, and SWE-bench task.
"""
from .config import Config
from .pipeline import run_pipeline
__all__=['Config','run_pipeline']
