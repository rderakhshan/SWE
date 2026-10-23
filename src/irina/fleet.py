"""Concurrent execution of independent Irina harness instances.

Input: ``jobs`` list[dict] with name/workdir/task, ``factory`` callable
(workdir)->Harness, and ``max_workers`` int. Output: iterator[dict] with
name/ok/report. Example: ``list(run_fleet(jobs, lambda p: Harness(p), 2))``.
The caller is responsible for providing separate workspaces.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed

def run_fleet(jobs: list[dict], factory, max_workers: int = 4):
    def execute(job):
        try:
            result = factory(job['workdir']).run(job['task'])
            return {'name': job['name'], 'ok': True, 'report': result['messages'][-1]['content']}
        except Exception as exc:
            return {'name': job['name'], 'ok': False, 'report': f'{type(exc).__name__}: {exc}'}
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(execute, job) for job in jobs]
        for future in as_completed(futures): yield future.result()
