# Irina validation record

## Confirmed locally

- Python source compilation.
- Unit tests in `tests/` on the available Python interpreter.
- CLI help and import smoke tests.
- Archive integrity and file inventory.

## Not confirmed in this environment

- End-to-end model-backed Qwen generation and LoRA update on a downloaded model.
- Live Hugging Face SWE-bench Verified dataset loading.
- Official SWE-bench Docker verification, including the report format of the installed version.
- Docker-backed agent tool execution against a real benchmark repository.
- Running a Colab notebook on hosted Colab hardware.
- Faithful behavior parity with original Odysseus Python source (not supplied).
- Exact crash recovery mid-tool, SSE web approvals, fully dynamic skill invocation, and multi-agent RL credit assignment.

A code path's presence is not proof that it was tested. Realistic-mode evaluation is an integration candidate that requires testing and possible adaptation to the installed official swebench version.

## Latest local checks

- 12/12 unit tests passed.
- `python -m compileall -q src` passed.
- `python -m irina --help` passed.
- `python -m irina.rl.cli doctor` passed; Docker, transformers, datasets, PEFT and swebench are unavailable here.
- `uv lock --offline` could not resolve dependencies from the local cache. No `uv.lock` is included; run `uv lock` with network access and commit the resulting lockfile before reproducible deployment.
