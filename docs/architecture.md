# Irina architecture

The full explanation, Mermaid diagrams, public APIs, data types, examples, reward provenance, execution boundaries, and module guide are in the repository root [README](../README.md).

## Implementation boundaries

- `irina/`: compact provider-independent coding agent and support capabilities.
- `irina/providers/qwen.py`: local trainable policy adapter.
- `irina/rl/tasks.py` and `workspace.py`: SWE-bench task and independent repository checkout.
- `irina/rl/rewards.py`: synthetic vs. official evaluation reward selection.
- `irina/rl/training.py`: grouped advantage calculation and clipped generated-token objective.
- `irina/rl/pipeline.py`: orchestration of the single training step.

## Important provenance note

The original Odysseus Python source was not among the uploaded files. This is an independently implemented Irina architecture, not a faithful source port. See [VALIDATION](../VALIDATION.md).
