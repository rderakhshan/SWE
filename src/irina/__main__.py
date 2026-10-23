"""Run the Irina coding-agent CLI with ``python -m irina``.

Input: CLI arguments as described by ``irina.cli.main``. Output: stdout
with provider-generated responses. Example: ``python -m irina --help``
prints supported flags and exits without loading a model.
"""
from .cli import main
main()
