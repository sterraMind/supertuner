"""SuperTuner engine.

A single-GPU training runtime. Everything that touches tensors lives here; the
TypeScript side owns orchestration, persistence and UI. The contract between the
two is JSON: a run config goes in, JSONL events come out.

Import policy: nothing in this package imports torch at module import time
unless it is inside a function or a module that is only loaded by the compute
path. That keeps `supertuner-engine plan` (pure arithmetic) usable on machines
with no CUDA, and keeps the test suite fast.
"""

from .version import __version__

__all__ = ["__version__"]
