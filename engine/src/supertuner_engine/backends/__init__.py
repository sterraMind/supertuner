"""Built-in training backends.

Importing this package is cheap: it only re-exports the classes. The heavy
training dependencies (torch, transformers, peft, bitsandbytes) are imported
inside `run_step` so that `compile`, `plan` and `plugins` work on machines that
have not installed the training stack yet.
"""

from __future__ import annotations

from .musubi import MusubiBackend
from .native_llm import NativeLLMBackend

__all__ = ["MusubiBackend", "NativeLLMBackend"]