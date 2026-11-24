"""JSONL run events: the engine's only channel to the supervisor.

One JSON object per line on stdout, flushed per event so the UI can chart a run
while it is still going. The wire shapes mirror `packages/core/src/events.ts`
field for field (snake_case, epoch-millisecond `ts`); `events.py` is the Python
side of that contract, and the the contract test compares both directions.

Event types for v0.1 (docs/ARCHITECTURE.md §7): `run_start`, `phase_start`,
`step`, `eval`, `checkpoint`, `log`, `error`, `phase_end`, `run_end`.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from typing import Any, Literal, Protocol, TextIO, runtime_checkable

from .schema import MemoryPlan, PlanStep

EventLevel = Literal["debug", "info", "warn", "error"]
RunStatus = Literal["completed", "failed", "cancelled"]
PhaseStatus = Literal["completed", "failed", "cancelled"]

EVENT_TYPES: frozenset[str] = frozenset(
    {
        "run_start",
        "phase_start",
        "step",
        "eval",
        "checkpoint",
        "log",
        "error",
        "phase_end",
        "run_end",
        "memory_report",
        "oom_retry",
        "eval_report",
    }
)


def now_ms() -> int:
    """Epoch milliseconds, the `ts` every event carries."""
    return int(time.time() * 1000)


@runtime_checkable
class Emit(Protocol):
    """What a backend is allowed to emit from inside a step."""

    def log(self, level: EventLevel, message: str) -> None: ...

    def step(
        self,
        step: int,
        *,
        loss: float | None = None,
        lr: float | None = None,
        grad_norm: float | None = None,
        tokens_per_s: float | None = None,
        vram_peak_mib: float | None = None,
        elapsed_s: float | None = None,
    ) -> None: ...

    def eval(self, step: int, metrics: dict[str, float]) -> None: ...

    def checkpoint(self, step: int, path: str, bytes: int | None = None) -> None: ...

    def error(self, message: str, detail: str | None = None) -> None: ...

    def memory_report(self, report: dict[str, Any]) -> None: ...

    def eval_report(self, report: dict[str, Any]) -> None: ...

    def oom_retry(
        self,
        *,
        attempt: int,
        max_retries: int,
        reason: str,
        new_blocks_to_swap: int,
        new_quant_base: str,
        new_vram_budget_mib: int,
    ) -> None: ...


class Emitter:
    """Writes validated event objects to a text stream, one line per event."""

    def __init__(self, stream: TextIO, run_id: str, *, clock: Callable[[], int] = now_ms) -> None:
        self._stream = stream
        self.run_id = run_id
        self._clock = clock
        self._step_count = 0

    # -- lifecycle ---------------------------------------------------------

    @property
    def step_count(self) -> int:
        """Highest training step number emitted so far (0 before any step)."""
        return self._step_count

    def emit(self, event: dict[str, Any]) -> None:
        """Write one event. Unknown/forwarded events are re-serialised as-is."""
        payload = {"ts": self._clock(), **event}
        self._stream.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n")
        self._stream.flush()

    def run_start(self, plan: MemoryPlan | None = None, resumed_from_step: int | None = None) -> None:
        self.emit(
            {
                "type": "run_start",
                "run_id": self.run_id,
                "plan": None if plan is None else plan.model_dump(),
                "resumed_from_step": resumed_from_step,
            }
        )

    def run_end(self, status: RunStatus, steps: int | None = None, reason: str | None = None) -> None:
        self.emit(
            {
                "type": "run_end",
                "status": status,
                "steps": self._step_count if steps is None else steps,
                "reason": reason,
            }
        )

    def phase_start(self, step: PlanStep) -> None:
        self.emit(
            {
                "type": "phase_start",
                "step_id": step.id,
                "phase": step.kind,
                "title": step.title,
            }
        )

    def phase_end(
        self,
        step: PlanStep,
        status: PhaseStatus,
        *,
        exit_code: int | None = None,
        reason: str | None = None,
    ) -> None:
        self.emit(
            {
                "type": "phase_end",
                "step_id": step.id,
                "phase": step.kind,
                "status": status,
                "exit_code": exit_code,
                "reason": reason,
            }
        )

    # -- payload events ----------------------------------------------------

    def log(self, level: EventLevel, message: str) -> None:
        self.emit({"type": "log", "level": level, "message": message})

    def step(
        self,
        step: int,
        *,
        loss: float | None = None,
        lr: float | None = None,
        grad_norm: float | None = None,
        tokens_per_s: float | None = None,
        vram_peak_mib: float | None = None,
        elapsed_s: float | None = None,
    ) -> None:
        self._step_count = max(self._step_count, step)
        self.emit(
            {
                "type": "step",
                "step": step,
                "loss": loss,
                "lr": lr,
                "grad_norm": grad_norm,
                "tokens_per_s": tokens_per_s,
                "vram_peak_mib": vram_peak_mib,
                "elapsed_s": elapsed_s,
            }
        )

    def eval(self, step: int, metrics: dict[str, float]) -> None:
        self.emit({"type": "eval", "step": step, "metrics": {k: float(v) for k, v in metrics.items()}})

    def checkpoint(self, step: int, path: str, bytes: int | None = None) -> None:  # noqa: A002 - wire field
        self.emit({"type": "checkpoint", "step": step, "path": path, "bytes": bytes})

    def error(self, message: str, detail: str | None = None) -> None:
        self.emit({"type": "error", "message": message, "detail": detail})

    def memory_report(self, report: dict[str, Any]) -> None:
        self.emit({"type": "memory_report", **report})

    def eval_report(self, report: dict[str, Any]) -> None:
        """v0.3 P0：训练结束后的自动评估报告（稳定性/过拟合/曲线摘要）。"""
        self.emit({"type": "eval_report", **report})

    def oom_retry(
        self,
        *,
        attempt: int,
        max_retries: int,
        reason: str,
        new_blocks_to_swap: int,
        new_quant_base: str,
        new_vram_budget_mib: int,
    ) -> None:
        """OOM 有限次自动重试（v0.2 T4 / G4）：第 `attempt` 次重试即将开始，
        计划已切换到更保守的候选配置（更多 block swap / 更低量化 + 更低预算）。"""
        self.emit(
            {
                "type": "oom_retry",
                "attempt": attempt,
                "max_retries": max_retries,
                "reason": reason,
                "new_blocks_to_swap": new_blocks_to_swap,
                "new_quant_base": new_quant_base,
                "new_vram_budget_mib": new_vram_budget_mib,
            }
        )

    def count_step(self, step: int) -> None:
        """Record a step number that arrived through a forwarded subprocess event."""
        self._step_count = max(self._step_count, step)


class NullEmitter:
    """Discards events; used by compile-only paths and dry runs."""

    step_count = 0
    run_id = "null"

    def emit(self, event: dict[str, Any]) -> None:  # pragma: no cover - trivial
        return None

    def run_start(self, plan: MemoryPlan | None = None, resumed_from_step: int | None = None) -> None:
        return None

    def run_end(self, status: RunStatus, steps: int | None = None, reason: str | None = None) -> None:
        return None

    def phase_start(self, step: PlanStep) -> None:
        return None

    def phase_end(
        self, step: PlanStep, status: PhaseStatus, *, exit_code: int | None = None, reason: str | None = None
    ) -> None:
        return None

    def log(self, level: EventLevel, message: str) -> None:
        return None

    def step(self, step: int, **kwargs: Any) -> None:
        return None

    def eval(self, step: int, metrics: dict[str, float]) -> None:
        return None

    def checkpoint(self, step: int, path: str, bytes: int | None = None) -> None:  # noqa: A002
        return None

    def error(self, message: str, detail: str | None = None) -> None:
        return None

    def memory_report(self, report: dict[str, Any]) -> None:
        return None

    def eval_report(self, report: dict[str, Any]) -> None:
        return None

    def oom_retry(
        self,
        *,
        attempt: int,
        max_retries: int,
        reason: str,
        new_blocks_to_swap: int,
        new_quant_base: str,
        new_vram_budget_mib: int,
    ) -> None:
        return None

    def count_step(self, step: int) -> None:
        return None


def parse_event_line(line: str) -> dict[str, Any] | None:
    """Return the event dict when `line` is a well-formed run event, else None.

    Subprocess backends (musubi) and third-party plugins may print progress as
    events on stdout; the runtime forwards only lines that pass this check and
    treats everything else as plain log output.
    """
    text = line.strip()
    if not text.startswith("{"):
        return None
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    if payload.get("type") not in EVENT_TYPES:
        return None
    ts = payload.get("ts")
    if not isinstance(ts, int):
        return None
    return payload