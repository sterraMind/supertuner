"""Execute a compiled `RunPlan` and stream its run events.

The runtime owns the process model: one engine process per run, one stdout
JSONL channel (`--events-stdout`), and one child process at a time. Step kinds
`prepare`/`cache`/`export` are always external argv; the `train` step is either
external (musubi) or in-process (native LLM backend), decided by whether the
step carries a `command`.

Exit codes (docs/ARCHITECTURE.md §7, mirrored by the server):

| code | meaning                                                    |
| ---- | ---------------------------------------------------------- |
| 0    | completed                                                   |
| 2    | plan infeasible (refused before any work)                   |
| 3    | a step failed, or the run crashed                           |
| 130  | cancelled by signal or by a `cancel` flag file in run dir   |
"""

from __future__ import annotations

import json
import os
import queue
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, TextIO

from .compiler import compile_spec
from .config import Context
from .control import Cancelled, CancelToken
from .events import Emit, Emitter, NullEmitter, parse_event_line
from .plan import resolve_config
from .plugins import Registry, load_registry
from .schema import LLM_FAMILIES, VIDEO_FAMILIES, ExecutionPlan, PlanStep, RunPlan, RunSpec

EXIT_COMPLETED = 0
EXIT_INFEASIBLE = 2
EXIT_FAILED = 3
EXIT_CANCELLED = 130

#: How long a cancelled child process gets to exit before SIGKILL.
CANCEL_GRACE_S = 10.0
#: Poll interval while waiting for child output.
_POLL_S = 0.2

#: v0.2 T4 / G4: each OOM retry tightens the VRAM ceiling by this many MiB
#: before re-running the auto-budget search, so the candidate that just OOMed
#: (whose planned peak sat under the old ceiling) can no longer be picked.
OOM_RETRY_BUDGET_STEP = 1024  # MiB

#: How conservative a quant is, by weight bytes per parameter (fewer bytes =
#: more VRAM headroom). Used only to compare configs in
#: ``_escalate_for_oom``; it never feeds the planner arithmetic.
_AUTO_CONSERVATIVE_QUANT_RANK = {
    "nf4": 0,  # 0.55 B/param
    "fp8": 1,  # 1.0 B/param
    "int8": 2,  # 1.05 B/param
    "none": 3,  # 2.0 B/param (bf16)
}

RunStatus = Literal["completed", "failed", "cancelled", "infeasible"]


@dataclass(frozen=True)
class RunOutcome:
    """What `execute` decided. The CLI maps `exit_code` straight to `sys.exit`."""

    exit_code: int
    status: RunStatus
    steps_completed: int = 0
    reason: str | None = None
    # v0.2 T4 / G4: id of the step that ran out of memory (None when the run
    # did not fail on an OOM, or the OOM came from a command step whose step
    # id the caller still knows from the plan).
    oom_step: str | None = None

    @property
    def ok(self) -> bool:
        return self.status == "completed"


class _SignalGuard:
    """Turn SIGINT/SIGTERM/SIGBREAK into a cooperative cancellation request."""

    def __init__(self, token: CancelToken) -> None:
        self.token = token
        self._previous: list[tuple[int, Any]] = []

    def __enter__(self) -> _SignalGuard:
        if threading.current_thread() is not threading.main_thread():
            return self
        names = ["SIGINT", "SIGTERM"]
        if hasattr(signal, "SIGBREAK"):  # Windows console Ctrl+Break
            names.append("SIGBREAK")
        for name in names:
            signum = getattr(signal, name, None)
            if signum is None:
                continue
            try:
                previous = signal.getsignal(signum)
                signal.signal(signum, self._handle)
            except (OSError, ValueError, RuntimeError):  # pragma: no cover - non-main thread
                continue
            self._previous.append((signum, previous))
        return self

    def _handle(self, signum: int, _frame: Any) -> None:
        name = signal.Signals(signum).name if hasattr(signal, "Signals") else str(signum)
        self.token.request(f"signal {name}")

    def __exit__(self, *_exc: object) -> None:
        for signum, previous in self._previous:
            try:
                signal.signal(signum, previous)
            except (OSError, ValueError, RuntimeError):  # pragma: no cover
                continue
        self._previous.clear()


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    """Write `payload` as JSON via a temp file + rename (crash-safe)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _pump(stream: Any, sink: queue.Queue[str | None]) -> None:
    try:
        for line in stream:
            sink.put(line)
    except (OSError, ValueError):  # pragma: no cover - stream closed mid-read
        pass
    finally:
        sink.put(None)


def _terminate(proc: subprocess.Popen[str], emit: Emit) -> None:
    if proc.poll() is not None:
        return
    emit.log("warn", "取消请求：正在终止子进程")
    try:
        proc.terminate()
        proc.wait(timeout=CANCEL_GRACE_S)
    except subprocess.TimeoutExpired:
        emit.log("warn", f"子进程 {CANCEL_GRACE_S:.0f}s 内未退出，发送 kill")
        proc.kill()
        try:
            proc.wait(timeout=CANCEL_GRACE_S)
        except subprocess.TimeoutExpired:  # pragma: no cover - unkillable child
            emit.log("error", "子进程无法终止；请手动结束该进程")
    except OSError as exc:  # pragma: no cover - already gone
        emit.log("warn", f"终止子进程时出错：{exc}")


#: Lower-cased stderr markers that identify an out-of-memory failure in an
#: external (musubi) step's output. `out of memory` covers torch's
#: "CUDA out of memory" and "Cuda error: out of memory" lines; `cuda_error`
#: covers CUDA_ERROR_OUT_OF_MEMORY / bare CUDA_ERROR codes.
OOM_STDERR_MARKERS = ("out of memory", "cuda_error")


def run_command_step(
    step: PlanStep,
    ctx: Context,
    emit: Emit,
    token: CancelToken,
    *,
    stderr: TextIO | None = None,
) -> tuple[int, bool]:
    """Run one external step and forward its stdout.

    Returns ``(returncode, oom_signature)``: the exit code is unchanged, and
    the flag is True when the step's captured stderr contains one of the
    :data:`OOM_STDERR_MARKERS` (the v0.2 T4 OOM retry uses ``rc != 0 and
    oom_signature`` to detect an out-of-memory step).

    Lines that parse as run events are forwarded verbatim (a musubi wrapper or a
    third-party plugin can report real progress); everything else becomes a
    `log` event so the engine's stdout stays pure JSONL.
    """
    assert step.command is not None, "run_command_step requires an argv"
    cwd = Path(step.cwd) if step.cwd else (ctx.run_dir or ctx.home)
    cwd.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.update(step.env or {})

    emit.log("info", f"$ {' '.join(step.command)}  (cwd={cwd})")
    proc = subprocess.Popen(  # noqa: S603 - argv list, shell=False, no interpolation
        list(step.command),
        cwd=str(cwd),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        shell=False,
    )

    sink: queue.Queue[str | None] = queue.Queue()
    err_sink: queue.Queue[str | None] = queue.Queue()
    reader = threading.Thread(target=_pump, args=(proc.stdout, sink), daemon=True)
    err_reader = threading.Thread(target=_pump, args=(proc.stderr, err_sink), daemon=True)
    reader.start()
    err_reader.start()

    oom_signature = False

    def drain_err() -> None:
        nonlocal oom_signature
        while True:
            try:
                line = err_sink.get_nowait()
            except queue.Empty:
                return
            if line is None:
                return
            text = line.rstrip()
            if not text:
                continue
            if not oom_signature and any(marker in text.lower() for marker in OOM_STDERR_MARKERS):
                oom_signature = True
            if stderr is not None:
                stderr.write(text + "\n")
                stderr.flush()
            emit.log("debug", text)

    done = False
    while not done:
        if token.poll():
            _terminate(proc, emit)
            done = True
            break
        try:
            line = sink.get(timeout=_POLL_S)
        except queue.Empty:
            drain_err()
            continue
        if line is None:
            done = True
            break
        drain_err()
        text = line.rstrip("\r\n")
        if not text.strip():
            continue
        event = parse_event_line(text)
        forward = getattr(emit, "emit", None)
        if event is not None and callable(forward):
            forward(event)
            step_number = event.get("step")
            if isinstance(step_number, int):
                counter = getattr(emit, "count_step", None)
                if callable(counter):
                    counter(step_number)
        else:
            emit.log("info", text)

    reader.join(timeout=5.0)
    err_reader.join(timeout=5.0)
    drain_err()
    returncode = proc.wait()
    if token.poll():
        raise Cancelled(token.reason or "cancelled")
    return returncode, oom_signature


def _is_oom_error(exc: BaseException) -> bool:
    """True when an in-process backend exception is a CUDA / GPU OOM.

    Matches torch's ``OutOfMemoryError`` (class name) and any exception whose
    message carries the ``out of memory`` text (torch also raises plain
    ``RuntimeError``/``CudaError`` with ``CUDA out of memory`` payloads).
    """
    if "OutOfMemory" in type(exc).__name__:
        return True
    return "out of memory" in str(exc).lower()


def _escalate_for_oom(
    spec: RunSpec,
    plan: RunPlan,
    ctx: Context,
) -> RunSpec | None:
    """Build the strictly-more-conservative spec for the next OOM retry.

    v0.2 T4 / G4. The lever is the auto-budget search: the escalated spec
    turns ``auto_budget`` on and lowers ``vram_budget_mib`` by
    :data:`OOM_RETRY_BUDGET_STEP` (bounded below by the desktop/browser
    overhead ``overlap_windows_mib``), so ``resolve_config`` ->
    ``_auto_budget_resolve`` re-runs the ``quant_base x blocks_to_swap x
    train_blocks`` search against the tighter ceiling and returns a strictly
    more conservative candidate (more block swap / lower quant) than the
    config that just OOMed.

    "Strictly more conservative" is judged against the config that just OOMed:
    the new effective config must have *more* ``blocks_to_swap``, a more
    aggressive ``quant_base``, or a strictly lower ``vram_budget_mib``. The
    budget case matters because the auto search always returns the lowest-peak
    candidate, so once the run is at that pick a further retry can only tighten
    the ceiling — still a more conservative plan, and what bounds the retry
    chain together with ``max_oom_retries``.

    Returns ``None`` (keep the OOM failure, no retry) when:

    - the family is unknown,
    - the tightened ceiling leaves no feasible candidate at all, or
    - the search returns a config that is neither more swapped, more quantised,
      nor cheaper than the one that just failed.

    The original spec is never mutated; the result is a fresh ``RunSpec``.
    """
    cfg = spec.config
    fam = LLM_FAMILIES.get(cfg.family) or VIDEO_FAMILIES.get(cfg.family)
    if fam is None:
        return None

    new_budget = max(
        plan.memory.vram_budget_mib - OOM_RETRY_BUDGET_STEP,
        cfg.overlap_windows_mib,
    )
    escalated_cfg = cfg.model_copy(
        update={
            "auto_budget": True,
            "vram_budget_mib": new_budget,
        }
    )
    try:
        eff_cfg, mem = resolve_config(escalated_cfg, spec.calibration)
    except ValueError:
        return None
    if not mem.feasible:
        return None

    cur_swap = cfg.blocks_to_swap
    new_swap = eff_cfg.blocks_to_swap
    cur_rank = _AUTO_CONSERVATIVE_QUANT_RANK.get(cfg.quant_base.value, 99)
    new_rank = _AUTO_CONSERVATIVE_QUANT_RANK.get(eff_cfg.quant_base.value, 99)
    # A strictly lower ceiling counts as more conservative too: the auto search
    # always returns the lowest-peak candidate, so once the run is at that pick
    # a further retry can only tighten the budget (same pick, lower ceiling).
    strictly_more_conservative = (
        (new_swap > cur_swap)
        or (new_rank < cur_rank)
        or (eff_cfg.vram_budget_mib < cfg.vram_budget_mib)
    )
    if not strictly_more_conservative:
        return None

    return spec.model_copy(update={"config": eff_cfg})


def _execute_plan(
    compiled: RunPlan,
    spec: RunSpec,
    ctx: Context,
    registry: Registry,
    emit: Emit,
    *,
    stderr: TextIO | None = None,
) -> tuple[RunOutcome, bool]:
    """Run every step of one compiled plan; return ``(outcome, oomed)``.

    ``oomed`` is True iff the run failed because a step ran out of memory:
    an in-process backend raised an OOM exception, or a command step exited
    non-zero with an OOM signature in its captured stderr. The caller
    (``execute``) uses it to decide whether a retry with a more conservative
    config is allowed (v0.2 T4 / G4).

    For a run that does not retry, the emitted events and exit semantics are
    exactly the pre-retry behaviour: ``run_start`` brackets the steps,
    infeasible plans still fail before any step, cancellation wins at every
    poll point, and optional-step tolerance is unchanged.
    """
    ctx.ensure_run_dirs()
    if ctx.run_dir is not None:
        plan_path = ctx.run_dir / "plan.json"
        if not plan_path.exists():
            write_json_atomic(plan_path, compiled.model_dump(mode="json"))
        spec_path = ctx.run_dir / "spec.json"
        if not spec_path.exists():
            write_json_atomic(spec_path, spec.model_dump(mode="json"))

    resumed_from_step = 0 if spec.train.resume_from else None
    emit.run_start(compiled.memory, resumed_from_step)

    if not compiled.feasible:
        errors = [i for i in compiled.issues if i.severity == "error"]
        for item in errors:
            detail = item.hint or None
            emit.error(f"{item.code}: {item.message}", detail)
        reason = "plan infeasible; see issues"
        emit.run_end("failed", None, reason)
        return RunOutcome(EXIT_INFEASIBLE, "infeasible", 0, reason), False

    backend = registry.backend(compiled.execution.backend) if compiled.execution.backend else None
    steps = compiled.execution.steps
    completed = 0

    with _SignalGuard(ctx.cancel):
        for step in steps:
            if ctx.cancel.poll():
                emit.run_end("cancelled", None, ctx.cancel.reason or "cancelled")
                return RunOutcome(EXIT_CANCELLED, "cancelled", completed, ctx.cancel.reason), False

            emit.phase_start(step)
            started = _now()

            def fail(
                reason: str,
                *,
                code: int = EXIT_FAILED,
                step: PlanStep = step,
                done: int = completed,
                oom: bool = False,
            ) -> RunOutcome:
                emit.error(reason)
                emit.phase_end(step, "failed", exit_code=code)
                emit.run_end("failed", None, reason)
                return RunOutcome(code, "failed", done, reason, oom_step=step.id if oom else None)

            step_oomed = False
            try:
                if step.command is not None:
                    rc, oom_signature = run_command_step(step, ctx, emit, ctx.cancel, stderr=stderr)
                    step_oomed = rc != 0 and oom_signature
                elif backend is not None:
                    result = backend.run_step(spec, step, ctx, emit)
                    rc = 0 if result is None else int(result)
                else:
                    rc = 1
                    emit.error(f"step '{step.id}' 需要进程内执行，但 backend 不可用")

            except Cancelled as exc:
                emit.phase_end(step, "cancelled", exit_code=EXIT_CANCELLED, reason=str(exc))
                emit.run_end("cancelled", None, str(exc))
                return RunOutcome(EXIT_CANCELLED, "cancelled", completed, str(exc)), False

            except Exception as exc:  # noqa: BLE001 - any backend bug becomes a run failure
                import traceback

                detail = traceback.format_exc()
                emit.log("error", detail)
                step_oomed = _is_oom_error(exc)
                return fail(f"step '{step.id}' 抛出 {type(exc).__name__}: {exc}", oom=step_oomed), step_oomed

            elapsed = _now() - started
            if rc != 0:
                if step.optional:
                    emit.log("warn", f"可选步骤 '{step.id}' 失败（exit {rc}），继续执行")
                    emit.phase_end(step, "completed", exit_code=rc, reason="optional step failed")
                    continue
                return fail(f"step '{step.id}' 失败，exit code {rc}", oom=step_oomed), step_oomed

            emit.phase_end(step, "completed", exit_code=rc)
            emit.log("debug", f"step '{step.id}' 完成，用时 {elapsed:.1f}s")
            completed += 1

    if ctx.cancel.poll():
        emit.run_end("cancelled", None, ctx.cancel.reason or "cancelled")
        return RunOutcome(EXIT_CANCELLED, "cancelled", completed, ctx.cancel.reason), False

    emit.run_end("completed", None, f"{completed}/{len(steps)} steps")
    return RunOutcome(EXIT_COMPLETED, "completed", completed), False


def execute(
    spec: RunSpec,
    ctx: Context,
    emit: Emit | None = None,
    *,
    registry: Registry | None = None,
    plan: RunPlan | None = None,
    dry_run: bool = False,
    stderr: TextIO | None = None,
    run_id: str | None = None,
) -> RunOutcome:
    """Compile (unless `plan` is given), then execute `plan` step by step.

    OOM bounded retry (v0.2 T4 / G4): when a step fails with an out-of-memory
    error, the whole plan is re-executed with a strictly more conservative
    config (see :func:`_escalate_for_oom`), at most
    ``spec.config.max_oom_retries`` times (default 2; 0 disables the retry).
    Each retry emits an ``oom_retry`` event and a fresh ``run_start`` for the
    escalated plan so the UI can show "retry N with more swap". A non-OOM
    failure, a cancelled run, or an infeasible plan never retries.
    """
    reg = registry if registry is not None else load_registry()
    first_plan = plan if plan is not None else compile_spec(spec, ctx, reg)

    if dry_run:
        return RunOutcome(
            exit_code=EXIT_COMPLETED if first_plan.feasible else EXIT_INFEASIBLE,
            status="completed" if first_plan.feasible else "infeasible",
            reason=None if first_plan.feasible else "plan infeasible",
        )

    if emit is None:
        emit = Emitter(sys.stdout, run_id or _default_run_id(ctx))

    max_retries = max(spec.config.max_oom_retries, 0)
    current_spec = spec
    current_plan = first_plan
    attempt = 0
    while True:
        outcome, oomed = _execute_plan(current_plan, current_spec, ctx, reg, emit, stderr=stderr)
        if not oomed or attempt >= max_retries:
            return outcome
        escalated = _escalate_for_oom(current_spec, current_plan, ctx)
        if escalated is None or escalated == current_spec:
            # No strictly more conservative config exists (already at the
            # heaviest quant with maximum swap, or nothing fits under the
            # tightened budget): keep the OOM failure as-is.
            return outcome
        emit.oom_retry(
            attempt=attempt + 1,
            max_retries=max_retries,
            reason=f"step '{outcome.oom_step}' CUDA OOM" if outcome.oom_step else "CUDA OOM",
            new_blocks_to_swap=escalated.config.blocks_to_swap,
            new_quant_base=escalated.config.quant_base.value,
            new_vram_budget_mib=escalated.config.vram_budget_mib,
        )
        current_spec = escalated
        current_plan = compile_spec(current_spec, ctx, reg)
        attempt += 1


def _now() -> float:
    return time.monotonic()


def _default_run_id(ctx: Context) -> str:
    if ctx.run_dir is not None:
        return ctx.run_dir.name
    return "adhoc"


def plan_for(
    spec: RunSpec,
    ctx: Context,
    registry: Registry | None = None,
    measured_vram_mib: int | None = None,
    measured_ram_mib: int | None = None,
) -> RunPlan:
    """Compile without executing; exposed for `run --dry-run` and tests.

    实测进预算（v0.2 T2 / G3 = AP-01）：两个实测值原样透传给 compile_spec；
    调用方不传时 compile_spec 按 None 处理，行为与改动前一致。
    """
    return compile_spec(
        spec,
        ctx,
        registry,
        measured_vram_mib=measured_vram_mib,
        measured_ram_mib=measured_ram_mib,
    )


def planned_commands(plan: ExecutionPlan) -> list[list[str]]:
    """The exact argv of every external step, for dry-run display and tests."""
    return [list(step.command) for step in plan.steps if step.command]


__all__ = [
    "EXIT_CANCELLED",
    "EXIT_COMPLETED",
    "EXIT_FAILED",
    "EXIT_INFEASIBLE",
    "NullEmitter",
    "RunOutcome",
    "execute",
    "plan_for",
    "planned_commands",
    "run_command_step",
    "write_json_atomic",
]