"""T4 / G4: bounded OOM auto-retry.

A training step that the memory planner believed would fit (but whose real
peak exceeded the budget) now triggers a bounded retry with a strictly more
conservative config (the auto-budget search re-run against a lowered ceiling),
emitting one ``oom_retry`` event per retry.

The fake backend below raises a fake ``OutOfMemoryError`` a fixed number of
times and then succeeds, so the retry loop can be exercised end-to-end
(``execute`` -> ``compile_spec`` -> ``_execute_plan`` -> ``_escalate_for_oom``)
without a GPU.
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

from supertuner_engine.config import Context, EngineConfig
from supertuner_engine.events import Emitter, parse_event_line
from supertuner_engine.plugins import load_registry
from supertuner_engine.runtime import (
    EXIT_COMPLETED,
    EXIT_FAILED,
    RunOutcome,
    execute,
    plan_for,
)
from supertuner_engine.schema import (
    DatasetKind,
    DatasetSpec,
    ExecutionPlan,
    PluginManifest,
    PlanStep,
    QuantBase,
    RunConfig,
    RunSpec,
    Stage,
)

FAMILY = "wan2.1-t2v-14b"  # video, block-swap capable -> auto-budget search runs
# Budget high enough that the auto search's cheapest candidate stays feasible
# across several 1024-MiB escalations (see the feasibility ladder in plan.py).
FEASIBLE_BUDGET_MIB = 18432


class FakeOutOfMemoryError(RuntimeError):
    """Stand-in for torch's ``OutOfMemoryError`` / a CUDA out-of-memory."""


class FlakyOOMBackend:
    """In-process backend whose ``run_step`` OOMs ``oom_calls`` times, then runs.

    ``oom_calls == -1`` means "always OOM". ``calls`` counts ``run_step``
    invocations so tests can assert how many attempts actually ran;
    ``seen_configs`` records the quant of each attempt's effective config.
    """

    def __init__(self, oom_calls: int, manifest: PluginManifest) -> None:
        self.oom_calls = oom_calls
        self.manifest = manifest
        self.calls = 0
        self.seen_configs: list[QuantBase] = []

    def probe(self, ctx: Context) -> list:
        return []

    def supports(self, spec: RunSpec) -> bool:
        return spec.config.family in self.manifest.families

    def compile(self, spec: RunSpec, ctx: Context) -> ExecutionPlan:
        step = PlanStep(
            id="train",
            kind="train",
            title=f"{spec.config.family} {spec.config.stage.value}",
            command=None,  # in-process: executed via run_step
        )
        return ExecutionPlan(
            backend=self.manifest.id,
            backend_version=self.manifest.version,
            steps=[step],
        )

    def run(self, spec: RunSpec, plan: ExecutionPlan, ctx: Context, emit) -> None:
        for step in plan.steps:
            self.run_step(spec, step, ctx, emit)

    def run_step(self, spec: RunSpec, step: PlanStep, ctx: Context, emit) -> None:
        self.calls += 1
        self.seen_configs.append(spec.config.quant_base)
        if self.oom_calls == -1 or self.calls <= self.oom_calls:
            raise FakeOutOfMemoryError("CUDA out of memory (fake)")


def _backend_manifest() -> PluginManifest:
    return PluginManifest(
        id="fake-oom",
        name="fake-oom",
        version="1.0.0",
        kind="backend",
        capabilities=["train.video.lora"],
        families=[FAMILY],
        priority=200,  # beats native-llm (100) and musubi (90)
    )


def _ctx(tmp_path: Path) -> Context:
    config = EngineConfig.from_env(
        {
            "SUPERTUNER_HOME": str(tmp_path / "home"),
            "SUPERTUNER_MODELS_DIR": str(tmp_path / "models"),
            "SUPERTUNER_DATASETS_DIR": str(tmp_path / "datasets"),
            "SUPERTUNER_CACHE_DIR": str(tmp_path / "cache"),
        }
    )
    return config.context(tmp_path / "run")


def _spec(backend: FlakyOOMBackend, *, max_oom_retries: int = 2) -> RunSpec:
    return RunSpec(
        config=RunConfig(
            name="oom-retry",
            family=FAMILY,
            stage=Stage.SFT,
            adapter="lora",
            auto_budget=True,
            vram_budget_mib=FEASIBLE_BUDGET_MIB,
            max_oom_retries=max_oom_retries,
        ),
        dataset=DatasetSpec(path="clips", kind=DatasetKind.VIDEO, format="directory"),
    )


def _registry(backend: FlakyOOMBackend):
    return load_registry(entry_points_fn=lambda group: [], extra_backends=[backend])


def _events(stream: io.StringIO) -> list[dict]:
    return [
        parsed
        for line in stream.getvalue().splitlines()
        if (parsed := parse_event_line(line))
    ]


def _oom_retry_events(stream: io.StringIO) -> list[dict]:
    return [e for e in _events(stream) if e.get("type") == "oom_retry"]


def _run(tmp_path: Path, spec: RunSpec, backend: FlakyOOMBackend, stream: io.StringIO):
    ctx = _ctx(tmp_path)
    registry = _registry(backend)
    emit = Emitter(stream, "oom")
    return execute(spec, ctx, emit, registry=registry)


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------


def test_oom_then_success_retries_with_escalated_config(tmp_path):
    """One OOM -> one retry with a strictly more conservative config -> success."""
    backend = FlakyOOMBackend(oom_calls=1, manifest=_backend_manifest())
    spec = _spec(backend, max_oom_retries=2)
    stream = io.StringIO()

    outcome = _run(tmp_path, spec, backend, stream=stream)

    assert isinstance(outcome, RunOutcome)
    assert outcome.exit_code == EXIT_COMPLETED, outcome
    assert outcome.status == "completed"
    assert backend.calls == 2  # initial attempt + one retry

    retries = _oom_retry_events(stream)
    assert len(retries) == 1
    event = retries[0]
    assert event["attempt"] == 1
    assert event["max_retries"] == 2
    # Escalated to a strictly heavier config than the original request (swap 0).
    assert event["new_blocks_to_swap"] > 0
    assert event["new_quant_base"] != QuantBase.NONE.value
    assert event["new_vram_budget_mib"] < FEASIBLE_BUDGET_MIB
    # The second attempt actually ran a different (more aggressive) quant.
    assert backend.seen_configs[1] != QuantBase.NONE


def test_always_oom_fails_after_max_retries(tmp_path):
    """max_oom_retries=2 with a backend that never recovers -> exactly 2 retries."""
    backend = FlakyOOMBackend(oom_calls=-1, manifest=_backend_manifest())
    spec = _spec(backend, max_oom_retries=2)
    stream = io.StringIO()

    outcome = _run(tmp_path, spec, backend, stream=stream)

    assert outcome.exit_code == EXIT_FAILED
    assert outcome.status == "failed"
    assert outcome.oom_step == "train"
    assert outcome.reason is not None
    assert "out of memory" in outcome.reason.lower() or "OutOfMemory" in outcome.reason

    retries = _oom_retry_events(stream)
    assert [e["attempt"] for e in retries] == [1, 2]
    assert backend.calls == 3  # initial attempt + two retries


def test_retry_chain_stops_when_no_feasible_candidate_remains(tmp_path):
    """A large retry budget still terminates: the ceiling hits the floor."""
    backend = FlakyOOMBackend(oom_calls=-1, manifest=_backend_manifest())
    spec = _spec(backend, max_oom_retries=8)
    stream = io.StringIO()

    outcome = _run(tmp_path, spec, backend, stream=stream)

    assert outcome.exit_code == EXIT_FAILED
    retries = _oom_retry_events(stream)
    # Each retry lowers the ceiling by 1024 MiB; below ~14.5 GiB nothing
    # feasible remains for this family, so the chain stops well before 8.
    assert 1 <= len(retries) < 8
    assert [e["attempt"] for e in retries] == list(range(1, len(retries) + 1))


def test_max_oom_retries_zero_disables_retry(tmp_path):
    backend = FlakyOOMBackend(oom_calls=-1, manifest=_backend_manifest())
    spec = _spec(backend, max_oom_retries=0)
    stream = io.StringIO()

    outcome = _run(tmp_path, spec, backend, stream=stream)

    assert outcome.exit_code == EXIT_FAILED
    assert outcome.status == "failed"
    assert backend.calls == 1  # no retry at all
    assert _oom_retry_events(stream) == []


def test_non_oom_failure_does_not_retry(tmp_path):
    """A generic backend error is not an OOM: no retry, no oom_retry event."""
    backend = FlakyOOMBackend(oom_calls=-1, manifest=_backend_manifest())

    def run_step(spec, step, ctx, emit):
        backend.calls += 1
        raise RuntimeError("loss exploded: NaN")

    backend.run_step = run_step  # type: ignore[method-assign]
    spec = _spec(backend, max_oom_retries=2)
    stream = io.StringIO()

    outcome = _run(tmp_path, spec, backend, stream=stream)

    assert outcome.exit_code == EXIT_FAILED
    assert outcome.oom_step is None
    assert "OutOfMemory" not in (outcome.reason or "")
    assert backend.calls == 1
    assert _oom_retry_events(stream) == []


def test_oom_signature_from_stderr_in_command_step(tmp_path):
    """A command step exiting non-zero with an OOM marker is treated as OOM."""
    backend = FlakyOOMBackend(oom_calls=-1, manifest=_backend_manifest())
    spec = _spec(backend, max_oom_retries=1)
    ctx = _ctx(tmp_path)
    ctx.ensure_run_dirs()
    registry = _registry(backend)

    plan = plan_for(spec, ctx, registry)
    plan.execution.steps = [
        PlanStep(
            id="train",
            kind="train",
            title="oom-cmd",
            command=[
                sys.executable,
                "-c",
                "import sys; print('CUDA out of memory', file=sys.stderr); sys.exit(1)",
            ],
        )
    ]

    stream = io.StringIO()
    outcome = execute(spec, ctx, Emitter(stream, "oom"), registry=registry, plan=plan)

    assert outcome.oom_step == "train"
    retries = _oom_retry_events(stream)
    assert [e["attempt"] for e in retries] == [1]


def test_run_config_max_oom_retries_defaults_to_two():
    config = RunConfig(name="t", family=FAMILY, stage=Stage.SFT)
    assert config.max_oom_retries == 2
