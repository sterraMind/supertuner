"""Runtime dry-run and step-execution tests. No GPU, no torch, no weights."""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

from supertuner_engine.cli import app
from supertuner_engine.config import Context, EngineConfig
from supertuner_engine.events import Emitter, parse_event_line
from supertuner_engine.plan import plan as memory_plan
from supertuner_engine.runtime import (
    EXIT_CANCELLED,
    EXIT_COMPLETED,
    EXIT_FAILED,
    EXIT_INFEASIBLE,
    RunOutcome,
    execute,
)
from supertuner_engine.schema import (
    DatasetKind,
    DatasetSpec,
    ExecutionPlan,
    PlanStep,
    RunConfig,
    RunPlan,
    RunSpec,
    Stage,
)

PY = sys.executable


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


def _spec(**config_kw) -> RunSpec:
    config_kw.setdefault("name", "runtime")
    config_kw.setdefault("family", "qwen3-1.7b")
    config_kw.setdefault("stage", Stage.SFT)
    return RunSpec(
        config=RunConfig(**config_kw),
        dataset=DatasetSpec(path="demo.jsonl", kind=DatasetKind.TEXT, format="jsonl"),
    )


def _plan(steps: list[PlanStep], *, feasible: bool = True, backend: str = "") -> RunPlan:
    cfg = RunConfig(name="runtime", family="qwen3-1.7b", stage=Stage.SFT)
    return RunPlan(
        feasible=feasible,
        memory=memory_plan(cfg),
        execution=ExecutionPlan(
            backend=backend,
            backend_version="test",
            steps=steps,
            issues=[],
            notes=[],
            expected_artifacts=[],
        ),
        issues=[],
    )


def _events(stream: io.StringIO) -> list[dict]:
    return [parsed for line in stream.getvalue().splitlines() if (parsed := parse_event_line(line))]


def _print_event(message: str) -> list[str]:
    code = (
        "import json;"
        f"print(json.dumps({{'ts': 7, 'type': 'log', 'level': 'info', 'message': {message!r}}}))"
    )
    return [PY, "-c", code]


def test_dry_run_feasible_returns_zero_and_emits_nothing(tmp_path):
    ctx = _ctx(tmp_path)
    stream = io.StringIO()
    emit = Emitter(stream, "run")
    plan = _plan([PlanStep(id="train", kind="train", title="t", command=None)])

    outcome = execute(_spec(), ctx, emit, plan=plan, dry_run=True)

    assert isinstance(outcome, RunOutcome)
    assert outcome.exit_code == EXIT_COMPLETED
    assert stream.getvalue() == ""
    # A dry run must not create the run layout either.
    assert not (ctx.run_dir / "plan.json").exists()


def test_dry_run_infeasible_returns_two(tmp_path):
    ctx = _ctx(tmp_path)
    outcome = execute(_spec(), ctx, Emitter(io.StringIO(), "run"), plan=_plan([], feasible=False), dry_run=True)
    assert outcome.exit_code == EXIT_INFEASIBLE


def test_command_step_forwards_child_events(tmp_path):
    ctx = _ctx(tmp_path)
    ctx.ensure_run_dirs()
    stream = io.StringIO()
    emit = Emitter(stream, "run")
    step = PlanStep(
        id="cache-text",
        kind="cache",
        title="缓存",
        command=_print_event("from-child"),
    )
    plan = _plan([step])

    outcome = execute(_spec(), ctx, emit, plan=plan)

    assert outcome.exit_code == EXIT_COMPLETED
    assert outcome.steps_completed == 1
    events = _events(stream)
    types = [event["type"] for event in events]
    assert types[0] == "run_start"
    assert types[-1] == "run_end"
    assert "phase_start" in types and "phase_end" in types
    assert events[-1]["status"] == "completed"

    forwarded = [event for event in events if event.get("message") == "from-child"]
    assert forwarded and forwarded[0]["ts"] == 7  # child timestamp preserved
    phase_end = next(event for event in events if event["type"] == "phase_end")
    assert phase_end["step_id"] == "cache-text"
    assert phase_end["status"] == "completed"


def test_phase_events_bracket_each_step_in_order(tmp_path):
    ctx = _ctx(tmp_path)
    ctx.ensure_run_dirs()
    stream = io.StringIO()
    steps = [
        PlanStep(id="a", kind="prepare", title="A", command=[PY, "-c", "pass"]),
        PlanStep(id="b", kind="train", title="B", command=[PY, "-c", "pass"]),
    ]
    outcome = execute(_spec(), ctx, Emitter(stream, "run"), plan=_plan(steps))
    assert outcome.exit_code == EXIT_COMPLETED

    events = _events(stream)
    phases = [event for event in events if event["type"] in {"phase_start", "phase_end"}]
    assert [(event["type"], event["step_id"]) for event in phases] == [
        ("phase_start", "a"),
        ("phase_end", "a"),
        ("phase_start", "b"),
        ("phase_end", "b"),
    ]


def test_child_env_and_cwd_are_honoured(tmp_path):
    ctx = _ctx(tmp_path)
    ctx.ensure_run_dirs()
    workdir = ctx.run_dir / "work" / "inner"
    workdir.mkdir(parents=True)
    code = "import os; print(os.environ['SUPERTUNER_TEST_VAR'] + '|' + os.getcwd())"
    step = PlanStep(
        id="train",
        kind="train",
        title="env",
        command=[PY, "-c", code],
        cwd=str(workdir),
        env={"SUPERTUNER_TEST_VAR": "hello"},
    )
    stream = io.StringIO()
    outcome = execute(_spec(), ctx, Emitter(stream, "run"), plan=_plan([step]))
    assert outcome.exit_code == EXIT_COMPLETED
    logs = [event["message"] for event in _events(stream) if event["type"] == "log"]
    assert any(message.startswith("hello|") for message in logs)


def test_failing_step_returns_three(tmp_path):
    ctx = _ctx(tmp_path)
    ctx.ensure_run_dirs()
    stream = io.StringIO()
    step = PlanStep(id="train", kind="train", title="boom", command=[PY, "-c", "import sys; sys.exit(4)"])
    outcome = execute(_spec(), ctx, Emitter(stream, "run"), plan=_plan([step]))

    assert outcome.exit_code == EXIT_FAILED
    events = _events(stream)
    assert events[-1]["type"] == "run_end"
    assert events[-1]["status"] == "failed"
    assert any(event["type"] == "error" for event in events)


def test_optional_step_failure_continues(tmp_path):
    ctx = _ctx(tmp_path)
    ctx.ensure_run_dirs()
    stream = io.StringIO()
    steps = [
        PlanStep(id="opt", kind="cache", title="optional", command=[PY, "-c", "import sys; sys.exit(1)"], optional=True),
        PlanStep(id="train", kind="train", title="train", command=[PY, "-c", "pass"]),
    ]
    outcome = execute(_spec(), ctx, Emitter(stream, "run"), plan=_plan(steps))

    assert outcome.exit_code == EXIT_COMPLETED
    assert outcome.steps_completed == 1
    assert any(event["type"] == "log" and event["level"] == "warn" for event in _events(stream))


def test_cancel_flag_file_stops_before_first_step(tmp_path):
    ctx = _ctx(tmp_path)
    ctx.ensure_run_dirs()
    (ctx.run_dir / "cancel").write_text("stop\n", encoding="utf-8")
    stream = io.StringIO()
    step = PlanStep(id="train", kind="train", title="t", command=[PY, "-c", "raise SystemExit(9)"])
    outcome = execute(_spec(), ctx, Emitter(stream, "run"), plan=_plan([step]))

    assert outcome.exit_code == EXIT_CANCELLED
    events = _events(stream)
    assert events[-1]["type"] == "run_end"
    assert events[-1]["status"] == "cancelled"


def test_cancel_flag_named_cancel_flag_also_works(tmp_path):
    ctx = _ctx(tmp_path)
    ctx.ensure_run_dirs()
    (ctx.run_dir / "cancel.flag").write_text("stop\n", encoding="utf-8")
    outcome = execute(_spec(), ctx, Emitter(io.StringIO(), "run"), plan=_plan([]))
    assert outcome.exit_code == EXIT_CANCELLED


def test_engine_writes_plan_and_spec_but_not_events_jsonl(tmp_path):
    ctx = _ctx(tmp_path)
    ctx.ensure_run_dirs()
    step = PlanStep(id="train", kind="train", title="t", command=[PY, "-c", "pass"])
    outcome = execute(_spec(), ctx, Emitter(io.StringIO(), "run"), plan=_plan([step]))

    assert outcome.exit_code == EXIT_COMPLETED
    assert (ctx.run_dir / "plan.json").is_file()
    assert (ctx.run_dir / "spec.json").is_file()
    # The server owns events.jsonl; the engine is the only writer of stdout.
    assert not (ctx.run_dir / "events.jsonl").exists()


def test_infeasible_plan_emits_error_events_and_exits_two(tmp_path):
    ctx = _ctx(tmp_path)
    from supertuner_engine.schema import Issue

    plan = _plan([], feasible=False)
    plan.issues = [Issue(severity="error", code="E_MEMORY_BUDGET", message="放不下", hint="降分辨率")]
    stream = io.StringIO()
    outcome = execute(_spec(), ctx, Emitter(stream, "run"), plan=plan)

    assert outcome.exit_code == EXIT_INFEASIBLE
    events = _events(stream)
    assert any(event["type"] == "error" and "E_MEMORY_BUDGET" in event["message"] for event in events)
    assert events[-1]["status"] == "failed"


def test_missing_in_process_backend_fails_loudly(tmp_path):
    ctx = _ctx(tmp_path)
    ctx.ensure_run_dirs()
    stream = io.StringIO()
    step = PlanStep(id="train", kind="train", title="native", command=None)
    outcome = execute(_spec(), ctx, Emitter(stream, "run"), plan=_plan([step], backend="native-llm"))

    assert outcome.exit_code == EXIT_FAILED
    assert any(event["type"] == "error" for event in _events(stream))


# -- CLI exit codes -------------------------------------------------------


@pytest.fixture()
def cli_env(tmp_path, monkeypatch):
    for name, value in {
        "SUPERTUNER_HOME": str(tmp_path / "home"),
        "SUPERTUNER_MODELS_DIR": str(tmp_path / "models"),
        "SUPERTUNER_DATASETS_DIR": str(tmp_path / "datasets"),
        "SUPERTUNER_CACHE_DIR": str(tmp_path / "cache"),
        "SUPERTUNER_MUSUBI_PATH": "",
    }.items():
        monkeypatch.setenv(name, value)
    return tmp_path


def _write_spec(tmp_path: Path, payload: dict) -> Path:
    path = tmp_path / "spec.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_cli_compile_exit_zero_for_feasible_spec(cli_env):
    spec_path = _write_spec(
        cli_env,
        {
            "schema_version": 1,
            "config": {"name": "ok", "family": "qwen3-1.7b", "stage": "sft", "adapter": "lora"},
            "dataset": {"path": "demo.jsonl", "kind": "text", "format": "jsonl"},
        },
    )
    result = CliRunner().invoke(app, ["compile", "--spec", str(spec_path), "--json"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["ok"] is True
    assert payload["plan"]["execution"]["backend"] == "native-llm"


def test_cli_compile_exit_two_when_infeasible(cli_env):
    spec_path = _write_spec(
        cli_env,
        {
            "schema_version": 1,
            "config": {
                "name": "too-big",
                "family": "wan2.1-t2v-14b",
                "stage": "sft",
                "adapter": "none",
                "width": 832,
                "height": 480,
                "frames": 49,
                "vram_budget_mib": 8192,
            },
            "dataset": {"path": "clips", "kind": "video", "format": "directory"},
        },
    )
    result = CliRunner().invoke(app, ["compile", "--spec", str(spec_path), "--json"])
    assert result.exit_code == 2
    payload = json.loads(result.output)
    assert payload["ok"] is True
    assert payload["plan"]["feasible"] is False
    codes = {issue["code"] for issue in payload["plan"]["issues"] if issue["severity"] == "error"}
    assert "E_MEMORY_BUDGET" in codes


def test_cli_compile_invalid_spec_is_ok_false(cli_env):
    path = cli_env / "bad.json"
    path.write_text("{not json", encoding="utf-8")
    result = CliRunner().invoke(app, ["compile", "--spec", str(path), "--json"])
    assert result.exit_code == 2
    payload = json.loads(result.output)
    assert payload["ok"] is False
    assert payload["issues"] == []


def test_cli_compile_missing_spec_file(cli_env):
    result = CliRunner().invoke(app, ["compile", "--spec", str(cli_env / "nope.json"), "--json"])
    assert result.exit_code == 2
    assert json.loads(result.output)["ok"] is False


def test_cli_run_dry_run_prints_plan_without_events(cli_env):
    spec_path = _write_spec(
        cli_env,
        {
            "schema_version": 1,
            "config": {"name": "dry", "family": "qwen3-1.7b", "stage": "sft", "adapter": "lora"},
            "dataset": {"path": "demo.jsonl", "kind": "text", "format": "jsonl"},
        },
    )
    run_dir = cli_env / "run"
    result = CliRunner().invoke(
        app, ["run", "--spec", str(spec_path), "--run-dir", str(run_dir), "--dry-run", "--json"]
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["ok"] is True
    assert payload["plan"]["execution"]["steps"][0]["command"] is None
    assert not (run_dir / "plan.json").exists()


def test_cli_run_dry_run_infeasible_exits_two(cli_env):
    spec_path = _write_spec(
        cli_env,
        {
            "schema_version": 1,
            "config": {"name": "dmd", "family": "qwen3-1.7b", "stage": "distill", "adapter": "lora"},
            "dataset": {"path": "demo.jsonl", "kind": "text", "format": "jsonl"},
        },
    )
    result = CliRunner().invoke(
        app, ["run", "--spec", str(spec_path), "--run-dir", str(cli_env / "r2"), "--dry-run", "--json"]
    )
    assert result.exit_code == 2
    assert json.loads(result.output)["plan"]["feasible"] is False


def test_cli_plugins_json_shape(cli_env):
    result = CliRunner().invoke(app, ["plugins", "--json"])
    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert set(payload) == {"plugins", "errors"}


def test_cli_env_json_is_bare_report(cli_env):
    result = CliRunner().invoke(app, ["env", "--json"])
    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert "python" in payload and "packages" in payload
    assert "ok" not in payload


def test_cli_families_json_shape(cli_env):
    result = CliRunner().invoke(app, ["families", "--json"])
    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert set(payload) == {"llm", "video"}
    assert any(family["key"] == "minimax-h3" for family in payload["video"])


def test_cli_version(cli_env):
    result = CliRunner().invoke(app, ["version"])
    assert result.exit_code == 0
    assert "version" in json.loads(result.output)


def test_cli_plan_preserves_legacy_shape(cli_env):
    cfg = {"name": "legacy", "family": "qwen3-0.6b", "stage": "pretrain"}
    path = cli_env / "config.json"
    path.write_text(json.dumps(cfg), encoding="utf-8")
    result = CliRunner().invoke(app, ["plan", str(path)])
    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["ok"] is True
    assert "training_peak_mib" in payload["plan"]