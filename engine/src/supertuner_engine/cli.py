"""SuperTuner engine CLI.

Machine contract (stable, used by the TypeScript server):

* ``supertuner-engine compile --spec <file> --json`` → one JSON object
  ``{"ok": true, "plan": RunPlan}`` or ``{"ok": false, "error": str, "issues": [Issue]}``.
  Exit code 0 when the plan is feasible, 2 when it is infeasible/refused,
  3 on an engine crash.
* ``supertuner-engine run --spec <file> --run-dir <dir> --events-stdout`` →
  one JSON object per line (RunEvent) on stdout, flushed per line; stderr is
  free-form. Exit codes: 0 completed, 2 infeasible, 3 failed, 130 cancelled.
* ``supertuner-engine plugins --json`` → ``{"plugins": [...], "errors": [...]}``.
* ``supertuner-engine env --json`` → bare ``EnvReport``.
* ``supertuner-engine families --json`` → ``{"llm": [...], "video": [...]}``.

Every command accepts ``--json`` for machine-only stdout (single-line JSON).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Annotated, Any

import typer

from .config import Context, cached_config, load_context
from .env import available_vram_mib, probe
from .events import Emitter
from .plan import plan
from .plugins import load_registry
from .runtime import (
    EXIT_COMPLETED,
    EXIT_FAILED,
    EXIT_INFEASIBLE,
    execute,
    plan_for,
)
from .schema import LLM_FAMILIES, VIDEO_FAMILIES, Issue, RunConfig, RunSpec
from .version import __version__

app = typer.Typer(help="SuperTuner engine: single-GPU training runtime.", no_args_is_help=True)

JsonOption = Annotated[bool, typer.Option("--json", help="机器可读输出：stdout 只有 JSON")]


def _dump(payload: Any, *, as_json: bool, pretty: bool = True) -> str:
    if as_json:
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    return json.dumps(payload, ensure_ascii=False, indent=2 if pretty else None)


def _echo_json(payload: Any, *, as_json: bool, pretty: bool = True) -> None:
    sys.stdout.write(_dump(payload, as_json=as_json, pretty=pretty) + "\n")
    sys.stdout.flush()


def _issues(issues: list[Issue]) -> list[dict[str, Any]]:
    return [issue.model_dump(mode="json") for issue in issues]


def _fail(message: str, *, issues: list[Issue] | None = None, code: int = EXIT_FAILED) -> None:
    payload = {"ok": False, "error": message, "issues": _issues(issues or [])}
    sys.stdout.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n")
    sys.stdout.flush()
    raise typer.Exit(code)


def _load_spec(path: Path) -> RunSpec:
    raw = path.read_text(encoding="utf-8")
    return RunSpec.model_validate_json(raw)


def _context(run_dir: Path | None) -> Context:
    base = load_context()
    if run_dir is None:
        return base
    return base.with_run_dir(run_dir)


@app.command()
def version() -> None:
    """Print engine version as JSON."""
    typer.echo(json.dumps({"version": __version__}))


@app.command()
def plan_cmd(
    config: Annotated[Path | None, typer.Argument(help="Path to RunConfig JSON; omit to read stdin")] = None,
    pretty: Annotated[bool, typer.Option(help="Pretty-print JSON")] = True,
) -> None:
    """Plan memory usage for a run config and print the MemoryPlan JSON."""
    raw = config.read_text(encoding="utf-8") if config else sys.stdin.read()
    try:
        cfg = RunConfig.model_validate_json(raw)
    except Exception as exc:  # noqa: BLE001 - user input error
        payload: dict[str, Any] = {"ok": False, "error": f"invalid RunConfig: {exc}"}
        typer.echo(json.dumps(payload, ensure_ascii=False, indent=2 if pretty else None))
        raise typer.Exit(EXIT_FAILED) from exc
    # 实测进预算（v0.2 T2 / G3 = AP-01）：探测本机实测 VRAM/RAM，交给 plan()
    # 按 cfg.budget_source 决定是否采用。probe() 永不抛异常（无 torch 也安全）。
    env_report = probe()
    measured_vram_mib = available_vram_mib(env_report)
    measured_ram_mib = env_report.ram_mib
    try:
        result = plan(cfg, measured_vram_mib=measured_vram_mib, measured_ram_mib=measured_ram_mib)
        payload = {"ok": True, "plan": result.model_dump()}
    except ValueError as e:
        payload = {"ok": False, "error": str(e)}
    typer.echo(json.dumps(payload, ensure_ascii=False, indent=2 if pretty else None))


# the human-facing command name is `plan`
app.command("plan")(plan_cmd)


@app.command()
def compile(
    spec: Annotated[Path, typer.Option("--spec", help="RunSpec JSON 文件")],
    run_dir: Annotated[Path | None, typer.Option("--run-dir", help="运行目录（用于解析输出路径）")] = None,
    out: Annotated[Path | None, typer.Option("--out", help="把编译结果写入该文件")] = None,
    as_json: JsonOption = False,
) -> None:
    """Compile a RunSpec into a RunPlan (no training happens)."""
    try:
        run_spec = _load_spec(spec)
    except FileNotFoundError:
        _fail(f"spec 文件不存在：{spec}", code=EXIT_INFEASIBLE)
    except Exception as exc:  # noqa: BLE001 - invalid user input
        _fail(f"无法解析 spec：{exc}", code=EXIT_INFEASIBLE)

    ctx = _context(run_dir)
    compiled = plan_for(run_spec, ctx)

    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(compiled.model_dump(mode="json"), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    _echo_json({"ok": True, "plan": compiled.model_dump(mode="json")}, as_json=as_json)
    if not compiled.feasible:
        raise typer.Exit(EXIT_INFEASIBLE)


@app.command()
def plugins(as_json: JsonOption = False) -> None:
    """List discovered plugins and discovery errors."""
    registry = load_registry()
    _echo_json(registry.to_response(), as_json=as_json)


@app.command()
def env(as_json: JsonOption = False) -> None:
    """Probe the local environment (GPU, RAM, packages, musubi)."""
    report = probe(cached_config())
    if as_json:
        _echo_json(report.model_dump(mode="json"), as_json=True)
        return
    lines = [
        f"python: {report.python}",
        f"platform: {report.platform}",
        f"ram_mib: {report.ram_mib}",
        f"ffmpeg: {report.ffmpeg or '未找到'}",
        f"musubi: {report.musubi or '未找到'}",
    ]
    for gpu in report.gpus:
        lines.append(
            f"gpu[{gpu.index}] {gpu.name}: {gpu.vram_total_mib} MiB total / "
            f"{gpu.vram_free_mib} MiB free (driver {gpu.driver or '未知'})"
        )
    for pkg in report.packages:
        lines.append(f"{pkg.name}: {pkg.version or '未安装'}{'' if pkg.ok else ' (缺依赖)'}")
    for warning in report.warnings:
        lines.append(f"warning: {warning}")
    sys.stdout.write("\n".join(lines) + "\n")


@app.command()
def families(as_json: JsonOption = False) -> None:
    """List known model families (llm and video)."""
    payload = {
        "llm": [family.model_dump(mode="json") for family in LLM_FAMILIES.values()],
        "video": [family.model_dump(mode="json") for family in VIDEO_FAMILIES.values()],
    }
    _echo_json(payload, as_json=as_json)


@app.command()
def run(
    spec: Annotated[Path, typer.Option("--spec", help="RunSpec JSON 文件")],
    run_dir: Annotated[Path | None, typer.Option("--run-dir", help="运行目录")] = None,
    events_stdout: Annotated[
        bool, typer.Option("--events-stdout/--no-events-stdout", help="把 RunEvent 逐行写到 stdout")
    ] = True,
    dry_run: Annotated[bool, typer.Option("--dry-run", help="只编译并打印计划，不执行")] = False,
    run_id: Annotated[str | None, typer.Option("--run-id", help="运行 ID（默认取运行目录名）")] = None,
    as_json: JsonOption = False,
) -> None:
    """Execute a RunSpec, emitting RunEvent JSON lines."""
    try:
        run_spec = _load_spec(spec)
    except FileNotFoundError:
        _fail(f"spec 文件不存在：{spec}", code=EXIT_INFEASIBLE)
    except Exception as exc:  # noqa: BLE001 - invalid user input
        _fail(f"无法解析 spec：{exc}", code=EXIT_INFEASIBLE)

    ctx = _context(run_dir)
    emitter = Emitter(sys.stdout, run_id or (ctx.run_dir.name if ctx.run_dir else "adhoc"))

    if dry_run:
        compiled = plan_for(run_spec, ctx)
        _echo_json({"ok": True, "plan": compiled.model_dump(mode="json")}, as_json=True)
        raise typer.Exit(EXIT_COMPLETED if compiled.feasible else EXIT_INFEASIBLE)

    outcome = execute(
        run_spec,
        ctx,
        emitter if events_stdout else None,
        stderr=sys.stderr,
        run_id=run_id,
    )
    if not events_stdout:
        sys.stdout.write(
            _dump(
                {"ok": outcome.ok, "status": outcome.status, "steps": outcome.steps_completed},
                as_json=as_json,
            )
            + "\n"
        )
    raise typer.Exit(outcome.exit_code)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
