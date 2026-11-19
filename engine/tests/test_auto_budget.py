"""Auto-budget planner tests: quant_base x block-swap search for video families.

These pin the behavior of `RunConfig.auto_budget`:

- it only activates for block-swap-capable video families;
- it picks the feasible candidate with the lowest step peak (quant + swap are
  recorded in notes and a dedicated `自动预算选择` line);
- it returns an infeasible plan with readable risks instead of raising when no
  candidate fits;
- with `auto_budget=False` (the default) the legacy arithmetic is unchanged.
"""

from __future__ import annotations

import re
from pathlib import Path

from supertuner_engine.backends import musubi as musubi_backend
from supertuner_engine.compiler import compile_spec
from supertuner_engine.config import EngineConfig
from supertuner_engine.plan import plan, resolve_config
from supertuner_engine.schema import (
    Adapter,
    CachePlan,
    DatasetKind,
    DatasetSpec,
    QuantBase,
    RunConfig,
    RunSpec,
    Stage,
    OutputSpec,
)

_MUSUBI_SCRIPTS = (
    "wan_train_network.py",
    "wan_cache_latents.py",
    "wan_cache_text_encoder_outputs.py",
    "minimax_h3_train_network.py",
    "minimax_h3_cache_latents.py",
    "minimax_h3_cache_text_encoder_outputs.py",
)


def _musubi_root(tmp_path: Path) -> Path:
    root = tmp_path / "musubi"
    root.mkdir(parents=True, exist_ok=True)
    for name in _MUSUBI_SCRIPTS:
        (root / name).write_text("raise SystemExit('stub')\n", encoding="utf-8")
    return root


def _ctx(tmp_path: Path):
    paths = {
        "SUPERTUNER_HOME": str(tmp_path / "home"),
        "SUPERTUNER_MODELS_DIR": str(tmp_path / "models"),
        "SUPERTUNER_DATASETS_DIR": str(tmp_path / "datasets"),
        "SUPERTUNER_CACHE_DIR": str(tmp_path / "cache"),
        "SUPERTUNER_MUSUBI_PATH": str(_musubi_root(tmp_path)),
    }
    return EngineConfig.from_env(paths).context(tmp_path / "run")


def _video(family: str = "minimax-h3", **kw) -> RunConfig:
    base = dict(
        name=f"auto-{family}",
        family=family,
        stage=Stage.SFT,
        adapter="lora",
        quant_base=QuantBase.NONE,
        blocks_to_swap=0,
        width=320,
        height=576,
        frames=49,
        cache=CachePlan.BOTH,
        vram_budget_mib=8192,
        ram_budget_mib=32768,
    )
    base.update(kw)
    return RunConfig(**base)


def _chosen_swap(p) -> int:
    note = next(n for n in p.notes if "自动预算选择" in n)
    m = re.search(r"blocks_to_swap=(\d+)/", note)
    assert m is not None, f"no blocks_to_swap in auto note: {note!r}"
    return int(m.group(1))


def test_h3_320p_8gb_auto_budget_picks_nf4_large_swap():
    """320x576x49f on 8GB + 32GB RAM: auto search lands on nf4 + big swap.

    With the unified ladder (v0.2 T3) NF4 is the most compressed candidate,
    so at an equal activation floor it carries the lowest peak and wins; the
    full-block vs partial-window tie then resolves toward train_blocks.
    """
    p = plan(_video(auto_budget=True))
    assert p.feasible, f"H3 320p should auto-fit 8GB, peak {p.headline_mib} MiB"
    assert p.ram_peak_mib > 0
    # selection is recorded both in notes and in a dedicated line
    note = next(n for n in p.notes if "自动预算选择" in n)
    assert "quant_base=nf4" in note
    assert "候选" in note
    line = next(item for item in p.lines if item.label == "自动预算选择")
    assert "nf4" in line.note and "swap" in line.note
    # large block swap: musubi keeps >= 2 blocks resident, so 48/50 here
    assert _chosen_swap(p) >= 40
    # the unified search records every evaluated candidate (v0.2 T3)
    assert p.auto_candidates, "auto_candidates must be populated"


def test_h3_480p_tight_budget_auto_budget_infeasible():
    """480x832x49f against a 6GB budget: every candidate misses, infeasible."""
    p = plan(_video(width=480, height=832, vram_budget_mib=6144, auto_budget=True))
    assert not p.feasible
    assert any("自动预算选择" in r for r in p.risks)
    assert any("全部不可行" in r for r in p.risks)
    # readable risk names the closest miss, not a bare exception
    assert any("最接近" in r for r in p.risks)


def test_wan_1_3b_auto_budget_stays_feasible():
    """Wan2.1 1.3B keeps a working config under auto search."""
    p = plan(_video(family="wan2.1-t2v-1.3b", width=480, height=832, auto_budget=True))
    assert p.feasible, f"Wan 1.3B should auto-fit 8GB, peak {p.headline_mib} MiB"
    assert any(line.label == "自动预算选择" for line in p.lines)
    assert p.ram_peak_mib >= 0


def test_auto_budget_default_off_keeps_legacy_numbers():
    """auto_budget=False (and the default) must not change legacy results."""
    legacy = plan(_video())  # auto_budget defaults to False
    explicit_off = plan(_video(auto_budget=False))
    assert legacy.headline_mib == explicit_off.headline_mib
    assert legacy.feasible == explicit_off.feasible
    assert legacy.ram_peak_mib == explicit_off.ram_peak_mib
    assert [ln.label for ln in legacy.lines] == [ln.label for ln in explicit_off.lines]
    # no auto-budget artifacts leak into a default run
    joined = " ".join(legacy.notes + [ln.note for ln in legacy.lines])
    assert "自动预算选择" not in joined


def test_auto_budget_ignores_llm_and_non_swap_families():
    """auto_budget=True is a no-op for LLM families (legacy path)."""
    p = plan(_video(family="qwen3-0.6b", auto_budget=True))
    assert not any(line.label == "自动预算选择" for line in p.lines)
    assert not any("自动预算选择" in n for n in p.notes)


def test_resolve_config_returns_effective_config():
    """resolve_config returns the auto-picked config, not just the plan."""
    cfg = _video(auto_budget=True)
    effective, p = resolve_config(cfg)
    assert p.feasible
    # the effective config is a real RunConfig the compiler can compile
    assert isinstance(effective, RunConfig)
    assert effective.quant_base == QuantBase.NF4
    assert effective.blocks_to_swap >= 40
    # the unified search prefers the partial window over full blocks on a tie
    assert effective.train_blocks == "20-49"
    # and plan(cfg) is exactly the plan half of resolve_config
    assert plan(cfg) == p


def test_compile_spec_applies_auto_budget_to_backend(tmp_path, monkeypatch):
    """auto_budget's chosen quant/swap must reach the musubi argv."""
    monkeypatch.setattr(musubi_backend, "nvidia_vram_mib", lambda: 0)
    spec = RunSpec(
        config=_video(auto_budget=True),
        dataset=DatasetSpec(path="clips", kind=DatasetKind.VIDEO, format="directory"),
        output=OutputSpec(dir="out"),
    )
    plan = compile_spec(spec, _ctx(tmp_path))

    codes = {i.code for i in plan.issues}
    assert "I_AUTO_BUDGET_APPLIED" in codes
    # the train step argv carries the auto-selected values. The spec asked for
    # blocks_to_swap=0; the auto search picked a large swap, which must reach
    # the argv. INT8 is the published H3 precision and emits no quant flag, so
    # the FP8 marker must be absent (the search did not pick the fp8 candidate).
    train = next(step for step in plan.execution.steps if step.id == "train")
    assert train.command is not None
    cmd = train.command
    assert "--blocks_to_swap" in cmd
    assert int(cmd[cmd.index("--blocks_to_swap") + 1]) >= 40
    assert "--fp8_base" not in cmd
    # the memory plan is the auto-budget plan
    assert any(line.label == "自动预算选择" for line in plan.memory.lines)


def test_compile_spec_auto_budget_off_leaves_argv_alone(tmp_path, monkeypatch):
    """auto_budget=False: the typed values reach the argv unchanged."""
    monkeypatch.setattr(musubi_backend, "nvidia_vram_mib", lambda: 0)
    spec = RunSpec(
        config=_video(auto_budget=False, quant_base=QuantBase.NONE, blocks_to_swap=0),
        dataset=DatasetSpec(path="clips", kind=DatasetKind.VIDEO, format="directory"),
        output=OutputSpec(dir="out"),
    )
    plan = compile_spec(spec, _ctx(tmp_path))
    assert "I_AUTO_BUDGET_APPLIED" not in {i.code for i in plan.issues}
    train = next(step for step in plan.execution.steps if step.id == "train")
    assert train.command is not None
    assert "--quant_base" not in train.command
    assert "--blocks_to_swap" not in train.command
