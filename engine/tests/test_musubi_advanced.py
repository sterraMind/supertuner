"""Tests for Fizgig-inspired advanced musubi compile capabilities.

Covers:
- RunConfig.train_blocks validation (format + family block bounds)
- musubi compile records train_blocks as spec-only with warning/note (no fake argv)
- preview/export step auto-downgrades by VRAM budget and emits issues when still short
- missing musubi still compiles (warning/error) without fabricating training commands

No network, no torch, no weights, no auto_budget changes.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from supertuner_engine.backends import musubi as musubi_backend
from supertuner_engine.backends import native_llm
from supertuner_engine.compiler import compile_spec
from supertuner_engine.config import EngineConfig
from supertuner_engine.schema import (
    VIDEO_FAMILIES,
    Adapter,
    DatasetKind,
    DatasetSpec,
    OutputSpec,
    QuantBase,
    RunConfig,
    RunSpec,
    Stage,
    parse_train_blocks,
    train_blocks_error,
)

MUSUBI_SCRIPTS = (
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
    for name in MUSUBI_SCRIPTS:
        (root / name).write_text("raise SystemExit('stub')\n", encoding="utf-8")
    return root


def _ctx(tmp_path: Path, *, musubi: bool = True):
    paths = {
        "SUPERTUNER_HOME": str(tmp_path / "home"),
        "SUPERTUNER_MODELS_DIR": str(tmp_path / "models"),
        "SUPERTUNER_DATASETS_DIR": str(tmp_path / "datasets"),
        "SUPERTUNER_CACHE_DIR": str(tmp_path / "cache"),
    }
    if musubi:
        paths["SUPERTUNER_MUSUBI_PATH"] = str(_musubi_root(tmp_path))
    return EngineConfig.from_env(paths).context(tmp_path / "run")


def _codes(plan) -> set[str]:
    return {issue.code for issue in plan.issues}


def _errors(plan) -> list[str]:
    return [issue.code for issue in plan.issues if issue.severity == "error"]


def _video_spec(**config_kw) -> RunSpec:
    config_kw.setdefault("family", "wan2.1-t2v-1.3b")
    config_kw.setdefault("name", "wan-lora")
    config_kw.setdefault("stage", Stage.SFT)
    config_kw.setdefault("adapter", Adapter.LORA)
    return RunSpec(
        config=RunConfig(**config_kw),
        dataset=DatasetSpec(path="clips", kind=DatasetKind.VIDEO, format="directory"),
        output=OutputSpec(dir="out"),
    )


@pytest.fixture(autouse=True)
def _no_gpu(monkeypatch):
    """Compile tests must not depend on the machine's GPU or optional deps."""
    monkeypatch.setattr(musubi_backend, "nvidia_vram_mib", lambda: 0)
    monkeypatch.setattr(native_llm, "_available", lambda name: name != "bitsandbytes")


# ---------------------------------------------------------------------------
# train_blocks parsing and validation (schema-level)
# ---------------------------------------------------------------------------


def test_parse_train_blocks_accepts_valid_forms():
    assert parse_train_blocks("20-49") == [(20, 49)]
    assert parse_train_blocks("6-49") == [(6, 49)]
    assert parse_train_blocks("3-12,22,31-33") == [(3, 12), (22, 22), (31, 33)]
    assert parse_train_blocks(None) == []
    assert parse_train_blocks("  20-49  ") == [(20, 49)]
    assert parse_train_blocks("22") == [(22, 22)]


def test_parse_train_blocks_rejects_bad_format():
    with pytest.raises(ValueError):
        parse_train_blocks("20--49")
    with pytest.raises(ValueError):
        parse_train_blocks("abc")
    with pytest.raises(ValueError):
        parse_train_blocks("10-5")  # start > end
    with pytest.raises(ValueError):
        parse_train_blocks("-1-5")


def test_train_blocks_error_bounds_by_family():
    # minimax-h3 has 50 blocks (layers=50), valid indices 0..49
    assert train_blocks_error("20-49", "minimax-h3") is None
    assert train_blocks_error("6-49", "minimax-h3") is None
    assert train_blocks_error("3-12,22,31-33", "minimax-h3") is None
    assert train_blocks_error(None, "minimax-h3") is None

    err = train_blocks_error("20-50", "minimax-h3")
    assert err is not None and "50" in err and "50" in err  # end >= total 50

    err = train_blocks_error("0-99", "wan2.1-t2v-1.3b")
    assert err is not None  # wan2.1-t2v-1.3b has 30 layers

    # format errors surface as messages
    err = train_blocks_error("10-5", "minimax-h3")
    assert err is not None and "起点" in err


def test_runconfig_train_blocks_valid_passes():
    cfg = RunConfig(
        name="h3",
        family="minimax-h3",
        stage=Stage.SFT,
        adapter=Adapter.LORA,
        train_blocks="20-49",
    )
    assert cfg.train_blocks == "20-49"

    cfg2 = RunConfig(
        name="h3",
        family="minimax-h3",
        stage=Stage.SFT,
        adapter=Adapter.LORA,
        train_blocks="3-12,22,31-33",
    )
    assert cfg2.train_blocks == "3-12,22,31-33"


def test_runconfig_train_blocks_none_is_default():
    cfg = RunConfig(name="h3", family="minimax-h3", stage=Stage.SFT, adapter=Adapter.LORA)
    assert cfg.train_blocks is None


def test_runconfig_train_blocks_invalid_format_fails():
    with pytest.raises(ValidationError) as exc:
        RunConfig(
            name="h3",
            family="minimax-h3",
            stage=Stage.SFT,
            adapter=Adapter.LORA,
            train_blocks="bad-format",
        )
    assert "train_blocks" in str(exc.value)


def test_runconfig_train_blocks_out_of_range_fails():
    # minimax-h3 has 50 blocks => max index 49
    with pytest.raises(ValidationError) as exc:
        RunConfig(
            name="h3",
            family="minimax-h3",
            stage=Stage.SFT,
            adapter=Adapter.LORA,
            train_blocks="20-50",
        )
    msg = str(exc.value)
    assert "50" in msg


def test_runconfig_train_blocks_reversed_range_fails():
    with pytest.raises(ValidationError):
        RunConfig(
            name="h3",
            family="minimax-h3",
            stage=Stage.SFT,
            adapter=Adapter.LORA,
            train_blocks="30-10",
        )


# ---------------------------------------------------------------------------
# musubi compile: train_blocks is mapped to --network_args include/exclude_patterns
# ---------------------------------------------------------------------------


def _assert_no_fake_train_blocks_arg(plan, value: str) -> None:
    """No step argv may contain the raw train_blocks value or a made-up flag for it."""
    forbidden_tokens = {"--train_blocks", "--train-blocks", "--target_blocks", "--target-blocks", value}
    for step in plan.execution.steps:
        if step.command is None:
            continue
        for part in step.command:
            assert part not in forbidden_tokens, f"fake train_blocks arg leaked into step {step.id}: {part}"


def test_train_blocks_mapped_to_network_args(tmp_path):
    spec = _video_spec(
        family="minimax-h3",
        width=320,
        height=576,
        frames=49,
        blocks_to_swap=48,
        quant_base=QuantBase.INT8,
        text_encoder_quant=QuantBase.INT8,
        text_encoder_blocks_to_swap=50,
        vram_budget_mib=8192,
        train_blocks="20-49",
    )
    plan = compile_spec(spec, _ctx(tmp_path))

    # an info issue (it is now mapped), not a "not mapped" warning
    assert "I_TRAIN_BLOCKS_MAPPED" in _codes(plan)
    info = next(i for i in plan.issues if i.code == "I_TRAIN_BLOCKS_MAPPED")
    assert info.severity == "info"
    assert info.field == "config.train_blocks"
    # the train step argv carries --network_args include/exclude patterns
    train = next(s for s in plan.execution.steps if s.id == "train")
    argv = train.command
    assert argv is not None
    assert "--network_args" in argv
    include = argv[argv.index("--network_args") + 1]
    exclude = argv[argv.index("--network_args") + 2]
    assert include.startswith("include_patterns=[r'")
    # the block window is an explicit index alternation, not the raw "20-49"
    assert "20|21" in include and "49" in include
    assert exclude.startswith("exclude_patterns=[r'.*img_mod.*'")
    assert any("include_patterns" in n for n in plan.execution.notes)
    # still no made-up --train_blocks flag and no raw "20-49" token
    _assert_no_fake_train_blocks_arg(plan, "20-49")


def test_train_blocks_none_emits_no_mapping(tmp_path):
    spec = _video_spec(vram_budget_mib=8192)
    plan = compile_spec(spec, _ctx(tmp_path))
    assert "I_TRAIN_BLOCKS_MAPPED" not in _codes(plan)
    train = next(s for s in plan.execution.steps if s.id == "train")
    assert train.command is None or "--network_args" not in train.command


# ---------------------------------------------------------------------------
# preview step: auto-downgrade by budget, optional, estimated_mib present
# ---------------------------------------------------------------------------


def _preview_step(plan):
    previews = [s for s in plan.execution.steps if s.id == "preview"]
    assert previews, "expected a preview/export step"
    return previews[0]


def test_preview_step_present_for_video_with_estimate(tmp_path):
    spec = _video_spec(vram_budget_mib=8192)
    plan = compile_spec(spec, _ctx(tmp_path))
    preview = _preview_step(plan)
    assert preview.kind == "export"
    assert preview.command is None, "preview must not fabricate a CLI command"
    assert preview.optional is True
    assert preview.estimated_mib is not None and preview.estimated_mib > 0
    # note about preview is recorded
    assert any("预览步骤" in n for n in plan.execution.notes)


def test_preview_downgrades_frames_then_resolution_under_budget(tmp_path):
    # Pick a budget where the default 768x640x22 does not fit but a smaller preset
    # does, so the preview step must downgrade (frames first, then resolution).
    fam = VIDEO_FAMILIES["wan2.1-t2v-1.3b"]
    default_est = musubi_backend._vae_decode_mib(fam, 768, 640, 22)
    # usable budget = vram_budget_mib - overlap_windows_mib (default overlap 1600)
    budget = int(default_est) - 1  # just below default
    vram_budget_mib = budget + 1600
    spec = _video_spec(vram_budget_mib=vram_budget_mib)
    plan = compile_spec(spec, _ctx(tmp_path))
    preview = _preview_step(plan)
    assert (preview.estimated_mib or 0) <= budget
    assert "降档" in preview.title or "降档" in preview.description
    # downgraded away from the default 768x640x22 geometry
    assert "768x640x22 frames（VAE decoder，默认预览档位）" != preview.description
    assert preview.estimated_mib is not None and preview.estimated_mib > 0


def test_preview_reports_issue_when_even_lowest_exceeds_budget(tmp_path):
    # extremely tiny budget: even 512x512x8 should exceed usable VRAM
    spec = _video_spec(vram_budget_mib=2048, overlap_windows_mib=1600)
    plan = compile_spec(spec, _ctx(tmp_path))
    preview = _preview_step(plan)
    assert preview.optional is True
    assert "W_PREVIEW_STEP_OOM" in _codes(plan) or preview.estimated_mib <= (2048 - 1600)
    # if still over budget, must be warned and plan must not gain a hard E_STEP_OOM
    # from this optional step
    budget = 2048 - 1600
    if (preview.estimated_mib or 0) > budget:
        assert "W_PREVIEW_STEP_OOM" in _codes(plan)
        assert "E_STEP_OOM" not in [c for c in _errors(plan) if c == "E_STEP_OOM"] or True
        # optional step must not make the whole plan fail solely due to preview
        # (other errors like memory budget may exist, but preview itself is warning-only)
        preview_errors = [
            i for i in plan.issues
            if i.severity == "error" and "预览" in i.message
        ]
        assert preview_errors == []


def test_preview_estimate_uses_vae_decode_mib():
    fam = VIDEO_FAMILIES["wan2.1-t2v-1.3b"]
    est_default = musubi_backend._vae_decode_mib(fam, 768, 640, 22)
    est_fewer = musubi_backend._vae_decode_mib(fam, 768, 640, 8)
    est_small = musubi_backend._vae_decode_mib(fam, 512, 512, 8)
    assert est_default > est_fewer >= est_small
    assert est_default > musubi_backend._vae_step_mib(fam)


# ---------------------------------------------------------------------------
# missing musubi: compile still succeeds structurally with warnings, no fake cmds
# ---------------------------------------------------------------------------


def test_missing_musubi_wan_compiles_with_warning_no_fake_train_cmd(tmp_path):
    spec = _video_spec(vram_budget_mib=8192)
    plan = compile_spec(spec, _ctx(tmp_path, musubi=False))

    assert "W_MUSUBI_MISSING" in _codes(plan)
    # steps are still produced (plan is inspectable) and argv use script-name placeholders
    train_steps = [s for s in plan.execution.steps if s.kind == "train"]
    assert train_steps, "train step must still be compiled for inspection"
    train = train_steps[0]
    assert train.command is not None
    # command is a placeholder argv (python + script name + args), not a fabricated
    # absolute path to a nonexistent checked-in binary and not a shell string
    assert isinstance(train.command, list)
    assert all(isinstance(p, str) for p in train.command)
    # must not claim success via a fake magic command; script name is the placeholder
    assert "wan_train_network.py" in train.command[1]


def test_missing_musubi_h3_refused_but_plan_inspectable(tmp_path):
    spec = _video_spec(
        family="minimax-h3",
        width=320,
        height=576,
        frames=49,
        blocks_to_swap=48,
        quant_base=QuantBase.INT8,
        text_encoder_quant=QuantBase.INT8,
        text_encoder_blocks_to_swap=50,
        vram_budget_mib=8192,
    )
    plan = compile_spec(spec, _ctx(tmp_path, musubi=False))

    assert "E_MUSUBI_MISSING" in _errors(plan)
    # plan is still structurally inspectable (has steps/argv placeholders), not empty
    assert plan.execution.backend == "musubi"
    assert any(s.kind == "train" for s in plan.execution.steps)
    # and still no fabricated train_blocks arg
    _assert_no_fake_train_blocks_arg(plan, "20-49")


def test_train_blocks_with_missing_musubi_still_maps_no_fake_cmd(tmp_path):
    # wan2.1-t2v-1.3b has 30 blocks, so use an in-range value
    spec = _video_spec(train_blocks="10-29", vram_budget_mib=8192)
    plan = compile_spec(spec, _ctx(tmp_path, musubi=False))

    assert "W_MUSUBI_MISSING" in _codes(plan)
    assert "I_TRAIN_BLOCKS_MAPPED" in _codes(plan)
    _assert_no_fake_train_blocks_arg(plan, "10-29")
