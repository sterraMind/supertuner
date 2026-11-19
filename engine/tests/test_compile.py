"""Compile-golden tests: the five recipes the product promises to get right.

No network, no torch, no weights: `compile_spec` must decide feasibility,
backend, issues and argv purely from the spec plus the local filesystem.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from supertuner_engine.backends import musubi as musubi_backend
from supertuner_engine.backends import native_llm
from supertuner_engine.compiler import compile_spec, validate_spec
from supertuner_engine.config import Context, EngineConfig
from supertuner_engine.schema import (
    Adapter,
    DatasetKind,
    DatasetSpec,
    Optimizer,
    OutputSpec,
    Precision,
    QuantBase,
    RunConfig,
    RunSpec,
    Stage,
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


def _ctx(tmp_path: Path, *, musubi: bool = True) -> Context:
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


def _llm_spec(**config_kw) -> RunSpec:
    config_kw.setdefault("family", "qwen3-1.7b")
    config_kw.setdefault("name", "qwen-qlora")
    config_kw.setdefault("stage", Stage.SFT)
    return RunSpec(
        config=RunConfig(**config_kw),
        dataset=DatasetSpec(path="demo.jsonl", kind=DatasetKind.CHAT, format="jsonl"),
        output=OutputSpec(dir="out"),
    )


def _video_spec(**config_kw) -> RunSpec:
    config_kw.setdefault("family", "wan2.1-t2v-1.3b")
    config_kw.setdefault("name", "wan-lora")
    config_kw.setdefault("stage", Stage.SFT)
    return RunSpec(
        config=RunConfig(**config_kw),
        dataset=DatasetSpec(path="clips", kind=DatasetKind.VIDEO, format="directory"),
        output=OutputSpec(dir="out"),
    )


@pytest.fixture(autouse=True)
def _no_gpu(monkeypatch):
    """Compile goldens must not depend on the machine's GPU."""
    monkeypatch.setattr(musubi_backend, "nvidia_vram_mib", lambda: 0)
    # bitsandbytes is optional: pin it off so QLoRA warnings are deterministic.
    monkeypatch.setattr(native_llm, "_available", lambda name: name != "bitsandbytes")


def test_qwen3_1_7b_qlora_8gb_golden(tmp_path):
    spec = _llm_spec(
        adapter=Adapter.QLORA,
        quant_base=QuantBase.NF4,
        lora_rank=16,
        lora_alpha=32,
        seq_len=2048,
        batch=1,
        grad_accum=8,
        vram_budget_mib=8192,
    )
    plan = compile_spec(spec, _ctx(tmp_path))

    assert plan.feasible is True
    assert plan.memory.feasible is True
    assert plan.memory.training_peak_mib < 8192
    assert plan.execution.backend == "native-llm"
    assert [step.id for step in plan.execution.steps] == ["train"]
    assert plan.execution.steps[0].command is None  # in-process backend step
    assert "W_BITSANDBYTES_MISSING" in _codes(plan)
    assert "I_QLORA_NF4" in _codes(plan)
    assert plan.execution.expected_artifacts[0].endswith("adapter_model.safetensors")


def test_wan2_1_1_3b_lora_8gb_golden(tmp_path):
    spec = _video_spec(
        adapter=Adapter.LORA,
        lora_rank=32,
        lora_alpha=64,
        width=480,
        height=832,
        frames=49,
        batch=1,
        grad_accum=4,
        blocks_to_swap=0,
        vram_budget_mib=8192,
    )
    plan = compile_spec(spec, _ctx(tmp_path))

    assert plan.feasible is False
    assert plan.memory.feasible is False
    assert "E_MEMORY_BUDGET" in _errors(plan)
    assert "E_STEP_OOM" in _errors(plan)
    assert plan.execution.backend == "musubi"
    assert [step.id for step in plan.execution.steps] == [
        "prepare",
        "cache-text",
        "cache-latents",
        "train",
        "preview",
    ]
    assert [step.kind for step in plan.execution.steps] == ["prepare", "cache", "cache", "train", "export"]
    train = plan.execution.steps[3]
    assert train.command is not None
    assert "--task" in train.command
    assert train.command[train.command.index("--task") + 1] == "t2v-1.3B"
    assert train.command[train.command.index("--network_module") + 1] == "networks.lora_wan"
    assert "--blocks_to_swap" not in train.command
    assert train.command[train.command.index("--output_dir") + 1].startswith(str(tmp_path))
    preview = plan.execution.steps[4]
    assert preview.command is None
    assert preview.optional is True
    assert preview.estimated_mib is not None and preview.estimated_mib > 0


def test_minimax_h3_8gb_edge_golden(tmp_path):
    spec = _video_spec(
        family="minimax-h3",
        name="h3-8gb-edge",
        adapter=Adapter.LORA,
        quant_base=QuantBase.INT8,
        lora_rank=16,
        lora_alpha=16,
        width=320,
        height=576,
        frames=49,
        batch=1,
        grad_accum=8,
        blocks_to_swap=48,
        text_encoder_quant=QuantBase.INT8,
        text_encoder_blocks_to_swap=50,
        vram_budget_mib=8192,
    )
    plan = compile_spec(spec, _ctx(tmp_path))

    # The 8 GB edge profile only fits the training step; caching the 32B text
    # encoder needs ~25 GB, so the whole run is correctly reported infeasible.
    assert plan.feasible is False
    assert plan.memory.feasible is True
    assert "E_STEP_OOM" in _errors(plan)
    assert plan.execution.backend == "musubi"
    assert [step.id for step in plan.execution.steps] == [
        "prepare",
        "cache-text",
        "cache-latents",
        "train",
        "preview",
    ]

    cache_text = plan.execution.steps[1].command or []
    assert cache_text[cache_text.index("--task") + 1] == "t2va"
    assert "--text_encoder_blocks_to_swap" in cache_text
    assert cache_text[cache_text.index("--text_encoder_blocks_to_swap") + 1] == "50"

    cache_latents = plan.execution.steps[2].command or []
    assert "--video_vae" in cache_latents
    assert "--audio_vae" in cache_latents

    train = plan.execution.steps[3].command or []
    assert train[train.index("--blocks_to_swap") + 1] == "48"
    assert train[train.index("--network_module") + 1] == "networks.lora_minimax_h3"
    assert train[train.index("--task") + 1] == "t2va"
    assert _errors(plan) == ["E_STEP_OOM"]


def test_wan2_1_14b_full_ft_is_refused(tmp_path):
    spec = _video_spec(
        family="wan2.1-t2v-14b",
        name="wan14b-full",
        adapter=Adapter.NONE,
        width=832,
        height=480,
        frames=49,
        vram_budget_mib=8192,
    )
    plan = compile_spec(spec, _ctx(tmp_path))

    assert plan.feasible is False
    assert plan.memory.feasible is False
    assert "E_MEMORY_BUDGET" in _errors(plan)
    # No backend claims train.video.full in v0.1, and the plan says so loudly.
    assert "E_NO_BACKEND" in _errors(plan)
    assert plan.execution.steps == []


def test_dmd_distillation_is_refused(tmp_path):
    from supertuner_engine.schema import DistillSpec, TrainSpec

    spec = _video_spec(family="minimax-h3", adapter=Adapter.LORA, blocks_to_swap=48, quant_base=QuantBase.INT8)
    spec.train = TrainSpec(max_steps=200, distill=DistillSpec(teacher="h3-teacher", mode="dmd"))
    spec.config.stage = Stage.DISTILL
    plan = compile_spec(spec, _ctx(tmp_path))

    assert plan.feasible is False
    assert "E_DISTILL_UNSUPPORTED" in _errors(plan)
    assert plan.execution.steps == []


def test_h3_without_musubi_is_refused(tmp_path):
    spec = _video_spec(
        family="minimax-h3",
        adapter=Adapter.LORA,
        quant_base=QuantBase.INT8,
        width=320,
        height=576,
        frames=49,
        blocks_to_swap=48,
        vram_budget_mib=8192,
    )
    plan = compile_spec(spec, _ctx(tmp_path, musubi=False))

    assert plan.feasible is False
    assert "E_MUSUBI_MISSING" in _errors(plan)
    issue = next(issue for issue in plan.issues if issue.code == "E_MUSUBI_MISSING")
    assert issue.hint


def test_wan_without_musubi_is_a_warning(tmp_path):
    spec = _video_spec()
    plan = compile_spec(spec, _ctx(tmp_path, musubi=False))

    assert "W_MUSUBI_MISSING" in _codes(plan)
    # Without the text encoder cached, the cache step does not fit 8 GB.
    assert plan.feasible is False


@pytest.mark.parametrize("quant", [QuantBase.NF4, QuantBase.FP8, QuantBase.INT8])
def test_h3_small_card_edge_recipe_accepts_quantized_base(tmp_path, monkeypatch, quant):
    """On a small card, a quantized base + deep swap is a valid H3 edge recipe.

    The auto-search can now pick NF4/FP8/INT8 (all smaller than INT8's bf16
    baseline), so the H3 VRAM guard must accept all of them as the 8GB recipe.
    """
    monkeypatch.setattr(musubi_backend, "nvidia_vram_mib", lambda: 8192)
    spec = _video_spec(
        family="minimax-h3",
        adapter=Adapter.LORA,
        quant_base=quant,
        width=320,
        height=576,
        frames=49,
        blocks_to_swap=48,
        vram_budget_mib=8192,
    )
    plan = compile_spec(spec, _ctx(tmp_path))
    assert "E_H3_VRAM_LOW" not in _errors(plan), f"{quant} edge recipe wrongly rejected"


def test_h3_small_card_edge_recipe_rejects_bf16(tmp_path, monkeypatch):
    """bf16 (no base quant) on a small card is still rejected by the H3 guard."""
    monkeypatch.setattr(musubi_backend, "nvidia_vram_mib", lambda: 8192)
    spec = _video_spec(
        family="minimax-h3",
        adapter=Adapter.LORA,
        quant_base=QuantBase.NONE,
        width=320,
        height=576,
        frames=49,
        blocks_to_swap=48,
        vram_budget_mib=8192,
    )
    plan = compile_spec(spec, _ctx(tmp_path))
    assert "E_H3_VRAM_LOW" in _errors(plan)


def test_unknown_family_is_an_issue_not_an_exception(tmp_path):
    spec = _llm_spec(family="not-a-family")
    plan = compile_spec(spec, _ctx(tmp_path))
    assert plan.feasible is False
    assert "E_FAMILY_UNKNOWN" in _errors(plan)


def test_unknown_backend_request_is_reported(tmp_path):
    spec = _llm_spec()
    spec.backend = "ghost-backend"
    plan = compile_spec(spec, _ctx(tmp_path))
    assert plan.feasible is False
    assert "E_NO_BACKEND" in _errors(plan)


def test_qlora_with_int8_base_conflicts(tmp_path):
    spec = _llm_spec(adapter=Adapter.QLORA, quant_base=QuantBase.INT8)
    plan = compile_spec(spec, _ctx(tmp_path))
    assert "E_QUANT_CONFLICT" in _errors(plan)


def test_fp32_master_is_refused_not_downgraded(tmp_path):
    spec = _llm_spec(precision=Precision.FP32_MASTER)
    plan = compile_spec(spec, _ctx(tmp_path))
    assert "E_PRECISION_UNSUPPORTED" in _errors(plan)
    assert plan.feasible is False


def test_came_optimizer_is_refused(tmp_path):
    spec = _llm_spec(optimizer=Optimizer.CAME)
    plan = compile_spec(spec, _ctx(tmp_path))
    assert "E_OPTIMIZER_UNSUPPORTED" in _errors(plan)


def test_text_dataset_on_video_family_mismatches(tmp_path):
    spec = _video_spec()
    spec.dataset = DatasetSpec(path="x.jsonl", kind=DatasetKind.TEXT, format="jsonl")
    plan = compile_spec(spec, _ctx(tmp_path))
    assert "E_DATASET_KIND_MISMATCH" in _errors(plan)


def test_missing_dataset_is_refused(tmp_path):
    spec = _llm_spec()
    spec.dataset = None
    plan = compile_spec(spec, _ctx(tmp_path))
    assert "E_DATASET_REQUIRED" in _errors(plan)


def test_block_swap_with_full_ft_is_refused(tmp_path):
    spec = _llm_spec(adapter=Adapter.NONE, lora_targets=[])
    spec.dataset = None
    spec.config.stage = Stage.PRETRAIN
    spec.config.blocks_to_swap = 8
    issues = validate_spec(spec, _ctx(tmp_path))
    assert "E_BLOCK_SWAP_FULL_FT" in {issue.code for issue in issues}


def test_commands_are_argv_arrays_never_shell_strings(tmp_path):
    for spec in (_video_spec(), _video_spec(family="minimax-h3", blocks_to_swap=48, quant_base=QuantBase.INT8)):
        plan = compile_spec(spec, _ctx(tmp_path))
        for step in plan.execution.steps:
            if step.command is not None:
                assert isinstance(step.command, list)
                assert all(isinstance(part, str) and part for part in step.command)
                assert all("\n" not in part for part in step.command)


def test_steps_are_ordered_by_kind(tmp_path):
    plan = compile_spec(_video_spec(), _ctx(tmp_path))
    order = {"prepare": 0, "cache": 1, "train": 2, "export": 3}
    kinds = [order[step.kind] for step in plan.execution.steps]
    assert kinds == sorted(kinds)


def test_dataset_toml_is_rendered_by_prepare_step(tmp_path):
    ctx = _ctx(tmp_path)
    spec = _video_spec()
    plan = compile_spec(spec, ctx)
    assert plan.execution.steps[0].command is None

    backend = musubi_backend.MusubiBackend()
    captured: list[str] = []

    class Emit:
        def log(self, level, message):
            captured.append(message)

        def __getattr__(self, name):
            return lambda *args, **kwargs: None

    ctx.ensure_run_dirs()
    backend.run_step(spec, plan.execution.steps[0], ctx, Emit())
    toml_path = musubi_backend.dataset_toml_path(spec, ctx)
    assert toml_path.is_file()
    text = toml_path.read_text(encoding="utf-8")
    assert "[general]" in text
    assert "resolution = [480, 832]" in text
    assert "[[datasets]]" in text
    assert 'frame_extraction = "head"' in text
    assert captured and str(toml_path) in captured[0]


def test_wan_geometry_must_be_16_divisible(tmp_path):
    spec = _video_spec(width=481, height=832)
    plan = compile_spec(spec, _ctx(tmp_path))
    assert "E_VIDEO_GEOMETRY" in _errors(plan)


def test_wan_frames_must_be_4n_plus_1(tmp_path):
    spec = _video_spec(frames=50)
    plan = compile_spec(spec, _ctx(tmp_path))
    assert "E_VIDEO_FRAMES" in _errors(plan)