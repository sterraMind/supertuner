"""Native LLM backend compile tests: static analysis only, no torch, no weights."""

from __future__ import annotations

from pathlib import Path

import pytest

from supertuner_engine.backends import native_llm
from supertuner_engine.backends.native_llm import (
    NativeLLMBackend,
    _lora_targets,
    _pack,
    _use_packing,
    resolve_dataset_path,
)
from supertuner_engine.config import Context, EngineConfig
from supertuner_engine.schema import (
    Adapter,
    DatasetKind,
    DatasetSpec,
    Optimizer,
    Precision,
    QuantBase,
    RunConfig,
    RunSpec,
    Stage,
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


def _spec(**config_kw) -> RunSpec:
    config_kw.setdefault("name", "native")
    config_kw.setdefault("family", "qwen3-1.7b")
    config_kw.setdefault("stage", Stage.SFT)
    return RunSpec(
        config=RunConfig(**config_kw),
        dataset=DatasetSpec(path="demo.jsonl", kind=DatasetKind.TEXT, format="jsonl"),
    )


@pytest.fixture(autouse=True)
def _pinned_optional_deps(monkeypatch):
    monkeypatch.setattr(native_llm, "_available", lambda name: name != "bitsandbytes")


def _codes(issues) -> set[str]:
    return {issue.code for issue in issues}


def test_compile_emits_single_in_process_train_step(tmp_path):
    ctx = _ctx(tmp_path)
    plan = NativeLLMBackend().compile(_spec(adapter=Adapter.LORA), ctx)

    assert plan.backend == "native-llm"
    assert len(plan.steps) == 1
    step = plan.steps[0]
    assert step.kind == "train"
    assert step.command is None
    assert step.cwd == str(ctx.run_dir)
    assert plan.expected_artifacts == [
        str(ctx.run_dir / "output" / "adapter" / "adapter_model.safetensors"),
        str(ctx.run_dir / "output" / "eval_report.json"),
    ]


def test_full_finetune_artifact_is_a_full_model(tmp_path):
    ctx = _ctx(tmp_path)
    plan = NativeLLMBackend().compile(_spec(adapter=Adapter.NONE), ctx)
    assert plan.expected_artifacts == [
        str(ctx.run_dir / "output" / "model" / "model.safetensors"),
        str(ctx.run_dir / "output" / "eval_report.json"),
    ]


def test_preference_training_compiles_with_lora(tmp_path):
    plan = NativeLLMBackend().compile(_spec(stage=Stage.PREFERENCE), _ctx(tmp_path))
    assert "E_PREFERENCE_UNSUPPORTED" not in _codes(plan.issues)
    assert [step.id for step in plan.steps] == ["train"]


def test_preference_full_finetune_is_refused(tmp_path):
    spec = _spec(stage=Stage.PREFERENCE)
    spec.config.adapter = Adapter.NONE
    plan = NativeLLMBackend().compile(spec, _ctx(tmp_path))
    assert "E_PREFERENCE_FULL_UNSUPPORTED" in _codes(plan.issues)


def test_rl_training_compiles_with_lora(tmp_path):
    plan = NativeLLMBackend().compile(_spec(stage=Stage.RL), _ctx(tmp_path))
    assert "E_RL_UNSUPPORTED" not in _codes(plan.issues)
    assert [step.id for step in plan.steps] == ["train"]


def test_rl_full_finetune_is_refused(tmp_path):
    spec = _spec(stage=Stage.RL)
    spec.config.adapter = Adapter.NONE
    plan = NativeLLMBackend().compile(spec, _ctx(tmp_path))
    assert "E_RL_FULL_UNSUPPORTED" in _codes(plan.issues)


def test_distillation_is_refused(tmp_path):
    plan = NativeLLMBackend().compile(_spec(stage=Stage.DISTILL), _ctx(tmp_path))
    assert "E_DISTILL_UNSUPPORTED" in _codes(plan.issues)


def test_qlora_without_bitsandbytes_warns(tmp_path):
    plan = NativeLLMBackend().compile(_spec(adapter=Adapter.QLORA, quant_base=QuantBase.NF4), _ctx(tmp_path))
    warnings = [issue for issue in plan.issues if issue.code == "W_BITSANDBYTES_MISSING"]
    assert warnings and warnings[0].severity == "warning"
    assert warnings[0].hint


def test_eight_bit_optimizer_without_bitsandbytes_warns(tmp_path):
    plan = NativeLLMBackend().compile(_spec(optimizer=Optimizer.ADAMW_8BIT), _ctx(tmp_path))
    assert "W_BITSANDBYTES_MISSING" in _codes(plan.issues)


def test_came_optimizer_is_refused(tmp_path):
    plan = NativeLLMBackend().compile(_spec(optimizer=Optimizer.CAME), _ctx(tmp_path))
    assert "E_OPTIMIZER_UNSUPPORTED" in _codes(plan.issues)


def test_int8_base_is_refused_for_llm(tmp_path):
    plan = NativeLLMBackend().compile(_spec(quant_base=QuantBase.INT8), _ctx(tmp_path))
    assert "E_QUANT_UNSUPPORTED" in _codes(plan.issues)


def test_fp32_master_is_refused(tmp_path):
    plan = NativeLLMBackend().compile(_spec(precision=Precision.FP32_MASTER), _ctx(tmp_path))
    assert "E_PRECISION_UNSUPPORTED" in _codes(plan.issues)


def test_missing_dataset_is_refused(tmp_path):
    spec = _spec()
    spec.dataset = None
    plan = NativeLLMBackend().compile(spec, _ctx(tmp_path))
    assert "E_DATASET_REQUIRED" in _codes(plan.issues)


def test_unsupported_dataset_format_is_refused(tmp_path):
    spec = _spec()
    spec.dataset = DatasetSpec(path="clips", kind=DatasetKind.TEXT, format="webdataset")
    plan = NativeLLMBackend().compile(spec, _ctx(tmp_path))
    assert "E_DATASET_FORMAT" in _codes(plan.issues)


def test_missing_model_dir_warns(tmp_path):
    plan = NativeLLMBackend().compile(_spec(), _ctx(tmp_path))
    assert "W_MODEL_PATH_MISSING" in _codes(plan.issues)


def test_explicit_model_path_is_used_when_present(tmp_path):
    model_dir = tmp_path / "hf-snapshot"
    model_dir.mkdir()
    plan = NativeLLMBackend().compile(_spec(model_path=str(model_dir)), _ctx(tmp_path))
    assert "W_MODEL_PATH_MISSING" not in _codes(plan.issues)


def test_blocks_to_swap_is_warned_as_ignored(tmp_path):
    plan = NativeLLMBackend().compile(_spec(blocks_to_swap=8), _ctx(tmp_path))
    assert "W_BLOCK_SWAP_IGNORED" in _codes(plan.issues)


def test_probe_never_raises_and_reports_optional_bitsandbytes(tmp_path):
    issues = NativeLLMBackend().probe(_ctx(tmp_path))
    assert isinstance(issues, list)
    assert "I_BITSANDBYTES_OPTIONAL" in _codes(issues)


def test_supports_only_llm_families(tmp_path):
    backend = NativeLLMBackend()
    assert backend.supports(_spec())
    assert not backend.supports(_spec(family="minimax-h3"))


def test_video_family_is_refused_by_native_compile(tmp_path):
    plan = NativeLLMBackend().compile(_spec(family="minimax-h3"), _ctx(tmp_path))
    assert "E_FAMILY_UNSUPPORTED" in _codes(plan.issues)


def test_run_step_rejects_non_train_steps(tmp_path):
    from supertuner_engine.schema import PlanStep

    step = PlanStep(id="prepare", kind="prepare", title="prep")
    with pytest.raises(RuntimeError, match="prepare"):
        NativeLLMBackend().run_step(_spec(), step, _ctx(tmp_path), null_emit())


class null_emit:  # noqa: N801 - tiny local stub
    def log(self, *args, **kwargs):
        return None

    def __getattr__(self, name):
        return lambda *args, **kwargs: None


def test_lora_targets_map_short_names():
    assert _lora_targets(["q", "k", "v", "o"]) == ["q_proj", "k_proj", "v_proj", "o_proj"]
    assert _lora_targets(["mlp"]) == ["gate_proj", "up_proj", "down_proj"]
    assert _lora_targets(["q_proj"]) == ["q_proj"]
    assert _lora_targets(["q", "q"]) == ["q_proj"]


def test_packing_only_for_pretrain():
    assert _use_packing(RunConfig(name="p", family="qwen3-0.6b", stage=Stage.PRETRAIN))
    assert not _use_packing(RunConfig(name="s", family="qwen3-0.6b", stage=Stage.SFT))


def test_pack_produces_full_length_blocks():
    rows = [{"input_ids": [1, 2, 3, 4, 5]}, {"input_ids": [6, 7, 8, 9, 10]}]
    blocks = _pack(rows, 4)
    assert len(blocks) == 2
    assert all(len(block["input_ids"]) == 4 for block in blocks)
    assert blocks[0]["input_ids"] == [1, 2, 3, 4]
    assert blocks[1]["input_ids"] == [5, 6, 7, 8]


def test_pack_returns_empty_when_shorter_than_one_block():
    assert _pack([{"input_ids": [1, 2]}], 8) == []


def test_dataset_path_resolves_against_datasets_dir(tmp_path):
    ctx = _ctx(tmp_path)
    relative = DatasetSpec(path="demo.jsonl")
    assert resolve_dataset_path(relative, ctx) == ctx.datasets_dir / "demo.jsonl"

    absolute = tmp_path / "elsewhere.jsonl"
    spec = DatasetSpec(path=str(absolute))
    assert resolve_dataset_path(spec, ctx) == absolute


def test_run_dir_is_absolute_in_step_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config = EngineConfig.from_env({"SUPERTUNER_HOME": str(tmp_path / "home")})
    ctx = config.context(Path("relative-run"))
    assert ctx.run_dir is not None and ctx.run_dir.is_absolute()
    plan = NativeLLMBackend().compile(_spec(), ctx)
    assert plan.steps[0].cwd == str(ctx.run_dir)