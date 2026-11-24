"""Native LLM backend: transformers + peft supervised fine-tuning.

This backend is the one that really trains on a local GPU. It is deliberately a
single process with no accelerate `Trainer`, because the engine needs control
over cancellation, step events and memory reporting.

Supported in v0.1: LoRA / QLoRA / full fine-tuning of causal LMs on
`stage=pretrain` and `stage=sft`. Preference, RL and distillation are refused
with actionable issues instead of pretending to run (ADR §12).

Everything heavy is imported inside `train()`; `compile()` and `probe()` only
look at package metadata so planning works without torch installed.
"""

from __future__ import annotations

import importlib.util
import json
import math
import os
import random
import sys
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from ..config import Context
from ..events import Emit
from .eval_report import build_eval_report, write_eval_report
from ..schema import (
    LLM_FAMILIES,
    Adapter,
    DatasetKind,
    DatasetSpec,
    ExecutionPlan,
    Issue,
    Optimizer,
    PlanStep,
    PluginManifest,
    Precision,
    QuantBase,
    RunSpec,
    Stage,
)

BACKEND_ID = "native-llm"
BACKEND_VERSION = "0.1.0"

#: Short LoRA target names used by RunConfig -> module suffixes for Llama/Qwen.
_TARGET_SUFFIXES: dict[str, tuple[str, ...]] = {
    "q": ("q_proj",),
    "k": ("k_proj",),
    "v": ("v_proj",),
    "o": ("o_proj",),
    "attn": ("q_proj", "k_proj", "v_proj", "o_proj"),
    "mlp": ("gate_proj", "up_proj", "down_proj"),
    "gate": ("gate_proj",),
    "up": ("up_proj",),
    "down": ("down_proj",),
}

_DATASET_FORMATS = {"auto", "jsonl", "parquet", "directory", "txt", "text"}


class NativeLLMBackend:
    """transformers `AutoModelForCausalLM` + peft LoRA, one GPU, one process."""

    manifest = PluginManifest(
        id=BACKEND_ID,
        name="Native LLM (transformers + peft)",
        version=BACKEND_VERSION,
        kind="backend",
        description="LoRA/QLoRA/全参 SFT，transformers + peft，单卡进程内训练",
        capabilities=[
            "train.llm.pretrain",
            "train.llm.sft",
            "train.llm.lora",
            "train.llm.full",
            "train.llm.rl",
            "train.llm.preference",
        ],
        families=sorted(LLM_FAMILIES),
        entry_point="supertuner_engine.backends.native_llm:NativeLLMBackend",
        requires=["torch", "transformers", "peft", "datasets"],
        license="Apache-2.0",
        homepage="https://github.com/supertuner/supertuner",
        priority=100,
    )

    # -- discovery ---------------------------------------------------------

    def probe(self, ctx: Context) -> list[Issue]:
        """Report the local stack's readiness. Never imports torch."""
        issues: list[Issue] = []
        for module, dist, hint in (
            ("torch", "torch", "uv sync --project engine --extra torch"),
            ("transformers", "transformers", "uv sync --project engine --extra torch"),
            ("peft", "peft", "uv sync --project engine --extra torch"),
            ("datasets", "datasets", "uv sync --project engine --extra torch"),
        ):
            if not _available(module):
                issues.append(
                    Issue(
                        severity="warning",
                        code="W_BACKEND_DEPENDENCY_MISSING",
                        message=f"native LLM 后端缺少 {dist}",
                        hint=hint,
                        field=None,
                    )
                )
        if not _available("bitsandbytes"):
            issues.append(
                Issue(
                    severity="info",
                    code="I_BITSANDBYTES_OPTIONAL",
                    message="bitsandbytes 未安装：QLoRA 与 8-bit 优化器会在运行时被明确拒绝",
                    hint="uv add --project engine --extra quant（Windows 上不保证有 wheel）",
                )
            )
        if not _cuda_visible():
            issues.append(
                Issue(
                    severity="info",
                    code="I_NO_CUDA",
                    message="当前进程看不到 CUDA 设备；native 后端只能在 CPU 上以 fp32 试跑",
                    hint="确认显卡驱动与 torch 的 CUDA 版本匹配",
                )
            )
        return issues

    def supports(self, spec: RunSpec) -> bool:
        return spec.config.family in LLM_FAMILIES

    # -- compile -----------------------------------------------------------

    def compile(self, spec: RunSpec, ctx: Context) -> ExecutionPlan:
        cfg = spec.config
        issues = _static_issues(spec)

        if spec.dataset is not None:
            dataset_path = resolve_dataset_path(spec.dataset, ctx)
            if not dataset_path.exists():
                issues.append(
                    Issue(
                        severity="warning",
                        code="W_DATASET_PATH_MISSING",
                        message=f"数据集路径不存在：{dataset_path}",
                        hint="训练会在加载时失败；先确认 SUPERTUNER_DATASETS_DIR 或 dataset.path",
                        field="dataset.path",
                    )
                )

        model_path = ctx.model_path(cfg.family, cfg.model_path)
        if model_path is None or not model_path.exists():
            issues.append(
                Issue(
                    severity="warning",
                    code="W_MODEL_PATH_MISSING",
                    message=f"基座权重路径不存在：{model_path}",
                    hint="把 HF 快照放进 SUPERTUNER_MODELS_DIR/<family>，或在 config.model_path 里指定绝对路径",
                    field="config.model_path",
                )
            )

        train_step = PlanStep(
            id="train",
            kind="train",
            title=f"{cfg.family} {cfg.stage.value} ({cfg.adapter.value})",
            description=(
                f"native-llm：max_steps={spec.train.max_steps}，batch={cfg.batch}×"
                f"grad_accum={cfg.grad_accum}，seq_len={cfg.seq_len}，precision={cfg.precision.value}"
            ),
            command=None,
            cwd=str(ctx.run_dir) if ctx.run_dir is not None else None,
            env={},
            optional=False,
        )

        artifacts = [
            str(ctx.output_dir / "adapter" / "adapter_model.safetensors")
            if cfg.adapter != Adapter.NONE
            else str(ctx.output_dir / "model" / "model.safetensors"),
            str(ctx.output_dir / "eval_report.json"),
        ]

        notes = [
            "训练在 engine 进程内执行（step 的 command 为 null），事件直接写入 stdout JSONL。",
        ]
        if cfg.adapter == Adapter.QLORA:
            notes.append("QLoRA：基座 4-bit NF4 冻结，仅 LoRA 权重参与优化。")
        if cfg.grad_checkpointing:
            notes.append("已启用 gradient checkpointing（用计算换显存）。")

        return ExecutionPlan(
            backend=BACKEND_ID,
            backend_version=BACKEND_VERSION,
            steps=[train_step],
            issues=issues,
            notes=notes,
            expected_artifacts=artifacts,
        )

    # -- execution ---------------------------------------------------------

    def run(self, spec: RunSpec, plan: ExecutionPlan, ctx: Context, emit: Emit) -> None:
        for step in plan.steps:
            self.run_step(spec, step, ctx, emit)

    def run_step(self, spec: RunSpec, step: PlanStep, ctx: Context, emit: Emit) -> None:
        if step.kind != "train":
            raise RuntimeError(f"native-llm 不执行 '{step.kind}' 步骤（step id={step.id}）")
        train(spec, ctx, emit)


# ---------------------------------------------------------------------------
# static analysis (no torch)
# ---------------------------------------------------------------------------


def _available(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError, ModuleNotFoundError):
        return False


def _cuda_visible() -> bool:
    """Cheap check that does not import torch: nvidia-smi present and healthy."""
    import shutil
    import subprocess

    exe = shutil.which("nvidia-smi")
    if exe is None:
        return False
    try:
        proc = subprocess.run(  # noqa: S603 - fixed argv
            [exe, "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=8,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return proc.returncode == 0 and bool((proc.stdout or "").strip())


def _static_issues(spec: RunSpec) -> list[Issue]:
    cfg = spec.config
    dataset = spec.dataset
    issues: list[Issue] = []

    if cfg.family not in LLM_FAMILIES:
        issues.append(
            Issue(
                severity="error",
                code="E_FAMILY_UNSUPPORTED",
                message=f"native-llm 不训练视频家族 '{cfg.family}'",
                hint="视频请用 musubi backend（backend='musubi'）",
                field="config.family",
            )
        )

    if cfg.stage == Stage.PREFERENCE:
        if cfg.adapter == Adapter.NONE:
            issues.append(
                Issue(
                    severity="error",
                    code="E_PREFERENCE_FULL_UNSUPPORTED",
                    message="偏好训练只支持 adapter=lora/qlora：参考模型由关闭适配器的基座提供",
                    hint="改用 adapter=lora；全参偏好训练需要第二份基座权重，v0.1 未实现",
                    field="config.adapter",
                )
            )
        if dataset is not None and dataset.kind not in (DatasetKind.PREFERENCE, DatasetKind.AUTO):
            issues.append(
                Issue(
                    severity="warning",
                    code="W_PREFERENCE_DATASET_KIND",
                    message=f"dataset.kind={dataset.kind.value} 不是 preference；需要 {dataset.chosen_column}/{dataset.rejected_column} 两列",
                    hint="设置 dataset.kind=preference，并确认数据里有 chosen/rejected 字段",
                    field="dataset.kind",
                )
            )
    if cfg.stage == Stage.RL:
        if cfg.adapter == Adapter.NONE:
            issues.append(
                Issue(
                    severity="error",
                    code="E_RL_FULL_UNSUPPORTED",
                    message="RL 只支持 adapter=lora/qlora：参考模型由关闭适配器的基座提供",
                    hint="改用 adapter=lora；全参 RL 需要第二份基座权重，v0.1 未实现",
                    field="config.adapter",
                )
            )
        if dataset is not None and dataset.kind not in (DatasetKind.CHAT, DatasetKind.AUTO):
            issues.append(
                Issue(
                    severity="warning",
                    code="W_RL_DATASET_KIND",
                    message=f"dataset.kind={dataset.kind.value} 不是 chat；RL 需要 {dataset.prompt_column}/{dataset.answer_column} 两列",
                    hint="设置 dataset.kind=chat，并确认数据里有 prompt/answer 字段",
                    field="dataset.kind",
                )
            )
    if cfg.stage == Stage.DISTILL:
        issues.append(
            Issue(
                severity="error",
                code="E_DISTILL_UNSUPPORTED",
                message="v0.1 不执行 DMD/DMAD 蒸馏：需要教师/学生双模型与额外显存，单卡消费级方案未验证",
                hint="改用 stage=sft 训练学生模型；蒸馏支持计划在 v0.2（ADR #6）",
                field="config.stage",
            )
        )

    if cfg.adapter == Adapter.QLORA and not _available("bitsandbytes"):
        issues.append(
            Issue(
                severity="warning",
                code="W_BITSANDBYTES_MISSING",
                message="QLoRA 需要 bitsandbytes，但当前环境没有安装；运行时会明确失败",
                hint="uv add --project engine --extra quant；Windows 上若没有 wheel，改用 adapter=lora",
                field="config.adapter",
            )
        )
    if cfg.optimizer in (Optimizer.ADAMW_8BIT, Optimizer.ADAMW_PAGED_8BIT) and not _available(
        "bitsandbytes"
    ):
        issues.append(
            Issue(
                severity="warning",
                code="W_BITSANDBYTES_MISSING",
                message=f"optimizer={cfg.optimizer.value} 需要 bitsandbytes，但当前环境没有安装",
                hint="uv add --project engine --extra quant，或改用 optimizer=adamw",
                field="config.optimizer",
            )
        )
    if cfg.optimizer == Optimizer.CAME:
        issues.append(
            Issue(
                severity="error",
                code="E_OPTIMIZER_UNSUPPORTED",
                message="native-llm 未实现 CAME 优化器",
                hint="改用 adamw / adamw_8bit / adamw_paged_8bit",
                field="config.optimizer",
            )
        )

    if cfg.quant_base in (QuantBase.INT8, QuantBase.FP8, QuantBase.NVFP4):
        issues.append(
            Issue(
                severity="error",
                code="E_QUANT_UNSUPPORTED",
                message=f"LLM 家族不支持 quant_base={cfg.quant_base.value} 的权重存储",
                hint="LLM 只用 nf4（QLoRA）；int8/fp8/nvfp4 块交换是 musubi-tuner 的视频路线",
                field="config.quant_base",
            )
        )

    if cfg.precision == Precision.FP32_MASTER:
        issues.append(
            Issue(
                severity="error",
                code="E_PRECISION_UNSUPPORTED",
                message="v0.1 的 native 后端没有 fp32 master weights 实现，不会静默降级为 bf16",
                hint="改用 precision=bf16_pure（Ampere+）或 precision=fp32（占双倍显存）",
                field="config.precision",
            )
        )

    if cfg.stage in (Stage.SFT, Stage.PRETRAIN, Stage.PREFERENCE) and spec.dataset is None:
        issues.append(
            Issue(
                severity="error",
                code="E_DATASET_REQUIRED",
                message=f"stage={cfg.stage.value} 需要 dataset，但 RunSpec.dataset 为空",
                hint="提供 dataset.path 与 kind（text/chat），jsonl/txt/parquet 均可",
                field="dataset",
            )
        )
    if spec.dataset is not None and spec.dataset.format not in _DATASET_FORMATS:
        issues.append(
            Issue(
                severity="error",
                code="E_DATASET_FORMAT",
                message=f"native-llm 不支持 dataset.format={spec.dataset.format}",
                hint=f"支持：{', '.join(sorted(_DATASET_FORMATS))}",
                field="dataset.format",
            )
        )

    if cfg.cache.value != "none":
        issues.append(
            Issue(
                severity="info",
                code="I_CACHE_UNUSED",
                message="LLM 训练不使用 latents/text-encoder 缓存，config.cache 会被忽略",
                field="config.cache",
            )
        )
    if cfg.lora_rank > 256:
        issues.append(
            Issue(
                severity="warning",
                code="W_LORA_RANK_LARGE",
                message=f"lora_rank={cfg.lora_rank} 很大，接近全参微调的显存开销",
                hint="常用 8–64；再大请改用 adapter=none",
                field="config.lora_rank",
            )
        )
    if cfg.seq_len > 32768:
        issues.append(
            Issue(
                severity="warning",
                code="W_SEQ_LEN_LARGE",
                message=f"seq_len={cfg.seq_len} 超出单卡常见范围，attention 激活会主导显存",
                hint="先降到 8192 或启用 packing/gradient checkpointing",
                field="config.seq_len",
            )
        )
    if cfg.adapter != Adapter.NONE and cfg.blocks_to_swap > 0:
        issues.append(
            Issue(
                severity="warning",
                code="W_BLOCK_SWAP_IGNORED",
                message="native-llm 不支持 blocks_to_swap（只有 musubi 视频后端支持）",
                hint="LLM 请用 quant_base=nf4 + adapter=qlora 降显存",
                field="config.blocks_to_swap",
            )
        )
    return issues


def _lora_targets(targets: list[str]) -> list[str]:
    suffixes: list[str] = []
    for name in targets:
        key = name.strip().lower()
        mapped = _TARGET_SUFFIXES.get(key)
        if mapped is None:
            # Already a module suffix such as `q_proj` or a regex fragment.
            mapped = (name.strip(),)
        for suffix in mapped:
            if suffix and suffix not in suffixes:
                suffixes.append(suffix)
    return suffixes


def resolve_dataset_path(dataset: DatasetSpec, ctx: Context) -> Path:
    path = Path(dataset.path)
    if path.is_absolute():
        return path
    return ctx.datasets_dir / path


# ---------------------------------------------------------------------------
# training (torch imported here and only here)
# ---------------------------------------------------------------------------


def configure_cuda_allocator() -> None:
    """Use the async allocator on Windows.

    The default caching allocator fragments badly against the WDDM per-process
    video-memory budget: a real 0.6B LoRA run died at ~1.7 GiB with several GiB
    reported free, while `cudaMallocAsync` completed the same run at ~2.7 GiB
    peak. Keep a user-supplied `backend:` setting untouched.
    """
    if sys.platform != "win32":
        return
    current = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "").strip()
    if "backend:" in current:
        return
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = f"{current},backend:cudaMallocAsync".strip(",")


class _PeakTracker:
    """Running max of per-step VRAM peaks; the training-loop peak for the run."""

    def __init__(self) -> None:
        self.peak_mib = 0.0

    def record(self, peak_mib: float | None) -> None:
        if peak_mib is not None and peak_mib > 0:
            self.peak_mib = max(self.peak_mib, float(peak_mib))


def _planned_peak_mib(spec: RunSpec) -> float | None:
    """The planner's training-peak prediction for this exact config, if computable."""
    try:
        from ..plan import plan as _planner  # noqa: PLC0415

        return float(_planner(spec.config).training_peak_mib)
    except Exception:  # noqa: BLE001 - the report must never break a finished run
        return None


def _emit_memory_report(
    device: str,
    spec: RunSpec,
    ctx: Context,
    emit: Emit,
    tracker: _PeakTracker,
) -> None:
    """Write memory_report.json and emit a memory_report event (planned vs measured)."""
    measured = float(tracker.peak_mib) if tracker.peak_mib > 0 else None
    planned = _planned_peak_mib(spec)
    delta = (measured - planned) if (measured is not None and planned is not None) else None
    delta_pct = (delta / planned * 100.0) if (delta is not None and planned) else None
    report = {
        "planned_mib": planned,
        "measured_peak_mib": measured,
        "delta_mib": delta,
        "delta_pct": delta_pct,
        "device": device,
        "family": spec.config.family,
        "stage": spec.config.stage.value,
        "adapter": spec.config.adapter.value,
        # 跑后自校准（v0.2 T1）：补齐五元组校准键的两个维度，
        # 使 server 能按 (family, stage, adapter, quant_base, blocks_to_swap) 聚合。
        "quant_base": spec.config.quant_base.value,
        "blocks_to_swap": spec.config.blocks_to_swap,
    }
    out = ctx.output_dir / "memory_report.json"
    try:
        out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    except OSError as exc:  # pragma: no cover - disk failure must not kill the run
        emit.log("warn", f"memory_report.json 写入失败：{exc}")
    emit.memory_report(report)
    if measured is not None and planned is not None:
        emit.log(
            "info",
            f"显存对照：计划 {planned:.0f} MiB，实测峰值 {measured:.0f} MiB"
            f"（差 {delta:+.0f} MiB / {delta_pct:+.1f}%）",
        )


def train(spec: RunSpec, ctx: Context, emit: Emit) -> None:
    cfg = spec.config
    dataset = spec.dataset
    if dataset is None:
        raise RuntimeError("native-llm: sft/pretrain 需要 dataset")
    if cfg.family not in LLM_FAMILIES:
        raise RuntimeError(f"native-llm: 不支持家族 '{cfg.family}'")
    if cfg.stage not in (Stage.SFT, Stage.PRETRAIN, Stage.PREFERENCE, Stage.RL):
        raise RuntimeError(f"native-llm: v0.1 不支持 stage={cfg.stage.value}")
    if cfg.precision == Precision.FP32_MASTER:
        raise RuntimeError("native-llm: v0.1 不支持 precision=fp32_master，请改用 bf16_pure 或 fp32")

    configure_cuda_allocator()
    import torch  # noqa: PLC0415
    from peft import LoraConfig, PeftModel, get_peft_model  # noqa: PLC0415
    from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: PLC0415

    try:  # transformers >= 5 moved get_scheduler into transformers.optimization
        from transformers import get_scheduler  # noqa: PLC0415
    except ImportError:  # pragma: no cover - depends on the installed transformers
        from transformers.optimization import get_scheduler  # noqa: PLC0415

    device = _select_device(torch, cfg.precision, emit)
    dtype = _select_dtype(torch, cfg.precision, device, emit)
    tracker = _PeakTracker()
    train_curve: list[dict[str, Any]] = []
    eval_curve: list[dict[str, Any]] = []

    model_path = ctx.model_path(cfg.family, cfg.model_path)
    if model_path is None or not model_path.exists():
        raise RuntimeError(
            f"基座权重不存在：{model_path}。把 HF 快照放到 SUPERTUNER_MODELS_DIR/{cfg.family} "
            "或设置 config.model_path"
        )

    emit.log("info", f"加载 tokenizer：{model_path}")
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    pad_id = int(tokenizer.pad_token_id)

    dataset_path = resolve_dataset_path(dataset, ctx)
    if cfg.stage == Stage.PREFERENCE:
        pairs = _load_preference(dataset, dataset_path, tokenizer, cfg, emit, seed=spec.train.seed)
        train_batches = _make_pref_loader(pairs, cfg, pad_id, shuffle=True, seed=spec.train.seed)
        eval_batches = None
    elif cfg.stage == Stage.RL:
        rl_samples = _load_rl(dataset, dataset_path, tokenizer, cfg, emit, seed=spec.train.seed)
        train_batches = []
        eval_batches = None
    else:
        train_ds, eval_ds = _load_split(dataset, dataset_path, tokenizer, cfg, emit, seed=spec.train.seed)
        train_batches = _make_loader(train_ds, cfg, pad_id, shuffle=True, seed=spec.train.seed)
        eval_batches = (
            _make_loader(eval_ds, cfg, pad_id, shuffle=False, seed=spec.train.seed) if eval_ds else None
        )

    # -- model -------------------------------------------------------------
    quant_config = None
    if cfg.adapter == Adapter.QLORA:
        if not _available("bitsandbytes"):
            raise RuntimeError(
                "QLoRA 需要 bitsandbytes：请 `uv add --project engine --extra quant`，"
                "或在 Windows 上改用 adapter=lora"
            )
        from transformers import BitsAndBytesConfig  # noqa: PLC0415

        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype,
        )

    emit.log("info", f"加载基座：{model_path}（dtype={dtype}，adapter={cfg.adapter.value}）")
    load_kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "dtype": dtype,
    }
    if quant_config is not None:
        load_kwargs["quantization_config"] = quant_config
        load_kwargs["device_map"] = {"": 0} if device == "cuda" else None
    else:
        load_kwargs["low_cpu_mem_usage"] = True
    model = _load_causal_lm(AutoModelForCausalLM, str(model_path), load_kwargs)
    if quant_config is None:
        # bnb 4-bit models are already placed by device_map; .to() on them is invalid.
        model.to(device)

    if cfg.grad_checkpointing:
        try:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        except TypeError:  # pragma: no cover - older transformers
            model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        emit.log("info", "gradient checkpointing 已启用")

    resume_dir = _resolve_resume(spec, ctx)
    if cfg.adapter != Adapter.NONE:
        resume = resume_dir is not None and (resume_dir / "adapter_config.json").is_file()
        if resume_dir is not None and not resume:
            raise RuntimeError(
                f"resume_from={resume_dir} 不是有效的 LoRA 适配器目录（缺少 adapter_config.json）"
            )
        if resume:
            emit.log("info", f"从适配器恢复：{resume_dir}")
            model = PeftModel.from_pretrained(model, str(resume_dir), is_trainable=True)
        else:
            lora = LoraConfig(
                r=cfg.lora_rank,
                lora_alpha=cfg.lora_alpha,
                lora_dropout=0.0,
                bias="none",
                task_type="CAUSAL_LM",
                target_modules=_lora_targets(cfg.lora_targets),
            )
            model = get_peft_model(model, lora)
        trainable = [p for p in model.parameters() if p.requires_grad]
        emit.log("info", f"可训练参数：{sum(p.numel() for p in trainable):,}")
    else:
        if resume_dir is not None:
            emit.log("warn", "全参训练暂不支持从目录恢复，忽略 train.resume_from")
        trainable = [p for p in model.parameters() if p.requires_grad]

    if not trainable:
        raise RuntimeError("所选 LoRA 目标没有匹配到任何模块，无法训练；检查 config.lora_targets")

    optimizer = _build_optimizer(torch, cfg.optimizer, spec.train, trainable)
    total_steps = int(spec.train.max_steps)
    if total_steps <= 0:
        raise RuntimeError("train.max_steps 必须 > 0")
    warmup = min(int(spec.train.warmup_steps), total_steps)
    scheduler = get_scheduler(
        name=spec.train.scheduler,
        optimizer=optimizer,
        num_warmup_steps=warmup,
        num_training_steps=total_steps,
    )

    start_step = 0
    if resume_dir is not None and (resume_dir / "optimizer.pt").is_file():
        state = json.loads((resume_dir / "trainer_state.json").read_text(encoding="utf-8"))
        start_step = int(state.get("global_step", 0))
        optimizer.load_state_dict(torch.load(resume_dir / "optimizer.pt", map_location=device))
        if (resume_dir / "scheduler.pt").is_file():
            scheduler.load_state_dict(torch.load(resume_dir / "scheduler.pt", map_location=device))
        emit.log("info", f"恢复优化器状态：从 step {start_step} 继续")

    if cfg.stage == Stage.RL:
        _train_grpo(
            torch,
            model,
            rl_samples,
            optimizer,
            scheduler,
            tokenizer,
            spec,
            cfg,
            ctx,
            emit,
            device=device,
            pad_id=pad_id,
            start_step=start_step,
            total_steps=total_steps,
            tracker=tracker,
            train_curve=train_curve,
            eval_curve=eval_curve,
        )
        _emit_eval_report(
            spec,
            ctx,
            emit,
            train_curve,
            eval_curve,
            extra={"stage": Stage.RL.value},
            model=model,
            tokenizer=tokenizer,
            device=device,
        )
        _emit_memory_report(device, spec, ctx, emit, tracker)
        return

    if cfg.stage == Stage.PREFERENCE:
        _train_dpo(
            torch,
            model,
            train_batches,
            optimizer,
            scheduler,
            tokenizer,
            spec,
            cfg,
            ctx,
            emit,
            start_step=start_step,
            total_steps=total_steps,
            tracker=tracker,
            train_curve=train_curve,
            eval_curve=eval_curve,
        )
        _emit_eval_report(
            spec,
            ctx,
            emit,
            train_curve,
            eval_curve,
            extra={"stage": Stage.PREFERENCE.value},
            model=model,
            tokenizer=tokenizer,
            device=device,
        )
        _emit_memory_report(device, spec, ctx, emit, tracker)
        return

    amp_dtype = None
    if device == "cuda" and dtype != torch.float32:
        amp_dtype = dtype if cfg.adapter == Adapter.QLORA else None

    model.train()
    emit.log("info", f"开始训练：target {total_steps} steps，device={device}")

    grad_accum = max(int(cfg.grad_accum), 1)
    batch_stream = _forever(train_batches)
    step = start_step
    micro = 0
    window_tokens = 0
    window_loss = 0.0
    window_start = time.monotonic()
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    while step < total_steps:
        ctx.cancel.check()
        batch = next(batch_stream)
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        tokens = int(batch["attention_mask"].sum().item())

        if amp_dtype is not None:
            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                out = model(**batch)
        else:
            out = model(**batch)
        loss = out.loss / grad_accum
        if not math.isfinite(float(loss.detach())):
            raise RuntimeError(f"step {step + 1}: loss 非有限值（{float(loss.detach())}），停止训练")
        loss.backward()

        micro += 1
        window_tokens += tokens
        window_loss += float(loss.detach()) * grad_accum
        if micro < grad_accum:
            continue

        grad_norm = torch.nn.utils.clip_grad_norm_(trainable, float(spec.train.grad_clip))
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)

        step += 1
        micro = 0
        elapsed = time.monotonic() - window_start
        vram_peak = (
            torch.cuda.max_memory_allocated() / (1024 * 1024) if device == "cuda" else None
        )
        tracker.record(vram_peak)
        step_loss = window_loss / grad_accum
        step_lr = float(scheduler.get_last_lr()[0])
        step_tps = (window_tokens / elapsed) if elapsed > 0 else None
        emit.step(
            step,
            loss=step_loss,
            lr=step_lr,
            grad_norm=float(grad_norm),
            tokens_per_s=step_tps,
            vram_peak_mib=float(vram_peak) if vram_peak is not None else None,
            elapsed_s=elapsed,
        )
        train_curve.append(
            {
                "step": step,
                "loss": float(step_loss),
                "lr": step_lr,
                "grad_norm": float(grad_norm),
                "tokens_per_s": float(step_tps) if step_tps is not None else None,
                "vram_peak_mib": float(vram_peak) if vram_peak is not None else None,
                "elapsed_s": float(elapsed),
            }
        )

        if eval_batches is not None and spec.train.eval_every > 0 and step % spec.train.eval_every == 0:
            metrics = _evaluate(torch, model, eval_batches, device, amp_dtype, emit)
            if metrics:
                emit.eval(step, metrics)
                eval_curve.append({"step": step, "metrics": dict(metrics)})

        if spec.train.save_every > 0 and step % spec.train.save_every == 0 and step < total_steps:
            _save_checkpoint(
                model, tokenizer, optimizer, scheduler, step, spec, ctx, emit,
                ctx.output_dir / f"checkpoint-{step}",
            )

        window_tokens = 0
        window_loss = 0.0
        window_start = time.monotonic()
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()

    final_dir = ctx.output_dir / ("adapter" if cfg.adapter != Adapter.NONE else "model")
    _save_checkpoint(model, tokenizer, optimizer, scheduler, step, spec, ctx, emit, final_dir)
    emit.log("info", f"训练完成：{step} steps，产物在 {final_dir}")
    _emit_eval_report(
        spec,
        ctx,
        emit,
        train_curve,
        eval_curve,
        extra={"stage": cfg.stage.value},
        model=model,
        tokenizer=tokenizer,
        device=device,
        amp_dtype=amp_dtype,
    )
    _emit_memory_report(device, spec, ctx, emit, tracker)



def _emit_eval_report(
    spec: RunSpec,
    ctx: Context,
    emit: Emit,
    train_curve: list[dict[str, Any]],
    eval_curve: list[dict[str, Any]],
    extra: dict[str, Any] | None = None,
    *,
    prompt_eval: dict[str, Any] | None = None,
    model: Any = None,
    tokenizer: Any = None,
    device: str | None = None,
    amp_dtype: Any = None,
) -> dict[str, Any]:
    eval_cfg = getattr(spec, "eval", None)
    prompts_path = getattr(eval_cfg, "prompts_path", None) if eval_cfg is not None else None
    enabled = True if eval_cfg is None else bool(getattr(eval_cfg, "enabled", True))
    if prompt_eval is None and enabled and prompts_path and model is not None and tokenizer is not None and device is not None:
        try:
            from .llm_prompt_eval import run_llm_prompt_eval  # noqa: PLC0415

            prompt_eval = run_llm_prompt_eval(
                model=model,
                tokenizer=tokenizer,
                spec=spec,
                ctx=ctx,
                emit=emit,
                device=device,
                amp_dtype=amp_dtype,
            )
        except Exception as exc:  # noqa: BLE001 - prompt eval must not break training artifacts
            emit.log("warn", f"prompt eval 执行失败：{exc}")
            prompt_eval = {"status": "unavailable", "reason": "prompt_eval_exception", "error": str(exc)}
    try:
        report = build_eval_report(
            spec=spec,
            train_curve=train_curve,
            eval_curve=eval_curve,
            extra=extra,
            prompt_eval=prompt_eval,
        )
    except Exception as exc:
        emit.log("warn", f"eval_report 生成失败：{exc}")
        report = {"version": 1, "status": "unavailable", "reason": str(exc), "summary": {}, "contract": {}}
    out_path = ctx.output_dir / "eval_report.json"
    try:
        write_eval_report(out_path, report)
        path_str: str | None = str(out_path)
    except Exception as exc:
        emit.log("warn", f"eval_report.json 写入失败：{exc}")
        path_str = None
    emit.eval_report({"status": report.get("status", "ok"), "path": path_str, "report": report})
    return report
def _sequence_logprob(
    torch: Any,
    model: Any,
    input_ids: Any,
    attention_mask: Any,
    *,
    chunk: int = 128,
) -> Any:
    """Mean token log-probability per sequence (length-normalised)."""
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits[:, :-1, :]
    labels = input_ids[:, 1:]
    mask = attention_mask[:, 1:].to(torch.float32)
    totals = torch.zeros(input_ids.shape[0], device=input_ids.device, dtype=torch.float32)
    for start in range(0, labels.shape[1], chunk):
        piece = logits[:, start : start + chunk, :].float()
        target = labels[:, start : start + chunk]
        token_logp = torch.log_softmax(piece, dim=-1).gather(-1, target.unsqueeze(-1)).squeeze(-1)
        totals = totals + (token_logp * mask[:, start : start + chunk]).sum(dim=1)
    return totals / mask.sum(dim=1).clamp(min=1.0)


def _dpo_loss(
    torch: Any,
    policy_chosen: Any,
    policy_rejected: Any,
    ref_chosen: Any,
    ref_rejected: Any,
    beta: float,
) -> tuple[Any, Any]:
    """Direct Preference Optimization loss and the implicit reward margin."""
    margin = (policy_chosen - ref_chosen) - (policy_rejected - ref_rejected)
    loss = -torch.nn.functional.logsigmoid(beta * margin).mean()
    return loss, margin


def _train_dpo(
    torch: Any,
    model: Any,
    train_batches: Any,
    optimizer: Any,
    scheduler: Any,
    tokenizer: Any,
    spec: RunSpec,
    cfg: Any,
    ctx: Context,
    emit: Emit,
    *,
    start_step: int,
    total_steps: int,
    tracker: _PeakTracker | None = None,
    train_curve: list[dict[str, Any]] | None = None,
    eval_curve: list[dict[str, Any]] | None = None,
) -> None:
    """Preference optimisation with a LoRA policy and the frozen base as reference."""
    if cfg.adapter == Adapter.NONE:
        raise RuntimeError(
            "偏好训练需要 adapter=lora/qlora：参考模型由关闭适配器的基座提供，"
            "全参微调需要第二份基座权重，v0.1 未实现"
        )
    if not hasattr(model, "disable_adapter"):
        raise RuntimeError("当前模型包装不支持 disable_adapter，无法构造参考模型")

    trainable = [p for p in model.parameters() if p.requires_grad]
    beta = float(spec.train.dpo_beta)
    grad_accum = max(int(cfg.grad_accum), 1)
    batch_stream = _forever(train_batches)
    step = start_step
    micro = 0
    window_loss = 0.0
    window_margin = 0.0
    window_start = time.monotonic()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    emit.log("info", f"DPO 训练开始：beta={beta}，target {total_steps} steps")

    while step < total_steps:
        ctx.cancel.check()
        batch = next(batch_stream)
        batch = {k: v.to("cuda" if torch.cuda.is_available() else "cpu") for k, v in batch.items()}

        with torch.no_grad():
            with model.disable_adapter():
                ref_chosen = _sequence_logprob(
                    torch, model, batch["chosen_input_ids"], batch["chosen_attention_mask"]
                )
                ref_rejected = _sequence_logprob(
                    torch, model, batch["rejected_input_ids"], batch["rejected_attention_mask"]
                )

        policy_chosen = _sequence_logprob(
            torch, model, batch["chosen_input_ids"], batch["chosen_attention_mask"]
        )
        policy_rejected = _sequence_logprob(
            torch, model, batch["rejected_input_ids"], batch["rejected_attention_mask"]
        )
        loss, margin = _dpo_loss(
            torch, policy_chosen, policy_rejected, ref_chosen, ref_rejected, beta
        )
        if not math.isfinite(float(loss.detach())):
            raise RuntimeError(f"step {step + 1}: DPO loss 非有限值（{float(loss.detach())}），停止训练")
        (loss / grad_accum).backward()

        micro += 1
        window_loss += float(loss.detach())
        window_margin += float(margin.detach().mean())
        if micro < grad_accum:
            continue

        grad_norm = torch.nn.utils.clip_grad_norm_(trainable, float(spec.train.grad_clip))
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)

        step += 1
        micro = 0
        elapsed = time.monotonic() - window_start
        vram_peak = (
            torch.cuda.max_memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else None
        )
        if tracker is not None:
            tracker.record(vram_peak)
        step_lr = float(scheduler.get_last_lr()[0])
        emit.step(
            step,
            loss=window_loss,
            lr=step_lr,
            grad_norm=float(grad_norm),
            tokens_per_s=None,
            vram_peak_mib=float(vram_peak) if vram_peak is not None else None,
            elapsed_s=elapsed,
        )
        if train_curve is not None:
            train_curve.append(
                {
                    "step": step,
                    "loss": float(window_loss),
                    "lr": step_lr,
                    "grad_norm": float(grad_norm),
                    "vram_peak_mib": float(vram_peak) if vram_peak is not None else None,
                    "elapsed_s": float(elapsed),
                    "reward_margin": float(window_margin),
                }
            )
        emit.log("debug", f"DPO step {step}: 隐式奖励间隔均值 {window_margin:.4f}")
        window_loss = 0.0
        window_margin = 0.0
        window_start = time.monotonic()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        if spec.train.save_every > 0 and step % spec.train.save_every == 0 and step < total_steps:
            _save_checkpoint(
                model, tokenizer, optimizer, scheduler, step, spec, ctx, emit,
                ctx.output_dir / f"checkpoint-{step}",
            )

    final_dir = ctx.output_dir / "adapter"
    _save_checkpoint(model, tokenizer, optimizer, scheduler, step, spec, ctx, emit, final_dir)
    emit.log("info", f"DPO 训练完成：{step} steps，产物在 {final_dir}")


def _select_device(torch: Any, precision: Precision, emit: Emit) -> str:
    if torch.cuda.is_available():
        return "cuda"
    if precision == Precision.FP32:
        emit.log("warn", "没有 CUDA 设备：以 fp32 在 CPU 上训练（非常慢）")
        return "cpu"
    raise RuntimeError(
        "native-llm 需要 CUDA 才能以 bf16 训练；未检测到可用 GPU。"
        "确认 torch 的 CUDA 版本与驱动匹配，或显式使用 precision=fp32 在 CPU 上试跑"
    )


def _select_dtype(torch: Any, precision: Precision, device: str, emit: Emit) -> Any:
    if precision == Precision.FP32:
        return torch.float32
    if device == "cuda" and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    raise RuntimeError(
        "precision=bf16_pure 需要支持 bf16 的 GPU（Ampere 或更新）；"
        "当前设备不支持。不会静默降级为 fp16——请改用 precision=fp32 或换卡"
    )


def _load_causal_lm(model_cls: Any, path: str, kwargs: dict[str, Any]) -> Any:
    try:
        return model_cls.from_pretrained(path, **kwargs)
    except TypeError as exc:
        if "dtype" not in str(exc):
            raise
        legacy = dict(kwargs)
        legacy["torch_dtype"] = legacy.pop("dtype")
        return model_cls.from_pretrained(path, **legacy)


def _load_raw(dataset: DatasetSpec, path: Path, *, seed: int = 42) -> tuple[Any, Any | None]:
    """Load the dataset and an optional validation split, before tokenisation."""
    from datasets import load_dataset  # noqa: PLC0415

    if not path.exists():
        raise RuntimeError(f"数据集不存在：{path}")
    fmt = dataset.format
    if fmt == "auto":
        suffix = path.suffix.lower()
        if path.is_dir():
            fmt = "directory"
        elif suffix in (".jsonl", ".json"):
            fmt = "jsonl"
        elif suffix == ".parquet":
            fmt = "parquet"
        elif suffix in (".txt", ".text"):
            fmt = "txt"
        else:
            raise RuntimeError(f"无法从扩展名判断数据集格式：{path}；请设置 dataset.format")

    if fmt == "jsonl":
        raw = load_dataset("json", data_files=str(path), split="train")
    elif fmt == "parquet":
        raw = load_dataset("parquet", data_files=str(path), split="train")
    elif fmt == "txt":
        raw = load_dataset("text", data_files=str(path), split="train")
    elif fmt == "directory":
        files = sorted(str(p) for p in path.rglob("*") if p.suffix.lower() in (".jsonl", ".json", ".parquet", ".txt"))
        if not files:
            raise RuntimeError(f"目录里没有 jsonl/json/parquet/txt：{path}")
        kind = "parquet" if files[0].endswith(".parquet") else ("json" if files[0].endswith((".jsonl", ".json")) else "text")
        raw = load_dataset(kind, data_files=files, split="train")
    else:
        raise RuntimeError(f"native-llm 不支持 dataset.format={fmt}")

    if dataset.max_samples:
        raw = raw.select(range(min(int(dataset.max_samples), len(raw))))

    val = None
    if dataset.val_fraction and 0 < dataset.val_fraction < 1:
        split = raw.train_test_split(test_size=dataset.val_fraction, seed=seed)
        raw, val = split["train"], split["test"]
    return raw, val


def _load_split(
    dataset: DatasetSpec,
    path: Path,
    tokenizer: Any,
    cfg: Any,
    emit: Emit,
    *,
    seed: int = 42,
) -> tuple[list[dict[str, list[int]]], list[dict[str, list[int]]] | None]:
    raw, val = _load_raw(dataset, path, seed=seed)

    text_column = dataset.text_column
    messages_column = dataset.messages_column
    chat = dataset.kind == DatasetKind.CHAT

    def tokenize(example: dict[str, Any]) -> dict[str, list[int]]:
        if chat:
            messages = example.get(messages_column)
            if isinstance(messages, str):
                messages = json.loads(messages)
            if not isinstance(messages, list):
                raise RuntimeError(f"chat 样本缺少 {messages_column} 列表字段")
            ids = tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=False
            )
        else:
            text = example.get(text_column)
            if text is None:
                for candidate in ("text", "content", "prompt"):
                    if candidate in example:
                        text = example[candidate]
                        break
            if text is None:
                raise RuntimeError(f"text 样本缺少非空字段 {text_column}")
            if not isinstance(text, str):
                raise RuntimeError(f"text 字段 {text_column} 不是字符串：{type(text).__name__}")
            if not text.strip():
                # Blank lines are normal in plain-text corpora (wikitext is full
                # of them). Drop the row instead of aborting the whole run.
                return {"input_ids": []}
            ids = tokenizer(text, add_special_tokens=True)["input_ids"]
        return {"input_ids": list(ids)[: int(cfg.seq_len)]}

    train_ds = [tokenize(row) for row in raw]
    train_ds = [row for row in train_ds if row["input_ids"]]
    emit.log("info", f"数据集：{len(train_ds)} 条训练样本")
    if not train_ds:
        raise RuntimeError(f"数据集为空或全部样本被截断：{path}")

    eval_ds = None
    if val is not None and len(val) > 0:
        eval_ds = [tokenize(row) for row in val]
        eval_ds = [row for row in eval_ds if row["input_ids"]]
        emit.log("info", f"验证集：{len(eval_ds)} 条样本")
    return train_ds, eval_ds


def _normalise_answer(text: str) -> str:
    """Lower-case, strip punctuation and whitespace so exact-match rewards are stable."""
    return "".join(ch for ch in text.lower() if ch.isalnum() or ch in ".-")


def _reward_matches(answer: str, completion: str) -> bool:
    """Numeric answers compare as integers; text answers compare normalised strings.

    Comparing numbers as integers avoids the substring trap where "7" would match
    an unrelated completion such as "17".
    """
    target = answer.strip()
    if target.lstrip("-").isdigit():
        import re

        match = re.search(r"-?\d+", completion)
        return bool(match) and int(match.group()) == int(target)
    return _normalise_answer(target) in _normalise_answer(completion)


def _load_rl(
    dataset: DatasetSpec,
    path: Path,
    tokenizer: Any,
    cfg: Any,
    emit: Emit,
    *,
    seed: int = 42,
) -> list[dict[str, str]]:
    """RL samples: a prompt plus a reference answer used by the exact-match reward."""
    raw, _ = _load_raw(dataset, path, seed=seed)
    rows: list[dict[str, str]] = []
    for example in raw:
        prompt = example.get(dataset.prompt_column)
        answer = example.get(dataset.answer_column)
        if not isinstance(prompt, str) or not isinstance(answer, str):
            raise RuntimeError(
                f"RL 样本缺少字符串字段 {dataset.prompt_column}/{dataset.answer_column}"
            )
        if not prompt.strip() or not answer.strip():
            continue
        rows.append({"prompt": prompt, "answer": answer})
    emit.log("info", f"RL 数据集：{len(rows)} 条 prompt/answer 样本")
    if not rows:
        raise RuntimeError(f"RL 数据集为空：{path}")
    return rows


def _group_advantages(torch: Any, rewards: Any, eps: float = 1e-6) -> Any:
    """Group-relative advantages inside one prompt's rollout group (GRPO)."""
    mean = rewards.mean()
    std = rewards.std(unbiased=False)
    return (rewards - mean) / (std + eps)


def _completion_logprobs(
    torch: Any,
    model: Any,
    input_ids: Any,
    attention_mask: Any,
    prompt_lengths: Any,
) -> Any:
    """Mean log-probability of the completion tokens only, one value per row."""
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits[:, :-1, :].float()
    labels = input_ids[:, 1:]
    mask = attention_mask[:, 1:].to(torch.float32)
    token_logp = torch.log_softmax(logits, dim=-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    # zero out the prompt part; prompt_lengths counts the prompt tokens before the first completion token
    completion_mask = mask.clone()
    for row, length in enumerate(prompt_lengths.tolist()):
        completion_mask[row, : max(int(length) - 1, 0)] = 0.0
    return (token_logp * completion_mask).sum(dim=1) / completion_mask.sum(dim=1).clamp(min=1.0)


def _train_grpo(
    torch: Any,
    model: Any,
    samples: list[dict[str, str]],
    optimizer: Any,
    scheduler: Any,
    tokenizer: Any,
    spec: RunSpec,
    cfg: Any,
    ctx: Context,
    emit: Emit,
    *,
    device: str,
    pad_id: int,
    start_step: int,
    total_steps: int,
    tracker: _PeakTracker | None = None,
    train_curve: list[dict[str, Any]] | None = None,
    eval_curve: list[dict[str, Any]] | None = None,
) -> None:
    """GRPO-style policy gradient with an exact-match reward and a KL term.

    The reward is computed from the dataset answer, so no separate reward model is
    needed. The reference model is the same base with its adapter disabled.
    """
    group = max(int(cfg.rl_group_size), 2)
    max_new = int(cfg.rl_max_new_tokens)
    seq_len = int(cfg.seq_len)
    beta = float(cfg.rl_beta)
    temperature = max(float(cfg.rl_temperature), 0.01)
    model.train()
    emit.log(
        "info",
        f"GRPO：group={group}，max_new_tokens={max_new}，reward=exact-match，beta={beta}",
    )
    step = start_step
    order: list[int] = []
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    window_start = time.monotonic()
    window_reward = 0.0
    window_loss = 0.0
    steps_done = 0

    while step < total_steps:
        ctx.cancel.check()
        if not order:
            order = list(range(len(samples)))
            random.Random(spec.train.seed + step).shuffle(order)
        item = samples[order.pop()]
        if getattr(tokenizer, "chat_template", None):
            prompt_text = tokenizer.apply_chat_template(
                [{"role": "user", "content": item["prompt"]}],
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            prompt_text = item["prompt"]
        prompt_ids = tokenizer(
            prompt_text, add_special_tokens=False, truncation=True, max_length=max(seq_len - max_new, 8)
        )["input_ids"]
        prompt_len = len(prompt_ids)
        prompt_tensor = torch.tensor([prompt_ids], device=device)
        prompt_mask = torch.ones_like(prompt_tensor)

        model.eval()
        with torch.no_grad():
            sequences = model.generate(
                input_ids=prompt_tensor,
                attention_mask=prompt_mask,
                max_new_tokens=max_new,
                do_sample=True,
                temperature=temperature,
                top_p=1.0,
                num_return_sequences=group,
                pad_token_id=pad_id,
            )
        completions = tokenizer.batch_decode(sequences[:, prompt_len:], skip_special_tokens=True)
        if step == start_step:
            emit.log("info", f"首个 rollout 示例：{completions[0]!r}（目标 {item['answer']}）")
        rewards = torch.tensor(
            [1.0 if _reward_matches(item["answer"], text) else 0.0 for text in completions],
            device=device,
            dtype=torch.float32,
        )
        advantages = _group_advantages(torch, rewards)

        attention_mask = (sequences != pad_id).to(torch.int64)
        attention_mask[:, :prompt_len] = 1
        prompt_lengths = torch.full((sequences.shape[0],), prompt_len, device=device, dtype=torch.int64)

        model.train()
        policy_logp = _completion_logprobs(torch, model, sequences, attention_mask, prompt_lengths)
        with torch.no_grad(), model.disable_adapter():
            ref_logp = _completion_logprobs(torch, model, sequences, attention_mask, prompt_lengths)
        ratio = ref_logp - policy_logp
        kl = (torch.exp(ratio) - ratio - 1.0).mean()
        loss = -(advantages.detach() * policy_logp).mean() + beta * kl

        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], float(spec.train.grad_clip)
        )
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        step += 1
        steps_done += 1
        window_reward += float(rewards.mean())
        window_loss += float(loss.detach())
        elapsed = time.monotonic() - window_start
        vram_peak = torch.cuda.max_memory_allocated() / (1024 * 1024) if device == "cuda" else None
        if tracker is not None:
            tracker.record(vram_peak)
        step_loss = window_loss / steps_done
        step_lr = float(scheduler.get_last_lr()[0])
        step_tps = float(sequences.shape[1] * group / elapsed) if elapsed > 0 else None
        mean_reward = window_reward / max(steps_done, 1)
        emit.step(
            step,
            loss=step_loss,
            lr=step_lr,
            grad_norm=float(grad_norm),
            tokens_per_s=step_tps,
            vram_peak_mib=float(vram_peak) if vram_peak is not None else None,
            elapsed_s=elapsed,
        )
        if train_curve is not None:
            train_curve.append(
                {
                    "step": step,
                    "loss": float(step_loss),
                    "lr": step_lr,
                    "grad_norm": float(grad_norm),
                    "tokens_per_s": step_tps,
                    "vram_peak_mib": float(vram_peak) if vram_peak is not None else None,
                    "elapsed_s": float(elapsed),
                    "mean_reward": float(mean_reward),
                }
            )
        if step % 10 == 0 or step == total_steps:
            emit.log("info", f"GRPO step {step}: mean reward {mean_reward:.3f}")
        if spec.train.save_every > 0 and step % spec.train.save_every == 0 and step < total_steps:
            _save_checkpoint(
                model, tokenizer, optimizer, scheduler, step, spec, ctx, emit,
                ctx.output_dir / f"checkpoint-{step}",
            )
        window_start = time.monotonic()
        window_reward = 0.0
        window_loss = 0.0
        steps_done = 0
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()

    final_dir = ctx.output_dir / ("adapter" if cfg.adapter != Adapter.NONE else "model")
    _save_checkpoint(model, tokenizer, optimizer, scheduler, step, spec, ctx, emit, final_dir)
    emit.log("info", f"RL 训练完成：{step} steps，产物在 {final_dir}")


def _load_preference(
    dataset: DatasetSpec,
    path: Path,
    tokenizer: Any,
    cfg: Any,
    emit: Emit,
    *,
    seed: int = 42,
) -> list[dict[str, list[int]]]:
    """Preference pairs: one chosen and one rejected completion per row."""
    raw, _ = _load_raw(dataset, path, seed=seed)
    rows: list[dict[str, list[int]]] = []
    seq_len = int(cfg.seq_len)
    for example in raw:
        chosen = example.get(dataset.chosen_column)
        rejected = example.get(dataset.rejected_column)
        if not isinstance(chosen, str) or not isinstance(rejected, str):
            raise RuntimeError(
                f"preference 样本缺少字符串字段 {dataset.chosen_column}/{dataset.rejected_column}"
            )
        chosen_ids = list(tokenizer(chosen, add_special_tokens=True)["input_ids"])[:seq_len]
        rejected_ids = list(tokenizer(rejected, add_special_tokens=True)["input_ids"])[:seq_len]
        if not chosen_ids or not rejected_ids:
            continue
        rows.append({"chosen_ids": chosen_ids, "rejected_ids": rejected_ids})
    emit.log("info", f"偏好数据集：{len(rows)} 对样本")
    if not rows:
        raise RuntimeError(f"偏好数据集为空：{path}")
    return rows


def _make_pref_loader(
    rows: list[dict[str, list[int]]],
    cfg: Any,
    pad_id: int,
    *,
    shuffle: bool,
    seed: int = 42,
) -> Any:
    import torch  # noqa: PLC0415
    from torch.utils.data import DataLoader  # noqa: PLC0415

    seq_len = int(cfg.seq_len)

    def collate(batch: list[dict[str, list[int]]]) -> dict[str, Any]:
        def pad_side(key: str) -> tuple[Any, Any]:
            width = min(max(len(row[key]) for row in batch), seq_len)
            ids = torch.full((len(batch), width), pad_id, dtype=torch.long)
            mask = torch.zeros((len(batch), width), dtype=torch.long)
            for index, row in enumerate(batch):
                values = row[key][:width]
                ids[index, : len(values)] = torch.tensor(values, dtype=torch.long)
                mask[index, : len(values)] = 1
            return ids, mask

        chosen_ids, chosen_mask = pad_side("chosen_ids")
        rejected_ids, rejected_mask = pad_side("rejected_ids")
        return {
            "chosen_input_ids": chosen_ids,
            "chosen_attention_mask": chosen_mask,
            "rejected_input_ids": rejected_ids,
            "rejected_attention_mask": rejected_mask,
        }

    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        rows,
        batch_size=int(cfg.batch),
        shuffle=shuffle,
        collate_fn=collate,
        generator=generator,
    )


def _make_loader(
    rows: list[dict[str, list[int]]] | None,
    cfg: Any,
    pad_id: int,
    *,
    shuffle: bool,
    seed: int = 42,
) -> Any:
    import torch  # noqa: PLC0415
    from torch.utils.data import DataLoader  # noqa: PLC0415

    if rows is None:
        return None

    seq_len = int(cfg.seq_len)
    packed_rows = _pack(rows, seq_len) if _use_packing(cfg) else None
    data = packed_rows if packed_rows else rows

    def collate(batch: list[dict[str, list[int]]]) -> dict[str, Any]:
        max_len = max(len(row["input_ids"]) for row in batch)
        max_len = min(max_len, seq_len)
        input_ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
        attention = torch.zeros_like(input_ids)
        labels = torch.full_like(input_ids, -100)
        for index, row in enumerate(batch):
            ids = row["input_ids"][:max_len]
            if not ids:
                continue
            tensor = torch.tensor(ids, dtype=torch.long)
            input_ids[index, : len(ids)] = tensor
            attention[index, : len(ids)] = 1
            labels[index, : len(ids)] = tensor
        return {"input_ids": input_ids, "attention_mask": attention, "labels": labels}

    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return DataLoader(
        data,
        batch_size=max(int(cfg.batch), 1),
        shuffle=shuffle,
        collate_fn=collate,
        generator=generator if shuffle else None,
        drop_last=False,
    )


def _use_packing(cfg: Any) -> bool:
    """Packing only for plain text corpora; chat samples keep their boundaries."""
    return cfg.stage == Stage.PRETRAIN


def _pack(rows: list[dict[str, list[int]]], seq_len: int) -> list[dict[str, list[int]]]:
    flat: list[int] = []
    for row in rows:
        flat.extend(row["input_ids"])
    if len(flat) < seq_len:
        return []
    blocks = [flat[i : i + seq_len] for i in range(0, len(flat) - seq_len + 1, seq_len)]
    return [{"input_ids": block} for block in blocks]


def _forever(loader: Any) -> Iterator[dict[str, Any]]:
    while True:
        yield from loader


def _build_optimizer(torch: Any, optimizer_kind: Optimizer, train: Any, params: list[Any]) -> Any:
    lr = float(train.learning_rate)
    weight_decay = float(train.weight_decay)
    if optimizer_kind in (Optimizer.ADAMW_8BIT, Optimizer.ADAMW_PAGED_8BIT):
        if not _available("bitsandbytes"):
            raise RuntimeError(f"optimizer={optimizer_kind.value} 需要 bitsandbytes，但未安装")
        import bitsandbytes as bnb  # noqa: PLC0415

        cls = bnb.optim.AdamW8bit if optimizer_kind == Optimizer.ADAMW_8BIT else bnb.optim.PagedAdamW8bit
        return cls(params, lr=lr, weight_decay=weight_decay)
    if optimizer_kind == Optimizer.CAME:
        raise RuntimeError("native-llm 未实现 CAME 优化器")
    return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)


def _evaluate(
    torch: Any,
    model: Any,
    loader: Any,
    device: str,
    amp_dtype: Any,
    emit: Emit,
    *,
    max_batches: int = 20,
) -> dict[str, float]:
    if loader is None:
        return {}
    was_training = model.training
    model.eval()
    total = 0.0
    count = 0
    try:
        with torch.no_grad():
            for index, batch in enumerate(loader):
                if index >= max_batches:
                    break
                batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
                if amp_dtype is not None:
                    with torch.autocast(device_type="cuda", dtype=amp_dtype):
                        out = model(**batch)
                else:
                    out = model(**batch)
                total += float(out.loss.detach())
                count += 1
    finally:
        model.train(was_training)
    return {"eval_loss": total / count} if count else {}


def _resolve_resume(spec: RunSpec, ctx: Context) -> Path | None:
    if not spec.train.resume_from:
        return None
    path = Path(spec.train.resume_from)
    if not path.is_absolute():
        base = ctx.run_dir if ctx.run_dir is not None else ctx.output_dir
        path = base / path
    return path


def _save_checkpoint(
    model: Any,
    tokenizer: Any,
    optimizer: Any,
    scheduler: Any,
    step: int,
    spec: RunSpec,
    ctx: Context,
    emit: Emit,
    target: Path,
) -> None:
    import torch  # noqa: PLC0415

    target.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(target), safe_serialization=True)
    tokenizer.save_pretrained(str(target))
    torch.save(optimizer.state_dict(), target / "optimizer.pt")
    torch.save(scheduler.state_dict(), target / "scheduler.pt")
    state = {
        "global_step": step,
        "family": spec.config.family,
        "adapter": spec.config.adapter.value,
    }
    (target / "trainer_state.json").write_text(json.dumps(state, indent=2), encoding="utf-8")
    emit.checkpoint(step, str(target), _dir_bytes(target))
    emit.log("info", f"checkpoint 已保存：{target}")


def _dir_bytes(path: Path) -> int:
    total = 0
    for file in path.rglob("*"):
        try:
            if file.is_file():
                total += file.stat().st_size
        except OSError:
            continue
    return total