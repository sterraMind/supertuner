"""Run configuration and result schemas for the SuperTuner engine.

Everything crossing the TypeScript/Python boundary is a pydantic model here and
plain JSON on the wire. The planner (`plan.py`) consumes `RunConfig` and
produces `MemoryPlan`; neither imports torch, so both run on any machine.

Units convention:
- parameter counts are floats in billions (7.5 == 7.5B params)
- memory numbers are MiB (matching nvidia-smi so users can compare directly)
"""

from __future__ import annotations

import re
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field, model_validator


class Stage(StrEnum):
    PRETRAIN = "pretrain"
    SFT = "sft"
    PREFERENCE = "preference"  # DPO / KTO / ORPO
    RL = "rl"  # GRPO / PPO
    DISTILL = "distill"  # DMD-style few-step distillation


class Adapter(StrEnum):
    NONE = "none"  # full finetune / from-scratch pretrain
    LORA = "lora"
    QLORA = "qlora"  # 4-bit frozen base + LoRA


class AttnImpl(StrEnum):
    FLASH = "flash"
    SDPA = "sdpa"  # torch scaled-dot-product, memory-efficient backends
    MATH = "math"  # O(n^2) scores materialized; planner will hard-warn


class Precision(StrEnum):
    BF16_PURE = "bf16_pure"  # weights bf16, optimizer states in param dtype
    FP32_MASTER = "fp32_master"  # bf16 weights + fp32 master weights (AMP)
    FP32 = "fp32"


class Optimizer(StrEnum):
    ADAMW = "adamw"
    ADAMW_8BIT = "adamw_8bit"
    ADAMW_PAGED_8BIT = "adamw_paged_8bit"
    CAME = "came"  # third-state optimizer used by musubi-tuner for 14B LoRA


class QuantBase(StrEnum):
    NONE = "none"
    NF4 = "nf4"  # bitsandbytes 4-bit (QLoRA)
    FP8 = "fp8"  # weight-only fp8 storage (musubi-style block swap)
    INT8 = "int8"  # ConvRot/weight-only int8 (musubi MiniMax-H3 path)
    NVFP4 = "nvfp4"  # NVFP4 4-bit float (4-bit weights + E4M3 block scales, ~0.5625 B/param)


class CachePlan(StrEnum):
    """Precompute expensive forwards to disk so they never live in VRAM."""

    NONE = "none"
    TEXT = "text"  # cache text-encoder outputs (T5/UMT5 embeddings)
    LATENTS = "latents"  # cache VAE latents for video datasets
    BOTH = "both"


class ModelFamily(BaseModel):
    """Architecture descriptor used by the planner arithmetic.

    `params_b` is the transformer (DiT or LLM) parameter count in billions.
    Video families additionally carry the text encoder / VAE that ride along
    unless the run plan caches them to disk.
    """

    key: str
    kind: Literal["llm", "video"]
    params_b: float | None = None  # None => planner refuses until verified
    hidden: int
    layers: int
    heads: int
    kv_heads: int = 0  # GQA key/value heads; 0 => same as heads
    head_dim: int
    ffn: int
    vocab: int = 0
    tie_embeddings: bool = False
    # video-only extras
    text_encoder_params_b: float = 0.0
    vae_params_b: float = 0.0
    patch: tuple[int, int, int] = (1, 2, 2)  # (t, h, w) patchify on latents
    vae_temporal_stride: int = 4
    vae_spatial_stride: int = 8
    latent_channels: int = 16
    fps: int = 0
    # MiniMax-H3 releases a pruned transformer (AdaLN branches removed, ~13B)
    # for inference/LoRA. Full-weight training must load params_b instead.
    pruned_params_b: float | None = None
    adaln_params_b: float = 0.0
    supports_block_swap: bool = False
    # text-encoder geometry used only when `cache` leaves it resident
    text_encoder_layers: int = 0
    approx: bool = False  # True => numbers flagged as approximations


# ---------------------------------------------------------------------------
# Registry. params_b values follow official checkpoints; `approx` marks rows
# where the community has not pinned an exact figure yet. GQA models set
# kv_heads < heads (Qwen3 uses 8 KV heads across sizes; Llama 3.2 uses 8 too).
# ---------------------------------------------------------------------------

LLM_FAMILIES: dict[str, ModelFamily] = {
    "smollm2-360m": ModelFamily(key="smollm2-360m", kind="llm", params_b=0.36, hidden=576, layers=24, heads=9, kv_heads=3, head_dim=64, ffn=1536, vocab=49152),
    "qwen3-0.6b": ModelFamily(key="qwen3-0.6b", kind="llm", params_b=0.6, hidden=1024, layers=28, heads=16, kv_heads=8, head_dim=128, ffn=3072, vocab=151936, tie_embeddings=True),
    "qwen3-1.7b": ModelFamily(key="qwen3-1.7b", kind="llm", params_b=1.7, hidden=2048, layers=28, heads=16, kv_heads=8, head_dim=128, ffn=6144, vocab=151936, tie_embeddings=True),
    "qwen3-4b": ModelFamily(key="qwen3-4b", kind="llm", params_b=4.0, hidden=2560, layers=36, heads=32, kv_heads=8, head_dim=128, ffn=9728, vocab=151936, tie_embeddings=True),
    "qwen3-8b": ModelFamily(key="qwen3-8b", kind="llm", params_b=8.0, hidden=4096, layers=36, heads=32, kv_heads=8, head_dim=128, ffn=12288, vocab=151936),
    "llama-3.2-1b": ModelFamily(key="llama-3.2-1b", kind="llm", params_b=1.2, hidden=2048, layers=16, heads=32, kv_heads=8, head_dim=64, ffn=8192, vocab=128256, tie_embeddings=True),
    "llama-3.2-3b": ModelFamily(key="llama-3.2-3b", kind="llm", params_b=3.2, hidden=3072, layers=28, heads=24, kv_heads=8, head_dim=128, ffn=8192, vocab=128256),
}

VIDEO_FAMILIES: dict[str, ModelFamily] = {
    # Wan2.1 1.3B: 30 layers, hidden 1536; UMT5-XXL text encoder ~5.6B; VAE ~0.15B
    "wan2.1-t2v-1.3b": ModelFamily(
        key="wan2.1-t2v-1.3b", kind="video", params_b=1.3, hidden=1536, layers=30, heads=12, head_dim=128,
        ffn=8960, text_encoder_params_b=5.6, vae_params_b=0.15, supports_block_swap=True,
        text_encoder_layers=24, fps=16,
    ),
    "wan2.1-t2v-14b": ModelFamily(
        key="wan2.1-t2v-14b", kind="video", params_b=14.0, hidden=5120, layers=40, heads=40, head_dim=128,
        ffn=13824, text_encoder_params_b=5.6, vae_params_b=0.15, supports_block_swap=True,
        text_encoder_layers=24, fps=16,
    ),
    "wan2.2-ti2v-5b": ModelFamily(
        key="wan2.2-ti2v-5b", kind="video", params_b=5.0, hidden=3072, layers=30, heads=16, head_dim=128,
        ffn=14336, text_encoder_params_b=5.6, vae_params_b=0.15, approx=True,
        supports_block_swap=True, text_encoder_layers=24, fps=24,
        # Wan2.2 VAE compresses 16x spatially, and the patch size adds another
        # 2x, so the latent grid is H/32 x W/32 (docs/02-feasibility.md).
        vae_spatial_stride=16,
    ),
    "cogvideox-2b": ModelFamily(
        key="cogvideox-2b", kind="video", params_b=2.0, hidden=1920, layers=30, heads=30, head_dim=64,
        ffn=10240, text_encoder_params_b=4.7, vae_params_b=0.19, approx=True,
        supports_block_swap=True, text_encoder_layers=24, fps=8,
    ),
    "ltx-video-2b": ModelFamily(
        key="ltx-video-2b", kind="video", params_b=2.0, hidden=2048, layers=28, heads=16, head_dim=64,
        ffn=8192, text_encoder_params_b=0.58, vae_params_b=0.05, approx=True,
        supports_block_swap=True, text_encoder_layers=24, fps=24,
    ),
    # MiniMax-H3: 33B dense omni transformer (≈13B in AdaLN branches), 50
    # blocks, f16t4d24 visual VAE (effective 32x spatial after patchify),
    # Qwen3-VL-32B encoder, 24 fps. Musubi ships a pruned 20B transformer
    # (40GB bf16 / 21GB INT8) for LoRA training with block swap.
    "minimax-h3": ModelFamily(
        key="minimax-h3", kind="video", params_b=33.0, hidden=6144, layers=50, heads=48, head_dim=128,
        ffn=24576, text_encoder_params_b=32.0, vae_params_b=2.5,
        patch=(1, 2, 2), vae_temporal_stride=4, vae_spatial_stride=16,
        latent_channels=24, fps=24, pruned_params_b=20.0, adaln_params_b=13.0,
        supports_block_swap=True, text_encoder_layers=64, approx=True,
    ),
}


# ---------------------------------------------------------------------------
# train_blocks: the target transformer-block window a video LoRA may train.
# A token is either a single block ("22") or an inclusive range ("20-49"), and a
# value is a comma list of those ("3-12,22,31-33"). It is validated here against
# the family's block count so an out-of-window spec fails fast, not mid-compile.
# ---------------------------------------------------------------------------

_TRAIN_BLOCK_TOKEN = re.compile(r"^\s*(\d+)(?:\s*-\s*(\d+))?\s*$")


def parse_train_blocks(value: str | None) -> list[tuple[int, int]]:
    """Parse a `train_blocks` string into an ordered list of `(start, end)` ranges.

    Raises `ValueError` with a clear message when a token is not a non-negative
    integer / `A-B` range, or when a range's start is greater than its end.
    An empty / whitespace value parses to `[]` (meaning "use the backend default").
    """
    if value is None:
        return []
    text = value.strip()
    if not text:
        return []
    ranges: list[tuple[int, int]] = []
    for raw in text.split(","):
        token = raw.strip()
        if not token:
            continue
        match = _TRAIN_BLOCK_TOKEN.match(token)
        if match is None:
            raise ValueError(f"train_blocks 的片段 '{token}' 不是合法的整数或 A-B 区间")
        start = int(match.group(1))
        end = int(match.group(2)) if match.group(2) is not None else start
        if start > end:
            raise ValueError(f"train_blocks 区间 '{token}' 的起点 {start} 大于终点 {end}")
        ranges.append((start, end))
    return ranges


def train_blocks_error(value: str | None, family: str) -> str | None:
    """Return an error message when `value` is invalid for `family`, else `None`."""
    if value is None:
        return None
    try:
        ranges = parse_train_blocks(value)
    except ValueError as exc:
        return str(exc)
    if not ranges:
        return f"train_blocks '{value}' 没有解析出任何 block"
    fam = LLM_FAMILIES.get(family) or VIDEO_FAMILIES.get(family)
    if fam is None:
        return None  # unknown family is reported elsewhere; only format is checked
    total = fam.layers
    for start, end in ranges:
        if start < 0 or end < 0:
            return f"train_blocks '{value}' 的 block 下标必须 ≥ 0"
        if end >= total:
            return f"train_blocks '{value}' 的终点 {end} 超出 '{family}' 的 {total} 个 block（须 < {total}）"
    return None


class RunConfig(BaseModel):
    """A training-run configuration as submitted by the UI/server."""

    name: str = Field(min_length=1, max_length=120)
    family: str  # key into the registries, or "generic"
    stage: Stage
    adapter: Adapter = Adapter.LORA
    precision: Precision = Precision.BF16_PURE
    optimizer: Optimizer = Optimizer.ADAMW
    quant_base: QuantBase = QuantBase.NONE
    lora_rank: int = 16
    lora_alpha: int = 16
    lora_targets: list[str] = ["q", "k", "v", "o", "mlp"]
    # video geometry (pixel space)
    width: int = 480
    height: int = 832
    frames: int = 49
    # LLM sequence length
    seq_len: int = 2048
    # GRPO rollout settings (stage=rl)
    rl_group_size: int = 4
    rl_max_new_tokens: int = 64
    rl_temperature: float = 0.8
    rl_beta: float = 0.04
    batch: int = 1
    grad_accum: int = 4
    grad_checkpointing: bool = True
    attn_impl: AttnImpl = AttnImpl.SDPA
    cache: CachePlan = CachePlan.BOTH  # video: default to caching everything
    # block swap trades VRAM for system RAM. 0 disables it; only valid for
    # LoRA/QLoRA (full fine-tuning cannot stream optimizer states).
    blocks_to_swap: int = 0
    # 目标 block 范围（视频 LoRA）：形如 "20-49" / "6-49" / "3-12,22,31-33"。
    # None 表示沿用后端默认（不限制可训练 block）。仅对视频 family 有意义。
    train_blocks: str | None = None
    text_encoder_quant: QuantBase = QuantBase.NONE
    text_encoder_blocks_to_swap: int = 0
    vram_budget_mib: int = 8192  # hard ceiling; planner fails past it
    ram_budget_mib: int = 32768
    overlap_windows_mib: int = 1600  # desktop/browser overhead on Windows
    # explicit override for families whose param count is unverified
    params_b_override: float | None = None
    # optional absolute path to the base weights. When empty the engine
    # resolves `<SUPERTUNER_MODELS_DIR>/<family>`; the server normally leaves
    # this unset and manages the models directory instead.
    model_path: str | None = None
    # When True and the family is a block-swap-capable video model, the planner
    # searches `quant_base x blocks_to_swap x train_blocks` and picks the
    # feasible candidate with the lowest step peak instead of planning the
    # caller's exact values. Off by default so the legacy planner arithmetic
    # is byte-for-byte unchanged.
    auto_budget: bool = False
    # 预算来源（v0.2 T2 实测进预算 / G3 = AP-01）："auto" = 机器实测 VRAM/RAM 可用
    # 时优先用实测值，否则回落到内置默认；"manual" = 直接使用上面显式给出的
    # vram_budget_mib / ram_budget_mib。
    budget_source: str = "auto"
    # OOM 有限次自动重试（v0.2 T4 / G4）：训练步骤真实峰值撞穿显存预算（计划
    # 认为可行但实际 OOM）时，engine 用更保守的候选（auto_budget 搜索 + 更低
    # 的显存预算）重跑整个计划，最多重试 max_oom_retries 次；0 表示禁用重试。
    # 每次重试都会发一条 oom_retry 事件。
    max_oom_retries: int = 2

    @model_validator(mode="after")
    def _validate_train_blocks(self) -> RunConfig:
        """Reject a malformed / out-of-window target block range."""
        error = train_blocks_error(self.train_blocks, self.family)
        if error is not None:
            raise ValueError(error)
        return self

    @model_validator(mode="after")
    def _validate_budget_source(self) -> RunConfig:
        """Reject a budget_source that is neither "auto" nor "manual"."""
        if self.budget_source not in ("auto", "manual"):
            raise ValueError(
                f"budget_source 必须是 'auto'（用机器实测值）或 'manual'（用手填值），"
                f"得到 {self.budget_source!r}"
            )
        return self


class MemoryLine(BaseModel):
    label: str
    mib: float
    note: str = ""


class AutoCandidate(BaseModel):
    """One candidate the unified auto-search considered (v0.2 T3)."""

    quant_base: str
    blocks_to_swap: int
    train_blocks: str | None
    feasible: bool
    headline_mib: float
    # short human note, e.g. "峰值 5.1 GB" or "缺 2.2 GB 显存"
    note: str = ""


class MemoryPlan(BaseModel):
    """Result of planning one RunConfig against one GPU budget."""

    feasible: bool
    vram_budget_mib: int
    ram_budget_mib: int = 32768
    training_peak_mib: float
    sampling_peak_mib: float
    headline_mib: float  # max of the two peaks
    headroom_mib: float  # budget - headline (may be negative)
    ram_peak_mib: float = 0
    ram_headroom_mib: float = 0
    lines: list[MemoryLine]
    risks: list[str]
    tokens_per_step: int
    notes: list[str]
    # 跑后自校准（v0.2 T1）：server 规划前按校准键注入的聚合偏差。engine 只读
    # 它做标注（notes 里也会写一句），绝不据此改动任何预算字段。
    calibration: Calibration | None = None
    # 预算来源（v0.2 T2 实测进预算 / G3 = AP-01）：解析后的实际来源。
    # "measured" = vram_budget_mib / ram_budget_mib 取自机器实测值；
    # "manual" = 来自手填的 vram_budget_mib / ram_budget_mib（或内置默认）。
    budget_source: str = "manual"
    # 实际采用的实测 VRAM / RAM（MiB）；该维度未使用实测值时为 None。
    budget_measured_vram_mib: int | None = None
    budget_measured_ram_mib: int | None = None
    # 统一自动选择（v0.2 T3 / G2 = AP-02）：auto_budget 评估过的全部候选，
    # 可行在前（按 headline_mib 升序）、不可行在后（按缺口升序）。仅当
    # auto_budget 真正执行了搜索时非空；legacy/LLM 计划保持 []。
    auto_candidates: list[AutoCandidate] = []
    # 单向 block swap 搬运时间带（v0.2 T6 / AP-08）：只统计被换出权重在
    # PCIe 上的搬运时间，不含注意力/计算。无 block swap 时保持 None。
    # swap_floor_s: 单向（H2D only）每步搬运下限（秒）。
    swap_floor_s: float | None = None
    # swap_roundtrip_floor_s: 往返（H2D+D2H）下限 = 2 x 单向（秒）。
    swap_roundtrip_floor_s: float | None = None
    # step_time_band_s: 面向 UI 的时间带说明行；无 swap 时为空串。
    step_time_band_s: str = ""


# ---------------------------------------------------------------------------
# Run specification: memory planning is one slice; a run also needs data,
# optimization hyper-parameters and an output contract. `RunConfig` stays
# flat because existing planner tests/CLI consumers depend on it; `RunSpec`
# nests it so the two concerns evolve independently.
# ---------------------------------------------------------------------------


class DatasetKind(StrEnum):
    AUTO = "auto"
    TEXT = "text"
    CHAT = "chat"
    PREFERENCE = "preference"
    VIDEO = "video"
    IMAGE = "image"


class DatasetSpec(BaseModel):
    path: str = Field(min_length=1)
    kind: DatasetKind = DatasetKind.AUTO
    format: Literal["auto", "jsonl", "parquet", "directory", "webdataset"] = "auto"
    text_column: str = "text"
    caption_column: str = "caption"
    messages_column: str = "messages"
    chosen_column: str = "chosen"
    rejected_column: str = "rejected"
    prompt_column: str = "prompt"
    answer_column: str = "answer"
    video_column: str = "video"
    audio_column: str = "audio"
    max_samples: int | None = Field(default=None, ge=1)
    val_fraction: float = Field(default=0.0, ge=0.0, le=0.5)
    notes: str = ""


class DistillSpec(BaseModel):
    teacher: str | None = None
    mode: Literal["dmd", "dmad", "dmd2"] = "dmd"
    student_steps: int = Field(default=4, ge=1, le=64)
    guidance: float = Field(default=1.0, ge=0.0, le=20.0)


class TrainSpec(BaseModel):
    learning_rate: float = Field(default=1e-4, gt=0.0, le=1.0)
    scheduler: Literal["constant", "cosine", "linear", "wsd"] = "cosine"
    warmup_steps: int = Field(default=20, ge=0)
    weight_decay: float = Field(default=0.01, ge=0.0)
    max_steps: int = Field(default=1000, ge=1)
    epochs: int = Field(default=1, ge=1)
    save_every: int = Field(default=200, ge=1)
    eval_every: int = Field(default=100, ge=1)
    seed: int = 42
    grad_clip: float = Field(default=1.0, gt=0.0)
    resume_from: str | None = None
    dpo_beta: float = Field(default=0.1, gt=0.0)
    distill: DistillSpec | None = None


class OutputSpec(BaseModel):
    dir: str | None = None  # defaults to <run_dir>/output
    save_format: Literal["safetensors", "pt"] = "safetensors"
    export: list[Literal["safetensors", "gguf", "onnx", "diffusers"]] = Field(default_factory=list)
    push_to_hub: bool = False
    hub_repo: str | None = None


class LicenceAck(BaseModel):
    """Explicit acceptance of a model-weight licence before a gated run starts."""

    family: str
    license_id: str
    version: str = "1"
    accepted_at: str


class Calibration(BaseModel):
    """跑后自校准（v0.2 T1）：server 规划前按校准键查库后注入的聚合偏差。

    engine 只把它当只读标注用（写进 MemoryPlan.notes 做提示），不据此修改
    vram_budget_mib 等任何预算字段。
    """

    count: int = Field(default=0, ge=0)  # 该校准键累计观测到的运行次数
    avg_delta_pct: float | None = None  # 平均偏差（实测-计划）/计划*100；无有效样本时为 None


class RunSpec(BaseModel):
    schema_version: Literal[1] = 1
    config: RunConfig
    dataset: DatasetSpec | None = None
    train: TrainSpec = Field(default_factory=TrainSpec)
    output: OutputSpec = Field(default_factory=OutputSpec)
    backend: str | None = None  # None => auto-select from registry
    licence_ack: LicenceAck | None = None
    # 规划前由 server 按校准键查库注入；engine 只读它做备注，不改动预算。
    calibration: Calibration | None = None


# ---------------------------------------------------------------------------
# Plugin / compilation contracts (also mirrored in packages/core/src).
# ---------------------------------------------------------------------------


class Issue(BaseModel):
    severity: Literal["error", "warning", "info"]
    code: str
    message: str
    hint: str = ""
    field: str | None = None


class PlanStep(BaseModel):
    id: str
    kind: Literal["prepare", "cache", "train", "export"]
    title: str
    description: str = ""
    command: list[str] | None = None  # argv, never a shell string
    cwd: str | None = None
    env: dict[str, str] = Field(default_factory=dict)
    optional: bool = False
    estimated_mib: float | None = None


class ExecutionPlan(BaseModel):
    backend: str
    backend_version: str = ""
    steps: list[PlanStep] = Field(default_factory=list)
    issues: list[Issue] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    expected_artifacts: list[str] = Field(default_factory=list)


class RunPlan(BaseModel):
    feasible: bool
    memory: MemoryPlan
    execution: ExecutionPlan
    issues: list[Issue] = Field(default_factory=list)
    estimated_minutes: float | None = None


class PluginManifest(BaseModel):
    id: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]*$")
    name: str
    version: str
    kind: Literal["model", "dataset", "strategy", "optimizer", "exporter", "ui", "backend"]
    description: str = ""
    capabilities: list[str] = Field(default_factory=list)
    families: list[str] = Field(default_factory=list)
    entry_point: str = ""
    requires: list[str] = Field(default_factory=list)
    license: str = "unknown"
    homepage: str = ""
    enabled: bool = True
    experimental: bool = False
    priority: int = 100


class EnvPackage(BaseModel):
    name: str
    version: str | None = None
    ok: bool
    hint: str = ""


class GpuInfo(BaseModel):
    index: int
    name: str
    vram_mib: int  # total VRAM in MiB (same number nvidia-smi reports)
    driver: str = ""
    vram_total_mib: int = 0
    vram_free_mib: int = 0


class EnvReport(BaseModel):
    python: str
    platform: str
    packages: list[EnvPackage] = Field(default_factory=list)
    gpus: list[GpuInfo] = Field(default_factory=list)
    ram_mib: int = 0
    ffmpeg: str | None = None
    warnings: list[str] = Field(default_factory=list)
    # detected musubi-tuner installation (path or "module:<name>")
    musubi: str | None = None
    # resolved engine paths (SUPERTUNER_* env vars)
    home: str = ""
    models_dir: str = ""
    datasets_dir: str = ""
    cache_dir: str = ""
    # 可选加速 kernel 探测结果（v0.2 T7）：import 检查，永不在缺失时抛错。
    # 键固定为 {"triton": bool, "liger": bool, "sageattention": bool}。
    capabilities: dict[str, bool] = {}
