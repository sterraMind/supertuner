"""GPU memory planner: the arithmetic that decides what fits on one card.

This module is pure math -- no torch, no CUDA. It exists so the UI can answer
"will this run OOM?" BEFORE the user burns hours of a consumer GPU on a crash.
Every term stays a separate MemoryLine so the UI can show a budget breakdown
and tests can assert each term independently.

Modeling decisions (each is deliberate and documented):
- Attention activations assume flash/SDPA linear-memory behavior; "math"
  attention is refused outright rather than estimated.
- PyTorch AdamW creates optimizer states with ``torch.zeros_like(p)``, so
  states follow the *trainable param dtype* unless an fp32 master copy exists
  (then states are fp32 too).
- Grad checkpointing stores one input tensor per layer boundary and pays one
  full layer's activations during recompute.
- The CE-loss logits upcast to fp32 is modeled explicitly -- at 150k vocab it
  is often the largest single activation term for small-LLM pretraining.
- Quantized weight storage: NF4 ~0.55 B/param (block scales included),
  weight-only fp8 = 1 B/param, int8 = 1.05 B/param (ConvRot scales).
- Block swap (musubi-tuner) keeps only a small ring of DiT blocks in VRAM and
  streams the rest from system RAM. We model the resident subset plus the
  swapped bytes as a RAM requirement; full fine-tuning cannot use it because
  optimizer states cannot be streamed the same way.
- MiniMax-H3 LoRA can load the pruned 20B transformer (AdaLN branches are
  precomputable and need not be trained); full fine-tuning must load all 33B.
"""

from __future__ import annotations

import dataclasses
import re

from .schema import (
    LLM_FAMILIES,
    VIDEO_FAMILIES,
    Adapter,
    AttnImpl,
    AutoCandidate,
    CachePlan,
    Calibration,
    MemoryLine,
    MemoryPlan,
    ModelFamily,
    Optimizer,
    Precision,
    QuantBase,
    RunConfig,
    Stage,
)

MIB = 1024 * 1024

# Chunked cross-entropy (liger-kernel / chunked-CE style): only this many tokens'
# logits are materialised and upcast to fp32 at once. Without chunking the logits
# tensor alone is tokens x vocab x 6 bytes and dominates small-model peaks.
CE_CHUNK_TOKENS = 512

# bytes per parameter for weight storage (bf16/fp32), overridden per-quant below
DTYPE_BYTES = {Precision.BF16_PURE: 2, Precision.FP32_MASTER: 2, Precision.FP32: 4}

# PCIe 4.0 x16 practical bandwidth used for the block-swap lower bound. Laptop
# cards are often x8; the number is advisory, not a promise.
PCIE_GB_PER_S = 8.0

# Fixed per-process cost the tensor ledger cannot see: CUDA context, allocator
# arenas, cuBLAS workspaces. Measured on this machine at ~0.9 GB regardless of
# model size (a 0.6B run peaked at 1787 MiB allocated but ~2.7 GB process RSS
# on the GPU). Override with SUPERTUNER_CUDA_CONTEXT_MIB.
DEFAULT_CONTEXT_MIB = 900.0


def _context_mib() -> float:
    import os

    raw = os.environ.get("SUPERTUNER_CUDA_CONTEXT_MIB")
    if raw:
        try:
            return max(float(raw), 0.0)
        except ValueError:
            pass
    return DEFAULT_CONTEXT_MIB


def _opt_state_factor(opt: Optimizer) -> float:
    # number of state tensors AdamW-family keeps (CAME carries a third)
    return {
        Optimizer.ADAMW: 2.0,
        Optimizer.ADAMW_8BIT: 2.0,
        Optimizer.ADAMW_PAGED_8BIT: 2.0,
        Optimizer.CAME: 3.0,
    }[opt]


def _opt_state_bytes_per_param(opt: Optimizer, precision: Precision) -> float:
    if opt in (Optimizer.ADAMW_8BIT, Optimizer.ADAMW_PAGED_8BIT):
        return 2 * 1.0  # two states at ~8 bits
    n = _opt_state_factor(opt)
    dtype = 4 if precision == Precision.FP32_MASTER else DTYPE_BYTES[precision]
    return n * dtype


# NVFP4 = 4-bit weights with an E4M3 scale per 16-element block:
# 4 bits + 8/16 bits of scales = 4.5 bits/param = 0.5625 B/param.
NVFP4_BYTES_PER_PARAM = 0.5625


def _weight_bytes_per_param(cfg: RunConfig) -> float:
    if cfg.adapter == Adapter.QLORA or cfg.quant_base == QuantBase.NF4:
        return 0.55  # NF4 + block scales, empirical average
    if cfg.quant_base == QuantBase.NVFP4:
        return NVFP4_BYTES_PER_PARAM
    if cfg.quant_base == QuantBase.FP8:
        return 1.0
    if cfg.quant_base == QuantBase.INT8:
        return 1.05  # ConvRot int8: one byte + per-group scales
    return float(DTYPE_BYTES[cfg.precision])


def _quant_bytes_per_param(quant: QuantBase, precision: Precision) -> float:
    if quant == QuantBase.NF4:
        return 0.55
    if quant == QuantBase.NVFP4:
        return NVFP4_BYTES_PER_PARAM
    if quant == QuantBase.FP8:
        return 1.0
    if quant == QuantBase.INT8:
        return 1.05
    return float(DTYPE_BYTES[precision])


def _lora_params_b(fam: ModelFamily, rank: int, targets: list[str]) -> float:
    """LoRA parameter count in billions for the selected target projections."""
    h, ffn, r = fam.hidden, fam.ffn, rank
    per_layer = 0.0
    t = set(targets)
    if "q" in t:
        per_layer += r * (h + h)
    if "k" in t:
        per_layer += r * (h + h)
    if "v" in t:
        per_layer += r * (h + h)
    if "o" in t:
        per_layer += r * (h + h)
    if "mlp" in t or "gate" in t or "up" in t or "down" in t:
        # gate and up: hidden->ffn, down: ffn->hidden
        per_layer += 3 * r * (h + ffn)
    return per_layer * fam.layers / 1e9


def _video_latent_tokens(cfg: RunConfig, fam: ModelFamily) -> int:
    """DiT token count for one video sample after VAE + patchify."""
    if cfg.frames < 1:
        return 0
    t_lat = max(1, (cfg.frames - 1) // max(1, fam.vae_temporal_stride) + 1)
    h_lat = max(1, cfg.height // max(1, fam.vae_spatial_stride))
    w_lat = max(1, cfg.width // max(1, fam.vae_spatial_stride))
    p_t, p_h, p_w = fam.patch
    p_t, p_h, p_w = max(1, p_t), max(1, p_h), max(1, p_w)
    return ((t_lat + p_t - 1) // p_t) * ((h_lat + p_h - 1) // p_h) * ((w_lat + p_w - 1) // p_w)


def _llm_activation_mib_per_layer(seq_tokens: int, fam: ModelFamily, dtype: int) -> float:
    """One full (non-checkpointed) transformer layer's activation bytes."""
    h, ffn = fam.hidden, fam.ffn
    # attn: qkv (3h) + attn-out (h); mlp: gate, up, silu, down-in (4*ffn)
    # norms/residual: 2h
    elems = seq_tokens * (4 * h + 4 * ffn + 2 * h)
    return elems * dtype / MIB


def _llm_boundary_mib_per_layer(seq_tokens: int, fam: ModelFamily, dtype: int) -> float:
    return seq_tokens * fam.hidden * dtype / MIB


def _swap_resident_fraction(requested: int, layers: int) -> tuple[int, float, bool]:
    """Return (effective_swapped_blocks, resident_fraction, clamped).

    musubi-tuner keeps at least two blocks resident (one executing, one
    prefetched) and documents 48/50 as the working maximum.
    """
    if requested <= 0 or layers <= 0:
        return 0, 1.0, False
    max_swap = max(0, layers - 2)
    effective = min(requested, max_swap)
    resident = max(2, layers - effective)
    return effective, resident / layers, effective != requested


def _calibration_note(calibration: Calibration) -> str:
    """把跑后自校准聚合结果渲染成一条只读备注。

    样本足够（count >= 3）且已有平均偏差时给出「已校准 N 次（平均偏差 x%）」；
    样本不足（N<3）或尚无有效偏差时只标「已校准 N 次（样本 N 次）」，不强行
    给修正值。无论哪种，都不据此改 vram_budget_mib。
    """
    count = calibration.count
    avg = calibration.avg_delta_pct
    if count >= 3 and avg is not None:
        sign = "+" if avg >= 0 else ""
        return f"已校准 {count} 次（平均偏差 {sign}{avg:.1f}%，仅供参考，未调整预算）"
    return f"已校准 {count} 次（样本 {count} 次，样本不足，不调整预算）"


def plan(
    cfg: RunConfig,
    calibration: Calibration | None = None,
    measured_vram_mib: int | None = None,
    measured_ram_mib: int | None = None,
) -> MemoryPlan:
    """Public planner entry point (the plan half of ``resolve_config``).

    ``calibration`` is the server-injected 跑后自校准 aggregation (v0.2 T1) for
    this config's calibration key; it is read-only annotation and never
    changes the budget arithmetic.

    ``measured_vram_mib`` / ``measured_ram_mib`` are the machine's ACTUALLY
    measured total VRAM / RAM in MiB (from the environment probe). Together
    with ``cfg.budget_source`` they decide which budget numbers the plan is
    computed against (v0.2 T2 实测进预算 / G3 = AP-01):

    * ``budget_source == "manual"`` — the caller's ``vram_budget_mib`` /
      ``ram_budget_mib`` are used as-is; the measured values are ignored.
    * ``budget_source == "auto"`` — a measured value is used when it is a
      positive int, otherwise the caller's built-in default applies. The
      resolved provenance is recorded on the returned ``MemoryPlan`` as
      ``budget_source`` ("measured" if any measured value was actually
      applied, else "manual") plus ``budget_measured_vram_mib`` /
      ``budget_measured_ram_mib`` (the measured figures that were used, or
      None for a dimension that fell back to the default).

    The measured figures only *select* the budget ceiling; the planning
    arithmetic (peaks, headroom, feasibility) is unchanged.
    """
    resolved = _resolve_budget(cfg, measured_vram_mib, measured_ram_mib)
    # Re-plan against an effective config carrying the resolved ceilings so the
    # existing arithmetic (feasible, refuse, and auto-budget paths alike) sees
    # exactly the budget the UI will display.
    effective = cfg.model_copy(
        update={
            "vram_budget_mib": resolved.vram_budget_mib,
            "ram_budget_mib": resolved.ram_budget_mib,
        }
    )
    plan_out = resolve_config(effective, calibration=calibration)[1]
    # Every construction site (_plan_once main path, its refuse() helper, and
    # _auto_budget_resolve's output) funnels through this single returned
    # model, so stamping the provenance here covers the feasible, infeasible,
    # and auto-budget paths alike.
    plan_out.budget_source = resolved.budget_source
    plan_out.budget_measured_vram_mib = resolved.measured_vram_mib
    plan_out.budget_measured_ram_mib = resolved.measured_ram_mib
    return plan_out


@dataclasses.dataclass(frozen=True)
class _ResolvedBudget:
    """The budget ceilings a plan should be computed against, plus provenance."""

    vram_budget_mib: int
    ram_budget_mib: int
    budget_source: str  # "measured" | "manual"
    measured_vram_mib: int | None
    measured_ram_mib: int | None


def _resolve_budget(
    cfg: RunConfig,
    measured_vram_mib: int | None,
    measured_ram_mib: int | None,
) -> _ResolvedBudget:
    """Resolve which VRAM/RAM ceilings a plan is computed against.

    ``budget_source == "manual"`` keeps the caller's explicit ceilings.
    ``budget_source == "auto"`` prefers a positive measured value for each
    dimension and falls back to the caller's ceiling otherwise; the provenance
    is "measured" only when at least one dimension actually used a measured
    value.
    """
    if cfg.budget_source == "manual":
        return _ResolvedBudget(
            vram_budget_mib=cfg.vram_budget_mib,
            ram_budget_mib=cfg.ram_budget_mib,
            budget_source="manual",
            measured_vram_mib=None,
            measured_ram_mib=None,
        )

    vram = cfg.vram_budget_mib
    ram = cfg.ram_budget_mib
    used_vram: int | None = None
    used_ram: int | None = None
    if isinstance(measured_vram_mib, int) and measured_vram_mib > 0:
        vram = measured_vram_mib
        used_vram = measured_vram_mib
    if isinstance(measured_ram_mib, int) and measured_ram_mib > 0:
        ram = measured_ram_mib
        used_ram = measured_ram_mib
    used_any = used_vram is not None or used_ram is not None
    return _ResolvedBudget(
        vram_budget_mib=vram,
        ram_budget_mib=ram,
        budget_source="measured" if used_any else "manual",
        measured_vram_mib=used_vram,
        measured_ram_mib=used_ram,
    )


def resolve_config(
    cfg: RunConfig,
    calibration: Calibration | None = None,
) -> tuple[RunConfig, MemoryPlan]:
    """Resolve the effective config a run should actually be planned/compiled with.

    With ``auto_budget`` enabled on a block-swap-capable video family this is
    the candidate the auto search picked (quant_base x blocks_to_swap x
    train_blocks, v0.2 T3); every other config comes back unchanged.
    ``plan(cfg)`` is the plan half of this call, so a single search serves both
    entry points. ``calibration`` is passed through to ``_plan_once`` as
    read-only annotation.
    """
    if not cfg.auto_budget:
        return cfg, _plan_once(cfg, calibration)
    fam = LLM_FAMILIES.get(cfg.family) or VIDEO_FAMILIES.get(cfg.family)
    if fam is None or fam.kind != "video" or not fam.supports_block_swap:
        return cfg, _plan_once(cfg, calibration)
    return _auto_budget_resolve(cfg, fam, calibration)


def _plan_once(cfg: RunConfig, calibration: Calibration | None = None) -> MemoryPlan:
    """Legacy single-config planner (the exact arithmetic tests pin)."""
    fam = LLM_FAMILIES.get(cfg.family) or VIDEO_FAMILIES.get(cfg.family)
    if fam is None:
        raise ValueError(f"unknown family: {cfg.family}")

    notes: list[str] = []
    risks: list[str] = []

    # 跑后自校准标注（v0.2 T1）：只读标注，不改任何预算项。放在最前，保证
    # refuse 早退路径和正常路径都能带上这条备注。
    if calibration is not None and calibration.count > 0:
        notes.append(_calibration_note(calibration))

    params_b = fam.params_b
    if params_b is None:
        if cfg.params_b_override is None:
            raise ValueError(
                f"family '{cfg.family}' has no verified parameter count; "
                "pass params_b_override explicitly once verified."
            )
        params_b = cfg.params_b_override
        notes.append(f"params_b={params_b} 来自手动 override（官方数字未核实）")

    def refuse(reason: str) -> MemoryPlan:
        risks.append(reason)
        return MemoryPlan(
            feasible=False,
            vram_budget_mib=cfg.vram_budget_mib,
            ram_budget_mib=cfg.ram_budget_mib,
            training_peak_mib=0,
            sampling_peak_mib=0,
            headline_mib=0,
            headroom_mib=-cfg.overlap_windows_mib,
            ram_peak_mib=0,
            ram_headroom_mib=cfg.ram_budget_mib,
            lines=[],
            risks=risks,
            tokens_per_step=0,
            notes=notes,
            calibration=calibration,
        )

    if cfg.attn_impl == AttnImpl.MATH:
        return refuse("attn_impl=math 会显式实例化 O(n^2) 注意力矩阵，已拒绝执行；请用 flash/sdpa")

    if fam.approx:
        notes.append(f"family '{fam.key}' 的参数量/结构是近似值，测算结果有 ±10~15% 误差")

    # Block swap is a LoRA-only optimisation. Full fine-tuning would need
    # optimizer states for the streamed weights, which defeats the purpose.
    swap_blocks = 0
    resident_fraction = 1.0
    # 单向/往返 PCIe 搬运下限（v0.2 T6 / AP-08）：无 swap 时保持 None/""。
    swap_floor_s: float | None = None
    swap_roundtrip_floor_s: float | None = None
    step_time_band_s = ""
    if cfg.blocks_to_swap > 0:
        if cfg.adapter == Adapter.NONE:
            return refuse("block swap 只支持 LoRA/QLoRA：全参训练的优化器状态无法随块流式搬运")
        if not fam.supports_block_swap:
            return refuse(f"family '{fam.key}' 没有 block swap 实现；请改用 LoRA 或更大的显卡")
        if not cfg.grad_checkpointing:
            return refuse("block swap 必须同时开启梯度检查点，否则激活值会让显存峰值反弹")
        swap_blocks, resident_fraction, clamped = _swap_resident_fraction(cfg.blocks_to_swap, fam.layers)
        if clamped:
            notes.append(
                f"blocks_to_swap={cfg.blocks_to_swap} 超过 {fam.key} 的可用上限；"
                f"实际按 {swap_blocks}/{fam.layers} 计算（至少保留 2 块常驻）"
            )

    dtype_w = _weight_bytes_per_param(cfg)
    dtype_act = 2  # activations always bf16
    is_video = fam.kind == "video"
    tokens_one_sample = _video_latent_tokens(cfg, fam) if is_video else cfg.seq_len
    tokens = cfg.batch * tokens_one_sample
    tokens_per_step = tokens * cfg.grad_accum

    lines: list[MemoryLine] = []

    # ------------------------------------------------------------------
    # 1) resident weights
    # ------------------------------------------------------------------
    # LoRA on H3 can use the pruned transformer; full fine-tuning cannot.
    base_params_b = params_b
    if cfg.adapter != Adapter.NONE and fam.pruned_params_b is not None and cfg.params_b_override is None:
        base_params_b = fam.pruned_params_b
        notes.append(
            f"LoRA 使用 pruned transformer（{base_params_b}B，AdaLN 分支已裁剪）；"
            f"全参微调需加载 {params_b}B 完整权重"
        )

    base_bytes = base_params_b * 1e9 * dtype_w
    resident_base_bytes = base_bytes * resident_fraction
    swapped_base_bytes = base_bytes - resident_base_bytes

    quant_note = ""
    if cfg.adapter == Adapter.QLORA or cfg.quant_base == QuantBase.NF4:
        quant_note = "NF4 量化基座（0.55 B/参数）"
    elif cfg.quant_base == QuantBase.FP8:
        quant_note = "fp8 权重存储（1 B/参数）"
    elif cfg.quant_base == QuantBase.INT8:
        quant_note = "int8 权重存储（1.05 B/参数，ConvRot）"
    elif cfg.quant_base == QuantBase.NVFP4:
        quant_note = f"NVFP4 量化基座（{NVFP4_BYTES_PER_PARAM} B/参数）"

    if swap_blocks:
        quant_note = (quant_note + "；" if quant_note else "") + f"block swap {swap_blocks}/{fam.layers}"

    if cfg.adapter == Adapter.NONE:
        trainable_b = params_b
        lines.append(MemoryLine(label="模型权重（全参训练）", mib=resident_base_bytes / MIB, note=quant_note))
    else:
        trainable_b = _lora_params_b(fam, cfg.lora_rank, cfg.lora_targets)
        lines.append(
            MemoryLine(
                label=f"冻结基座权重常驻（{cfg.adapter.value}，{base_params_b:g}B）",
                mib=resident_base_bytes / MIB,
                note=quant_note,
            )
        )

    ram_peak_bytes = swapped_base_bytes
    if swap_blocks:
        lines.append(
            MemoryLine(
                label=f"块交换暂存到系统内存（{swap_blocks}/{fam.layers} 块）",
                mib=swapped_base_bytes / MIB,
                note="block swap 的 CPU 侧权重",
            )
        )
        if swapped_base_bytes > 0:
            # 单向（H2D-only）与往返搬运下限分开给出（v0.2 T6 / AP-08）：
            # 单向流式换块实测比往返快约 6.4 倍（docs/research/
            # 06-operator-architecture-optimization.md），不能只报往返数。
            swap_floor_s = swapped_base_bytes / (PCIE_GB_PER_S * 1e9)
            swap_roundtrip_floor_s = 2 * swap_floor_s
            step_time_band_s = (
                f"≈{swap_floor_s:.1f}s–{swap_roundtrip_floor_s:.1f}s/step "
                "(swap transfer only, excludes attention/compute)"
            )
            notes.append(
                f"block swap 每步单向搬运下限 {swap_floor_s:.1f}s、往返下限 "
                f"{swap_roundtrip_floor_s:.1f}s（PCIe {PCIE_GB_PER_S:g}GB/s 上限）；"
                "实际还受 CPU 解压/固定内存影响，建议 NVMe + 大内存"
            )

    # video companions: text encoder / VAE stay out of VRAM when cached
    companion_bytes = 0.0
    if is_video:
        if cfg.cache in (CachePlan.TEXT, CachePlan.BOTH):
            notes.append(f"文本编码器（~{fam.text_encoder_params_b:g}B）输出已预缓存到磁盘，不占显存")
        else:
            te_bytes_per_param = _quant_bytes_per_param(cfg.text_encoder_quant, cfg.precision)
            te_bytes = fam.text_encoder_params_b * 1e9 * te_bytes_per_param
            te_swap, te_fraction, te_clamped = _swap_resident_fraction(
                cfg.text_encoder_blocks_to_swap, fam.text_encoder_layers
            )
            if te_clamped:
                notes.append(
                    f"text_encoder_blocks_to_swap={cfg.text_encoder_blocks_to_swap} 超过上限；"
                    f"实际按 {te_swap}/{fam.text_encoder_layers} 计算"
                )
            if te_swap and cfg.text_encoder_quant == QuantBase.NONE and fam.text_encoder_params_b >= 10:
                risks.append("大文本编码器未量化；int8/fp8 可把它的常驻权重再降一半")
            resident_te = te_bytes * te_fraction
            companion_bytes += resident_te
            ram_peak_bytes += te_bytes - resident_te
            label = "文本编码器权重（未缓存）"
            if te_swap:
                label += f"，swap {te_swap}/{fam.text_encoder_layers}"
            lines.append(MemoryLine(label=label, mib=resident_te / MIB))
        if cfg.cache not in (CachePlan.LATENTS, CachePlan.BOTH):
            vae_bytes = fam.vae_params_b * 1e9 * 2
            companion_bytes += vae_bytes
            lines.append(MemoryLine(label="VAE 权重（latents 未缓存）", mib=vae_bytes / MIB))
        else:
            notes.append("VAE latents 已预缓存到磁盘；VAE 只在数据预处理/采样时短暂进显存")

    # LoRA weights themselves
    lora_bytes = 0.0
    if cfg.adapter != Adapter.NONE:
        lora_bytes = trainable_b * 1e9 * 2
        lines.append(MemoryLine(label=f"LoRA 权重 rank={cfg.lora_rank}", mib=lora_bytes / MIB))

    # ------------------------------------------------------------------
    # 2) optimizer states + grads (trainable params only)
    # ------------------------------------------------------------------
    opt_bytes = trainable_b * 1e9 * _opt_state_bytes_per_param(cfg.optimizer, cfg.precision)
    opt_name = {
        Optimizer.ADAMW: "AdamW",
        Optimizer.ADAMW_8BIT: "AdamW-8bit",
        Optimizer.ADAMW_PAGED_8BIT: "AdamW-paged-8bit",
        Optimizer.CAME: "CAME",
    }[cfg.optimizer]
    lines.append(MemoryLine(label=f"优化器状态（{opt_name}）", mib=opt_bytes / MIB))

    grad_bytes = trainable_b * 1e9 * dtype_act
    lines.append(MemoryLine(label="梯度（bf16）", mib=grad_bytes / MIB))

    master_bytes = 0.0
    if cfg.precision == Precision.FP32_MASTER:
        master_bytes = trainable_b * 1e9 * 4
        lines.append(MemoryLine(label="fp32 主权重副本", mib=master_bytes / MIB))

    # ------------------------------------------------------------------
    # 3) activations (all values below are MiB)
    # ------------------------------------------------------------------
    ckpt_note = "开启梯度检查点" if cfg.grad_checkpointing else "未开梯度检查点"
    per_layer_full = _llm_activation_mib_per_layer(tokens, fam, dtype_act)
    per_layer_boundary = _llm_boundary_mib_per_layer(tokens, fam, dtype_act)
    # checkpointing keeps one input tensor per layer; without it every layer's
    # full activation set stays alive from forward until backward
    act = fam.layers * per_layer_boundary + per_layer_full
    if not cfg.grad_checkpointing:
        act += fam.layers * per_layer_full

    logits_mib = 0.0
    if not is_video and cfg.stage in (Stage.PRETRAIN, Stage.SFT, Stage.PREFERENCE):
        # CE is computed in token chunks (liger/chunked-CE): only CE_CHUNK_TOKENS
        # logits are upcast to fp32 at a time. bf16 logits + fp32 copy per chunk.
        chunk = min(tokens, CE_CHUNK_TOKENS)
        logits_mib = chunk * fam.vocab * (dtype_act + 4) / MIB
        lines.append(
            MemoryLine(
                label=f"输出 logits+CE（分块 {chunk} tokens，vocab={fam.vocab}）",
                mib=logits_mib,
                note="按 512-token 分块计算交叉熵，避免全量 vocab logits 常驻",
            )
        )
        act += logits_mib

    act_note = "含一层重算峰值" if cfg.grad_checkpointing else "全部层激活常驻"
    if cfg.stage == Stage.PREFERENCE and not is_video:
        # DPO keeps the chosen and the rejected graph alive until backward, so
        # two activation sets are resident at the same time.
        act *= 2
        act_note = f"{act_note}；偏好训练保留 chosen/rejected 两份激活"
    if cfg.stage == Stage.RL and not is_video:
        # GRPO keeps one policy forward per rollout in the group alive for backward,
        # and the rollout itself needs a KV cache for the whole group.
        group = max(int(cfg.rl_group_size), 1)
        act *= group
        act_note = f"{act_note}；GRPO 保留 {group} 条 rollout 的激活"
        kv_bytes = 2 * 2 * fam.layers * fam.kv_heads * fam.head_dim
        rollout_mib = group * (int(cfg.seq_len) + int(cfg.rl_max_new_tokens)) * kv_bytes / MIB
        lines.append(
            MemoryLine(
                label=f"GRPO rollout 缓存（group={group}）",
                mib=rollout_mib,
                note="生成阶段每条 rollout 的 K/V 缓存；生成结束即释放",
            )
        )
    if is_video:
        lines.append(MemoryLine(label=f"DiT 激活值（{ckpt_note}，{tokens} tokens）", mib=act, note=act_note))
    else:
        lines.append(MemoryLine(label=f"LLM 激活值（{ckpt_note}，{tokens} tokens）", mib=act, note=act_note))

    # ------------------------------------------------------------------
    # 4) training peak (MiB)
    # ------------------------------------------------------------------
    resident_bytes = resident_base_bytes + companion_bytes
    training_peak = (
        resident_bytes + lora_bytes + opt_bytes + grad_bytes + master_bytes
    ) / MIB + act

    # ------------------------------------------------------------------
    # 5) sampling/eval peak: frozen forward (weights + KV cache or latents)
    # ------------------------------------------------------------------
    if is_video:
        sampling_peak = resident_bytes / MIB
        if fam.vae_params_b >= 1:
            notes.append("采样/解码峰值未计入 VAE decoder；大 VAE 需要 tiled decode 才能放进小显存")
    else:
        kv_heads = fam.kv_heads or fam.heads
        kv_mib = 2 * fam.layers * kv_heads * fam.head_dim * tokens * dtype_act / MIB
        rollout_factor = max(int(cfg.rl_group_size), 1) if cfg.stage == Stage.RL else 1
        sampling_peak = resident_bytes / MIB + kv_mib * rollout_factor
        lines.append(MemoryLine(label="采样峰值（权重+KV cache）", mib=sampling_peak, note="推理/评估时"))

    headline = max(training_peak, sampling_peak)
    context_mib = _context_mib()
    lines.append(
        MemoryLine(
            label="CUDA 上下文与分配器保留",
            mib=context_mib,
            note="进程级固定开销，与模型大小无关；可用 SUPERTUNER_CUDA_CONTEXT_MIB 覆盖",
        )
    )
    training_peak += context_mib
    sampling_peak += context_mib
    headline = max(training_peak, sampling_peak)
    budget = cfg.vram_budget_mib - cfg.overlap_windows_mib
    headroom = budget - headline
    ram_peak_mib = ram_peak_bytes / MIB
    ram_headroom = cfg.ram_budget_mib - ram_peak_mib
    feasible = headroom >= 0 and ram_headroom >= 0

    # ------------------------------------------------------------------
    # 6) advice
    # ------------------------------------------------------------------
    if headroom < 0:
        deficit = -headroom
        risks.append(f"超出预算 {deficit:.0f} MiB 显存（预算已扣除桌面/浏览器占用 {cfg.overlap_windows_mib} MiB）")
        if cfg.adapter == Adapter.NONE:
            risks.append(
                "全参训练装不下：优先考虑 LoRA/QLoRA；坚持全参请减小 seq_len/batch 并开启 fp32_master+paged 优化器"
            )
        else:
            if cfg.optimizer == Optimizer.ADAMW:
                risks.append("可改用 AdamW-8bit/paged-8bit 降低优化器状态")
            if cfg.grad_checkpointing is False:
                risks.append("开启梯度检查点可显著降低激活值")
            if is_video and cfg.cache != CachePlan.BOTH:
                risks.append("开启 TEXT+LATENTS 缓存可把文本编码器和 VAE 移出显存")
            if is_video and cfg.quant_base == QuantBase.NONE:
                risks.append("基座换 fp8/int8 权重存储（musubi-tuner 路线）可省一半权重显存")
            if is_video and cfg.blocks_to_swap == 0 and fam.supports_block_swap:
                risks.append("启用 blocks_to_swap 可把大部分 DiT 权重放到系统内存，只留 2 块常驻")
            if is_video:
                risks.append("降低训练分辨率/帧数会平方级减少 token 数与激活值；H3 原生 768p 不适合小卡训练")

    if ram_headroom < 0:
        risks.append(
            f"系统内存不足 {-ram_headroom:.0f} MiB：block swap 的 CPU 侧权重放不下；"
            "换更大内存、量化基座或减少 swap 块数（但会占用更多显存）"
        )

    if cfg.precision == Precision.BF16_PURE and cfg.adapter == Adapter.NONE:
        risks.append("bf16 纯训练（优化器状态也是 bf16）对预训练有数值稳定性风险；建议 fp32_master 或 8bit 优化器+fp32 状态")
    if cfg.grad_checkpointing is False and tokens > 8192:
        risks.append("长序列/大 token 量下未开梯度检查点，激活值将线性爆炸")

    return MemoryPlan(
        feasible=feasible,
        vram_budget_mib=cfg.vram_budget_mib,
        ram_budget_mib=cfg.ram_budget_mib,
        training_peak_mib=round(training_peak, 1),
        sampling_peak_mib=round(sampling_peak, 1),
        headline_mib=round(headline, 1),
        headroom_mib=round(headroom, 1),
        ram_peak_mib=round(ram_peak_mib, 1),
        ram_headroom_mib=round(ram_headroom, 1),
        lines=[MemoryLine(label=line.label, mib=round(line.mib, 1), note=line.note) for line in lines],
        risks=risks,
        tokens_per_step=tokens_per_step,
        notes=notes,
        calibration=calibration,
        swap_floor_s=swap_floor_s,
        swap_roundtrip_floor_s=swap_roundtrip_floor_s,
        step_time_band_s=step_time_band_s,
    )


# Rank quants for tie-breaking when step peaks land within model noise: int8
# carries per-group ConvRot scales and is the published MiniMax-H3 precision,
# so prefer it over fp8 over bf16 at an equal peak. NF4 (0.55 B/param) and
# NVFP4 (0.5625 B/param) are the most aggressive 4-bit tiers and share the
# last rank on a tie: they are only picked when nothing else fits.
_AUTO_QUANT_RANK = {
    QuantBase.INT8: 0,
    QuantBase.FP8: 1,
    QuantBase.NONE: 2,
    QuantBase.NF4: 3,
    QuantBase.NVFP4: 3,
}
# Peaks closer than this (or 2% of the headline, whichever is larger) are
# treated as the same and resolved by quant rank instead.
_AUTO_PEAK_TOLERANCE_MIB = 32.0


def _budget_deficit(plan: MemoryPlan) -> float:
    """How far a plan misses the budgets (0 when it fits)."""
    return max(0.0, -plan.headroom_mib) + max(0.0, -plan.ram_headroom_mib)


_TRAIN_BLOCK_TOKEN = re.compile(r"^\s*(\d+)(?:\s*-\s*(\d+))?\s*$")


def _trained_block_count(train_blocks: str | None, layers: int) -> int:
    """How many transformer blocks a ``train_blocks`` spec actually trains.

    ``None`` (or an empty / malformed spec) means the backend default, which
    trains every block. A range like ``"20-49"`` counts its span; comma
    separated ranges add up.
    """
    if train_blocks is None:
        return layers
    text = train_blocks.strip()
    if not text:
        return layers
    count = 0
    for token in text.split(","):
        token = token.strip()
        if not token:
            continue
        match = _TRAIN_BLOCK_TOKEN.match(token)
        if match is None:
            return layers
        start = int(match.group(1))
        end = int(match.group(2)) if match.group(2) is not None else start
        if start > end:
            return layers
        count += end - start + 1
    return max(0, count)


def _pick_best_candidate(
    feasible: list[tuple[RunConfig, MemoryPlan]],
    layers: int,
) -> tuple[RunConfig, MemoryPlan]:
    """Return the feasible candidate with the lowest step peak.

    Candidates whose peaks land within the model noise floor are treated as
    tied and resolved by quant rank (int8 > fp8 > bf16), then by more swapped
    blocks (a larger swap keeps a lower VRAM resident at equal precision),
    then by fewer trained blocks: with ``train_blocks`` set, a partial window
    trains a subset of the layers, so it is faster and less overfit-prone
    than re-running the full stack at an equal peak.
    """
    best: tuple[RunConfig, MemoryPlan] | None = None
    for candidate, cand_plan in feasible:
        if best is None:
            best = (candidate, cand_plan)
            continue
        best_cfg, best_plan = best
        tol = max(_AUTO_PEAK_TOLERANCE_MIB, best_plan.headline_mib * 0.02)
        if cand_plan.headline_mib < best_plan.headline_mib - tol:
            best = (candidate, cand_plan)
        elif abs(cand_plan.headline_mib - best_plan.headline_mib) <= tol:
            rank_now = _AUTO_QUANT_RANK.get(candidate.quant_base, 9)
            rank_best = _AUTO_QUANT_RANK.get(best_cfg.quant_base, 9)
            if rank_now < rank_best:
                best = (candidate, cand_plan)
            elif rank_now == rank_best:
                fewer_now = _trained_block_count(candidate.train_blocks, layers)
                fewer_best = _trained_block_count(best_cfg.train_blocks, layers)
                if candidate.blocks_to_swap > best_cfg.blocks_to_swap:
                    best = (candidate, cand_plan)
                elif (
                    candidate.blocks_to_swap == best_cfg.blocks_to_swap
                    and fewer_now < fewer_best
                ):
                    best = (candidate, cand_plan)
    assert best is not None  # feasible is non-empty by construction
    return best


def _auto_budget_resolve(
    cfg: RunConfig,
    fam: ModelFamily,
    calibration: Calibration | None = None,
) -> tuple[RunConfig, MemoryPlan]:
    """Search ``quant_base x blocks_to_swap x train_blocks`` for the cheapest
    feasible plan (v0.2 T3 / G2 = AP-02).

    Only LoRA candidates are generated: block swap is a LoRA-only
    optimisation and full fine-tuning must never be picked by the auto search.
    Gradient checkpointing is forced on and the caller's cache plan is left
    untouched. H3 LoRA candidates keep using the pruned transformer via
    ``pruned_params_b`` (see ``_plan_once``). Returns the chosen candidate
    config alongside its plan so the compiler can run that exact config.

    The full precision ladder is ``NONE -> FP8 -> INT8 -> NF4 -> NVFP4``
    (NF4/NVFP4 are the most aggressive 4-bit compression). For video
    block-swap families the
    search also varies ``train_blocks`` over a small preset set: all blocks
    (``None``) plus the Fizgig-style partial window ``"20-49"`` (a
    user-supplied ``train_blocks`` is added as an extra preset). Non-video
    families keep the dimension at ``[None]``. Every evaluated candidate is
    recorded on the returned plan as ``auto_candidates`` (feasible first,
    ascending peak; then infeasible, ascending deficit) so the UI can show
    the "what can train on this GPU" menu instead of hiding it.
    """
    max_swap = max(0, fam.layers - 2)
    quants = [QuantBase.NONE, QuantBase.FP8, QuantBase.INT8, QuantBase.NF4, QuantBase.NVFP4]
    train_presets: list[str | None]
    if fam.kind == "video":
        train_presets = [None, "20-49"]
    else:
        train_presets = [None]
    if cfg.train_blocks is not None and cfg.train_blocks not in train_presets:
        train_presets = [*train_presets, cfg.train_blocks]

    evaluated: list[tuple[RunConfig, MemoryPlan]] = []
    for quant in quants:
        for swap in range(max_swap + 1):
            for train in train_presets:
                candidate = cfg.model_copy(
                    update={
                        "adapter": Adapter.LORA,
                        "quant_base": quant,
                        "blocks_to_swap": swap,
                        "grad_checkpointing": True,
                        "train_blocks": train,
                    }
                )
                evaluated.append((candidate, _plan_once(candidate, calibration)))

    # Display ledger: every evaluated candidate with a short status note.
    # Feasible rows come first (ascending peak), then misses (ascending gap),
    # with a deterministic (quant, swap, train) tie order.
    _quant_order = {q.value: i for i, q in enumerate(quants)}
    _train_order = {preset: i for i, preset in enumerate(train_presets) if preset is not None}

    def _display_key(item: tuple[RunConfig, MemoryPlan]) -> tuple[int, float, int, int, int]:
        cfg_i, p = item
        gap = 0.0 if p.feasible else _budget_deficit(p)
        return (
            0 if p.feasible else 1,
            p.headline_mib if p.feasible else gap,
            _quant_order.get(cfg_i.quant_base.value, 9),
            cfg_i.blocks_to_swap,
            _train_order.get(cfg_i.train_blocks or "", 9),
        )

    auto_candidates = [
        AutoCandidate(
            quant_base=cfg_i.quant_base.value,
            blocks_to_swap=cfg_i.blocks_to_swap,
            train_blocks=cfg_i.train_blocks,
            feasible=p.feasible,
            headline_mib=p.headline_mib,
            note=(
                f"峰值 {p.headline_mib:.0f} MiB"
                if p.feasible
                else f"缺 {_budget_deficit(p):.0f} MiB"
            ),
        )
        for cfg_i, p in sorted(evaluated, key=_display_key)
    ]

    feasible = [(c, p) for c, p in evaluated if p.feasible]
    summary = (
        f"自动预算选择：共评估 {len(evaluated)} 个候选"
        f"（{len(quants)} 种量化 x {max_swap + 1} 档 block swap"
        f" x {len(train_presets)} 档 train_blocks）"
    )
    if feasible:
        best_cfg, best_plan = _pick_best_candidate(feasible, fam.layers)
        summary_note = (
            summary
            + f"，选中 quant_base={best_cfg.quant_base.value}、"
            + f"blocks_to_swap={best_cfg.blocks_to_swap}/{fam.layers}，"
            + f"步骤峰值 {best_plan.headline_mib:.0f} MiB"
        )
        lines = list(best_plan.lines)
        lines.append(
            MemoryLine(
                label="自动预算选择",
                mib=0,
                note=(
                    f"候选 {len(evaluated)} 个（可行 {len(feasible)} 个）；"
                    f"选中 {best_cfg.quant_base.value} + swap "
                    f"{best_cfg.blocks_to_swap}/{fam.layers}"
                    + (f" + train_blocks {best_cfg.train_blocks}" if best_cfg.train_blocks else "")
                    + f"；峰值 {best_plan.headline_mib:.0f} MiB"
                ),
            )
        )
        return best_cfg, best_plan.model_copy(
            update={
                "notes": [*best_plan.notes, summary_note],
                "lines": lines,
                "auto_candidates": auto_candidates,
            }
        )

    # Every candidate misses at least one budget: report the closest miss.
    nearest_cfg, nearest_plan = min(evaluated, key=lambda cp: _budget_deficit(cp[1]))
    vram_gap = max(0.0, -nearest_plan.headroom_mib)
    ram_gap = max(0.0, -nearest_plan.ram_headroom_mib)
    risks = list(nearest_plan.risks)
    risks.append(
        summary
        + "全部不可行；最接近的方案 "
        + f"quant_base={nearest_cfg.quant_base.value}、"
        + f"blocks_to_swap={nearest_cfg.blocks_to_swap}/{fam.layers} "
        + f"仍缺 {vram_gap:.0f} MiB 显存 / {ram_gap:.0f} MiB 内存；"
        + "请降低分辨率或帧数、增大预算，或换更大显存/内存的机器"
    )
    lines = list(nearest_plan.lines)
    lines.append(
        MemoryLine(
            label="自动预算选择",
            mib=0,
            note=(
                f"候选 {len(evaluated)} 个，全部不可行；最接近 "
                f"{nearest_cfg.quant_base.value} + swap "
                f"{nearest_cfg.blocks_to_swap}/{fam.layers}"
            ),
        )
    )
    return nearest_cfg, nearest_plan.model_copy(
        update={
            "feasible": False,
            "risks": risks,
            "notes": [*nearest_plan.notes, summary + "，全部不可行"],
            "lines": lines,
            "auto_candidates": auto_candidates,
        }
    )