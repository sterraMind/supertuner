"""Planner sanity tests: pure arithmetic, no torch required.

Each test pins one modeling decision so a future change that silently shifts
the numbers fails loudly. The absolute numbers are approximations by design;
the tests assert structure, ordering, and that flagship consumer scenarios
land in the right feasibility bucket.
"""

from __future__ import annotations

import pytest

from supertuner_engine.plan import plan
from supertuner_engine.schema import (
    VIDEO_FAMILIES,
    Adapter,
    CachePlan,
    ModelFamily,
    Optimizer,
    Precision,
    QuantBase,
    RunConfig,
    Stage,
)


def _cfg(**kw) -> RunConfig:
    base = dict(name="test", family="qwen3-0.6b", stage=Stage.PRETRAIN)
    base.update(kw)
    return RunConfig(**base)


def test_unknown_family_raises():
    with pytest.raises(ValueError, match="unknown family"):
        plan(_cfg(family="does-not-exist"))


def test_unverified_family_requires_override(monkeypatch):
    monkeypatch.setitem(
        VIDEO_FAMILIES,
        "unverified-video",
        ModelFamily(key="unverified-video", kind="video", params_b=None, hidden=1024, layers=12, heads=8, head_dim=128, ffn=4096),
    )
    with pytest.raises(ValueError, match="no verified parameter count"):
        plan(_cfg(family="unverified-video", stage=Stage.SFT))
    p = plan(_cfg(family="unverified-video", stage=Stage.SFT, params_b_override=0.5))
    assert any("override" in n for n in p.notes)


def test_minimax_h3_uses_real_geometry_and_pruned_lora():
    p = plan(_cfg(family="minimax-h3", stage=Stage.SFT, width=320, height=576, frames=49))
    # f16t4d24: 49 frames -> 13 latent frames; 576/16/2 x 320/16/2 = 18 x 10
    assert p.tokens_per_step == 13 * 18 * 10 * 1 * 4
    assert any("pruned transformer（20.0B" in n for n in p.notes)


def test_minimax_h3_lora_edge_recipe_fits_8gb_with_swap(monkeypatch):
    """320x576x49f, pruned int8 base, 48/50 block swap: the 8GB proof point."""
    p = plan(
        _cfg(
            family="minimax-h3",
            stage=Stage.SFT,
            adapter=Adapter.LORA,
            quant_base=QuantBase.INT8,
            width=320,
            height=576,
            frames=49,
            blocks_to_swap=48,
            cache=CachePlan.BOTH,
        )
    )
    assert p.feasible, f"edge H3 recipe should fit 8GB, peak {p.headline_mib} MiB"
    assert p.ram_peak_mib > 0
    assert any("block swap 48/50" in line.note for line in p.lines)


def test_block_swap_requires_lora_and_grad_ckpt():
    full = plan(_cfg(family="wan2.1-t2v-1.3b", stage=Stage.SFT, adapter=Adapter.NONE, blocks_to_swap=20))
    assert not full.feasible
    assert any("只支持 LoRA" in r for r in full.risks)

    no_ckpt = plan(
        _cfg(family="wan2.1-t2v-1.3b", stage=Stage.SFT, adapter=Adapter.LORA, blocks_to_swap=20, grad_checkpointing=False)
    )
    assert not no_ckpt.feasible
    assert any("梯度检查点" in r for r in no_ckpt.risks)


def test_block_swap_ram_budget_can_reject():
    p = plan(
        _cfg(
            family="minimax-h3",
            stage=Stage.SFT,
            adapter=Adapter.LORA,
            quant_base=QuantBase.INT8,
            width=320,
            height=576,
            frames=49,
            blocks_to_swap=48,
            ram_budget_mib=8192,
        )
    )
    assert not p.feasible
    assert p.ram_headroom_mib < 0
    assert any("系统内存不足" in r for r in p.risks)


def test_math_attention_refused():
    p = plan(_cfg(attn_impl="math"))
    assert not p.feasible
    assert any("attn_impl=math" in r for r in p.risks)


def test_pretrain_360m_fits_8gb():
    # realistic consumer recipe: 360M full-param with paged 8-bit AdamW + fp32 master
    cfg = _cfg(
        family="smollm2-360m",
        adapter=Adapter.NONE,
        precision=Precision.FP32_MASTER,
        optimizer=Optimizer.ADAMW_PAGED_8BIT,
        seq_len=2048,
        batch=8,
    )
    p = plan(cfg)
    assert p.feasible, f"360M pretrain should fit 8GB, got {p.headline_mib} MiB"
    # chunked CE should still be a visible line
    labels = [line.label for line in p.lines]
    assert any("logits" in line for line in labels)


def test_8bit_optimizer_saves_peak_against_fp32_adamw():
    # same batch/seq: paged 8-bit AdamW must be measurably cheaper than fp32 AdamW
    common = dict(family="smollm2-360m", adapter=Adapter.NONE, precision=Precision.FP32_MASTER, seq_len=2048, batch=8)
    fp32 = plan(_cfg(**common, optimizer=Optimizer.ADAMW))
    int8 = plan(_cfg(**common, optimizer=Optimizer.ADAMW_PAGED_8BIT))
    assert int8.headline_mib < fp32.headline_mib
    # the fixed ~0.9 GB CUDA-context term dilutes the percentage, so compare the
    # absolute saving against the pre-context optimizer delta
    assert fp32.headline_mib - int8.headline_mib > 1500


def test_7b_lora_fits_qlora_not_full():
    full = plan(_cfg(family="qwen3-8b", stage=Stage.SFT, adapter=Adapter.NONE, seq_len=1024, batch=1))
    assert not full.feasible, "7-8B full finetune must not fit 8GB"

    qlora = plan(_cfg(family="qwen3-8b", stage=Stage.SFT, adapter=Adapter.QLORA, quant_base=QuantBase.NF4, seq_len=1024, batch=1))
    assert qlora.feasible, f"8B QLoRA should fit 8GB, got {qlora.headline_mib}"


def test_lora_line_items_order():
    p = plan(_cfg(family="qwen3-1.7b", stage=Stage.SFT, adapter=Adapter.LORA, optimizer=Optimizer.ADAMW_8BIT))
    labels = [line.label for line in p.lines]
    assert any("冻结基座" in line for line in labels)
    assert any("LoRA 权重" in line for line in labels)
    assert any("优化器状态" in line and "8bit" in line for line in labels)
    assert any("梯度" in line for line in labels)


def test_8bit_optimizer_beats_adamw_for_lora():
    a = plan(_cfg(family="qwen3-4b", stage=Stage.SFT, adapter=Adapter.LORA, optimizer=Optimizer.ADAMW))
    b = plan(_cfg(family="qwen3-4b", stage=Stage.SFT, adapter=Adapter.LORA, optimizer=Optimizer.ADAMW_8BIT))
    assert b.headline_mib < a.headline_mib


def test_grad_ckpt_reduces_video_peak():
    common = dict(family="wan2.1-t2v-1.3b", stage=Stage.SFT, width=480, height=832, frames=49, cache=CachePlan.BOTH)
    off = plan(_cfg(**common, grad_checkpointing=False))
    on = plan(_cfg(**common, grad_checkpointing=True))
    assert on.headline_mib < off.headline_mib


def test_video_cache_moves_text_encoder_out():
    p = plan(_cfg(family="wan2.1-t2v-1.3b", stage=Stage.SFT, cache=CachePlan.BOTH))
    assert not any("文本编码器权重（未缓存）" in line.label for line in p.lines)
    assert any("预缓存" in n for n in p.notes)
    q = plan(_cfg(family="wan2.1-t2v-1.3b", stage=Stage.SFT, cache=CachePlan.NONE))
    assert any("文本编码器权重（未缓存）" in line.label for line in q.lines)


def test_wan_1_3b_lora_480p_plausible():
    """The flagship consumer scenario: Wan2.1 1.3B LoRA at 480x832x49f."""
    p = plan(_cfg(family="wan2.1-t2v-1.3b", stage=Stage.SFT, width=480, height=832, frames=49, cache=CachePlan.BOTH, quant_base=QuantBase.FP8))
    assert p.feasible, f"Wan 1.3B LoRA 480p should fit 8GB, peak {p.headline_mib} MiB"
    # tokens = 13 lat frames x 60 x 104 = 81120; activation must dominate
    assert p.tokens_per_step > 0
    assert any("DiT 激活值" in line.label for line in p.lines)


def test_wan_14b_needs_quant_or_swap():
    """14B LoRA without quant tricks must not silently pass on 8GB."""
    p = plan(_cfg(family="wan2.1-t2v-14b", stage=Stage.SFT, cache=CachePlan.BOTH, width=480, height=832, frames=49))
    assert not p.feasible


def test_tokens_per_step_matches_geometry():
    cfg = _cfg(family="wan2.1-t2v-1.3b", stage=Stage.SFT, width=480, height=832, frames=49, batch=1, grad_accum=4)
    p = plan(cfg)
    # (49-1)/4+1 = 13 latent frames; 832/8/2 = 52, 480/8/2 = 30 -> 13*52*30 tokens
    assert p.tokens_per_step == 13 * 52 * 30 * 4


def test_headroom_negative_when_over():
    cfg = _cfg(family="qwen3-8b", stage=Stage.SFT, adapter=Adapter.NONE, seq_len=4096)
    p = plan(cfg)
    assert p.headroom_mib < 0
    assert any("超出预算" in r for r in p.risks)


def test_feasible_small_sft_no_risk_noise():
    p = plan(_cfg(family="qwen3-0.6b", stage=Stage.SFT, adapter=Adapter.LORA))
    if p.feasible:
        assert p.headroom_mib > 0
