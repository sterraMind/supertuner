"""NVFP4 量化档位测试（v0.2 T7）。

Pins:

- NVFP4 is a valid ``QuantBase`` member that the planner accepts;
- its per-param bytes are 0.5625 (4-bit weights + E4M3 block scales);
- the unified auto-budget search evaluates NVFP4 candidates and the
  NVFP4 tier behaves like the other low-VRAM 4-bit/8-bit tiers (it never
  out-peaks NF4, which stays the search's most aggressive 4-bit tier).
"""

from __future__ import annotations

from supertuner_engine.plan import (
    NVFP4_BYTES_PER_PARAM,
    _quant_bytes_per_param,
    _weight_bytes_per_param,
    plan,
)
from supertuner_engine.schema import (
    Precision,
    QuantBase,
    RunConfig,
    Stage,
)


def _video(family: str = "wan2.1-t2v-1.3b", **kw) -> RunConfig:
    base = dict(
        name="nvfp4-test",
        family=family,
        stage=Stage.SFT,
        adapter="lora",
        quant_base="none",
        width=480,
        height=832,
        frames=49,
        cache="both",
        vram_budget_mib=8192,
        ram_budget_mib=32768,
    )
    base.update(kw)
    return RunConfig(**base)


def test_nvfp4_is_a_valid_quant_base():
    assert QuantBase.NVFP4.value == "nvfp4"
    # pydantic accepts the string and resolves to the enum member
    cfg = RunConfig(name="t", family="qwen3-0.6b", stage=Stage.SFT, quant_base="nvfp4")
    assert cfg.quant_base is QuantBase.NVFP4


def test_nvfp4_bytes_per_param():
    assert NVFP4_BYTES_PER_PARAM == 0.5625
    cfg = RunConfig(name="t", family="qwen3-0.6b", stage=Stage.SFT, quant_base="nvfp4")
    assert _weight_bytes_per_param(cfg) == 0.5625
    assert _quant_bytes_per_param(QuantBase.NVFP4, Precision.BF16_PURE) == 0.5625
    # still a low-VRAM tier: below fp8/int8/bf16 storage
    assert _weight_bytes_per_param(cfg) < 1.0
    assert _weight_bytes_per_param(cfg) < _weight_bytes_per_param(
        RunConfig(name="t", family="qwen3-0.6b", stage=Stage.SFT)
    )


def test_nvfp4_plan_is_feasible_for_lora():
    p = plan(_video(quant_base="nvfp4"))
    assert p.feasible, f"NVFP4 LoRA on Wan 1.3B should fit 8GB, peak {p.headline_mib}"
    # the quant note surfaces on the resident-weights line
    assert any("NVFP4" in line.note for line in p.lines)


def test_nvfp4_auto_budget_evaluates_nvfp4_candidates():
    """The unified search's candidate ledger records NVFP4 rows."""
    p = plan(_video(auto_budget=True))
    assert p.feasible
    quants = {c.quant_base for c in p.auto_candidates}
    assert QuantBase.NVFP4.value in quants
    # NVFP4 is the most aggressive 4-bit tier alongside NF4: it never wins the
    # selection outright when NF4 is available, but it is a valid candidate
    # and carries a per-candidate peak in the ledger.
    nvfp4_rows = [c for c in p.auto_candidates if c.quant_base == QuantBase.NVFP4.value]
    assert nvfp4_rows
    assert all(isinstance(c.headline_mib, float) for c in nvfp4_rows)


def test_nvfp4_never_outranks_nf4_on_peaks():
    """NVFP4 (0.5625 B/p) must not beat NF4 (0.55 B/p) at an equal swap."""
    base = dict(
        family="wan2.1-t2v-1.3b",
        stage=Stage.SFT,
        adapter="lora",
        width=480,
        height=832,
        frames=49,
        cache="both",
        blocks_to_swap=20,
        vram_budget_mib=16384,
        ram_budget_mib=65536,
    )
    nf4 = plan(RunConfig(name="nf4", **base, quant_base=QuantBase.NF4))
    nvfp4 = plan(RunConfig(name="nvfp4", **base, quant_base=QuantBase.NVFP4))
    assert nvfp4.headline_mib >= nf4.headline_mib
    assert nvfp4.ram_peak_mib >= nf4.ram_peak_mib
