"""单向 block swap 搬运时间带测试（v0.2 T6 / AP-08）。

Pins the structured swap-band fields on `MemoryPlan`:

- when block swap is active the one-way (H2D-only) PCIe floor and the
  round-trip floor are exposed as structured fields, with the round-trip
  exactly 2x the one-way floor and a human band string;
- without swap the fields stay None / empty (backward compatible).
"""

from __future__ import annotations

import pytest

from supertuner_engine.plan import plan
from supertuner_engine.schema import (
    Adapter,
    QuantBase,
    RunConfig,
    Stage,
)


def _cfg(**kw) -> RunConfig:
    base = dict(name="swap-band", family="qwen3-0.6b", stage=Stage.SFT)
    base.update(kw)
    return RunConfig(**base)


def test_swap_band_populated_when_swap_active():
    """H3 LoRA 48/50 swap: structured one-way vs round-trip floors appear."""
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
        )
    )
    assert p.swap_floor_s is not None and p.swap_floor_s > 0
    assert p.swap_roundtrip_floor_s is not None
    # round-trip is exactly 2x the one-way floor
    assert p.swap_roundtrip_floor_s == pytest.approx(2 * p.swap_floor_s)
    # the band string is human-readable and names both bounds
    assert p.step_time_band_s
    assert f"{p.swap_floor_s:.1f}s" in p.step_time_band_s
    assert f"{p.swap_roundtrip_floor_s:.1f}s" in p.step_time_band_s
    assert "swap transfer only" in p.step_time_band_s
    # the legacy note now references the structured one-way/round-trip numbers
    assert any("单向" in n and "往返" in n for n in p.notes)


def test_swap_band_floor_matches_pcie_math():
    """The one-way floor is swapped bytes / (8 GB/s) for the H3 edge recipe."""
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
        )
    )
    # pruned 20B base x 1.05 B/param int8, 48/50 blocks swapped out
    swapped_bytes = (48.0 / 50.0) * 20e9 * 1.05
    expected = swapped_bytes / (8.0 * 1e9)
    assert p.swap_floor_s is not None
    assert p.swap_floor_s == pytest.approx(expected, rel=1e-6)


def test_swap_band_empty_when_no_swap():
    """No block swap: all three fields keep their empty defaults."""
    p = plan(_cfg())
    assert p.swap_floor_s is None
    assert p.swap_roundtrip_floor_s is None
    assert p.step_time_band_s == ""
