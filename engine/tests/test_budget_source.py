"""实测进预算（v0.2 T2 / G3 = AP-01）：预算来源与实测值的溯源行为。

钉住 `plan()` 在 `budget_source` 为 auto / manual 时如何解析 VRAM/RAM 预算，
以及 `MemoryPlan` 上 `budget_source` / `budget_measured_*` 三个字段的取值。
算术本身不变，这里只验证"预算数字从哪来"。
"""

from __future__ import annotations

from supertuner_engine.plan import plan
from supertuner_engine.schema import RunConfig, Stage


def _cfg(**kw) -> RunConfig:
    base = dict(name="budget", family="qwen3-0.6b", stage=Stage.SFT)
    base.update(kw)
    return RunConfig(**base)


def test_auto_with_measured_values_uses_them():
    cfg = _cfg(budget_source="auto", vram_budget_mib=8192, ram_budget_mib=32768)
    p = plan(cfg, measured_vram_mib=24576, measured_ram_mib=65536)
    assert p.budget_source == "measured"
    assert p.budget_measured_vram_mib == 24576
    assert p.budget_measured_ram_mib == 65536
    # 计划里的预算上限就是实测值。
    assert p.vram_budget_mib == 24576
    assert p.ram_budget_mib == 65536


def test_manual_ignores_measured_values():
    cfg = _cfg(budget_source="manual", vram_budget_mib=8192, ram_budget_mib=32768)
    p = plan(cfg, measured_vram_mib=24576, measured_ram_mib=65536)
    assert p.budget_source == "manual"
    assert p.budget_measured_vram_mib is None
    assert p.budget_measured_ram_mib is None
    # 手填值原样保留。
    assert p.vram_budget_mib == 8192
    assert p.ram_budget_mib == 32768


def test_auto_without_measured_falls_back_to_defaults():
    cfg = _cfg(budget_source="auto", vram_budget_mib=8192, ram_budget_mib=32768)
    p = plan(cfg, measured_vram_mib=None, measured_ram_mib=None)
    assert p.budget_source == "manual"
    assert p.budget_measured_vram_mib is None
    assert p.budget_measured_ram_mib is None
    assert p.vram_budget_mib == 8192
    assert p.ram_budget_mib == 32768


def test_auto_partial_measured_only_vram():
    cfg = _cfg(budget_source="auto", vram_budget_mib=8192, ram_budget_mib=32768)
    p = plan(cfg, measured_vram_mib=24576, measured_ram_mib=None)
    assert p.budget_source == "measured"
    assert p.budget_measured_vram_mib == 24576
    assert p.budget_measured_ram_mib is None
    # VRAM 用实测值，RAM 回落到手填默认。
    assert p.vram_budget_mib == 24576
    assert p.ram_budget_mib == 32768


def test_default_budget_source_is_auto_and_falls_back():
    # 未显式给 budget_source 时默认 "auto"；没给实测值则回落默认（"manual"）。
    cfg = _cfg(vram_budget_mib=8192, ram_budget_mib=32768)
    assert cfg.budget_source == "auto"
    p = plan(cfg)
    assert p.budget_source == "manual"
    assert p.budget_measured_vram_mib is None
    assert p.budget_measured_ram_mib is None
    assert p.vram_budget_mib == 8192


def test_infeasible_refused_plan_still_carries_budget_fields():
    # attn_impl=math 会被拒绝；拒绝路径（refuse()）也必须带预算来源字段。
    cfg = _cfg(budget_source="auto", vram_budget_mib=8192, ram_budget_mib=32768, attn_impl="math")
    p = plan(cfg, measured_vram_mib=4096, measured_ram_mib=8192)
    assert not p.feasible
    assert p.budget_source == "measured"
    assert p.budget_measured_vram_mib == 4096
    assert p.budget_measured_ram_mib == 8192
    assert p.vram_budget_mib == 4096
    assert p.ram_budget_mib == 8192


def test_infeasible_over_budget_plan_carries_budget_fields():
    # 配置太大导致不可行（非 refuse 的负 headroom 路径）同样要带预算字段。
    cfg = _cfg(family="qwen3-8b", stage=Stage.SFT, adapter="none", seq_len=4096)
    p = plan(cfg, measured_vram_mib=8192, measured_ram_mib=32768)
    assert not p.feasible
    assert p.budget_source == "measured"
    assert p.budget_measured_vram_mib == 8192
    assert p.budget_measured_ram_mib == 32768
    assert p.vram_budget_mib == 8192


def test_zero_measured_falls_back():
    # measured=0 视为"未提供"，回落到 cfg 值。
    cfg = _cfg(budget_source="auto", vram_budget_mib=8192, ram_budget_mib=32768)
    p = plan(cfg, measured_vram_mib=0, measured_ram_mib=0)
    assert p.budget_source == "manual"
    assert p.budget_measured_vram_mib is None
    assert p.budget_measured_ram_mib is None
    assert p.vram_budget_mib == 8192
