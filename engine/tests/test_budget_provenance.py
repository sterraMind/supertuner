"""compile 路径的实测进预算溯源（v0.2 T2 / G3 = AP-01）。

`plan()` 已按 budget_source 解析预算并盖溯源戳；这里钉住 `compile_spec`
走同样的语义：auto + 实测值 → "measured" 且计划按实测预算计算；manual 或
没给实测值 → "manual" 且手填/默认预算原样保留。算术本身不在此验证。
"""

from __future__ import annotations

import pytest

from supertuner_engine.backends import musubi as musubi_backend
from supertuner_engine.backends import native_llm
from supertuner_engine.compiler import compile_spec
from supertuner_engine.schema import Adapter, RunSpec

from test_compile import _ctx, _llm_spec


@pytest.fixture(autouse=True)
def _no_gpu(monkeypatch):
    """与 test_compile.py 相同：编译结果不依赖本机 GPU / bitsandbytes。"""
    monkeypatch.setattr(musubi_backend, "nvidia_vram_mib", lambda: 0)
    monkeypatch.setattr(native_llm, "_available", lambda name: name != "bitsandbytes")


def _spec(**config_kw) -> RunSpec:
    config_kw.setdefault("family", "qwen3-0.6b")
    config_kw.setdefault("adapter", Adapter.LORA)
    config_kw.setdefault("budget_source", "auto")
    return _llm_spec(**config_kw)


def _codes(plan) -> set[str]:
    return {issue.code for issue in plan.issues}


def test_compile_auto_with_measured_stamps_provenance(tmp_path):
    spec = _spec(budget_source="auto", vram_budget_mib=8192, ram_budget_mib=32768)
    plan = compile_spec(spec, _ctx(tmp_path), measured_vram_mib=8191, measured_ram_mib=32452)

    mem = plan.memory
    assert mem.budget_source == "measured"
    assert mem.budget_measured_vram_mib == 8191
    assert mem.budget_measured_ram_mib == 32452
    # 计划按实测预算算：预算上限就是实测值。
    assert mem.vram_budget_mib == 8191
    assert mem.ram_budget_mib == 32452
    # 供持久化的原 spec 不许被改动。
    assert spec.config.vram_budget_mib == 8192
    # auto_budget 搜索没有换候选，不能误报 I_AUTO_BUDGET_APPLIED。
    assert "I_AUTO_BUDGET_APPLIED" not in _codes(plan)


def test_compile_manual_ignores_measured_values(tmp_path):
    spec = _spec(budget_source="manual", vram_budget_mib=6144, ram_budget_mib=16384)
    plan = compile_spec(spec, _ctx(tmp_path), measured_vram_mib=8191, measured_ram_mib=32452)

    mem = plan.memory
    assert mem.budget_source == "manual"
    assert mem.budget_measured_vram_mib is None
    assert mem.budget_measured_ram_mib is None
    # 手填预算原样保留。
    assert mem.vram_budget_mib == 6144
    assert mem.ram_budget_mib == 16384


def test_compile_auto_without_measured_falls_back_to_manual(tmp_path):
    spec = _spec(budget_source="auto", vram_budget_mib=8192, ram_budget_mib=32768)
    plan = compile_spec(spec, _ctx(tmp_path))

    mem = plan.memory
    assert mem.budget_source == "manual"
    assert mem.budget_measured_vram_mib is None
    assert mem.budget_measured_ram_mib is None
    assert mem.vram_budget_mib == 8192
    assert mem.ram_budget_mib == 32768
