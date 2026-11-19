"""跑后自校准（v0.2 T1）：plan() 只读注入的 calibration 做只读标注。

这些测试固定三条行为，防止未来回归：
  1. 样本足够（count>=3）且已有平均偏差时，notes 里出现「已校准 N 次（平均偏差 x%）」；
  2. 样本不足（N<3）时只标「样本 N 次」，不强行给偏差；
  3. 注入的 calibration 对象互不串档，且绝不改动任何预算字段。
"""

from __future__ import annotations

from supertuner_engine.plan import plan
from supertuner_engine.schema import (
    Calibration,
    RunConfig,
    Stage,
)


def _cfg(**kw) -> RunConfig:
    base = dict(name="test", family="qwen3-0.6b", stage=Stage.PRETRAIN)
    base.update(kw)
    return RunConfig(**base)


def _calib_notes(p):
    """返回该 plan 里所有「已校准」备注。"""
    return [n for n in p.notes if "已校准" in n]


def test_note_shows_count_and_avg_when_enough_samples():
    p = plan(_cfg(), calibration=Calibration(count=5, avg_delta_pct=2.0))
    assert p.calibration is not None
    assert p.calibration.count == 5
    assert p.calibration.avg_delta_pct == 2.0
    notes = _calib_notes(p)
    assert len(notes) == 1
    assert "已校准 5 次" in notes[0]
    assert "平均偏差 +2.0%" in notes[0]


def test_note_marks_sample_when_few_samples():
    p = plan(_cfg(), calibration=Calibration(count=2, avg_delta_pct=1.0))
    notes = _calib_notes(p)
    assert len(notes) == 1
    assert "已校准 2 次" in notes[0]
    # 样本不足（N<3）只标样本数，不出现「平均偏差」
    assert "样本 2 次" in notes[0]
    assert "平均偏差" not in notes[0]


def test_no_calibration_no_note():
    p = plan(_cfg(), calibration=None)
    assert p.calibration is None
    assert _calib_notes(p) == []


def test_zero_count_not_annotated():
    p = plan(_cfg(), calibration=Calibration(count=0, avg_delta_pct=None))
    assert p.calibration is not None
    assert p.calibration.count == 0
    # count=0 视为尚未校准，不写「已校准」备注
    assert _calib_notes(p) == []


def test_zero_avg_rendered_as_signed_zero():
    # count>=3 且 avg=0.0 时应给「平均偏差 +0.0%」
    p = plan(_cfg(), calibration=Calibration(count=3, avg_delta_pct=0.0))
    notes = _calib_notes(p)
    assert len(notes) == 1
    assert "平均偏差 +0.0%" in notes[0]


def test_distinct_calibrations_do_not_cross():
    a = plan(_cfg(), calibration=Calibration(count=3, avg_delta_pct=1.5))
    b = plan(_cfg(), calibration=Calibration(count=7, avg_delta_pct=-2.25))
    na = _calib_notes(a)
    nb = _calib_notes(b)
    assert any("已校准 3 次" in n for n in na)
    assert any("已校准 7 次" in n for n in nb)
    # a 不应混入 b 的计数，反之亦然
    assert not any("已校准 7 次" in n for n in na)
    assert not any("已校准 3 次" in n for n in nb)


def test_calibration_does_not_change_budget():
    base = plan(_cfg())
    with_cal = plan(_cfg(), calibration=Calibration(count=10, avg_delta_pct=50.0))
    # 只读标注：预算、峰值、步 token 数都保持不变
    assert base.vram_budget_mib == with_cal.vram_budget_mib
    assert base.ram_budget_mib == with_cal.ram_budget_mib
    assert base.headline_mib == with_cal.headline_mib
    assert base.training_peak_mib == with_cal.training_peak_mib
    assert base.tokens_per_step == with_cal.tokens_per_step
