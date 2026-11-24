"""Tests for v0.3 P0 eval_report prompt_eval integration (pure)."""

from __future__ import annotations

import json

from supertuner_engine.backends.eval_report import build_eval_report, summarise_stability, write_eval_report
from supertuner_engine.schema import RunSpec


def _spec(**overrides):
    base = {
        "schema_version": 1,
        "config": {"name": "t", "family": "qwen3-0.6b", "stage": "sft", "adapter": "lora"},
        "dataset": {"path": "x", "kind": "text", "val_fraction": 0.1},
        "train": {"max_steps": 10, "eval_every": 5},
    }
    base.update(overrides)
    return RunSpec.model_validate(base)


def test_summarise_stability_basic():
    s = summarise_stability(
        train_curve=[
            {"step": 1, "loss": 2.0, "grad_norm": 1.0},
            {"step": 2, "loss": 1.5, "grad_norm": 1.1},
            {"step": 3, "loss": 1.2, "grad_norm": 0.9},
            {"step": 4, "loss": 1.1, "grad_norm": 1.0},
        ],
        eval_curve=[
            {"step": 2, "metrics": {"eval_loss": 1.6}},
            {"step": 4, "metrics": {"eval_loss": 1.3}},
        ],
    )
    assert s["train"]["steps"] == 4
    assert s["train"]["loss_last"] == 1.1
    assert s["eval"]["count"] == 2
    assert s["eval"]["loss_last"] == 1.3
    assert s["overfitting"]["train_eval_gap"] == 1.1 - 1.3
    assert s["grad_norm"]["count"] == 4


def test_summarise_stability_insufficient_eval():
    s = summarise_stability(train_curve=[{"step": 1, "loss": 1.0, "grad_norm": 1.0}], eval_curve=[])
    assert s["overfitting"]["signal"] == "insufficient_eval"
    assert s["overfitting"]["train_eval_gap"] is None


def test_build_eval_report_ok_and_contract():
    spec = _spec(eval={"enabled": True, "prompts_path": None, "holdout": True, "source_dir": None, "metrics": ["clip_prompt_following"]})
    report = build_eval_report(
        spec=spec,
        train_curve=[{"step": 1, "loss": 2.0, "grad_norm": 1.0}, {"step": 2, "loss": 1.2, "grad_norm": 0.8}],
        eval_curve=[{"step": 2, "metrics": {"eval_loss": 1.3}}],
        extra={"stage": "sft"},
    )
    assert report["version"] == 1
    assert report["status"] == "ok"
    assert report["extra"] == {"stage": "sft"}
    assert report["contract"]["holdout"] is True
    assert report["contract"]["dataset_val_fraction"] == 0.1
    assert report["contract"]["requested_metrics"] == ["clip_prompt_following"]
    assert "train_loss_curve" in report["contract"]["llm_p0_metrics"]
    assert "clip_prompt_following" in report["contract"]["video_p0_contract_metrics"]


def test_build_eval_report_disabled():
    spec = _spec(eval={"enabled": False})
    report = build_eval_report(spec=spec, train_curve=[{"step": 1, "loss": 1.0}], eval_curve=[])
    assert report["status"] == "skipped"
    assert report["reason"] == "eval.disabled"


def test_write_eval_report(tmp_path):
    out = tmp_path / "nested" / "eval_report.json"
    payload = {"version": 1, "status": "ok", "summary": {}}
    write_eval_report(out, payload)
    assert json.loads(out.read_text(encoding="utf-8")) == payload


def test_build_eval_report_with_prompt_eval_ok():
    spec = _spec(eval={"enabled": True, "prompts_path": "docs/development/eval/prompts/llm_smoke_v1.json"})
    prompt_eval = {
        "status": "ok",
        "backend": "llm_prompt_eval",
        "count": 6,
        "ok_count": 6,
        "aggregate": {"non_empty_rate": 1.0, "exact_match_rate": 0.5, "mean_token_overlap_f1": 0.7},
    }
    report = build_eval_report(
        spec=spec,
        train_curve=[{"step": 1, "loss": 2.0, "grad_norm": 1.0}],
        eval_curve=[],
        prompt_eval=prompt_eval,
    )
    assert report["status"] == "ok"
    assert report["summary"]["prompt_eval"]["count"] == 6
    assert "prompt_eval_non_empty_rate" in report["contract"]["llm_p0_metrics"]
    assert any("prompt-level eval status: ok" in n for n in report["notes"])


def test_build_eval_report_prompt_eval_unavailable_degrades_status():
    spec = _spec(eval={"enabled": True, "prompts_path": "missing.json"})
    report = build_eval_report(
        spec=spec,
        train_curve=[{"step": 1, "loss": 2.0, "grad_norm": 1.0}],
        eval_curve=[],
        prompt_eval={"status": "unavailable", "reason": "prompt_set_invalid", "error": "nope"},
    )
    assert report["status"] == "partial"
    assert report["summary"]["prompt_eval"]["status"] == "unavailable"
