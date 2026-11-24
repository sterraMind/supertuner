"""Training/eval summary report generation (v0.3 P0).

Produces a JSON-serialisable eval_report from in-run metrics curves.
Pure Python: no torch dependency, so it can be unit-tested cheaply.
"""

from __future__ import annotations

import json
import math
import statistics
from pathlib import Path
from typing import Any


def _finite(values: list[float]) -> list[float]:
    return [v for v in values if math.isfinite(v)]


def _safe_float(value: Any) -> float | None:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def summarise_stability(
    train_curve: list[dict[str, Any]],
    eval_curve: list[dict[str, Any]],
    *,
    tail_window: int = 20,
) -> dict[str, Any]:
    """Compute stability and overfitting indicators from step/eval curves."""
    losses = [v for row in train_curve if (v := _safe_float(row.get("loss"))) is not None]
    grad_norms = [v for row in train_curve if (v := _safe_float(row.get("grad_norm"))) is not None]
    eval_losses = [
        (int(row.get("step", 0)), float(metrics["eval_loss"]))
        for row in eval_curve
        for metrics in [row.get("metrics") or {}]
        if isinstance(metrics, dict) and _safe_float(metrics.get("eval_loss")) is not None
    ]

    tail = _finite(losses[-tail_window:]) if losses else []
    grad_finite = _finite(grad_norms)
    grad_median = statistics.median(grad_finite) if grad_finite else 0.0
    spike_threshold = max(grad_median * 10.0, 5.0) if grad_median > 0 else 10.0
    grad_spikes = [g for g in grad_finite if g > spike_threshold]

    train_last = tail[-1] if tail else (losses[-1] if losses else None)
    eval_last = eval_losses[-1][1] if eval_losses else None
    overfit_gap = None
    if train_last is not None and eval_last is not None:
        overfit_gap = float(train_last - eval_last)

    # simple trend: compare mean of first half vs second half of tail
    trend = None
    if len(tail) >= 4:
        mid = len(tail) // 2
        first = statistics.mean(tail[:mid])
        second = statistics.mean(tail[mid:])
        if second < first * 0.98:
            trend = "decreasing"
        elif second > first * 1.02:
            trend = "increasing"
        else:
            trend = "flat"

    eval_trend = None
    if len(eval_losses) >= 3:
        vals = [v for _, v in eval_losses]
        mid = len(vals) // 2
        first = statistics.mean(vals[:mid])
        second = statistics.mean(vals[mid:])
        if second < first * 0.98:
            eval_trend = "decreasing"
        elif second > first * 1.02:
            eval_trend = "increasing"
        else:
            eval_trend = "flat"

    return {
        "train": {
            "steps": len(train_curve),
            "loss_first": losses[0] if losses else None,
            "loss_last": losses[-1] if losses else None,
            "loss_min": min(losses) if losses else None,
            "loss_max": max(losses) if losses else None,
            "loss_mean_tail": statistics.mean(tail) if tail else None,
            "loss_std_tail": statistics.pstdev(tail) if len(tail) >= 2 else None,
            "loss_trend_tail": trend,
        },
        "grad_norm": {
            "count": len(grad_finite),
            "max": max(grad_finite) if grad_finite else None,
            "mean": statistics.mean(grad_finite) if grad_finite else None,
            "median": grad_median if grad_finite else None,
            "spike_threshold": spike_threshold if grad_finite else None,
            "spike_count": len(grad_spikes),
        },
        "eval": {
            "count": len(eval_losses),
            "loss_first": eval_losses[0][1] if eval_losses else None,
            "loss_last": eval_last,
            "loss_min": min(v for _, v in eval_losses) if eval_losses else None,
            "loss_max": max(v for _, v in eval_losses) if eval_losses else None,
            "trend": eval_trend,
            "points": [{"step": s, "eval_loss": v} for s, v in eval_losses],
        },
        "overfitting": {
            "train_eval_gap": overfit_gap,
            "signal": (
                "likely_overfitting"
                if overfit_gap is not None and overfit_gap > 0.1 and eval_trend == "increasing"
                else "possible_overfitting"
                if overfit_gap is not None and overfit_gap > 0.05
                else "insufficient_eval"
                if overfit_gap is None
                else "no_strong_signal"
            ),
        },
    }


def build_eval_report(
    *,
    spec: Any,
    train_curve: list[dict[str, Any]],
    eval_curve: list[dict[str, Any]],
    extra: dict[str, Any] | None = None,
    prompt_eval: dict[str, Any] | None = None,
) -> dict[str, Any]:
    eval_cfg = getattr(spec, "eval", None)
    enabled = True if eval_cfg is None else bool(getattr(eval_cfg, "enabled", True))
    holdout = True if eval_cfg is None else bool(getattr(eval_cfg, "holdout", True))
    prompts_path = None if eval_cfg is None else getattr(eval_cfg, "prompts_path", None)
    source_dir = None if eval_cfg is None else getattr(eval_cfg, "source_dir", None)
    requested_metrics = [] if eval_cfg is None else list(getattr(eval_cfg, "metrics", []) or [])

    if not enabled:
        report: dict[str, Any] = {
            "version": 1,
            "status": "skipped",
            "reason": "eval.disabled",
            "summary": {},
            "contract": {},
        }
    else:
        stability = summarise_stability(train_curve, eval_curve)
        dataset = getattr(spec, "dataset", None)
        val_fraction = float(getattr(dataset, "val_fraction", 0.0) or 0.0) if dataset else 0.0
        has_eval = stability["eval"]["count"] > 0
        overfit = dict(stability["overfitting"])
        if not holdout:
            overfit["note"] = "holdout disabled in eval config"
        elif not has_eval:
            overfit["note"] = "no eval points recorded; set dataset.val_fraction>0 and train.eval_every to measure overfitting gap"
        if val_fraction <= 0.0 and not has_eval:
            overfit["data_holdout_fraction"] = val_fraction

        status = "ok"
        notes: list[str] = []
        if not train_curve:
            status = "partial"
            notes.append("no training step metrics recorded")
        if prompts_path:
            p = Path(prompts_path)
            notes.append(f"frozen prompt set configured: {prompts_path} (exists={p.exists()})")
        if source_dir:
            notes.append(f"source_dir configured: {source_dir}")
        if requested_metrics:
            notes.append(f"requested metrics recorded as contract (not all may be computed in LLM P0): {requested_metrics}")
        if prompt_eval is not None:
            pe_status = str(prompt_eval.get("status") or "unknown")
            notes.append(f"prompt-level eval status: {pe_status}")
            if pe_status in {"unavailable", "error"}:
                status = "partial"
            elif pe_status == "partial" and status == "ok":
                status = "partial"
        elif prompts_path:
            notes.append("prompt-level eval not executed in this run")

        report = {
            "version": 1,
            "status": status,
            "summary": {
                "stability": stability["train"],
                "grad_norm": stability["grad_norm"],
                "eval_curve": stability["eval"],
                "overfitting": overfit,
            },
            "contract": {
                "holdout": holdout,
                "dataset_val_fraction": val_fraction,
                "prompts_path": prompts_path,
                "source_dir": source_dir,
                "requested_metrics": requested_metrics,
                "llm_p0_metrics": ["train_loss_curve", "grad_norm_stability", "eval_loss_curve", "train_eval_gap"],
                "video_p0_contract_metrics": ["clip_prompt_following", "lpips_edit_fidelity", "temporal_consistency"],
            },
            "notes": notes,
        }
        if prompt_eval is not None:
            report["summary"]["prompt_eval"] = prompt_eval
            report["contract"]["llm_p0_metrics"] = list(report["contract"]["llm_p0_metrics"]) + [
                "prompt_eval_non_empty_rate",
                "prompt_eval_exact_match_rate",
                "prompt_eval_token_overlap_f1",
            ]

    if extra:
        report["extra"] = extra
    return report


def write_eval_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
