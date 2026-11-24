"""LLM prompt-set evaluation helpers (v0.3 P0).

Loads a frozen prompt set (json/jsonl) and runs lightweight, dependency-free
scoring on generated completions. Generation itself stays in native_llm because
it needs torch/model/tokenizer already in memory after training.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


class PromptSetError(ValueError):
    """Invalid or unreadable frozen prompt set."""


def _tokenize_text(text: str) -> list[str]:
    return [t for t in re.split(r"\s+", (text or "").strip().lower()) if t]


def _normalize_match(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def score_completion(reference: str | None, completion: str) -> dict[str, Any]:
    """Lightweight heuristic scores for one completion (no external models)."""
    ref = reference or ""
    comp = completion or ""
    norm_ref = _normalize_match(ref)
    norm_comp = _normalize_match(comp)
    exact = bool(norm_ref) and norm_comp == norm_ref
    contains = bool(norm_ref) and norm_ref in norm_comp
    ref_tokens = _tokenize_text(ref)
    comp_tokens = _tokenize_text(comp)
    overlap = 0.0
    if ref_tokens and comp_tokens:
        ref_set = set(ref_tokens)
        comp_set = set(comp_tokens)
        inter = len(ref_set & comp_set)
        prec = inter / max(len(comp_set), 1)
        rec = inter / max(len(ref_set), 1)
        overlap = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
    return {
        "non_empty": bool(comp.strip()),
        "char_count": len(comp),
        "token_count": len(comp_tokens),
        "exact_match": exact if ref else None,
        "contains_reference": contains if ref else None,
        "token_overlap_f1": round(overlap, 6) if ref else None,
    }


def _coerce_prompt_entry(raw: Any, index: int) -> dict[str, Any]:
    if isinstance(raw, str):
        raw = {"prompt": raw}
    if not isinstance(raw, dict):
        raise PromptSetError(f"prompt[{index}] must be an object or string")
    prompt = raw.get("prompt")
    messages = raw.get("messages")
    if not isinstance(prompt, str) and messages is None:
        raise PromptSetError(f"prompt[{index}] requires 'prompt' (string) or 'messages'")
    entry: dict[str, Any] = {
        "id": str(raw.get("id") or f"p{index + 1:03d}"),
        "prompt": prompt if isinstance(prompt, str) else None,
        "messages": messages if isinstance(messages, list) else None,
        "reference": raw.get("reference") if isinstance(raw.get("reference"), str) else None,
        "tags": [str(t) for t in raw.get("tags", [])] if isinstance(raw.get("tags"), list) else [],
        "max_new_tokens": raw.get("max_new_tokens"),
    }
    return entry


def load_prompt_set(path: str | Path) -> dict[str, Any]:
    """Load a frozen prompt set from JSON or JSONL.

    JSON: {"version":1,"name":"...","prompts":[...]} or a bare list [...].
    JSONL: one prompt entry per line (blank/comment lines ignored).
    """
    p = Path(path)
    if not p.exists():
        raise PromptSetError(f"prompt set not found: {p}")
    text = p.read_text(encoding="utf-8")
    suffix = p.suffix.lower()
    meta: dict[str, Any] = {"version": 1, "name": p.stem, "description": None, "seed": None, "max_new_tokens": 64}
    items: list[Any] = []
    try:
        if suffix == ".jsonl":
            for line in text.splitlines():
                s = line.strip()
                if not s or s.startswith("#"):
                    continue
                items.append(json.loads(s))
        else:
            data = json.loads(text)
            if isinstance(data, list):
                items = data
            elif isinstance(data, dict):
                if isinstance(data.get("prompts"), list):
                    items = data["prompts"]
                elif isinstance(data.get("samples"), list):
                    items = data["samples"]
                else:
                    raise PromptSetError("JSON prompt set must contain a 'prompts' (or 'samples') list or be a bare list")
                for key in ("version", "name", "description", "seed", "max_new_tokens"):
                    if key in data and data[key] is not None:
                        meta[key] = data[key]
            else:
                raise PromptSetError("unsupported prompt set JSON structure")
    except json.JSONDecodeError as exc:
        raise PromptSetError(f"invalid JSON in prompt set {p}: {exc}") from exc

    prompts = [_coerce_prompt_entry(item, i) for i, item in enumerate(items)]
    if not prompts:
        raise PromptSetError(f"prompt set is empty: {p}")
    return {**meta, "path": str(p), "prompts": prompts}


def aggregate_prompt_scores(results: list[dict[str, Any]]) -> dict[str, Any]:
    if not results:
        return {"count": 0}
    scored = [r.get("scores") or {} for r in results]
    non_empty = [1.0 if s.get("non_empty") else 0.0 for s in scored]
    lengths = [float(s.get("char_count") or 0) for s in scored]
    tokens = [float(s.get("token_count") or 0) for s in scored]
    def _mean_bool(key: str) -> float | None:
        vals = [1.0 if s.get(key) else 0.0 for s in scored if s.get(key) is not None]
        return round(sum(vals) / len(vals), 6) if vals else None
    def _mean_num(key: str) -> float | None:
        vals = [float(s[key]) for s in scored if s.get(key) is not None]
        return round(sum(vals) / len(vals), 6) if vals else None
    refs = sum(1 for r in results if r.get("reference"))
    return {
        "count": len(results),
        "non_empty_rate": round(sum(non_empty) / len(non_empty), 6),
        "mean_char_count": round(sum(lengths) / len(lengths), 3),
        "mean_token_count": round(sum(tokens) / len(tokens), 3),
        "reference_count": refs,
        "exact_match_rate": _mean_bool("exact_match"),
        "contains_reference_rate": _mean_bool("contains_reference"),
        "mean_token_overlap_f1": _mean_num("token_overlap_f1"),
    }
