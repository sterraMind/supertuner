"""Frozen prompt-set evaluation for LLM runs (v0.3 P0).

Runs after training while model/tokenizer are still loaded. Uses greedy
generation with fixed seed for comparability. Metrics are lightweight and
dependency-free (no CLIP/LPIPS/external judge models).
"""

from __future__ import annotations

import time
from typing import Any

from .prompt_eval import PromptSetError, aggregate_prompt_scores, load_prompt_set, score_completion


def _format_prompt(tokenizer: Any, entry: dict[str, Any]) -> str | None:
    messages = entry.get("messages")
    if isinstance(messages, list) and messages:
        try:
            if hasattr(tokenizer, "apply_chat_template"):
                return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            return "\n".join(str(m.get("content", "")) for m in messages if isinstance(m, dict))
    prompt = entry.get("prompt")
    return prompt if isinstance(prompt, str) else None


def run_llm_prompt_eval(
    *,
    model: Any,
    tokenizer: Any,
    spec: Any,
    ctx: Any,
    emit: Any,
    device: str,
    amp_dtype: Any = None,
) -> dict[str, Any]:
    eval_cfg = getattr(spec, "eval", None)
    prompts_path = getattr(eval_cfg, "prompts_path", None) if eval_cfg is not None else None
    if not prompts_path:
        return {"status": "skipped", "reason": "no_prompts_path"}

    try:
        prompt_set = load_prompt_set(prompts_path)
    except PromptSetError as exc:
        emit.log("warn", f"prompt eval: 无法加载冻结 prompt 集：{exc}")
        return {"status": "unavailable", "reason": "prompt_set_invalid", "error": str(exc), "path": str(prompts_path)}
    except Exception as exc:  # noqa: BLE001
        emit.log("warn", f"prompt eval: 读取 prompt 集失败：{exc}")
        return {"status": "unavailable", "reason": "prompt_set_error", "error": str(exc), "path": str(prompts_path)}

    try:
        import torch  # noqa: PLC0415
    except Exception as exc:  # noqa: BLE001
        return {"status": "unavailable", "reason": "torch_unavailable", "error": str(exc)}

    prompts = list(prompt_set.get("prompts") or [])
    default_max_new = int(prompt_set.get("max_new_tokens") or 64)
    seed = prompt_set.get("seed")
    if seed is not None:
        try:
            torch.manual_seed(int(seed))
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(int(seed))
        except Exception:
            pass

    was_training = getattr(model, "training", False)
    model.eval()
    results: list[dict[str, Any]] = []
    emit.log("info", f"prompt eval: 运行冻结 prompt 集 {prompt_set.get('name')}（{len(prompts)} 条）")
    t0 = time.monotonic()
    for entry in prompts:
        text = _format_prompt(tokenizer, entry)
        if not text:
            results.append({
                "id": entry.get("id"),
                "status": "skipped",
                "reason": "empty_prompt",
                "reference": entry.get("reference"),
                "scores": score_completion(entry.get("reference"), ""),
            })
            continue
        max_new = int(entry.get("max_new_tokens") or default_max_new)
        max_new = max(1, min(max_new, 512))
        try:
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            input_len = int(inputs["input_ids"].shape[-1])
            gen_kwargs: dict[str, Any] = {
                "max_new_tokens": max_new,
                "do_sample": False,
                "pad_token_id": getattr(tokenizer, "pad_token_id", None) or getattr(tokenizer, "eos_token_id", None),
                "eos_token_id": getattr(tokenizer, "eos_token_id", None),
            }
            with torch.inference_mode():
                if amp_dtype is not None and device == "cuda":
                    with torch.autocast(device_type="cuda", dtype=amp_dtype):
                        out = model.generate(**inputs, **gen_kwargs)
                else:
                    out = model.generate(**inputs, **gen_kwargs)
            generated_ids = out[0][input_len:]
            completion = tokenizer.decode(generated_ids, skip_special_tokens=True)
            scores = score_completion(entry.get("reference"), completion)
            results.append({
                "id": entry.get("id"),
                "status": "ok",
                "prompt": text,
                "reference": entry.get("reference"),
                "completion": completion,
                "generated_tokens": int(generated_ids.shape[-1]),
                "scores": scores,
                "tags": entry.get("tags") or [],
            })
        except Exception as exc:  # noqa: BLE001
            emit.log("warn", f"prompt eval: {entry.get('id')} 失败：{exc}")
            results.append({
                "id": entry.get("id"),
                "status": "error",
                "reason": str(exc),
                "reference": entry.get("reference"),
                "scores": score_completion(entry.get("reference"), ""),
            })
        if device == "cuda":
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

    if was_training:
        try:
            model.train()
        except Exception:
            pass
    elapsed = time.monotonic() - t0
    aggregate = aggregate_prompt_scores(results)
    ok_count = sum(1 for r in results if r.get("status") == "ok")
    status = "ok" if ok_count == len(results) else ("partial" if ok_count > 0 else "unavailable")
    report = {
        "status": status,
        "backend": "llm_prompt_eval",
        "prompt_set": {
            "name": prompt_set.get("name"),
            "version": prompt_set.get("version"),
            "path": prompt_set.get("path"),
            "seed": prompt_set.get("seed"),
            "max_new_tokens_default": default_max_new,
        },
        "count": len(results),
        "ok_count": ok_count,
        "elapsed_s": round(elapsed, 3),
        "aggregate": aggregate,
        "samples": results,
        "notes": [
            "heuristic scoring only (exact/contains/token-overlap, non-empty rate); no external judge/CLIP/LPIPS",
            "generation uses greedy decode with fixed seed for comparability",
        ],
    }
    emit.log("info", f"prompt eval 完成：{ok_count}/{len(results)} ok，{elapsed:.1f}s")
    return report
