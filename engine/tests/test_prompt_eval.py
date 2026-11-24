"""Tests for v0.3 P0 frozen prompt-set scoring/loading (pure, no torch)."""

from __future__ import annotations

import json

import pytest

from supertuner_engine.backends.prompt_eval import (
    PromptSetError,
    aggregate_prompt_scores,
    load_prompt_set,
    score_completion,
)


def test_score_completion_with_reference():
    s = score_completion("7", "the answer is 7")
    assert s["non_empty"] is True
    assert s["exact_match"] is False
    assert s["contains_reference"] is True
    assert s["token_overlap_f1"] is not None and s["token_overlap_f1"] > 0
    assert score_completion("Paris", "  paris ")["exact_match"] is True


def test_score_completion_without_reference():
    s = score_completion(None, "hello world")
    assert s["non_empty"] is True
    assert s["exact_match"] is None
    assert s["contains_reference"] is None
    assert s["token_overlap_f1"] is None
    assert score_completion(None, "   ")["non_empty"] is False


def test_load_prompt_set_json_object(tmp_path):
    p = tmp_path / "set.json"
    p.write_text(
        json.dumps(
            {
                "version": 1,
                "name": "demo",
                "seed": 42,
                "max_new_tokens": 32,
                "prompts": [
                    {"id": "a", "prompt": "hi", "reference": "yo"},
                    "plain prompt",
                    {"messages": [{"role": "user", "content": "q"}]},
                ],
            }
        ),
        encoding="utf-8",
    )
    d = load_prompt_set(p)
    assert d["name"] == "demo"
    assert d["seed"] == 42
    assert d["max_new_tokens"] == 32
    assert len(d["prompts"]) == 3
    assert d["prompts"][0]["id"] == "a"
    assert d["prompts"][1]["prompt"] == "plain prompt"
    assert d["prompts"][2]["messages"][0]["content"] == "q"


def test_load_prompt_set_jsonl_and_bare_list(tmp_path):
    jl = tmp_path / "set.jsonl"
    jl.write_text('{"prompt":"a"}\n\n{"prompt":"b","reference":"b"}\n', encoding="utf-8")
    d = load_prompt_set(jl)
    assert [x["prompt"] for x in d["prompts"]] == ["a", "b"]
    bare = tmp_path / "bare.json"
    bare.write_text(json.dumps([{"prompt": "x"}]), encoding="utf-8")
    assert len(load_prompt_set(bare)["prompts"]) == 1


def test_load_prompt_set_errors(tmp_path):
    missing = tmp_path / "nope.json"
    with pytest.raises(PromptSetError, match="not found"):
        load_prompt_set(missing)
    bad = tmp_path / "bad.json"
    bad.write_text('{"foo": 1}', encoding="utf-8")
    with pytest.raises(PromptSetError):
        load_prompt_set(bad)
    empty = tmp_path / "empty.json"
    empty.write_text(json.dumps({"prompts": []}), encoding="utf-8")
    with pytest.raises(PromptSetError, match="empty"):
        load_prompt_set(empty)


def test_aggregate_prompt_scores():
    results = [
        {"reference": "a", "scores": {"non_empty": True, "char_count": 5, "token_count": 2, "exact_match": True, "contains_reference": True, "token_overlap_f1": 1.0}},
        {"reference": "b", "scores": {"non_empty": True, "char_count": 3, "token_count": 1, "exact_match": False, "contains_reference": False, "token_overlap_f1": 0.0}},
        {"reference": None, "scores": {"non_empty": False, "char_count": 0, "token_count": 0, "exact_match": None, "contains_reference": None, "token_overlap_f1": None}},
    ]
    agg = aggregate_prompt_scores(results)
    assert agg["count"] == 3
    assert agg["non_empty_rate"] == pytest.approx(2 / 3)
    assert agg["reference_count"] == 2
    assert agg["exact_match_rate"] == pytest.approx(0.5)
    assert agg["mean_token_overlap_f1"] == pytest.approx(0.5)


def test_frozen_llm_smoke_prompt_set():
    d = load_prompt_set("docs/development/eval/prompts/llm_smoke_v1.json")
    assert d["name"] == "llm_smoke_v1"
    assert d["version"] == 1
    assert len(d["prompts"]) == 6
    ids = [p["id"] for p in d["prompts"]]
    assert "gsm8k_like_01" in ids and "instruction_zh_01" in ids
