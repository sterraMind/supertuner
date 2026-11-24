"""Tests for v0.3 P0 single-variable experiment matrix expansion (pure)."""

from __future__ import annotations

import json

import pytest

from supertuner_engine.experiment import MatrixError, expand_matrix, write_expanded_matrix


def _base_spec():
    return {
        "schema_version": 1,
        "config": {"name": "base", "family": "qwen3-0.6b", "stage": "sft", "adapter": "lora", "lora_rank": 8},
        "dataset": {"path": "data.jsonl", "kind": "text"},
        "train": {"max_steps": 10, "eval_every": 5},
        "output": {"dir": "out/base"},
    }


def test_expand_matrix_single_variable_sweep():
    matrix = {
        "name": "rank-sweep",
        "base": _base_spec(),
        "sweep": {"path": "config.lora_rank", "values": [8, 16, 32]},
    }
    variants = expand_matrix(matrix)
    assert len(variants) == 3
    ranks = [v["spec"]["config"]["lora_rank"] for v in variants]
    assert ranks == [8, 16, 32]
    assert [v["run_name"] for v in variants] == [
        "rank-sweep-lora_rank-1-8",
        "rank-sweep-lora_rank-2-16",
        "rank-sweep-lora_rank-3-32",
    ]
    assert variants[0]["sweep_path"] == "config.lora_rank"
    assert variants[1]["sweep_value"] == 16


def test_expand_matrix_output_dir_and_eval_prompts(tmp_path):
    prompts = tmp_path / "prompts.json"
    prompts.write_text(json.dumps({"prompts": [{"prompt": "hi"}]}), encoding="utf-8")
    matrix = {
        "name": "m",
        "base": _base_spec(),
        "sweep": {"path": "train.max_steps", "values": [5, 10]},
        "output_dir": str(tmp_path / "outs"),
        "eval_prompts": str(prompts),
    }
    variants = expand_matrix(matrix, base_dir=tmp_path)
    assert variants[0]["spec"]["output"]["dir"].endswith("m-max_steps-1-5")
    assert variants[1]["spec"]["output"]["dir"].endswith("m-max_steps-2-10")
    assert variants[0]["spec"]["eval"]["prompts_path"] == str(prompts.resolve())


def test_expand_matrix_loads_base_from_file_and_writes_manifest(tmp_path):
    base = tmp_path / "base.json"
    base.write_text(json.dumps(_base_spec()), encoding="utf-8")
    matrix_path = tmp_path / "matrix.json"
    matrix = {
        "name": "from-file",
        "base": "base.json",
        "sweep": {"path": "config.name", "values": ["a", "b"]},
    }
    matrix_path.write_text(json.dumps(matrix), encoding="utf-8")
    manifest = write_expanded_matrix(matrix, matrix_path=matrix_path, out_dir=tmp_path / "expanded")
    assert manifest["count"] == 2
    assert (tmp_path / "expanded" / "matrix.json").exists()
    specs = list((tmp_path / "expanded" / "specs").glob("*.json"))
    assert len(specs) == 2
    spec0 = json.loads(specs[0].read_text(encoding="utf-8"))
    assert spec0["config"]["name"] in {"from-file-name-1-a", "from-file-name-2-b"}


def test_expand_matrix_validates_specs_and_errors():
    bad = {"base": _base_spec(), "sweep": {"path": "config.stage", "values": ["not_a_stage"]}}
    with pytest.raises(Exception):
        expand_matrix(bad)
    with pytest.raises(MatrixError, match="sweep.values"):
        expand_matrix({"base": _base_spec(), "sweep": {"path": "config.name", "values": []}})
    with pytest.raises(MatrixError, match="base"):
        expand_matrix({"sweep": {"path": "config.name", "values": ["x"]}})
