"""Single-variable experiment matrix expansion (v0.3 P0).

Expands one base RunSpec into N RunSpecs by sweeping a single dotted field path.
No torch/network; deterministic and safe to unit-test.
"""

from __future__ import annotations

import copy
import json
import re
from pathlib import Path
from typing import Any

from .schema import RunSpec


class MatrixError(ValueError):
    """Invalid matrix definition."""


def _parse_path(path: str) -> list[str | int]:
    if not path or not isinstance(path, str):
        raise MatrixError("sweep.path must be a non-empty string")
    parts: list[str | int] = []
    for token in path.replace("[", ".").replace("]", "").split("."):
        if token == "":
            continue
        if token.isdigit():
            parts.append(int(token))
        else:
            parts.append(token)
    if not parts:
        raise MatrixError(f"invalid sweep.path: {path!r}")
    return parts


def _get_at(data: Any, parts: list[str | int]) -> Any:
    cur = data
    for part in parts:
        if isinstance(part, int):
            cur = cur[part]
        else:
            cur = cur[part]
    return cur


def _set_at(data: Any, parts: list[str | int], value: Any) -> None:
    cur = data
    for part in parts[:-1]:
        if isinstance(part, int):
            cur = cur[part]
        else:
            cur = cur[part]
    last = parts[-1]
    if isinstance(last, int):
        cur[last] = value
    else:
        cur[last] = value


def _slug(value: Any) -> str:
    text = json.dumps(value, ensure_ascii=False, separators=(",", ":")) if isinstance(value, (dict, list)) else str(value)
    text = text.strip().replace(" ", "_")
    text = re.sub(r"[^A-Za-z0-9._-]+", "-", text)
    text = re.sub(r"-+", "-", text).strip("-")
    return text[:80] or "value"


def load_base_spec(matrix: dict[str, Any], base_dir: Path) -> dict[str, Any]:
    base = matrix.get("base") if "base" in matrix else matrix.get("base_spec")
    if base is None:
        raise MatrixError("matrix requires 'base' (RunSpec path or inline object)")
    if isinstance(base, str):
        path = Path(base)
        if not path.is_absolute():
            path = (base_dir / path).resolve()
        return json.loads(path.read_text(encoding="utf-8"))
    if isinstance(base, dict):
        return copy.deepcopy(base)
    raise MatrixError("'base' must be a spec path string or an inline spec object")


def expand_matrix(matrix: dict[str, Any], *, base_dir: Path | None = None) -> list[dict[str, Any]]:
    """Expand a single-variable matrix into validated RunSpec dicts.

    Matrix format:
      {
        "name": "optional-matrix-name",
        "base": "spec.json" | { ...RunSpec... },
        "sweep": {"path": "config.lora_rank", "values": [8,16,32]},
        "name_prefix": "optional",
        "run_name_template": "{base_name}-{sweep_key}-{sweep_slug}",  # optional
        "output_dir": "runs/matrix",  # optional, relative to base_dir/cwd
        "eval_prompts": "docs/.../prompts.json"  # optional override applied to every variant
      }
    """
    base_dir = base_dir or Path.cwd()
    base_spec = load_base_spec(matrix, base_dir)
    sweep = matrix.get("sweep")
    if not isinstance(sweep, dict):
        raise MatrixError("matrix requires 'sweep': {path, values}")
    path = sweep.get("path")
    values = sweep.get("values")
    if not isinstance(values, list) or not values:
        raise MatrixError("sweep.values must be a non-empty list")
    parts = _parse_path(str(path))

    matrix_name = str(matrix.get("name") or matrix.get("id") or "matrix")
    name_prefix = str(matrix.get("name_prefix") or matrix_name)
    output_dir = matrix.get("output_dir")
    eval_prompts = matrix.get("eval_prompts")
    variants: list[dict[str, Any]] = []
    base_name = str((((base_spec.get("config") or {}).get("name")) or "run"))
    sweep_key = str(parts[-1])

    for idx, value in enumerate(values, start=1):
        spec = copy.deepcopy(base_spec)
        try:
            _set_at(spec, parts, copy.deepcopy(value))
        except Exception as exc:  # noqa: BLE001
            raise MatrixError(f"cannot set sweep.path {path!r} on base spec: {exc}") from exc

        slug = _slug(value)
        run_name = str(matrix.get("run_name_template") or "{prefix}-{key}-{idx}-{slug}").format(
            prefix=name_prefix,
            base_name=base_name,
            key=sweep_key,
            idx=idx,
            slug=slug,
            value=slug,
        )
        spec.setdefault("config", {})
        if isinstance(spec["config"], dict):
            spec["config"]["name"] = run_name
        if output_dir:
            out = Path(str(output_dir))
            if not out.is_absolute():
                out = (base_dir / out).resolve()
            spec.setdefault("output", {})
            if isinstance(spec["output"], dict):
                spec["output"]["dir"] = str(out / run_name)
        if eval_prompts:
            ep = Path(str(eval_prompts))
            if not ep.is_absolute():
                ep = (base_dir / ep).resolve()
            spec.setdefault("eval", {})
            if isinstance(spec["eval"], dict):
                spec["eval"]["prompts_path"] = str(ep)

        # validate
        RunSpec.model_validate(spec)
        variants.append(
            {
                "index": idx,
                "sweep_path": str(path),
                "sweep_value": value,
                "sweep_slug": slug,
                "run_name": run_name,
                "spec": spec,
            }
        )
    return variants


def write_expanded_matrix(
    matrix: dict[str, Any],
    *,
    matrix_path: Path,
    out_dir: Path,
) -> dict[str, Any]:
    base_dir = matrix_path.parent
    variants = expand_matrix(matrix, base_dir=base_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    specs_dir = out_dir / "specs"
    specs_dir.mkdir(parents=True, exist_ok=True)
    manifest_variants: list[dict[str, Any]] = []
    for variant in variants:
        spec_path = specs_dir / f"{variant['index']:02d}-{variant['sweep_slug']}.json"
        spec_path.write_text(json.dumps(variant["spec"], ensure_ascii=False, indent=2), encoding="utf-8")
        manifest_variants.append(
            {
                "index": variant["index"],
                "run_name": variant["run_name"],
                "sweep_path": variant["sweep_path"],
                "sweep_value": variant["sweep_value"],
                "spec_path": str(spec_path),
                "output_dir": ((variant["spec"].get("output") or {}).get("dir")),
            }
        )
    manifest = {
        "version": 1,
        "matrix_name": matrix.get("name") or matrix.get("id") or "matrix",
        "matrix_path": str(matrix_path),
        "count": len(manifest_variants),
        "sweep": matrix.get("sweep"),
        "variants": manifest_variants,
    }
    (out_dir / "matrix.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest
