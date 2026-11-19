"""Capability probe tests (v0.2 T7): optional acceleration kernels.

The probe records triton / liger / sageattention importability as booleans
on ``EnvReport.capabilities``. It must never raise, even when the import
machinery itself blows up, and must stay a plain import check (no module
execution).
"""

from __future__ import annotations

import importlib.util

from supertuner_engine import env as env_module
from supertuner_engine.config import EngineConfig


def _config(tmp_path, **overrides: str):
    base = {
        "SUPERTUNER_HOME": str(tmp_path / "home"),
        "SUPERTUNER_MODELS_DIR": str(tmp_path / "models"),
        "SUPERTUNER_DATASETS_DIR": str(tmp_path / "datasets"),
        "SUPERTUNER_CACHE_DIR": str(tmp_path / "cache"),
    }
    base.update(overrides)
    return env_module.EngineConfig.from_env(base)


def test_capability_probe_returns_bool_dict_for_fixed_keys():
    caps = env_module._probe_capabilities()
    assert caps == {
        "triton": bool(caps["triton"]),
        "liger": bool(caps["liger"]),
        "sageattention": bool(caps["sageattention"]),
    }
    assert set(caps) == {"triton", "liger", "sageattention"}
    for value in caps.values():
        assert isinstance(value, bool)


def test_capability_probe_never_raises_when_find_spec_fails(tmp_path, monkeypatch):
    def broken_find_spec(name, *args, **kwargs):
        raise RuntimeError("import machinery exploded")

    monkeypatch.setattr(env_module.importlib.util, "find_spec", broken_find_spec)
    caps = env_module._probe_capabilities()
    assert caps == {"triton": False, "liger": False, "sageattention": False}


def test_capability_probe_true_when_module_present(tmp_path, monkeypatch):
    def fake_find_spec(name, *args, **kwargs):
        # pretend every module under test is importable
        return importlib.util.find_spec(name) if name != "liger_kernel" else importlib.util.spec_from_loader("liger_kernel", loader=None)

    monkeypatch.setattr(env_module.importlib.util, "find_spec", fake_find_spec)
    caps = env_module._probe_capabilities()
    assert caps["liger"] is True
    # triton / sageattention follow whatever the real interpreter sees
    for key in ("triton", "sageattention"):
        assert isinstance(caps[key], bool)


def test_probe_report_carries_capabilities(tmp_path):
    report = env_module.probe(_config(tmp_path))
    assert isinstance(report.capabilities, dict)
    assert set(report.capabilities) == {"triton", "liger", "sageattention"}
    assert all(isinstance(v, bool) for v in report.capabilities.values())


def test_capabilities_field_serializes_with_default():
    """Existing EnvReport construction sites keep working (default {})."""
    from supertuner_engine.schema import EnvReport

    report = EnvReport(python="3.12", platform="test")
    assert report.capabilities == {}
    data = report.model_dump()
    assert data["capabilities"] == {}
    # and a full report round-trips
    report2 = EnvReport(
        python="3.12",
        platform="test",
        capabilities={"triton": True, "liger": False, "sageattention": True},
    )
    assert report2.capabilities["triton"] is True
    assert report2.capabilities["sageattention"] is True
