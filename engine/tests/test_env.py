"""Environment probe tests: never raise for missing optional dependencies."""

from __future__ import annotations

import sys

from supertuner_engine import env as env_module
from supertuner_engine.config import EngineConfig
from supertuner_engine.schema import EnvReport, GpuInfo

EXPECTED_PACKAGES = {
    "torch",
    "transformers",
    "peft",
    "accelerate",
    "bitsandbytes",
    "diffusers",
    "datasets",
}


def _config(tmp_path, **overrides: str) -> EngineConfig:
    base = {
        "SUPERTUNER_HOME": str(tmp_path / "home"),
        "SUPERTUNER_MODELS_DIR": str(tmp_path / "models"),
        "SUPERTUNER_DATASETS_DIR": str(tmp_path / "datasets"),
        "SUPERTUNER_CACHE_DIR": str(tmp_path / "cache"),
    }
    base.update(overrides)
    return EngineConfig.from_env(base)


def test_probe_returns_env_report(tmp_path):
    report = env_module.probe(_config(tmp_path))
    assert isinstance(report, EnvReport)
    assert report.python == f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    assert report.platform
    assert report.ram_mib > 0
    assert {pkg.name for pkg in report.packages} >= EXPECTED_PACKAGES


def test_probe_reports_resolved_paths(tmp_path):
    config = _config(tmp_path)
    report = env_module.probe(config)
    assert report.home == str(config.home)
    assert report.models_dir == str(config.models_dir)
    assert report.datasets_dir == str(config.datasets_dir)
    assert report.cache_dir == str(config.cache_dir)


def test_every_package_has_version_and_hint_fields(tmp_path):
    report = env_module.probe(_config(tmp_path))
    for package in report.packages:
        assert isinstance(package.ok, bool)
        assert isinstance(package.hint, str)
        assert package.version is None or isinstance(package.version, str)
        if package.ok:
            assert package.version is not None or package.name == "torch"


def test_missing_torch_does_not_raise(tmp_path, monkeypatch):
    """A machine without torch still gets a report, with torch marked unavailable."""
    monkeypatch.setattr(env_module, "_package_version", lambda name: None)

    def no_spec(name, *args, **kwargs):
        raise ImportError(f"no module named {name}")

    monkeypatch.setattr(env_module.importlib.util, "find_spec", no_spec)
    monkeypatch.setattr(env_module, "_probe_gpus", lambda: ([], ["torch 不可导入（ImportError）"]))
    monkeypatch.setattr(env_module, "detect_musubi", lambda config=None: None)

    report = env_module.probe(_config(tmp_path))
    torch_pkg = next(pkg for pkg in report.packages if pkg.name == "torch")
    assert torch_pkg.ok is False
    assert torch_pkg.hint
    assert any("torch" in warning for warning in report.warnings)
    assert report.gpus == []


def test_torch_failure_in_probe_gpus_is_reported(tmp_path, monkeypatch):
    class ExplodingTorch:
        def __getattr__(self, name):  # pragma: no cover - defensive
            raise AssertionError("should not be touched")

    def broken_import(name, *args, **kwargs):
        raise RuntimeError("CUDA driver mismatch")

    monkeypatch.setattr("builtins.__import__", broken_import)
    gpus, warnings = env_module._probe_gpus()
    assert gpus == []
    assert warnings and "CUDA driver mismatch" in warnings[0]


def test_detect_musubi_checkout(tmp_path):
    root = tmp_path / "musubi"
    root.mkdir()
    (root / "wan_train_network.py").write_text("print('x')\n", encoding="utf-8")
    report = env_module.probe(_config(tmp_path, SUPERTUNER_MUSUBI_PATH=str(root)))
    assert report.musubi == str(root)


def test_detect_musubi_missing_checkout_is_explained(tmp_path):
    missing = tmp_path / "nope"
    report = env_module.probe(_config(tmp_path, SUPERTUNER_MUSUBI_PATH=str(missing)))
    assert report.musubi is not None
    assert "路径不存在" in report.musubi
    assert any("musubi" in warning for warning in report.warnings)


def test_detect_musubi_empty_checkout_is_explained(tmp_path):
    root = tmp_path / "empty"
    root.mkdir()
    report = env_module.probe(_config(tmp_path, SUPERTUNER_MUSUBI_PATH=str(root)))
    assert report.musubi is not None
    assert "未找到" in report.musubi


def test_available_vram_uses_largest_gpu():
    report = EnvReport(
        python="3.12.0",
        platform="test",
        gpus=[
            GpuInfo(index=0, name="A", vram_mib=8192, driver="1"),
            GpuInfo(index=1, name="B", vram_mib=24576, driver="1"),
        ],
    )
    assert env_module.available_vram_mib(report) == 24576


def test_available_vram_without_gpus_is_zero():
    report = EnvReport(python="3.12.0", platform="test")
    assert env_module.available_vram_mib(report) == 0


def test_probe_torch_runtime_reports_cpu_without_cuda():
    runtime = env_module.probe_torch_runtime()
    assert "available" in runtime
    if runtime["available"]:
        assert runtime["device"] in {"cuda", "cpu"}
        assert isinstance(runtime["cuda"], bool)
    else:
        assert "error" in runtime


def test_invalid_log_level_is_a_warning_not_a_failure(tmp_path):
    config = _config(tmp_path, SUPERTUNER_ENGINE_LOG_LEVEL="shout")
    assert config.log_level == "info"
    assert any("SUPERTUNER_ENGINE_LOG_LEVEL" in warning for warning in config.warnings)
    report = env_module.probe(config)
    assert any("SUPERTUNER_ENGINE_LOG_LEVEL" in warning for warning in report.warnings)