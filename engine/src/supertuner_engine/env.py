"""Environment probe: what this machine can actually run.

`probe()` answers the questions the UI asks on its System page: GPU model and
VRAM, system RAM, ffmpeg, the training stack versions and whether musubi-tuner
is installed. It must never raise on a machine where torch is absent — planning
has to work before the user installs 2 GB of wheels.
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
import os
import platform as platform_module
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from .config import EngineConfig, cached_config
from .schema import EnvPackage, EnvReport, GpuInfo

#: Import name -> (display name, why it matters when missing).
_PACKAGES: tuple[tuple[str, str, str], ...] = (
    ("torch", "torch", "native LLM 训练与 GPU 枚举必需"),
    ("transformers", "transformers", "native LLM 后端必需"),
    ("peft", "peft", "LoRA/QLoRA 训练必需"),
    ("accelerate", "accelerate", "混合精度与显存分配必需"),
    ("bitsandbytes", "bitsandbytes", "QLoRA 与 8-bit 优化器必需（Windows 上经常不可用）"),
    ("diffusers", "diffusers", "部分扩散后端可选"),
    ("datasets", "datasets", "数据集加载（jsonl/parquet）必需"),
)

_MUSUBI_TRAIN_SCRIPTS = ("wan_train_network.py", "minimax_h3_train_network.py")


def _package_version(dist_name: str) -> str | None:
    try:
        return importlib.metadata.version(dist_name)
    except importlib.metadata.PackageNotFoundError:
        return None
    except Exception:  # noqa: BLE001 - broken metadata must not break the probe
        return None


def _package_available(module_name: str, dist_name: str) -> bool:
    if _package_version(dist_name) is not None:
        return True
    try:
        return importlib.util.find_spec(module_name) is not None
    except (ImportError, ValueError, ModuleNotFoundError):
        return False


def _probe_packages() -> tuple[list[EnvPackage], list[str]]:
    packages: list[EnvPackage] = []
    warnings: list[str] = []
    for module_name, dist_name, hint in _PACKAGES:
        version = _package_version(dist_name)
        available = version is not None or _package_available(module_name, dist_name)
        packages.append(
            EnvPackage(
                name=dist_name,
                version=version,
                ok=available,
                hint="" if available else hint,
            )
        )
        if not available and module_name == "torch":
            warnings.append("torch 未安装：无法枚举 GPU，也无法执行 native LLM 后端；`uv sync --project engine --extra torch`")
        elif not available and module_name == "bitsandbytes":
            warnings.append("bitsandbytes 未安装：QLoRA 与 AdamW-8bit 不可用，LoRA/全参训练不受影响")
    return packages, warnings


# 可选加速 kernel 探测（v0.2 T7）：键 -> import 名。只查可导入性，
# 从不执行模块体，Windows/headless 安全，缺失时永不抛错。
_CAPABILITY_MODULES: tuple[tuple[str, str], ...] = (
    ("triton", "triton"),
    ("liger", "liger_kernel"),
    ("sageattention", "sageattention"),
)


def _probe_capabilities() -> dict[str, bool]:
    """Detect optional acceleration kernels (import check only).

    Keys are fixed: ``triton`` / ``liger`` / ``sageattention``. A key is True
    only when its module is importable; anything else (missing module, a
    broken install that raises on find_spec) records False and never raises.
    """
    caps: dict[str, bool] = {}
    for key, module_name in _CAPABILITY_MODULES:
        try:
            caps[key] = importlib.util.find_spec(module_name) is not None
        except Exception:  # noqa: BLE001 - probe must never raise
            caps[key] = False
    return caps


def _nvidia_smi_driver() -> str:
    exe = shutil.which("nvidia-smi")
    if exe is None:
        return ""
    try:
        proc = subprocess.run(  # noqa: S603 - fixed argv, no shell
            [exe, "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=8,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    if proc.returncode != 0:
        return ""
    first = (proc.stdout or "").strip().splitlines()
    return first[0].strip() if first else ""


def _probe_gpus() -> tuple[list[GpuInfo], list[str]]:
    try:
        import torch  # noqa: PLC0415 - optional heavy dependency, import on demand
    except Exception as exc:  # noqa: BLE001 - torch can fail to load for many reasons
        return [], [f"torch 不可导入（{type(exc).__name__}: {exc}）；GPU 列表为空"]

    if not torch.cuda.is_available():
        backend = getattr(torch.version, "hip", None)
        runtime = "ROCm" if backend else "CPU"
        return [], [f"torch {torch.__version__} 可用但没有 CUDA 设备（runtime={runtime}）；训练将不可用或极慢"]

    driver = _nvidia_smi_driver()
    if not driver:
        cuda_version = getattr(torch.version, "cuda", None)
        driver = f"CUDA {cuda_version}" if cuda_version else "unknown"

    gpus: list[GpuInfo] = []
    for index in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(index)
        try:
            free_bytes, total_bytes = torch.cuda.mem_get_info(index)
        except Exception:  # noqa: BLE001 - mem_get_info can fail on MIG/virtualised GPUs
            free_bytes, total_bytes = 0, int(props.total_memory)
        gpus.append(
            GpuInfo(
                index=index,
                name=str(props.name),
                vram_mib=int(total_bytes // (1024 * 1024)),
                driver=driver,
                vram_total_mib=int(total_bytes // (1024 * 1024)),
                vram_free_mib=int(free_bytes // (1024 * 1024)),
            )
        )
    return gpus, []


def _probe_ram() -> tuple[int, list[str]]:
    try:
        import psutil  # noqa: PLC0415 - optional dependency

        return int(psutil.virtual_memory().total // (1024 * 1024)), []
    except Exception:  # noqa: BLE001 - fall through to the stdlib
        pass
    try:
        if hasattr(os, "sysconf") and "SC_PAGE_SIZE" in os.sysconf_names:
            pages = os.sysconf("SC_PHYS_PAGES")
            page_size = os.sysconf("SC_PAGE_SIZE")
            return int(pages * page_size // (1024 * 1024)), []
    except (OSError, ValueError):
        pass
    return 0, ["无法检测系统内存（psutil 未安装且平台不支持 sysconf）；block swap 的可行性判断会缺失 RAM 数据"]


def _probe_ffmpeg() -> tuple[str | None, list[str]]:
    exe = shutil.which("ffmpeg")
    if exe is None:
        return None, ["ffmpeg 未在 PATH 中：视频数据集的解码与帧抽取不可用"]
    try:
        proc = subprocess.run(  # noqa: S603 - fixed argv, no shell
            [exe, "-version"],
            capture_output=True,
            text=True,
            timeout=8,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return None, [f"ffmpeg 存在但执行失败（{type(exc).__name__}: {exc}）"]
    text = (proc.stdout or "") + (proc.stderr or "")
    match = re.search(r"ffmpeg version (\S+)", text)
    return (match.group(1) if match else "unknown"), []


def detect_musubi(config: EngineConfig | None = None) -> str | None:
    """Return a label for the musubi-tuner installation, or None.

    A checkout pointed at by `SUPERTUNER_MUSUBI_PATH` wins over an installed
    module because that is the path the musubi backend will actually use.
    """
    cfg = config or cached_config()
    if cfg.musubi_path is not None:
        root = Path(cfg.musubi_path)
        if root.is_dir():
            scripts = {p.name for p in root.rglob("*_train_network.py")}
            if scripts & set(_MUSUBI_TRAIN_SCRIPTS):
                return str(root)
            return f"{root}（未找到 wan_train_network.py / minimax_h3_train_network.py）"
        return f"{root}（路径不存在）"

    try:
        spec = importlib.util.find_spec("musubi_tuner")
    except (ImportError, ValueError, ModuleNotFoundError):
        return None
    if spec is None:
        return None
    version = _package_version("musubi-tuner")
    origin = spec.origin or ""
    return f"module:musubi_tuner {version or ''}".strip() if origin else "module:musubi_tuner"


def probe(config: EngineConfig | None = None) -> EnvReport:
    """Collect one `EnvReport`. Never raises for missing optional dependencies."""
    cfg = config or cached_config()

    packages, warnings = _probe_packages()
    gpus, gpu_warnings = _probe_gpus()
    warnings.extend(gpu_warnings)
    ram_mib, ram_warnings = _probe_ram()
    warnings.extend(ram_warnings)
    ffmpeg, ffmpeg_warnings = _probe_ffmpeg()
    warnings.extend(ffmpeg_warnings)
    warnings.extend(cfg.warnings)

    musubi = detect_musubi(cfg)
    if musubi is None:
        warnings.append(
            "musubi-tuner 未检测到：视频后端只能编译不能执行；设置 SUPERTUNER_MUSUBI_PATH "
            "指向含 wan_train_network.py 的仓库"
        )
    elif "未找到" in musubi or "路径不存在" in musubi:
        warnings.append(f"musubi-tuner 配置无效：{musubi}；视频后端只能编译不能执行")

    return EnvReport(
        python=f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        platform=platform_module.platform(),
        packages=packages,
        gpus=gpus,
        ram_mib=ram_mib,
        ffmpeg=ffmpeg,
        warnings=warnings,
        musubi=musubi,
        home=str(cfg.home),
        models_dir=str(cfg.models_dir),
        datasets_dir=str(cfg.datasets_dir),
        cache_dir=str(cfg.cache_dir),
        capabilities=_probe_capabilities(),
    )


def available_vram_mib(report: EnvReport | None = None) -> int:
    """Largest GPU's total VRAM in MiB (0 when no CUDA device is visible)."""
    env = report or probe()
    return max((gpu.vram_mib for gpu in env.gpus), default=0)


def probe_torch_runtime() -> dict[str, Any]:
    """Minimal torch capability map used by the native LLM backend at run time."""
    try:
        import torch  # noqa: PLC0415
    except Exception as exc:  # noqa: BLE001
        return {"available": False, "error": f"{type(exc).__name__}: {exc}"}
    cuda = bool(torch.cuda.is_available())
    bf16 = bool(cuda and getattr(torch.cuda, "is_bf16_supported", lambda: False)())
    return {
        "available": True,
        "version": torch.__version__,
        "cuda": cuda,
        "bf16": bf16,
        "device": "cuda" if cuda else "cpu",
    }