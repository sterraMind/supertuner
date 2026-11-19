"""musubi-tuner backend: video LoRA through an external, pinned CLI.

musubi-tuner is a separate project with its own venv and its own CLI; this
backend never re-implements it. `compile()` turns a `RunSpec` into the exact
argv arrays that musubi's scripts accept (wan_* or minimax_h3_*), plus one
in-process `prepare` step that writes the dataset TOML musubi requires.

The CLI is not a stable API (docs/ARCHITECTURE.md §13), so every flag lives in
this module only, and `--dry-run` prints the argv before anything runs. Missing
musubi is a refusal for MiniMax-H3 (the one family the brief pins to a hard
requirement) and a warning for Wan (the plan stays inspectable; execution fails).
"""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from ..config import Context
from ..events import Emit
from ..schema import (
    VIDEO_FAMILIES,
    Adapter,
    DatasetKind,
    ExecutionPlan,
    Issue,
    Optimizer,
    PlanStep,
    PluginManifest,
    QuantBase,
    RunSpec,
    Stage,
    parse_train_blocks,
)

BACKEND_ID = "musubi"
BACKEND_VERSION = "0.1.0"

#: Families musubi-tuner can actually train. Other video families have no backend.
MUSUBI_FAMILIES: tuple[str, ...] = (
    "wan2.1-t2v-1.3b",
    "wan2.1-t2v-14b",
    "minimax-h3",
)

#: family -> musubi `--task` value (docs/wan.md, docs/minimax_h3.md).
_TASK_BY_FAMILY: dict[str, str] = {
    "wan2.1-t2v-1.3b": "t2v-1.3B",
    "wan2.1-t2v-14b": "t2v-14B",
    "minimax-h3": "t2va",
}

#: Conventional filenames under SUPERTUNER_MODELS_DIR/<family>/ (all overridable
#: through config.model_path for the DiT; the companions are documented here).
_ROLE_FILES: dict[str, str] = {
    "dit": "dit.safetensors",
    "vae": "vae.safetensors",
    "t5": "t5.safetensors",
    "text_encoder": "text_encoder.safetensors",
    "audio_vae": "audio_vae.safetensors",
}

#: LoRA network module per family (musubi ships these names).
_NETWORK_MODULE: dict[str, str] = {
    "wan2.1-t2v-1.3b": "networks.lora_wan",
    "wan2.1-t2v-14b": "networks.lora_wan",
    "minimax-h3": "networks.lora_minimax_h3",
}

_OPTIMIZER_NAMES: dict[Optimizer, str] = {
    Optimizer.ADAMW: "AdamW",
    Optimizer.ADAMW_8BIT: "AdamW8bit",
    Optimizer.ADAMW_PAGED_8BIT: "PagedAdamW8bit",
}

#: MiniMax-H3 hard requirements from the brief: 24 GB, or a quantized base
#: (fp8/int8/nf4) plus deep block swap. NF4 and FP8 are smaller than INT8, so
#: any of them is a valid 8GB edge recipe.
H3_VRAM_FLOOR_MIB = 24 * 1024
H3_EDGE_BLOCKS_TO_SWAP = 40
#: Base quantizations that shrink the H3 DiT enough to fit a small card when
#: combined with deep block swap (i.e. everything except full-precision bf16).
H3_EDGE_QUANTS = (QuantBase.FP8, QuantBase.INT8, QuantBase.NF4, QuantBase.NVFP4)


class MusubiBackend:
    """Compile-only capable adapter for musubi-tuner video LoRA training."""

    manifest = PluginManifest(
        id=BACKEND_ID,
        name="musubi-tuner (video LoRA)",
        version=BACKEND_VERSION,
        kind="backend",
        description="通过 musubi-tuner CLI 训练 Wan2.1/2.2 与 MiniMax-H3 视频 LoRA",
        capabilities=["train.video.lora"],
        families=sorted(MUSUBI_FAMILIES),
        entry_point="supertuner_engine.backends.musubi:MusubiBackend",
        requires=["musubi-tuner", "torch"],
        license="Apache-2.0",
        homepage="https://github.com/kohya-ss/musubi-tuner",
        priority=90,
    )

    # 准备/导出阶段的预览（VAE decoder 采样）按可用显存自动降档：先降帧数再降分辨率。
    # 这些预设只用于 execution plan 与 estimated_mib，不修改 config。
    _PREVIEW_PRESETS: tuple[tuple[int, int, int], ...] = (
        (768, 640, 22),  # 默认
        (768, 640, 8),   # 帧数降档
        (512, 512, 8),   # 分辨率+帧数降档
    )

    # -- discovery ---------------------------------------------------------

    def probe(self, ctx: Context) -> list[Issue]:
        root = musubi_root(ctx)
        if root is None:
            return [
                Issue(
                    severity="warning",
                    code="W_MUSUBI_MISSING",
                    message="未检测到 musubi-tuner：视频计划可以编译，但不能执行",
                    hint=(
                        "git clone https://github.com/kohya-ss/musubi-tuner，"
                        "然后把 SUPERTUNER_MUSUBI_PATH 指向仓库根目录"
                    ),
                )
            ]
        return [
            Issue(
                severity="info",
                code="I_MUSUBI_FOUND",
                message=f"musubi-tuner 仓库：{root}",
            )
        ]

    def supports(self, spec: RunSpec) -> bool:
        return spec.config.family in MUSUBI_FAMILIES

    # -- compile -----------------------------------------------------------

    def compile(self, spec: RunSpec, ctx: Context) -> ExecutionPlan:
        cfg = spec.config
        issues: list[Issue] = []
        root = musubi_root(ctx)

        if cfg.family not in MUSUBI_FAMILIES:
            issues.append(
                Issue(
                    severity="error",
                    code="E_FAMILY_UNSUPPORTED",
                    message=f"musubi-tuner 不支持家族 '{cfg.family}'",
                    hint=f"musubi 覆盖：{', '.join(MUSUBI_FAMILIES)}；其他视频家族暂无可用后端",
                    field="config.family",
                )
            )
        if cfg.adapter == Adapter.NONE:
            issues.append(
                Issue(
                    severity="error",
                    code="E_VIDEO_FULL_UNSUPPORTED",
                    message="v0.1 的视频后端只训练 LoRA；全参微调未验证",
                    hint="把 adapter 设为 lora 或 qlora；全参视频训练需要多卡与更大显存",
                    field="config.adapter",
                )
            )
        if cfg.stage not in (Stage.SFT, Stage.PRETRAIN):
            issues.append(
                Issue(
                    severity="error",
                    code="E_STAGE_UNSUPPORTED",
                    message=f"musubi 后端不支持 stage={cfg.stage.value}",
                    hint="视频 LoRA 使用 stage=sft",
                    field="config.stage",
                )
            )
        if cfg.optimizer not in _OPTIMIZER_NAMES:
            issues.append(
                Issue(
                    severity="error",
                    code="E_OPTIMIZER_UNSUPPORTED",
                    message=f"musubi 后端不支持 optimizer={cfg.optimizer.value}",
                    hint=f"支持：{', '.join(o.value for o in _OPTIMIZER_NAMES)}",
                    field="config.optimizer",
                )
            )

        if root is None:
            severity = "error" if cfg.family == "minimax-h3" else "warning"
            issues.append(
                Issue(
                    severity=severity,
                    code="E_MUSUBI_MISSING" if severity == "error" else "W_MUSUBI_MISSING",
                    message="未找到 musubi-tuner（SUPERTUNER_MUSUBI_PATH 或已安装的 musubi_tuner 模块）",
                    hint=(
                        "git clone https://github.com/kohya-ss/musubi-tuner，"
                        "把 SUPERTUNER_MUSUBI_PATH 指向仓库根目录；"
                        "再用 SUPERTUNER_MUSUBI_PYTHON 指定装有 musubi 依赖的 python"
                    ),
                    field="config.family",
                )
            )

        if cfg.family == "minimax-h3":
            issues.extend(_h3_guard(cfg))
        else:
            issues.extend(_wan_geometry_guard(cfg))

        model_dir = ctx.model_path(cfg.family, None)
        assert model_dir is not None
        roles = _role_paths(cfg, model_dir)
        missing_roles = [f"{name}: {path}" for name, path in roles.items() if not path.exists()]
        if missing_roles:
            issues.append(
                Issue(
                    severity="warning",
                    code="W_MUSUBI_MODEL_PATHS",
                    message="以下权重文件不存在（训练启动时会失败）：" + "；".join(missing_roles),
                    hint=(
                        f"把权重放到 SUPERTUNER_MODELS_DIR/{cfg.family}/ 下的约定文件名，"
                        "或用 config.model_path 指向 DiT"
                    ),
                )
            )

        if cfg.cache.value == "none":
            issues.append(
                Issue(
                    severity="warning",
                    code="W_NO_CACHE",
                    message="cache=none：每次训练都会重新编码文本与 latents，显存与时间消耗更高",
                    hint="建议 cache=both（默认），磁盘换显存",
                    field="config.cache",
                )
            )

        if cfg.train_blocks is not None:
            mapped = _train_blocks_network_args(cfg.train_blocks)
            if mapped:
                issues.append(
                    Issue(
                        severity="info",
                        code="I_TRAIN_BLOCKS_MAPPED",
                        message="目标 block 范围已映射为 musubi --network_args 的 include/exclude_patterns",
                        hint=(
                            f"train_blocks={cfg.train_blocks} → 只在该 block 窗口内训练 LoRA，"
                            "并保留 musubi 默认的 img_mod/txt_mod/modulation 排除。"
                        ),
                        field="config.train_blocks",
                    )
                )
                notes_extra = [
                    f"train_blocks={cfg.train_blocks} 已映射为 --network_args include_patterns（只训该 block 窗口）"
                ]
            else:
                notes_extra = []
        else:
            notes_extra = []

        notes = _notes(spec, root)
        notes.extend(notes_extra)
        steps = self._build_steps(spec, ctx, roles, root, issues, notes)
        return ExecutionPlan(
            backend=BACKEND_ID,
            backend_version=BACKEND_VERSION,
            steps=steps,
            issues=issues,
            notes=notes,
            expected_artifacts=[str(_output_artifact(spec, ctx))],
        )

    # -- execution ---------------------------------------------------------

    def run(self, spec: RunSpec, plan: ExecutionPlan, ctx: Context, emit: Emit) -> None:
        for step in plan.steps:
            self.run_step(spec, step, ctx, emit)

    def run_step(self, spec: RunSpec, step: PlanStep, ctx: Context, emit: Emit) -> None:
        if step.kind == "prepare":
            if step.command is not None:
                raise RuntimeError("prepare 步骤应当是进程内步骤")
            dataset = spec.dataset
            if dataset is None:
                raise RuntimeError("musubi 训练需要 dataset")
            path = dataset_toml_path(spec, ctx)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(render_dataset_toml(spec, ctx), encoding="utf-8")
            emit.log("info", f"musubi 数据集配置已写入：{path}")
            return
        if step.id == "preview" and step.optional:
            emit.log(
                "info",
                f"预览步骤为记录项（{step.description}）；musubi 无稳定独立采样命令，跳过执行",
            )
            return
        raise RuntimeError(f"musubi 后端的 '{step.kind}' 步骤必须由 runtime 以 argv 执行")

    # -- preview / steps ---------------------------------------------------

    def _preview_step(self, spec: RunSpec, ctx: Context) -> list[PlanStep]:
        """Build an optional sampling/preview step, downgraded by budget if needed.

        The step is a record-only (command=None) export step: it carries the chosen
        preview geometry and an estimated VRAM peak for the VAE decoder, but the
        musubi CLI does not expose a stable standalone sampling command, so the
        runtime will not actually run it. Budget: vram_budget - overlap.
        """
        cfg = spec.config
        budget = cfg.vram_budget_mib - cfg.overlap_windows_mib
        fam = VIDEO_FAMILIES.get(cfg.family)
        if fam is None:
            return []
        default = self._PREVIEW_PRESETS[0]
        chosen = self._PREVIEW_PRESETS[-1]
        for preset in self._PREVIEW_PRESETS:
            w, h, f = preset
            est = _vae_decode_mib(fam, w, h, f)
            if est <= budget:
                chosen = preset
                break
            chosen = preset
        w, h, f = chosen
        est = _vae_decode_mib(fam, w, h, f)
        downgraded = chosen != default
        if downgraded:
            title = "采样/预览（自动降档）"
            desc = (
                f"默认 768x640x22 frames；预算 {budget:.0f} MiB 不足，"
                f"已降档到 {w}x{h}x{f} frames（先降帧数再降分辨率）"
            )
        else:
            title = "采样/预览"
            desc = f"{w}x{h}x{f} frames（VAE decoder，默认预览档位）"
        return [
            PlanStep(
                id="preview",
                kind="export",
                title=title,
                description=desc,
                command=None,
                optional=True,
                estimated_mib=est,
            )
        ]

    def _build_steps(
        self,
        spec: RunSpec,
        ctx: Context,
        roles: dict[str, Path],
        root: Path | None,
        issues: list[Issue],
        notes: list[str],
    ) -> list[PlanStep]:
        cfg = spec.config
        family = cfg.family
        python = musubi_python(root)
        pythonpath = _pythonpath(root)
        env = {"PYTHONPATH": pythonpath} if pythonpath else {}
        toml = dataset_toml_path(spec, ctx)
        is_h3 = family == "minimax-h3"
        h3 = is_h3

        steps: list[PlanStep] = [
            PlanStep(
                id="prepare",
                kind="prepare",
                title="写入 musubi 数据集配置",
                description=f"生成 {toml.name}（resolution={cfg.width}x{cfg.height}，frames={cfg.frames}）",
                command=None,
                cwd=str(ctx.run_dir) if ctx.run_dir is not None else None,
                env=env,
            )
        ]

        if cfg.cache.value in ("text", "both"):
            if h3:
                name = "minimax_h3_cache_text_encoder_outputs.py"
                args = [
                    "--dataset_config",
                    str(toml),
                    "--task",
                    _TASK_BY_FAMILY[family],
                    "--text_encoder",
                    str(roles["text_encoder"]),
                    "--text_cache_dtype",
                    "bf16",
                    "--skip_existing",
                ]
                if cfg.text_encoder_blocks_to_swap > 0:
                    args += ["--text_encoder_blocks_to_swap", str(cfg.text_encoder_blocks_to_swap)]
            else:
                name = "wan_cache_text_encoder_outputs.py"
                args = ["--dataset_config", str(toml), "--t5", str(roles["t5"])]
                if cfg.text_encoder_quant in (QuantBase.FP8, QuantBase.INT8):
                    args.append("--fp8_t5")
            steps.append(
                PlanStep(
                    id="cache-text",
                    kind="cache",
                    title="缓存文本编码器输出",
                    description=f"{name} --dataset_config …（每份数据集运行一次）",
                    command=_argv(root, python, name, args),
                    cwd=str(root) if root is not None else None,
                    env=env,
                    estimated_mib=_text_encoder_step_mib(cfg, VIDEO_FAMILIES[family]),
                )
            )

        if cfg.cache.value in ("latents", "both"):
            if h3:
                name = "minimax_h3_cache_latents.py"
                args = [
                    "--dataset_config",
                    str(toml),
                    "--task",
                    _TASK_BY_FAMILY[family],
                    "--video_vae",
                    str(roles["vae"]),
                    "--audio_vae",
                    str(roles["audio_vae"]),
                    "--cache_seed",
                    str(int(spec.train.seed)),
                    "--skip_existing",
                ]
            else:
                name = "wan_cache_latents.py"
                args = ["--dataset_config", str(toml), "--vae", str(roles["vae"])]
                if cfg.blocks_to_swap == 0 and cfg.seq_len == 0:  # pragma: no cover - never true
                    args.append("--vae_cache_cpu")
            steps.append(
                PlanStep(
                    id="cache-latents",
                    kind="cache",
                    title="缓存 VAE latents",
                    description=f"{name} --dataset_config …（磁盘换显存）",
                    command=_argv(root, python, name, args),
                    cwd=str(root) if root is not None else None,
                    env=env,
                    estimated_mib=_vae_step_mib(VIDEO_FAMILIES[family]),
                )
            )

        steps.append(_train_step(spec, ctx, roles, root, env))

        # 预览/采样步骤（可选、记录型）：按预算自动降档，estimated_mib 供计划展示。
        preview = self._preview_step(spec, ctx)
        if preview:
            steps.extend(preview)
            budget = cfg.vram_budget_mib - cfg.overlap_windows_mib
            est = preview[0].estimated_mib or 0
            if est > budget:
                issues.append(
                    Issue(
                        severity="warning",
                        code="W_PREVIEW_STEP_OOM",
                        message=(
                            f"预览步骤估算峰值约 {est:.0f} MiB，超过可用 {budget:.0f} MiB；"
                            "已降到最低档位仍不足"
                        ),
                        hint="该步骤为可选；训练可继续，但无法在此预算内完成 VAE 预览解码",
                        field="config.vram_budget_mib",
                    )
                )
            notes.append(
                f"预览步骤（可选）：{preview[0].description}，估算 {est:.0f} MiB。"
            )
        return steps


# ---------------------------------------------------------------------------
# discovery helpers
# ---------------------------------------------------------------------------


def _script_names() -> tuple[str, ...]:
    return (
        "wan_train_network.py",
        "wan_cache_latents.py",
        "wan_cache_text_encoder_outputs.py",
        "minimax_h3_train_network.py",
        "minimax_h3_cache_latents.py",
        "minimax_h3_cache_text_encoder_outputs.py",
    )


def _text_encoder_step_mib(cfg, fam) -> float:
    """VRAM the cache-text step needs while the encoder runs on the card."""
    from ..plan import _swap_resident_fraction

    bytes_per_param = {
        QuantBase.NONE: 2.0,
        QuantBase.INT8: 1.05,
        QuantBase.FP8: 1.0,
    }[cfg.text_encoder_quant]
    total = fam.text_encoder_params_b * 1e9 * bytes_per_param
    if cfg.text_encoder_blocks_to_swap > 0 and fam.text_encoder_layers > 0:
        resident, _, _ = _swap_resident_fraction(
            cfg.text_encoder_blocks_to_swap, fam.text_encoder_layers
        )
        total *= resident / fam.text_encoder_layers
    return total / (1024 * 1024)


def _vae_step_mib(fam) -> float:
    """VRAM the cache-latents step needs for the VAE decoder."""
    return fam.vae_params_b * 1e9 * 2 / (1024 * 1024)


def _vae_decode_mib(fam: Any, width: int, height: int, frames: int) -> float:
    """Estimate VRAM (MiB) to decode one preview clip with the VAE decoder.

    Uses the VAE weights in bf16 plus the decoded video latent/feature grid
    (latent_channels * t_lat * h_lat * w_lat, bf16). The decoder dominates; this
    is a planner-level estimate, not a runtime measurement.
    """
    base = _vae_step_mib(fam)
    t_lat = max(1, (frames - 1) // max(1, fam.vae_temporal_stride) + 1)
    h_lat = max(1, width // max(1, fam.vae_spatial_stride))
    w_lat = max(1, height // max(1, fam.vae_spatial_stride))
    latent_mib = fam.latent_channels * t_lat * h_lat * w_lat * 2 / (1024 * 1024)
    # decoder activations / working buffers: rough 4x latent grid in bf16
    work_mib = latent_mib * 4
    return base + latent_mib + work_mib


def musubi_root(ctx: Context) -> Path | None:
    """Locate a musubi-tuner checkout (SUPERTUNER_MUSUBI_PATH wins over a module)."""
    if ctx.musubi_path is not None:
        root = Path(ctx.musubi_path)
        return root if _has_scripts(root) else None
    try:
        spec = importlib.util.find_spec("musubi_tuner")
    except (ImportError, ValueError, ModuleNotFoundError):
        return None
    if spec is None or not spec.origin:
        return None
    package_dir = Path(spec.origin).parent
    for candidate in (package_dir.parent, package_dir.parent.parent, package_dir):
        if _has_scripts(candidate):
            return candidate
    return package_dir if _has_scripts(package_dir) else None


def _has_scripts(root: Path) -> bool:
    if not root.is_dir():
        return False
    return any(
        (root / name).is_file() or (root / "src" / "musubi_tuner" / name).is_file()
        for name in ("wan_train_network.py", "minimax_h3_train_network.py")
    )


def _script(root: Path, name: str) -> Path:
    for candidate in (root / name, root / "src" / "musubi_tuner" / name, root / "musubi_tuner" / name):
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"musubi 仓库中找不到 {name}（root={root}）")


def musubi_python(root: Path | None) -> str:
    """Interpreter used for musubi scripts.

    Priority: SUPERTUNER_MUSUBI_PYTHON, then the checkout's own `.venv`, then
    the engine interpreter. musubi has heavier pins than the engine, so its own
    venv is the normal setup.
    """
    override = (os.environ.get("SUPERTUNER_MUSUBI_PYTHON") or "").strip()
    if override:
        return override
    if root is not None:
        for relative in (".venv/Scripts/python.exe", ".venv/bin/python"):
            candidate = root / relative
            if candidate.is_file():
                return str(candidate)
    return sys.executable


def _argv(root: Path | None, python: str, script_name: str, args: list[str]) -> list[str]:
    """`python <script> args...`; a missing root still produces a displayable argv."""
    if root is None:
        return [python, script_name, *args]
    return [python, str(_script(root, script_name)), *args]


def _pythonpath(root: Path | None) -> str:
    if root is None:
        return ""
    source = root / "src"
    return str(source if source.is_dir() else root)


def nvidia_vram_mib() -> int:
    """Largest GPU's total VRAM in MiB via nvidia-smi; 0 when unknown."""
    exe = shutil.which("nvidia-smi")
    if exe is None:
        return 0
    try:
        proc = subprocess.run(  # noqa: S603 - fixed argv
            [exe, "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=8,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return 0
    if proc.returncode != 0:
        return 0
    values = []
    for line in (proc.stdout or "").splitlines():
        line = line.strip()
        if line.isdigit():
            values.append(int(line))
    return max(values, default=0)


# ---------------------------------------------------------------------------
# guards
# ---------------------------------------------------------------------------


def _h3_guard(cfg: Any) -> list[Issue]:
    vram = nvidia_vram_mib()
    if vram <= 0:
        return [
            Issue(
                severity="info",
                code="I_VRAM_UNKNOWN",
                message="无法通过 nvidia-smi 检测显存：跳过 H3 的 24GB 门槛检查",
                hint="运行前确认 GPU 与驱动可用",
            )
        ]
    edge_recipe = (
        cfg.quant_base in H3_EDGE_QUANTS and cfg.blocks_to_swap >= H3_EDGE_BLOCKS_TO_SWAP
    )
    if vram >= H3_VRAM_FLOOR_MIB or edge_recipe:
        return []
    return [
        Issue(
            severity="error",
            code="E_H3_VRAM_LOW",
            message=(
                f"MiniMax-H3 需要 ≥{H3_VRAM_FLOOR_MIB} MiB 显存，"
                f"或 fp8/int8/nf4/nvfp4 + blocks_to_swap≥{H3_EDGE_BLOCKS_TO_SWAP} 的小卡配方；"
                f"检测到 {vram} MiB"
            ),
            hint=(
                "设置 quant_base=int8（或 fp8/nf4/nvfp4）、blocks_to_swap=48（小卡边缘配方），"
                "或换 ≥24GB 显存 GPU"
            ),
            field="config.vram_budget_mib",
        )
    ]


def _wan_geometry_guard(cfg: Any) -> list[Issue]:
    issues: list[Issue] = []
    if cfg.width % 16 or cfg.height % 16:
        issues.append(
            Issue(
                severity="error",
                code="E_VIDEO_GEOMETRY",
                message=f"Wan 分辨率必须是 16 的倍数，收到 {cfg.width}x{cfg.height}",
                hint="常见：480x832、832x480、720x1280",
                field="config.width",
            )
        )
    if cfg.frames < 1 or (cfg.frames - 1) % 4 != 0:
        issues.append(
            Issue(
                severity="error",
                code="E_VIDEO_FRAMES",
                message=f"Wan 帧数必须满足 4n+1，收到 {cfg.frames}",
                hint="常见：33、49、81",
                field="config.frames",
            )
        )
    return issues


# ---------------------------------------------------------------------------
# plan construction
# ---------------------------------------------------------------------------


def _role_paths(cfg: Any, model_dir: Path) -> dict[str, Path]:
    dit = Path(cfg.model_path) if cfg.model_path else model_dir / _ROLE_FILES["dit"]
    return {
        "dit": dit,
        "vae": model_dir / _ROLE_FILES["vae"],
        "t5": model_dir / _ROLE_FILES["t5"],
        "text_encoder": model_dir / _ROLE_FILES["text_encoder"],
        "audio_vae": model_dir / _ROLE_FILES["audio_vae"],
    }


def output_name(spec: RunSpec) -> str:
    raw = spec.output.dir or spec.config.name or f"{spec.config.family}-lora"
    return Path(raw).name or f"{spec.config.family}-lora"


def _output_artifact(spec: RunSpec, ctx: Context) -> Path:
    return ctx.output_dir / f"{output_name(spec)}.safetensors"


def dataset_toml_path(spec: RunSpec, ctx: Context) -> Path:
    base = ctx.run_dir if ctx.run_dir is not None else ctx.cache_dir
    return base / "work" / "dataset.toml"


def _latents_cache_dir(spec: RunSpec, ctx: Context) -> Path:
    base = ctx.run_dir if ctx.run_dir is not None else ctx.cache_dir
    return base / "cache" / "latents"


def render_dataset_toml(spec: RunSpec, ctx: Context) -> str:
    """Render the musubi dataset TOML (docs/dataset_config.md)."""
    dataset = spec.dataset
    if dataset is None:
        raise RuntimeError("musubi 数据集配置需要 dataset")
    cfg = spec.config
    video_dir = Path(dataset.path)
    if not video_dir.is_absolute():
        video_dir = ctx.datasets_dir / video_dir
    cache_dir = _latents_cache_dir(spec, ctx)
    caption_extension = ".txt"
    if dataset.caption_column and dataset.caption_column.startswith("."):
        caption_extension = dataset.caption_column

    lines = [
        "[general]",
        f"resolution = [{int(cfg.width)}, {int(cfg.height)}]",
        f"caption_extension = {json.dumps(caption_extension)}",
        f"batch_size = {max(int(cfg.batch), 1)}",
        "enable_bucket = true",
        "bucket_no_upscale = false",
        "",
        "[[datasets]]",
        f"video_directory = {json.dumps(str(video_dir.resolve()))}",
        f"cache_directory = {json.dumps(str(cache_dir.resolve()))}",
        f"target_frames = [{int(cfg.frames)}]",
        'frame_extraction = "head"',
        "",
    ]
    return "\n".join(lines)


def _block_index_alternation(ranges: list[tuple[int, int]]) -> str:
    """Explicit `|`-joined alternation of every block index covered by `ranges`."""
    indices: set[int] = set()
    for start, end in ranges:
        indices.update(range(start, end + 1))
    return "|".join(str(i) for i in sorted(indices))


def _train_blocks_network_args(train_blocks: str | None) -> list[str]:
    """Map a `train_blocks` range to musubi `--network_args` patterns.

    musubi-tuner has no direct ``--train_blocks`` flag: LoRA target modules are
    selected with ``include_patterns`` / ``exclude_patterns`` (complete-match
    regexes over module names such as ``transformer_blocks.20.attn.to_q`` or
    ``double_blocks.0.img_mod.linear``). We restrict LoRA to the requested block
    window while preserving musubi's default exclusions (img_mod / txt_mod /
    modulation). Returns the argv fragments that follow ``--network_args``; an
    empty list means "no train_blocks to map".
    """
    if not train_blocks:
        return []
    try:
        ranges = parse_train_blocks(train_blocks)
    except ValueError:
        return []
    if not ranges:
        return []
    alternation = _block_index_alternation(ranges)
    include = f".*blocks\\.(?:{alternation})\\..*"
    return [
        "--network_args",
        f"include_patterns=[r'{include}']",
        "exclude_patterns=[r'.*img_mod.*', r'.*txt_mod.*', r'.*modulation.*']",
    ]


def _train_step(
    spec: RunSpec,
    ctx: Context,
    roles: dict[str, Path],
    root: Path | None,
    env: dict[str, str],
) -> PlanStep:
    cfg = spec.config
    family = cfg.family
    python = musubi_python(root)
    is_h3 = family == "minimax-h3"
    script = "minimax_h3_train_network.py" if is_h3 else "wan_train_network.py"

    args: list[str] = [
        "--dataset_config",
        str(dataset_toml_path(spec, ctx)),
        "--task",
        _TASK_BY_FAMILY[family],
        "--dit",
        str(roles["dit"]),
        "--network_module",
        _NETWORK_MODULE[family],
        "--network_dim",
        str(int(cfg.lora_rank)),
        "--network_alpha",
        str(int(cfg.lora_alpha)),
        "--output_dir",
        str(ctx.output_dir),
        "--output_name",
        output_name(spec),
        "--learning_rate",
        f"{spec.train.learning_rate:g}",
        "--max_train_steps",
        str(int(spec.train.max_steps)),
        "--optimizer_type",
        _OPTIMIZER_NAMES[cfg.optimizer],
        "--mixed_precision",
        "bf16" if cfg.precision.value != "fp32" else "no",
        "--sdpa",
    ]
    if cfg.grad_checkpointing:
        args.append("--gradient_checkpointing")
    if cfg.grad_accum > 1:
        args += ["--gradient_accumulation_steps", str(int(cfg.grad_accum))]
    if cfg.blocks_to_swap > 0:
        args += ["--blocks_to_swap", str(int(cfg.blocks_to_swap))]
    args.extend(_train_blocks_network_args(cfg.train_blocks))
    if cfg.quant_base == QuantBase.FP8:
        args.append("--fp8_base")
    if spec.train.save_every > 0:
        args += ["--save_every_n_steps", str(int(spec.train.save_every))]
    if spec.train.seed:
        args += ["--seed", str(int(spec.train.seed))]
    if spec.train.scheduler != "constant":
        args += ["--lr_scheduler", spec.train.scheduler]
        args += ["--lr_warmup_steps", str(int(spec.train.warmup_steps))]
    if spec.train.grad_clip > 0:
        args += ["--max_grad_norm", f"{spec.train.grad_clip:g}"]
    if spec.train.resume_from:
        args += ["--resume", str(spec.train.resume_from)]
    if cfg.text_encoder_quant in (QuantBase.FP8, QuantBase.INT8) and not is_h3:
        args.append("--fp8_t5")
    if not is_h3:
        args += ["--vae", str(roles["vae"]), "--t5", str(roles["t5"])]
    if is_h3:
        # H3 trains the video+audio pair, so it wants the two H3-specific VAE flags.
        args += [
            "--video_vae",
            str(roles["vae"]),
            "--audio_vae",
            str(roles["audio_vae"]),
        ]

    return PlanStep(
        id="train",
        kind="train",
        title=f"{family} LoRA 训练",
        description=(
            f"{script}：{spec.train.max_steps} steps，blocks_to_swap={cfg.blocks_to_swap}，"
            f"cache={cfg.cache.value}"
        ),
        command=_argv(root, python, script, args),
        cwd=str(root) if root is not None else None,
        env=env,
    )


def _notes(spec: RunSpec, root: Path | None) -> list[str]:
    notes = [
        "musubi-tuner 的 CLI 不是稳定 API：argv 只在本模块构造，升级 musubi 后先跑 --dry-run 对照。",
        "cache 与 train 步骤是外部子进程，事件从子进程 stdout 的 JSON 行转发。",
    ]
    if root is None:
        notes.append("未找到 musubi 仓库：argv 以脚本名占位，执行会明确失败而不是假装成功。")
    else:
        python = musubi_python(root)
        notes.append(f"musubi 解释器：{python}")
    if spec.config.family == "minimax-h3":
        notes.append("H3 训练需要先缓存视频/音频 VAE 与 Qwen3-VL 文本编码器输出。")
    return notes


def dataset_kind_ok(spec: RunSpec) -> bool:
    if spec.dataset is None:
        return False
    return spec.dataset.kind in (DatasetKind.AUTO, DatasetKind.VIDEO, DatasetKind.IMAGE)
