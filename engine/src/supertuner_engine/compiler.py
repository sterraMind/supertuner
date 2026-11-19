"""Compile a portable `RunSpec` into an inspectable `RunPlan`.

The compiler is the only place that decides *whether* a run is allowed to
exist: it validates the combination, runs the memory planner, selects a
backend, asks that backend for an execution plan, and merges every issue into
one list. The result is deterministic JSON the UI can render and the server can
persist — nothing here touches torch, disk or the network beyond reading the
backend manifests.

Exit semantics used by the CLI (docs/ARCHITECTURE.md §3, §6):

- a spec that compiles keeps `ok: true` even when `plan.feasible` is false, so
  the caller can show the memory lines and the reasons; the CLI exits 2.
- only an unreadable spec, an unknown backend id or "no backend at all" is a
  hard compile failure (`ok: false`).
"""

from __future__ import annotations

from collections.abc import Iterable

from .config import Context
from .plan import _resolve_budget, resolve_config
from .plugins import Registry, Selection, load_registry
from .schema import (
    LLM_FAMILIES,
    VIDEO_FAMILIES,
    Adapter,
    ExecutionPlan,
    Issue,
    MemoryLine,
    MemoryPlan,
    PlanStep,
    PluginManifest,
    QuantBase,
    RunConfig,
    RunPlan,
    RunSpec,
    Stage,
)

# Stable step order inside a plan; within one kind the backend's order wins.
STEP_KIND_ORDER: dict[str, int] = {"prepare": 0, "cache": 1, "train": 2, "export": 3}
_SEVERITY_RANK: dict[str, int] = {"error": 0, "warning": 1, "info": 2}


def issue(
    severity: str,
    code: str,
    message: str,
    *,
    hint: str = "",
    field: str | None = None,
) -> Issue:
    return Issue(severity=severity, code=code, message=message, hint=hint, field=field)


def refusal_memory(cfg: RunConfig, reason: str) -> MemoryPlan:
    """MemoryPlan used when planning could not run at all (unknown family etc.)."""
    return MemoryPlan(
        feasible=False,
        vram_budget_mib=cfg.vram_budget_mib,
        ram_budget_mib=cfg.ram_budget_mib,
        training_peak_mib=0,
        sampling_peak_mib=0,
        headline_mib=0,
        headroom_mib=-cfg.overlap_windows_mib,
        ram_peak_mib=0,
        ram_headroom_mib=cfg.ram_budget_mib,
        lines=[MemoryLine(label="未计算", mib=0, note=reason)],
        risks=[reason],
        tokens_per_step=0,
        notes=[],
    )


def _family_kind(family: str) -> str | None:
    if family in LLM_FAMILIES:
        return "llm"
    if family in VIDEO_FAMILIES:
        return "video"
    return None


def _known_families() -> str:
    return ", ".join(sorted([*LLM_FAMILIES, *VIDEO_FAMILIES]))


def validate_spec(spec: RunSpec, ctx: Context) -> list[Issue]:
    """Architecture-level checks that do not need a backend.

    Backend-specific refusals (quant kernel missing, musubi absent, …) belong to
    the backend's `compile()`; this function only rejects combinations that are
    invalid for every backend.
    """
    cfg = spec.config
    issues: list[Issue] = []

    kind = _family_kind(cfg.family)
    if kind is None:
        issues.append(
            issue(
                "error",
                "E_FAMILY_UNKNOWN",
                f"未知模型家族 '{cfg.family}'",
                hint=f"可用家族：{_known_families()}；或在 schema.py 的 LLM_FAMILIES/VIDEO_FAMILIES 中注册",
                field="config.family",
            )
        )
        return issues

    if cfg.stage == Stage.DISTILL:
        issues.append(
            issue(
                "error",
                "E_DISTILL_UNSUPPORTED",
                "v0.1 不执行 DMD/DMAD 蒸馏：需要教师/学生双模型与额外显存，单卡消费级方案未验证",
                hint="改用 stage=sft 训练学生模型；蒸馏支持计划在 v0.2（ADR #6）",
                field="config.stage",
            )
        )

    if kind == "video" and cfg.stage in (Stage.PREFERENCE, Stage.RL):
        issues.append(
            issue(
                "error",
                "E_STAGE_UNSUPPORTED",
                f"视频家族 '{cfg.family}' 不支持 stage={cfg.stage.value}",
                hint="视频只支持 sft/pretrain（LoRA）；偏好/强化学习目前仅面向 LLM",
                field="config.stage",
            )
        )

    if cfg.adapter == Adapter.QLORA and cfg.quant_base not in (QuantBase.NONE, QuantBase.NF4):
        issues.append(
            issue(
                "error",
                "E_QUANT_CONFLICT",
                f"adapter=qlora 需要 4-bit 基座，但 quant_base={cfg.quant_base.value}",
                hint="QLoRA 只能用 nf4（或不填由引擎补 nf4）；int8/fp8 请配 adapter=lora",
                field="config.quant_base",
            )
        )

    if cfg.adapter != Adapter.NONE and not cfg.lora_targets:
        issues.append(
            issue(
                "error",
                "E_LORA_TARGETS_EMPTY",
                "adapter=lora/qlora 但 lora_targets 为空，没有可训练的参数",
                hint="至少选择 q/k/v/o 之一，或 mlp（gate/up/down）",
                field="config.lora_targets",
            )
        )

    if cfg.adapter == Adapter.NONE and cfg.blocks_to_swap > 0:
        issues.append(
            issue(
                "error",
                "E_BLOCK_SWAP_FULL_FT",
                "block swap 只支持 LoRA/QLoRA：全参训练的优化器状态无法随块流式搬运",
                hint="改用 adapter=lora（musubi 路线），或关闭 blocks_to_swap 并减少 batch/分辨率",
                field="config.blocks_to_swap",
            )
        )

    if cfg.stage in (Stage.SFT, Stage.PRETRAIN, Stage.PREFERENCE, Stage.RL) and spec.dataset is None:
        issues.append(
            issue(
                "error",
                "E_DATASET_REQUIRED",
                f"stage={cfg.stage.value} 需要 dataset，但 RunSpec.dataset 为空",
                hint="提供 dataset.path 与 kind（text/chat/video），服务器负责把 ~ 展开为绝对路径",
                field="dataset",
            )
        )

    if spec.dataset is not None:
        kind_mismatch = _dataset_kind_mismatch(cfg, spec)
        if kind_mismatch:
            issues.append(kind_mismatch)

    if cfg.adapter == Adapter.QLORA:
        issues.append(
            issue(
                "info",
                "I_QLORA_NF4",
                "adapter=qlora：基座按 4-bit NF4 加载并冻结，LoRA 权重在 bf16 中训练",
                field="config.adapter",
            )
        )

    if kind == "video" and cfg.cache.value in ("text", "both") and cfg.text_encoder_quant != QuantBase.NONE:
        issues.append(
            issue(
                "info",
                "I_TE_QUANT_UNUSED",
                "文本编码器输出已缓存到磁盘，text_encoder_quant 不会影响显存",
                field="config.text_encoder_quant",
            )
        )

    if kind == "llm" and cfg.quant_base in (
        QuantBase.INT8,
        QuantBase.FP8,
        QuantBase.NVFP4,
    ):
        issues.append(
            issue(
                "error",
                "E_QUANT_UNSUPPORTED",
                f"LLM 家族不支持 quant_base={cfg.quant_base.value} 的权重存储",
                hint="LLM 只用 nf4（QLoRA）；int8/fp8/nvfp4 块交换是 musubi-tuner 的视频路线",
                field="config.quant_base",
            )
        )

    return issues


def _dataset_kind_mismatch(cfg: RunConfig, spec: RunSpec) -> Issue | None:
    from .schema import DatasetKind

    assert spec.dataset is not None
    declared = spec.dataset.kind
    if declared == DatasetKind.AUTO:
        return None
    expected: DatasetKind
    if cfg.family in LLM_FAMILIES:
        allowed = {DatasetKind.TEXT, DatasetKind.CHAT}
        if cfg.stage == Stage.PREFERENCE:
            allowed.add(DatasetKind.PREFERENCE)
        expected = (
            DatasetKind.PREFERENCE
            if declared == DatasetKind.PREFERENCE
            else (DatasetKind.CHAT if declared == DatasetKind.CHAT else DatasetKind.TEXT)
        )
    else:
        allowed = {DatasetKind.VIDEO, DatasetKind.IMAGE}
        expected = DatasetKind.VIDEO
    if declared not in allowed:
        return issue(
            "error",
            "E_DATASET_KIND_MISMATCH",
            f"dataset.kind={declared.value} 与 family '{cfg.family}'（{expected.value}）不匹配",
            hint=f"该家族接受 {', '.join(sorted(k.value for k in allowed))}",
            field="dataset.kind",
        )
    return None


def _merge_issues(*groups: Iterable[Issue]) -> list[Issue]:
    seen: set[tuple[str, str, str]] = set()
    merged: list[Issue] = []
    for group in groups:
        for item in group:
            key = (item.severity, item.code, item.message)
            if key in seen:
                continue
            seen.add(key)
            merged.append(item)
    merged.sort(key=lambda i: (_SEVERITY_RANK.get(i.severity, 9), i.code, i.message))
    return merged


def _order_steps(steps: Iterable[PlanStep]) -> list[PlanStep]:
    indexed = list(enumerate(steps))
    indexed.sort(key=lambda pair: (STEP_KIND_ORDER.get(pair[1].kind, 99), pair[0]))
    return [step for _, step in indexed]


def compile_spec(
    spec: RunSpec,
    ctx: Context,
    registry: Registry | None = None,
    measured_vram_mib: int | None = None,
    measured_ram_mib: int | None = None,
) -> RunPlan:
    """Compile `spec` against `ctx`. Never raises for user input problems.

    ``auto_budget`` is resolved first: when the planner picks a different
    ``quant_base``/``blocks_to_swap`` candidate, that effective config is what
    gets validated, selected and compiled (so it reaches the backend argv),
    while the caller's original spec is left untouched for persistence.

    ``measured_vram_mib`` / ``measured_ram_mib`` 是环境探测拿到的本机实测
    总 VRAM / RAM（MiB），与 ``spec.config.budget_source`` 一起决定计划按哪个
    预算上限来算——语义和 ``plan()`` 完全一致（v0.2 T2 实测进预算 / G3 =
    AP-01）。解析出的溯源信息盖在返回的 ``RunPlan.memory`` 上（``budget_source``
    / ``budget_measured_*``）；不传这两个参数时预算保持原值、盖戳为 "manual"，
    与改动前行为相同。
    """
    reg = registry if registry is not None else load_registry()
    cfg = spec.config
    auto_applied: list[Issue] = []
    effective_cfg = cfg
    auto_search_picked = False
    memory: MemoryPlan | None = None
    memory_error: str | None = None
    try:
        # 实测进预算（v0.2 T2 / G3 = AP-01）：语义与 plan() 完全一致——先按
        # budget_source 解析出规划实际使用的预算上限，再用解析后的预算重建配置
        # 传给 resolve_config；resolve 成功后给 MemoryPlan 盖溯源戳。
        resolved = _resolve_budget(cfg, measured_vram_mib, measured_ram_mib)
        budgeted_cfg = cfg.model_copy(
            update={
                "vram_budget_mib": resolved.vram_budget_mib,
                "ram_budget_mib": resolved.ram_budget_mib,
            }
        )
        # 跑后自校准（v0.2 T1）：把 server 注入的 calibration 只读标注传给规划器，
        # 让它把「已校准 N 次」写进 MemoryPlan.notes；calibration 本身不改预算。
        effective_cfg, memory = resolve_config(budgeted_cfg, spec.calibration)
        # 盖溯源戳：与 plan() 相同，feasible / refuse 路径都覆盖（stamping 只在
        # resolve 成功后执行，ValueError 分支维持 memory=None 现状）。
        memory.budget_source = resolved.budget_source
        memory.budget_measured_vram_mib = resolved.measured_vram_mib
        memory.budget_measured_ram_mib = resolved.measured_ram_mib
        # spec_for_compile 的 auto_budget 分支语义保持不变（判断「auto_budget
        # 搜索是否换了候选」）：resolve_config 没做搜索时原样返回收到的
        # budgeted_cfg，换了候选才返回新对象；规划失败（ValueError）时异常发生在
        # 这行之前，flag 保持 False，effective_cfg 已退回 cfg，不会误入该分支。
        auto_search_picked = effective_cfg is not budgeted_cfg
    except ValueError as exc:
        # Planning error is surfaced below as E_PLAN_FAILED / E_PARAMS_UNVERIFIED.
        effective_cfg = cfg
        memory = None
        memory_error = str(exc)

    spec_for_compile = spec
    if auto_search_picked:
        spec_for_compile = spec.model_copy(update={"config": effective_cfg})
        auto_applied.append(
            issue(
                "info",
                "I_AUTO_BUDGET_APPLIED",
                "auto_budget 已选中并应用 "
                f"quant_base={effective_cfg.quant_base.value}、"
                f"blocks_to_swap={effective_cfg.blocks_to_swap}"
                "（原请求 "
                f"quant_base={cfg.quant_base.value}、blocks_to_swap={cfg.blocks_to_swap}）",
                field="config.auto_budget",
            )
        )

    family_kind = _family_kind(effective_cfg.family)

    validation = validate_spec(spec_for_compile, ctx)
    blocking = [i for i in validation if i.severity == "error"]

    if family_kind is None or any(i.code == "E_FAMILY_UNKNOWN" for i in blocking):
        execution = ExecutionPlan(
            backend="none",
            steps=[],
            issues=[i for i in validation if i.severity == "error"],
            notes=["未选择 backend：模型家族未知"],
        )
        return RunPlan(
            feasible=False,
            memory=refusal_memory(cfg, f"unknown family '{cfg.family}'"),
            execution=execution,
            issues=_merge_issues(auto_applied, validation),
            estimated_minutes=None,
        )

    if family_kind == "video" and cfg.stage in (Stage.PREFERENCE, Stage.RL):
        execution = ExecutionPlan(
            backend="none",
            steps=[],
            issues=[i for i in validation if i.severity == "error"],
            notes=["未选择 backend：该家族不支持此 stage"],
        )
        return RunPlan(
            feasible=False,
            memory=refusal_memory(cfg, f"stage {cfg.stage.value} unsupported for video families"),
            execution=execution,
            issues=_merge_issues(auto_applied, validation),
            estimated_minutes=None,
        )

    memory_issues: list[Issue] = []
    if memory is None:
        message = memory_error or "plan failed"
        code = "E_PARAMS_UNVERIFIED" if "no verified parameter count" in message else "E_PLAN_FAILED"
        hint = (
            "该家族的参数量未核实；确认后通过 config.params_b_override 显式给出"
            if code == "E_PARAMS_UNVERIFIED"
            else "检查 family/width/height/frames/seq_len 等取值"
        )
        memory_issues.append(issue("error", code, message, hint=hint, field="config.family"))
        memory = refusal_memory(effective_cfg, message)

    if not memory.feasible:
        memory_issues.append(
            issue(
                "error",
                "E_MEMORY_BUDGET",
                "内存预算不足：" + "；".join(memory.risks[:3]),
                hint="按 risks 里的建议降低分辨率/序列长度、换 8-bit 优化器、开启缓存或 block swap",
                field="config.vram_budget_mib",
            )
        )
    for risk in memory.risks:
        memory_issues.append(issue("warning", "W_MEMORY_RISK", risk))
    for note in memory.notes:
        memory_issues.append(issue("info", "I_PLANNER_NOTE", note))

    selection = reg.select(spec_for_compile)
    backend_issues: list[Issue] = []
    execution: ExecutionPlan

    if selection.backend is None:
        backend_issues.append(
            issue(
                "error",
                "E_NO_BACKEND",
                "没有 backend 能执行该组合",
                hint=" | ".join(selection.reasons) or "未发现任何 backend 插件",
            )
        )
        execution = ExecutionPlan(
            backend="none",
            steps=[],
            issues=backend_issues,
            notes=[f"required capability: {selection.capability}"],
        )
    else:
        backend = selection.backend
        manifest = backend.manifest
        if selection.reasons:
            backend_issues.append(
                issue(
                    "error",
                    "E_BACKEND_MISMATCH",
                    f"backend '{manifest.id}' 不能执行该组合",
                    hint=" | ".join(selection.reasons),
                )
            )

        try:
            probe_issues = list(backend.probe(ctx))
        except Exception as exc:  # noqa: BLE001 - a broken plugin must not kill compilation
            probe_issues = [
                issue(
                    "warning",
                    "W_BACKEND_PROBE_FAILED",
                    f"backend '{manifest.id}' 的 probe() 失败：{type(exc).__name__}: {exc}",
                )
            ]
        backend_issues.extend(probe_issues)

        try:
            execution = backend.compile(spec_for_compile, ctx)
        except Exception as exc:  # noqa: BLE001 - plugin code is untrusted
            backend_issues.append(
                issue(
                    "error",
                    "E_BACKEND_COMPILE_FAILED",
                    f"backend '{manifest.id}' 编译失败：{type(exc).__name__}: {exc}",
                )
            )
            execution = ExecutionPlan(backend=manifest.id, backend_version=manifest.version, steps=[])

        if execution.backend in ("", "auto", "none"):
            execution = execution.model_copy(update={"backend": manifest.id})
        if not execution.backend_version:
            execution = execution.model_copy(update={"backend_version": manifest.version})
        backend_issues.extend(execution.issues)

    execution = execution.model_copy(update={"steps": _order_steps(execution.steps)})

    # Every step has to fit, not just the training step: H3's cache-text step
    # loads the 32B text encoder even when the training peak is small.
    step_issues: list[Issue] = []
    step_budget = effective_cfg.vram_budget_mib - effective_cfg.overlap_windows_mib
    for step in execution.steps:
        if step.estimated_mib is None:
            continue
        if step.estimated_mib > step_budget:
            if step.optional:
                step_issues.append(
                    issue(
                        "warning",
                        "W_STEP_OOM",
                        f"可选步骤 '{step.title}' 需要约 {step.estimated_mib:.0f} MiB，"
                        f"超过可用 {step_budget:.0f} MiB",
                        hint="该步骤可跳过；降低该步骤的分辨率/帧数或换更大显存的卡",
                        field="config.vram_budget_mib",
                    )
                )
            else:
                step_issues.append(
                    issue(
                        "error",
                        "E_STEP_OOM",
                        f"步骤 '{step.title}' 需要约 {step.estimated_mib:.0f} MiB，"
                        f"超过可用 {step_budget:.0f} MiB",
                        hint="缓存文本编码器/VAE 需要额外显存；换更大显存的卡，或先把缓存产物准备好再训练",
                        field="config.vram_budget_mib",
                    )
                )

    all_issues = _merge_issues(auto_applied, validation, memory_issues, backend_issues, step_issues)
    has_error = any(i.severity == "error" for i in all_issues)
    feasible = bool(memory.feasible) and not has_error and bool(execution.steps)

    return RunPlan(
        feasible=feasible,
        memory=memory,
        execution=execution,
        issues=all_issues,
        estimated_minutes=None,
    )


def backend_manifest(plan: RunPlan, registry: Registry) -> PluginManifest | None:
    """Manifest of the backend a compiled plan points at (None when unknown)."""
    plugin = registry.backend(plan.execution.backend)
    return None if plugin is None else plugin.manifest


def selection_for(spec: RunSpec, registry: Registry) -> Selection:
    """Expose the raw selection for diagnostics commands."""
    return registry.select(spec)