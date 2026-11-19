"""Plugin registry and the `Backend` protocol.

Plugin objects are Python classes registered through entry points in the group
`supertuner.plugins` plus the built-ins compiled into this package. A plugin
either implements the `Backend` protocol or is manifest-only (model, dataset,
strategy, optimizer, exporter, ui); the UI only ever sees manifests.

Discovery never fails a command: import errors, bad entry points and id
conflicts are collected into `{plugins, errors}` (docs/ARCHITECTURE.md §5).
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from importlib import metadata as importlib_metadata
from typing import Any, Protocol, runtime_checkable

from .config import Context
from .events import Emit
from .schema import (
    VIDEO_FAMILIES,
    Adapter,
    ExecutionPlan,
    Issue,
    PlanStep,
    PluginManifest,
    RunSpec,
    Stage,
)

#: Entry-point group scanned for third-party plugins.
PLUGIN_GROUP = "supertuner.plugins"

#: Capability required per (stage, family kind, adapter). Mirrors
#: `requiredCapabilities` in packages/core/src/plugins.ts so the UI and the
#: engine disable the same combinations.
def required_capability(stage: Stage, kind: str, adapter: Adapter) -> str:
    if stage == Stage.DISTILL:
        return "train.distill.dmd"
    if stage == Stage.RL:
        return "train.llm.rl"
    if stage == Stage.PREFERENCE:
        return "train.llm.preference"
    if stage == Stage.PRETRAIN:
        return "train.llm.pretrain" if kind == "llm" else "train.video.full"
    if kind == "llm":
        return "train.llm.sft"
    return "train.video.full" if adapter == Adapter.NONE else "train.video.lora"


@runtime_checkable
class Backend(Protocol):
    """A compilable/executable training backend.

    `compile` and `probe` must work without torch installed: the UI plans on
    machines that may not have the training stack. Import heavy modules inside
    `run_step`/`run`.
    """

    manifest: PluginManifest

    def probe(self, ctx: Context) -> list[Issue]: ...

    def supports(self, spec: RunSpec) -> bool: ...

    def compile(self, spec: RunSpec, ctx: Context) -> ExecutionPlan: ...

    def run(self, spec: RunSpec, plan: ExecutionPlan, ctx: Context, emit: Emit) -> None: ...

    def run_step(self, spec: RunSpec, step: PlanStep, ctx: Context, emit: Emit) -> None: ...


@dataclass
class PluginError:
    """A plugin that could not be loaded. Reported, never fatal."""

    entry_point: str
    reason: str

    def model_dump(self) -> dict[str, str]:
        return {"entry_point": self.entry_point, "reason": self.reason}


@dataclass(frozen=True)
class Selection:
    """Outcome of backend selection.

    `backend` is None when nothing matched, and `reasons` then explains every
    rejected candidate so the compiler can turn it into an actionable Issue.
    When an explicitly requested backend exists but does not fit, it stays in
    `backend` and `reasons` carries the mismatch (the compiler errors on it).
    """

    capability: str
    backend: Backend | None = None
    reasons: list[str] = field(default_factory=list)


@dataclass
class Registry:
    """Loaded plugins, sorted by id. Backends are the executable subset."""

    plugins: dict[str, Any] = field(default_factory=dict)
    errors: list[PluginError] = field(default_factory=list)

    # -- registration ------------------------------------------------------

    def register(self, plugin: Any, source: str = "") -> None:
        manifest = _manifest_of(plugin)
        if manifest is None:
            raise TypeError(f"{plugin!r} does not expose a PluginManifest as `.manifest`")
        existing = self.plugins.get(manifest.id)
        if existing is not None:
            other = _manifest_of(existing)
            version = other.version if other else "?"
            if version != manifest.version:
                raise ValueError(f"plugin id conflict: '{manifest.id}' {version} vs {manifest.version} ({source})")
            return
        self.plugins[manifest.id] = plugin

    def add_error(self, entry_point: str, reason: str) -> None:
        self.errors.append(PluginError(entry_point=entry_point, reason=reason))

    # -- queries -----------------------------------------------------------

    def enabled(self) -> list[Any]:
        return [p for p in self._sorted() if _manifest_of(p).enabled]

    def manifests(self) -> list[PluginManifest]:
        return [m for m in (_manifest_of(p) for p in self._sorted()) if m is not None]

    def backends(self) -> list[Any]:
        out: list[Any] = []
        for plugin in self.enabled():
            manifest = _manifest_of(plugin)
            if manifest is None or manifest.kind != "backend":
                continue
            if not _looks_like_backend(plugin):
                continue
            out.append(plugin)
        return out

    def get(self, plugin_id: str) -> Any | None:
        return self.plugins.get(plugin_id)

    def backend(self, backend_id: str) -> Any | None:
        plugin = self.plugins.get(backend_id)
        if plugin is None or not _looks_like_backend(plugin):
            return None
        return plugin

    def supports(self, capability: str) -> bool:
        return any(capability in m.capabilities for m in (m for m in self.manifests() if m.enabled))

    def families_for(self, capability: str) -> list[str]:
        out: set[str] = set()
        for manifest in (m for m in self.manifests() if m.enabled and capability in m.capabilities):
            out.update(manifest.families)
        return sorted(out)

    def to_response(self) -> dict[str, Any]:
        return {
            "plugins": [manifest.model_dump() for manifest in self.manifests()],
            "errors": [err.model_dump() for err in self.errors],
        }

    # -- selection ---------------------------------------------------------

    def select(self, spec: RunSpec) -> Selection:
        """Pick the highest-priority backend for `spec`.

        Priority is the manifest's `priority` value: **higher wins**, ties are
        broken alphabetically by id so selection is deterministic.
        """
        family = spec.config.family
        kind = _family_kind(family)
        capability = required_capability(spec.config.stage, kind, spec.config.adapter)
        reasons: list[str] = []

        if spec.backend:
            requested = self.backend(spec.backend)
            if requested is None:
                known = ", ".join(sorted(m.id for m in self.manifests() if m.kind == "backend")) or "（无）"
                reasons.append(f"请求的 backend '{spec.backend}' 不存在或不可用；已注册 backend：{known}")
                return Selection(capability=capability, backend=None, reasons=reasons)
            mismatch = _mismatch(requested, capability, family)
            if mismatch:
                reasons.append(f"backend '{spec.backend}' {mismatch}")
            return Selection(capability=capability, backend=requested, reasons=reasons)

        candidates: list[Any] = []
        for plugin in self.plugins.values():
            manifest = _manifest_of(plugin)
            if manifest is None or manifest.kind != "backend":
                continue
            if not manifest.enabled:
                reasons.append(f"backend '{manifest.id}' 已禁用（enabled=false）")
                continue
            if not _looks_like_backend(plugin):
                reasons.append(f"backend '{manifest.id}' 未实现 compile/supports/run（插件不完整）")
                continue
            mismatch = _mismatch(plugin, capability, family)
            if mismatch:
                reasons.append(f"backend '{manifest.id}' {mismatch}")
                continue
            candidates.append(plugin)

        if candidates:
            best = sorted(candidates, key=lambda p: (-_manifest_of(p).priority, _manifest_of(p).id))[0]
            return Selection(capability=capability, backend=best, reasons=[])

        reasons.append(
            f"没有已启用的 backend 声明 capability '{capability}' 且覆盖 family '{family}'；"
            "安装对应的插件或改用受支持的 stage/adapter"
        )
        return Selection(capability=capability, backend=None, reasons=reasons)

    # -- internals ---------------------------------------------------------

    def _sorted(self) -> list[Any]:
        return sorted(self.plugins.values(), key=lambda p: _manifest_of(p).id)


def _manifest_of(plugin: Any) -> PluginManifest | None:
    if isinstance(plugin, PluginManifest):
        return plugin
    manifest = getattr(plugin, "manifest", None)
    if isinstance(manifest, PluginManifest):
        return manifest
    if isinstance(manifest, dict):
        try:
            return PluginManifest.model_validate(manifest)
        except ValueError:
            return None
    return None


def _looks_like_backend(plugin: Any) -> bool:
    return all(callable(getattr(plugin, name, None)) for name in ("compile", "supports", "run"))


def _family_kind(family: str) -> str:
    from .schema import LLM_FAMILIES

    if family in LLM_FAMILIES:
        return "llm"
    if family in VIDEO_FAMILIES:
        return "video"
    return "unknown"


def _mismatch(plugin: Any, capability: str, family: str) -> str:
    """Return a human-readable mismatch reason, or "" when the plugin fits."""
    manifest = _manifest_of(plugin)
    if manifest is None:
        return "没有可读的 manifest"
    if capability not in manifest.capabilities:
        supported = ", ".join(manifest.capabilities) or "（无）"
        return f"未声明 capability '{capability}'（已声明：{supported}）"
    if manifest.families and family not in manifest.families:
        return f"未声明 family '{family}'（已声明：{', '.join(manifest.families)}）"
    return ""


def builtin_backends() -> list[Any]:
    """Instantiate the built-in backends. Imported lazily to avoid import cycles."""
    from .backends.musubi import MusubiBackend
    from .backends.native_llm import NativeLLMBackend

    return [NativeLLMBackend(), MusubiBackend()]


def load_registry(
    *,
    include_builtin: bool = True,
    entry_points_fn: Callable[[str], Iterable[Any]] | None = None,
    extra_backends: Sequence[Any] = (),
) -> Registry:
    """Load built-ins, extra backends and entry points; collect errors.

    `entry_points_fn` and `extra_backends` let tests exercise discovery without
    installing a package.
    """
    registry = Registry()

    if include_builtin:
        for backend in builtin_backends():
            _register_or_report(registry, backend, source="builtin")
    for backend in extra_backends:
        _register_or_report(registry, backend, source="injected")

    if entry_points_fn is None:
        try:
            discovered = list(importlib_metadata.entry_points(group=PLUGIN_GROUP))
        except Exception as exc:  # pragma: no cover - metadata backend failure
            registry.add_error(PLUGIN_GROUP, f"{type(exc).__name__}: {exc}")
            discovered = []
    else:
        discovered = list(entry_points_fn(PLUGIN_GROUP))

    for entry_point in discovered:
        label = f"{getattr(entry_point, 'name', '?')} = {getattr(entry_point, 'value', '?')}"
        try:
            target = entry_point.load()
            plugin = target() if isinstance(target, type) else target
            manifest = _manifest_of(plugin)
            if manifest is None:
                registry.add_error(label, "entry point did not return an object with a PluginManifest")
                continue
            if manifest.kind == "backend" and not _looks_like_backend(plugin):
                registry.add_error(label, "backend plugin is missing compile/supports/run")
                continue
            registry.register(plugin, source=label)
        except Exception as exc:  # noqa: BLE001 - plugin code is untrusted
            registry.add_error(label, f"{type(exc).__name__}: {exc}")

    return registry


def _register_or_report(registry: Registry, plugin: Any, source: str) -> None:
    try:
        registry.register(plugin, source=source)
    except Exception as exc:  # noqa: BLE001 - a broken built-in must not kill the CLI
        registry.add_error(source, f"{type(exc).__name__}: {exc}")