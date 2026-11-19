"""Plugin registry tests: discovery resilience and backend selection."""

from __future__ import annotations

import pytest

from supertuner_engine.config import Context, EngineConfig
from supertuner_engine.plugins import (
    PLUGIN_GROUP,
    Registry,
    load_registry,
    required_capability,
)
from supertuner_engine.schema import (
    Adapter,
    ExecutionPlan,
    PluginManifest,
    RunConfig,
    RunSpec,
    Stage,
)


def _ctx(tmp_path) -> Context:
    config = EngineConfig.from_env(
        {
            "SUPERTUNER_HOME": str(tmp_path / "home"),
            "SUPERTUNER_MODELS_DIR": str(tmp_path / "models"),
            "SUPERTUNER_DATASETS_DIR": str(tmp_path / "datasets"),
            "SUPERTUNER_CACHE_DIR": str(tmp_path / "cache"),
        }
    )
    return config.context(tmp_path / "run")


def _spec(
    family: str = "qwen3-1.7b",
    stage: Stage = Stage.SFT,
    adapter: Adapter = Adapter.LORA,
    backend: str | None = None,
) -> RunSpec:
    return RunSpec(
        config=RunConfig(name="test", family=family, stage=stage, adapter=adapter),
        backend=backend,
    )


def _manifest(
    plugin_id: str,
    *,
    families: tuple[str, ...] = ("qwen3-1.7b",),
    capabilities: tuple[str, ...] = ("train.llm.sft",),
    priority: int = 100,
    enabled: bool = True,
    kind: str = "backend",
) -> PluginManifest:
    return PluginManifest(
        id=plugin_id,
        name=plugin_id,
        version="1.0.0",
        kind=kind,  # type: ignore[arg-type]
        capabilities=list(capabilities),
        families=list(families),
        priority=priority,
        enabled=enabled,
    )


class FakeBackend:
    def __init__(self, manifest: PluginManifest) -> None:
        self.manifest = manifest
        self.compiled = 0

    def probe(self, ctx: Context):
        return []

    def supports(self, spec: RunSpec) -> bool:
        return spec.config.family in self.manifest.families

    def compile(self, spec: RunSpec, ctx: Context) -> ExecutionPlan:
        self.compiled += 1
        return ExecutionPlan(backend=self.manifest.id, backend_version=self.manifest.version)

    def run(self, spec: RunSpec, plan: ExecutionPlan, ctx: Context, emit) -> None:
        return None

    def run_step(self, spec: RunSpec, step, ctx: Context, emit) -> None:
        return None


class FakeEntryPoint:
    def __init__(self, name: str, value: str, loader) -> None:
        self.name = name
        self.value = value
        self._loader = loader

    def load(self):
        return self._loader()


class BrokenEntryPoint:
    name = "broken"
    value = "broken.module:missing"

    def load(self):
        raise ImportError("No module named 'broken'")


def test_builtin_backends_are_registered():
    registry = load_registry(entry_points_fn=lambda group: [])
    ids = {manifest.id for manifest in registry.manifests()}
    assert {"native-llm", "musubi"} <= ids
    assert registry.errors == []


def test_entry_point_instance_is_registered():
    backend = FakeBackend(_manifest("third-party"))
    registry = load_registry(
        entry_points_fn=lambda group: [FakeEntryPoint("third-party", "pkg:Backend", lambda: backend)]
    )
    assert registry.get("third-party") is backend
    assert registry.errors == []


def test_entry_point_class_is_instantiated():
    backend_cls = lambda: FakeBackend(_manifest("class-plugin"))  # noqa: E731
    registry = load_registry(
        entry_points_fn=lambda group: [FakeEntryPoint("class-plugin", "pkg:Backend", backend_cls)]
    )
    assert isinstance(registry.get("class-plugin"), FakeBackend)


def test_broken_entry_point_is_collected_not_fatal():
    registry = load_registry(entry_points_fn=lambda group: [BrokenEntryPoint()])
    ids = {manifest.id for manifest in registry.manifests()}
    assert {"native-llm", "musubi"} <= ids
    assert len(registry.errors) == 1
    assert registry.errors[0].entry_point.startswith("broken =")
    assert "ImportError" in registry.errors[0].reason


def test_entry_point_without_manifest_is_reported():
    registry = load_registry(
        entry_points_fn=lambda group: [FakeEntryPoint("bad", "pkg:thing", lambda: object())]
    )
    assert registry.errors
    assert "PluginManifest" in registry.errors[0].reason


def test_entry_point_group_is_supertuner_plugins():
    seen: list[str] = []

    def entry_points(group: str):
        seen.append(group)
        return []

    load_registry(entry_points_fn=entry_points)
    assert seen == [PLUGIN_GROUP]
    assert PLUGIN_GROUP == "supertuner.plugins"


def test_higher_priority_wins():
    high = FakeBackend(_manifest("zzz-high", priority=200))
    registry = load_registry(entry_points_fn=lambda group: [], extra_backends=[high])
    selection = registry.select(_spec())
    assert selection.backend is high
    assert selection.reasons == []


def test_lower_priority_loses_to_builtin():
    low = FakeBackend(_manifest("aaa-low", priority=1))
    registry = load_registry(entry_points_fn=lambda group: [], extra_backends=[low])
    selection = registry.select(_spec())
    assert selection.backend is not None
    assert selection.backend.manifest.id == "native-llm"


def test_equal_priority_breaks_tie_by_id():
    first = FakeBackend(_manifest("aaa-tie", priority=150))
    second = FakeBackend(_manifest("bbb-tie", priority=150))
    registry = load_registry(entry_points_fn=lambda group: [], extra_backends=[first, second])
    selection = registry.select(_spec())
    assert selection.backend is first


def test_no_matching_backend_reports_reasons():
    # cogvideox has no video backend in v0.1: capability train.video.lora is unmatched.
    registry = load_registry(entry_points_fn=lambda group: [])
    selection = registry.select(_spec(family="cogvideox-2b"))
    assert selection.backend is None
    assert selection.capability == "train.video.lora"
    joined = " | ".join(selection.reasons)
    assert "train.video.lora" in joined
    assert "cogvideox-2b" in joined


def test_disabled_backend_is_skipped_with_reason():
    disabled = FakeBackend(_manifest("disabled", priority=999, enabled=False))
    registry = load_registry(entry_points_fn=lambda group: [], extra_backends=[disabled])
    selection = registry.select(_spec())
    assert selection.backend is not None
    assert selection.backend.manifest.id == "native-llm"
    assert all(backend.manifest.id != "disabled" for backend in registry.backends())


def test_explicit_unknown_backend_is_reported():
    registry = load_registry(entry_points_fn=lambda group: [])
    selection = registry.select(_spec(backend="nope"))
    assert selection.backend is None
    assert any("nope" in reason for reason in selection.reasons)


def test_explicit_backend_kept_even_when_mismatched():
    registry = load_registry(entry_points_fn=lambda group: [])
    selection = registry.select(_spec(family="qwen3-1.7b", backend="musubi"))
    assert selection.backend is not None
    assert selection.backend.manifest.id == "musubi"
    joined = " | ".join(selection.reasons)
    assert joined and ("train.llm.sft" in joined or "family" in joined)


def test_manifest_only_plugins_are_not_backends():
    plugin = _manifest("dataset-pack", kind="dataset", capabilities=["data.text"])
    registry = load_registry(entry_points_fn=lambda group: [], extra_backends=[plugin])
    assert registry.get("dataset-pack") is plugin
    assert all(manifest.id != "dataset-pack" for manifest in (b.manifest for b in registry.backends()))


def test_duplicate_id_with_different_version_is_rejected():
    registry = Registry()
    registry.register(FakeBackend(_manifest("dup")))
    other = _manifest("dup")
    other.version = "2.0.0"
    with pytest.raises(ValueError, match="conflict"):
        registry.register(FakeBackend(other))


def test_duplicate_id_same_version_is_idempotent():
    registry = Registry()
    first = FakeBackend(_manifest("dup"))
    second = FakeBackend(_manifest("dup"))
    registry.register(first)
    registry.register(second)
    assert registry.get("dup") is first


def test_to_response_matches_wire_shape():
    registry = load_registry(entry_points_fn=lambda group: [BrokenEntryPoint()])
    payload = registry.to_response()
    assert set(payload) == {"plugins", "errors"}
    assert isinstance(payload["plugins"], list)
    assert set(payload["errors"][0]) == {"entry_point", "reason"}


@pytest.mark.parametrize(
    ("stage", "kind", "adapter", "expected"),
    [
        (Stage.SFT, "llm", Adapter.LORA, "train.llm.sft"),
        (Stage.SFT, "llm", Adapter.NONE, "train.llm.sft"),
        (Stage.PRETRAIN, "llm", Adapter.LORA, "train.llm.pretrain"),
        (Stage.PRETRAIN, "video", Adapter.LORA, "train.video.full"),
        (Stage.SFT, "video", Adapter.LORA, "train.video.lora"),
        (Stage.SFT, "video", Adapter.NONE, "train.video.full"),
        (Stage.PREFERENCE, "llm", Adapter.LORA, "train.llm.preference"),
        (Stage.RL, "llm", Adapter.LORA, "train.llm.rl"),
        (Stage.DISTILL, "video", Adapter.LORA, "train.distill.dmd"),
    ],
)
def test_required_capability_mirrors_typescript(stage, kind, adapter, expected):
    assert required_capability(stage, kind, adapter) == expected


def test_families_for_capability():
    registry = load_registry(entry_points_fn=lambda group: [])
    assert "minimax-h3" in registry.families_for("train.video.lora")
    assert "qwen3-1.7b" in registry.families_for("train.llm.sft")
    assert registry.families_for("train.distill.dmd") == []


def test_ctx_is_unused_by_registry_but_valid(tmp_path):
    # Selection does not need a Context; compile does. Keep the helper honest.
    assert _ctx(tmp_path).models_dir.name == "models"