"""T5 (AP-07) external-declaration alignment: musubi task firmware fixture.

The fixture ``fixtures/musubi_tasks.json`` records what upstream musubi-tuner
actually supports (docs/wan.md + docs/minimax_h3.md, fetched 2026-11). This
test pins ``MUSUBI_FAMILIES`` / ``_TASK_BY_FAMILY`` / ``_NETWORK_MODULE`` in
``supertuner_engine/backends/musubi.py`` against that fixture item-for-item, so
the ``--task`` / ``--network_module`` values we advertise cannot drift from
what the upstream CLI accepts.

Plain pytest, no GPU, no torch, no network.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from supertuner_engine.backends import musubi as musubi_backend

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "musubi_tasks.json"


@pytest.fixture(scope="module")
def firmware() -> dict:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def test_fixture_documents_the_upstream_firmware(firmware: dict) -> None:
    """The fixture itself must be well-formed and name its provenance."""
    assert firmware.get("source"), "fixture must record its upstream source"
    tasks = firmware.get("tasks")
    assert isinstance(tasks, dict) and tasks, "fixture must list the supported tasks"
    for family, entry in tasks.items():
        assert isinstance(entry.get("task"), str) and entry["task"], (
            f"task value missing for {family}"
        )
        assert isinstance(entry.get("network_module"), str) and entry["network_module"], (
            f"network_module missing for {family}"
        )
    unsupported = firmware.get("unsupported")
    assert isinstance(unsupported, list), "fixture must declare the unsupported list"
    for entry in unsupported:
        assert isinstance(entry.get("family"), str) and entry["family"]
        assert isinstance(entry.get("reason"), str) and entry["reason"]


def test_task_values_match_fixture(firmware: dict) -> None:
    """Every family in _TASK_BY_FAMILY must hit the fixture task 1:1."""
    tasks = firmware["tasks"]
    for family, task in musubi_backend._TASK_BY_FAMILY.items():
        assert family in tasks, f"fixture has no task entry for family {family!r}"
        assert tasks[family]["task"] == task, (
            f"family {family}: code says --task {task!r}, "
            f"upstream fixture says {tasks[family]['task']!r}"
        )


def test_network_modules_match_fixture(firmware: dict) -> None:
    """Every family in _NETWORK_MODULE must hit the fixture module 1:1."""
    tasks = firmware["tasks"]
    for family, module in musubi_backend._NETWORK_MODULE.items():
        assert family in tasks, f"fixture has no task entry for family {family!r}"
        assert tasks[family]["network_module"] == module, (
            f"family {family}: code says --network_module {module!r}, "
            f"fixture says {tasks[family]['network_module']!r}"
        )


def test_families_and_fixture_are_the_same_set(firmware: dict) -> None:
    """No orphan families on either side of the boundary."""
    fixture_families = set(firmware["tasks"])
    code_families = set(musubi_backend.MUSUBI_FAMILIES)
    assert code_families == fixture_families, (
        f"MUSUBI_FAMILIES {sorted(code_families)} != "
        f"fixture tasks {sorted(fixture_families)}"
    )
    assert set(musubi_backend._TASK_BY_FAMILY) == code_families
    assert set(musubi_backend._NETWORK_MODULE) == code_families


def test_wan22_ti2v_5b_is_not_trainable(firmware: dict) -> None:
    """wan2.2-ti2v-5b must be declared unsupported upstream and absent from code."""
    unsupported = {entry["family"] for entry in firmware["unsupported"]}
    assert "wan2.2-ti2v-5b" in unsupported
    assert "wan2.2-ti2v-5b" not in musubi_backend.MUSUBI_FAMILIES
    assert "wan2.2-ti2v-5b" not in musubi_backend._TASK_BY_FAMILY
    assert "wan2.2-ti2v-5b" not in musubi_backend._NETWORK_MODULE
