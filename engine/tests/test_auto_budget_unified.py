"""Unified auto-select tests (v0.2 T3 / G2 = AP-02).

Pins the extended candidate search introduced on top of the auto-budget
planner:

- the quant ladder is ``NONE -> FP8 -> INT8 -> NF4 -> NVFP4`` (both 4-bit
  tiers participate, v0.2 T7);
- video block-swap families also search ``train_blocks`` (``None`` and
  the Fizgig-style ``"20-49"`` partial window);
- ``MemoryPlan.auto_candidates`` records every evaluated candidate,
  feasible-first by ascending peak, then infeasible by ascending deficit;
- the chosen candidate stays the lowest-peak feasible one, with the
  fewer-trained-blocks candidate winning an exact tie;
- ``auto_candidates`` stays ``[]`` when the search did not run.
"""

from __future__ import annotations

import re

from supertuner_engine.plan import plan, resolve_config
from supertuner_engine.schema import QuantBase, RunConfig


def _video(family: str = "minimax-h3", **kw):
    base = dict(
        name=f"unified-{family}",
        family=family,
        stage="sft",
        adapter="lora",
        quant_base=QuantBase.NONE,
        blocks_to_swap=0,
        width=320,
        height=576,
        frames=49,
        cache="both",
        vram_budget_mib=8192,
        ram_budget_mib=32768,
        auto_budget=True,
    )
    base.update(kw)
    return RunConfig(**base)


def _llm(**kw):
    base = dict(
        name="unified-llm",
        family="qwen3-0.6b",
        stage="sft",
        adapter="lora",
        quant_base=QuantBase.NONE,
        width=480,
        height=832,
        frames=49,
        seq_len=2048,
        vram_budget_mib=8192,
        ram_budget_mib=32768,
    )
    base.update(kw)
    return RunConfig(**base)


def _deficit_from_note(note: str) -> float:
    m = re.search(r"缺 (\d+) MiB", note)
    assert m is not None, f"deficit note not found in {note!r}"
    return float(m.group(1))


def test_quant_ladder_includes_4bit_tiers():
    """The full precision ladder {NONE, FP8, INT8, NF4, NVFP4} is evaluated."""
    p = plan(_video())
    assert p.auto_candidates, "auto_candidates must be populated for auto search"
    quants = {c.quant_base for c in p.auto_candidates}
    assert quants == {
        QuantBase.NONE.value,
        QuantBase.FP8.value,
        QuantBase.INT8.value,
        QuantBase.NF4.value,
        QuantBase.NVFP4.value,
    }, f"ladder incomplete: {sorted(quants)}"
    assert any(c.quant_base == QuantBase.NF4.value for c in p.auto_candidates)
    assert any(c.quant_base == QuantBase.NVFP4.value for c in p.auto_candidates)


def test_train_blocks_dimension_present():
    """Video families search both full blocks and the 20-49 partial window."""
    p = plan(_video())
    train_values = {c.train_blocks for c in p.auto_candidates}
    assert None in train_values, "full-blocks (None) candidate must be evaluated"
    assert "20-49" in train_values, "the 20-49 partial window must be evaluated"
    # a user-supplied window is added as its own preset (deduped against it)
    p2 = plan(_video(train_blocks="3-12"))
    train_values2 = {c.train_blocks for c in p2.auto_candidates}
    assert train_values2 == {None, "20-49", "3-12"}


def test_auto_candidates_sorted_feasible_first():
    """Feasible rows come first (ascending peak), then infeasible (ascending gap)."""
    p = plan(_video(width=480, height=832))  # mixed feasible/infeasible set
    rows = p.auto_candidates
    feasible = [r for r in rows if r.feasible]
    infeasible = [r for r in rows if not r.feasible]
    assert feasible, "expected a mixed feasible/infeasible set at this budget"
    # feasible rows form the prefix, in ascending headline order
    assert [r.feasible for r in rows] == [True] * len(feasible) + [False] * len(infeasible)
    assert all(
        rows[i].headline_mib <= rows[i + 1].headline_mib
        for i in range(len(feasible) - 1)
    ), "feasible rows must be ascending by headline_mib"
    # infeasible rows are ordered by ascending deficit (encoded in the note)
    deficits = [_deficit_from_note(r.note) for r in infeasible]
    assert all(a <= b for a, b in zip(deficits, deficits[1:])), (
        f"infeasible rows must be ascending by deficit: {deficits[:10]}..."
    )


def test_chosen_candidate_is_lowest_peak_and_prefers_fewer_blocks():
    """Best = lowest feasible peak; an exact tie prefers the fewer blocks."""
    cfg = _video()
    effective, p = resolve_config(cfg)
    # the chosen row is the first (lowest-peak) row of the ledger
    first = p.auto_candidates[0]
    assert p.feasible and first.feasible
    assert first.headline_mib == p.headline_mib
    # the full-blocks counterpart of the pick must have been evaluated and
    # ties at the same peak (train_blocks does not change the arithmetic)
    full_row = next(
        (
            r
            for r in p.auto_candidates
            if r.quant_base == effective.quant_base.value
            and r.blocks_to_swap == effective.blocks_to_swap
            and r.train_blocks is None
            and r.feasible
        ),
        None,
    )
    assert full_row is not None, "full-blocks counterpart of the pick must be evaluated"
    assert full_row.headline_mib == first.headline_mib
    # on the exact tie the pick must prefer the fewer trained blocks
    assert effective.train_blocks == "20-49"


def test_auto_candidates_empty_for_plain_llm():
    """auto_budget off (and LLM families) must not populate auto_candidates."""
    p_off = plan(_llm())  # auto_budget defaults to False
    assert p_off.auto_candidates == []

    p_llm_auto = plan(_llm(auto_budget=True))  # LLM: search never activates
    assert p_llm_auto.auto_candidates == []


def test_infeasible_candidate_note_carries_deficit():
    """Infeasible rows carry a deficit note and feasible=False."""
    p = plan(_video(width=480, height=832, vram_budget_mib=6144))
    assert not p.feasible
    rows = p.auto_candidates
    assert rows, "infeasible runs still record the evaluated ledger"
    for r in rows:
        assert r.feasible is False
        assert re.match(r"^缺 \d+ MiB$", r.note), f"bad infeasible note: {r.note!r}"
        assert _deficit_from_note(r.note) > 0


def test_user_train_blocks_deduped_against_presets():
    """A user preset equal to a built-in preset is not duplicated."""
    p = plan(_video(train_blocks="20-49"))
    counts: dict = {}
    for c in p.auto_candidates:
        key = (c.quant_base, c.blocks_to_swap, c.train_blocks)
        counts[key] = counts.get(key, 0) + 1
    assert all(n == 1 for n in counts.values()), "candidate combos must be unique"
    assert "20-49" in {c.train_blocks for c in p.auto_candidates}
