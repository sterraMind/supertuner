"""Preference (DPO) helpers: loss maths, log-probability, dataset pairs."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from supertuner_engine.backends.native_llm import (
    _dpo_loss,
    _load_preference,
    _sequence_logprob,
)


class _FakeTokenizer:
    """Whitespace tokenizer: deterministic ids, no vocabulary file needed."""

    def __call__(self, text: str, add_special_tokens: bool = True) -> dict[str, list[int]]:
        del add_special_tokens
        return {"input_ids": [len(word) for word in text.split()]}


class _FakeModel:
    """Returns fixed logits so the log-probability maths can be checked by hand."""

    def __init__(self, logits: list[list[list[float]]]) -> None:
        self._logits = logits

    def __call__(self, input_ids, attention_mask):
        torch = pytest.importorskip("torch")
        tensor = torch.tensor(self._logits, dtype=torch.float32)
        return SimpleNamespace(logits=tensor)


def test_dpo_loss_uses_the_implicit_reward_margin() -> None:
    torch = pytest.importorskip("torch")
    policy_chosen = torch.tensor([2.0])
    policy_rejected = torch.tensor([-1.0])
    ref_chosen = torch.tensor([1.0])
    ref_rejected = torch.tensor([0.5])
    beta = 0.5

    loss, margin = _dpo_loss(torch, policy_chosen, policy_rejected, ref_chosen, ref_rejected, beta)

    expected_margin = (2.0 - 1.0) - (-1.0 - 0.5)
    assert float(margin[0]) == pytest.approx(expected_margin)
    expected_loss = -torch.nn.functional.logsigmoid(torch.tensor(beta * expected_margin))
    assert float(loss) == pytest.approx(float(expected_loss), rel=1e-6)


def test_dpo_loss_drops_when_the_policy_prefers_chosen() -> None:
    torch = pytest.importorskip("torch")
    strong, _ = _dpo_loss(
        torch,
        torch.tensor([3.0]),
        torch.tensor([0.0]),
        torch.tensor([0.0]),
        torch.tensor([0.0]),
        1.0,
    )
    weak, _ = _dpo_loss(
        torch,
        torch.tensor([0.2]),
        torch.tensor([0.0]),
        torch.tensor([0.0]),
        torch.tensor([0.0]),
        1.0,
    )
    assert float(strong) < float(weak)


def test_sequence_logprob_is_length_normalised() -> None:
    torch = pytest.importorskip("torch")
    # 2 tokens, vocab 3: the highest logit is always at index 2.
    logits = [[[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]]
    model = _FakeModel(logits)
    input_ids = torch.tensor([[2, 2, 2]])
    attention_mask = torch.tensor([[1, 1, 1]])

    logprob = _sequence_logprob(torch, model, input_ids, attention_mask)

    expected_token = torch.log_softmax(torch.tensor([0.0, 0.0, 1.0]), dim=-1)[2]
    assert float(logprob[0]) == pytest.approx(float(expected_token), rel=1e-5)


def test_load_preference_reads_chosen_and_rejected_columns(tmp_path: Path) -> None:
    rows = [
        {"chosen": "good answer here", "rejected": "bad"},
        {"chosen": "fine reply", "rejected": "worse reply here"},
    ]
    path = tmp_path / "pairs.jsonl"
    path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )

    dataset = SimpleNamespace(
        path=str(path),
        kind="preference",
        format="jsonl",
        text_column="text",
        caption_column="caption",
        messages_column="messages",
        chosen_column="chosen",
        rejected_column="rejected",
        video_column="video",
        audio_column="audio",
        max_samples=None,
        val_fraction=0.0,
        notes="",
    )
    cfg = SimpleNamespace(seq_len=8)

    pairs = _load_preference(dataset, path, _FakeTokenizer(), cfg, _SilentEmit())

    assert len(pairs) == 2
    assert pairs[0]["chosen_ids"] == [4, 6, 4]
    assert pairs[0]["rejected_ids"] == [3]


def test_load_preference_rejects_rows_without_pairs(tmp_path: Path) -> None:
    path = tmp_path / "pairs.jsonl"
    path.write_text(json.dumps({"chosen": "only one side"}) + "\n", encoding="utf-8")

    dataset = SimpleNamespace(
        path=str(path),
        kind="preference",
        format="jsonl",
        text_column="text",
        caption_column="caption",
        messages_column="messages",
        chosen_column="chosen",
        rejected_column="rejected",
        video_column="video",
        audio_column="audio",
        max_samples=None,
        val_fraction=0.0,
        notes="",
    )
    cfg = SimpleNamespace(seq_len=8)

    with pytest.raises(RuntimeError, match="preference"):
        _load_preference(dataset, path, _FakeTokenizer(), cfg, _SilentEmit())


class _SilentEmit:
    def log(self, level: str, message: str) -> None:
        del level, message
