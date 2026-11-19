"""Dataset loading must tolerate real plain-text corpora.

wikitext-2 is a directory of `.txt` files full of blank lines. The first
smoke run on a real GPU aborted on the first blank row, so blank lines are now
dropped while a missing/typed column still raises.
"""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

from supertuner_engine.backends.native_llm import _load_split, configure_cuda_allocator
from supertuner_engine.schema import DatasetKind, DatasetSpec


class _FakeTokenizer:
    def __call__(self, text: str, add_special_tokens: bool = True) -> dict[str, list[int]]:
        return {"input_ids": [ord(ch) for ch in text.strip()]}

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=False):  # noqa: ANN001
        return [ord(ch) for ch in str(messages)]


class _Emit:
    def log(self, level: str, message: str) -> None:  # noqa: ANN001
        pass


def _load(path: Path, dataset: DatasetSpec):
    return _load_split(
        dataset,
        path,
        _FakeTokenizer(),
        SimpleNamespace(seq_len=16, seed=42),
        _Emit(),
    )


def test_blank_lines_in_text_directory_are_skipped(tmp_path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "train.txt").write_text("\n\nhello\n\n \nworld\n", encoding="utf-8")

    train, val = _load(corpus, DatasetSpec(path=str(corpus), kind=DatasetKind.TEXT, format="directory"))

    assert val is None
    assert [row["input_ids"] for row in train] == [[ord(ch) for ch in "hello"], [ord(ch) for ch in "world"]]

def test_windows_allocator_defaults_to_async(monkeypatch):
    monkeypatch.setattr('sys.platform', 'win32')
    monkeypatch.delenv('PYTORCH_CUDA_ALLOC_CONF', raising=False)
    configure_cuda_allocator()
    assert os.environ['PYTORCH_CUDA_ALLOC_CONF'] == 'backend:cudaMallocAsync'


def test_windows_allocator_keeps_explicit_backend(monkeypatch):
    monkeypatch.setattr('sys.platform', 'win32')
    monkeypatch.setenv('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
    configure_cuda_allocator()
    assert os.environ['PYTORCH_CUDA_ALLOC_CONF'] == 'expandable_segments:True,backend:cudaMallocAsync'


def test_non_windows_leaves_allocator_unset(monkeypatch):
    monkeypatch.setattr('sys.platform', 'linux')
    monkeypatch.delenv('PYTORCH_CUDA_ALLOC_CONF', raising=False)
    configure_cuda_allocator()
    assert 'PYTORCH_CUDA_ALLOC_CONF' not in os.environ
