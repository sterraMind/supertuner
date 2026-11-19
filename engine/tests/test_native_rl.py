from __future__ import annotations

from supertuner_engine.backends.native_llm import _reward_matches


def test_numeric_reward_compares_values_not_substrings() -> None:
    assert _reward_matches("37", "37") is True
    assert _reward_matches("37", " 37. ") is True
    assert _reward_matches("37", "137") is False
    assert _reward_matches("-5", "the answer is -5") is True


def test_text_reward_is_case_and_punctuation_insensitive() -> None:
    assert _reward_matches("Paris", "paris!") is True
    assert _reward_matches("New York", "new   york!") is True
    assert _reward_matches("Paris", "London") is False
