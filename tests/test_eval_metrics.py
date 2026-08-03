from __future__ import annotations

import pytest

from qwen_cua.eval.metrics import compare_binary_acc, is_binary_pass


def test_binary_pass_rejects_fractional_reward() -> None:
    assert is_binary_pass(1.0)
    assert not is_binary_pass(0.9999999995)
    assert not is_binary_pass(0.999)
    assert not is_binary_pass(0.71)
    assert not is_binary_pass(0.0)


def test_compare_binary_acc_counts_paired_transitions() -> None:
    reference = {"a": 1.0, "b": 0.0, "c": 0.7, "d": 1.0}
    candidate = {"a": 1.0, "b": 1.0, "c": 0.7, "d": 0.0}

    result = compare_binary_acc(reference, candidate)

    assert result.completed == 4
    assert result.passed == 2
    assert result.accuracy == 0.5
    assert result.gained == 1
    assert result.lost == 1
    assert result.unchanged == 2
    assert result.missing_task_ids == ()


def test_compare_binary_acc_rejects_partial_candidate_by_default() -> None:
    with pytest.raises(ValueError, match="missing task ids"):
        compare_binary_acc({"a": 1.0, "b": 0.0}, {"a": 1.0})


def test_compare_binary_acc_allows_explicit_partial_candidate() -> None:
    result = compare_binary_acc(
        {"a": 1.0, "b": 0.0, "c": 1.0},
        {"a": 1.0},
        allow_partial=True,
    )

    assert result.completed == 1
    assert result.passed == 1
    assert result.accuracy == 1.0
    assert result.missing_task_ids == ("b", "c")
