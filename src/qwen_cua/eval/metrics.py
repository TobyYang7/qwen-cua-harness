from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


def is_binary_pass(score: float) -> bool:
    """A benchmark task counts as accurate only when its evaluator returns 1."""

    return float(score) == 1.0


def load_structured_scores(path: Path, *, domain: str) -> dict[str, float]:
    scores: dict[str, float] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("domain") == domain:
                scores[str(row["task_id"])] = float(row["score"])
    return scores


def load_raw_scores(task_root: Path) -> dict[str, float]:
    scores: dict[str, float] = {}
    for result in task_root.glob("*/result.txt"):
        scores[result.parent.name] = float(result.read_text(encoding="utf-8").strip())
    return scores


@dataclass(frozen=True, slots=True)
class BinaryAccComparison:
    completed: int
    passed: int
    accuracy: float
    gained: int
    lost: int
    unchanged: int
    missing_task_ids: tuple[str, ...]


def compare_binary_acc(
    reference: dict[str, float],
    candidate: dict[str, float],
    *,
    allow_partial: bool = False,
) -> BinaryAccComparison:
    unknown = candidate.keys() - reference.keys()
    if unknown:
        raise ValueError(f"candidate contains unknown task ids: {sorted(unknown)}")
    missing = reference.keys() - candidate.keys()
    if missing and not allow_partial:
        raise ValueError(f"candidate is missing task ids: {sorted(missing)}")
    passed = sum(is_binary_pass(score) for score in candidate.values())
    gained = sum(
        is_binary_pass(score) and not is_binary_pass(reference[task_id])
        for task_id, score in candidate.items()
    )
    lost = sum(
        not is_binary_pass(score) and is_binary_pass(reference[task_id])
        for task_id, score in candidate.items()
    )
    completed = len(candidate)
    return BinaryAccComparison(
        completed=completed,
        passed=passed,
        accuracy=passed / completed if completed else 0.0,
        gained=gained,
        lost=lost,
        unchanged=completed - gained - lost,
        missing_task_ids=tuple(sorted(missing)),
    )
