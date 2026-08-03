#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from qwen_cua.eval.metrics import (  # noqa: E402
    compare_binary_acc,
    load_raw_scores,
    load_structured_scores,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare OSWorld runs using binary task accuracy (result == 1)."
    )
    parser.add_argument("baseline", type=Path, help="structured baseline tasks.jsonl")
    parser.add_argument("run", type=Path, nargs="+", help="raw <domain> task directories")
    parser.add_argument("--domain", default="multi_apps")
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Allow runs missing baseline task IDs; unknown task IDs still fail.",
    )
    args = parser.parse_args()

    baseline = load_structured_scores(args.baseline, domain=args.domain)
    output = {
        "metric": "binary_accuracy",
        "pass_condition": "result == 1",
        "domain": args.domain,
        "baseline_tasks": len(baseline),
        "runs": [],
    }
    for task_root in args.run:
        scores = load_raw_scores(task_root)
        comparison = compare_binary_acc(
            baseline,
            scores,
            allow_partial=args.allow_partial,
        )
        output["runs"].append(
            {
                "task_root": str(task_root.resolve()),
                "completed": comparison.completed,
                "passed": comparison.passed,
                "accuracy": comparison.accuracy,
                "gained_vs_baseline": comparison.gained,
                "lost_vs_baseline": comparison.lost,
                "unchanged_vs_baseline": comparison.unchanged,
                "missing_count": len(comparison.missing_task_ids),
                "missing_task_ids": list(comparison.missing_task_ids),
            }
        )
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
