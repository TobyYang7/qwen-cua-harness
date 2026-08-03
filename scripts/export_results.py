#!/usr/bin/env python3
"""Export one raw OSWorld run to results/{config_name}/{run_id}."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from qwen_cua.actions import action_to_public_dict
from qwen_cua.deploy import load_profile
from qwen_cua.protocol import ToolCallParseError, parse_tool_calls

SCHEMA_VERSION = 1
HARNESS_ROOT = Path(__file__).resolve().parents[1]


def _json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _rel(path: Path) -> str:
    return os.path.relpath(path.resolve(), HARNESS_ROOT)


def _git_commit(path: Path) -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=path,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_trajectory(path: Path) -> tuple[list[dict[str, Any]], str | None]:
    steps: list[dict[str, Any]] = []
    error = None
    if not path.exists():
        return steps, error
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                error = f"invalid traj.jsonl line {line_number}: {exc}"
                continue
            if not isinstance(item, dict):
                error = f"invalid traj.jsonl line {line_number}: expected object"
            elif "Error" in item:
                error = str(item["Error"])
            elif "step_num" in item:
                steps.append(item)
    return steps, error


def _failure_type(error: str | None) -> str | None:
    if not error:
        return None
    lowered = error.lower()
    if "no proxy available" in lowered or "proxy pool" in lowered:
        return "proxy_pool_empty"
    if "no such container" in lowered or ("404" in lowered and "container" in lowered):
        return "docker_404"
    if "timeout" in lowered or "timed out" in lowered:
        return "timeout"
    if "initializ" in lowered:
        return "init_failure"
    return "environment_error"


def _timestamp(value: Any) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        return datetime.strptime(value, "%Y%m%d@%H%M%S").replace(tzinfo=UTC)
    except ValueError:
        return None


def _duration(steps: list[dict[str, Any]]) -> float | None:
    values = [stamp for step in steps if (stamp := _timestamp(step.get("action_timestamp")))]
    return (max(values) - min(values)).total_seconds() if values else None


def _score(path: Path) -> float | None:
    if not path.exists():
        return None
    text = path.read_text(encoding="utf-8").strip()
    if text.lower() in {"true", "false"}:
        return float(text.lower() == "true")
    try:
        return float(text)
    except ValueError:
        return None


def _diagnostics(path: Path) -> tuple[int, int, int]:
    if not path.exists():
        return 0, 0, 0
    text = path.read_text(encoding="utf-8", errors="replace")
    repairs = text.count("[qwencua] malformed tool call, one repair:")
    no_actions = text.count("[qwencua] actions=(no action)")
    return repairs, min(repairs, no_actions), no_actions


def _thinking_chars(response: str) -> int:
    return sum(len(match) for match in re.findall(r"<think>(.*?)</think>", response, re.DOTALL))


def _step_record(
    raw: dict[str, Any],
    *,
    config_name: str,
    run_id: str,
    domain: str,
    task_id: str,
    task_dir: Path,
) -> dict[str, Any]:
    response = str(raw.get("response") or "")
    parsed: list[dict[str, object]] = []
    parse_error = None
    try:
        parsed = [action_to_public_dict(action) for action in parse_tool_calls(response)]
    except ToolCallParseError as exc:
        parse_error = str(exc)
    screenshot = task_dir / str(raw.get("screenshot_file", ""))
    return {
        "schema_version": SCHEMA_VERSION,
        "config_name": config_name,
        "run_id": run_id,
        "task_id": task_id,
        "domain": domain,
        "step_num": raw.get("step_num"),
        "action_timestamp": raw.get("action_timestamp"),
        "model_response": response,
        "parsed_actions": parsed,
        "executed_action": raw.get("action"),
        "tool_call_valid": bool(parsed) and parse_error is None,
        "parse_error": parse_error,
        "thinking_chars": _thinking_chars(response),
        "reward": raw.get("reward"),
        "done": bool(raw.get("done", False)),
        "info": raw.get("info") or {},
        "screenshot_path": _rel(screenshot) if screenshot.is_file() else None,
    }


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _infer_osworld_root(raw_run: Path) -> Path:
    for candidate in (raw_run, *raw_run.parents):
        if candidate.name == "osworld_eval":
            return candidate
    raise ValueError("raw run must be located under an osworld_eval directory")


def _resolve_from(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _parse_log_times(path: Path | None) -> tuple[str | None, str | None]:
    if path is None or not path.exists():
        return None, None
    stamps = re.findall(
        r"(?m)^(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z)", path.read_text(errors="replace")
    )
    return (stamps[0], stamps[-1]) if stamps else (None, None)


def export_run(args: argparse.Namespace) -> Path:
    raw_run = args.raw_run.resolve()
    config = args.config.resolve()
    osworld_root = _infer_osworld_root(raw_run)
    profile = load_profile(config)
    config_name = config.stem
    run_id = args.run_id or raw_run.name
    output_root = args.output_root.resolve()
    output = output_root / config_name / run_id
    if output.exists():
        if not args.overwrite:
            raise FileExistsError(f"output exists (pass --overwrite): {output}")
        shutil.rmtree(output)
    output.mkdir(parents=True)

    args_paths = sorted(raw_run.glob("*/*/*/args.json"))
    if len(args_paths) != 1:
        raise ValueError(f"expected exactly one args.json, found {len(args_paths)}")
    source_args_path = args_paths[0]
    source_args = _json(source_args_path)
    task_base = source_args_path.parent
    task_manifest = (
        args.task_manifest.resolve()
        if args.task_manifest
        else _resolve_from(osworld_root, str(source_args["test_all_meta_path"])).resolve()
    )
    task_map = _json(task_manifest)
    domains = str(source_args.get("domain", "all"))
    if domains != "all":
        selected = {item.strip() for item in domains.split(",")}
        task_map = {key: value for key, value in task_map.items() if key in selected}

    shutil.copyfile(config, output / "config.yaml")
    episodes: list[dict[str, Any]] = []
    step_records: list[dict[str, Any]] = []
    for domain, task_ids in task_map.items():
        for task_id in task_ids:
            task_dir = task_base / str(domain) / str(task_id)
            trajectory_path = task_dir / "traj.jsonl"
            result_path = task_dir / "result.txt"
            runtime_log_path = task_dir / "runtime.log"
            recording_path = task_dir / "recording.mp4"
            steps, error = _read_trajectory(trajectory_path)
            failure_type = _failure_type(error)
            score = _score(result_path)
            if failure_type in {"proxy_pool_empty", "docker_404", "timeout", "init_failure"}:
                status = "infra_failed"
            elif error:
                status = "failed"
            elif score is not None:
                status = "completed"
            else:
                status = "incomplete"
            repairs, repair_failures, no_actions = _diagnostics(runtime_log_path)
            episodes.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "config_name": config_name,
                    "run_id": run_id,
                    "task_id": str(task_id),
                    "domain": str(domain),
                    "status": status,
                    "score": score,
                    "steps": len(steps),
                    "duration_seconds_observed": _duration(steps),
                    "failure_type": failure_type,
                    "failure_message": error,
                    "repairs": repairs,
                    "repair_failures": repair_failures,
                    "no_action_turns": no_actions,
                    "trajectory_path": _rel(trajectory_path) if trajectory_path.exists() else None,
                    "result_path": _rel(result_path) if result_path.exists() else None,
                    "runtime_log_path": _rel(runtime_log_path)
                    if runtime_log_path.exists()
                    else None,
                    "recording_path": _rel(recording_path) if recording_path.exists() else None,
                }
            )
            step_records.extend(
                _step_record(
                    step,
                    config_name=config_name,
                    run_id=run_id,
                    domain=str(domain),
                    task_id=str(task_id),
                    task_dir=task_dir,
                )
                for step in steps
            )

    with (output / "episodes.jsonl").open("w", encoding="utf-8") as handle:
        for item in episodes:
            handle.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")
    with (output / "steps.jsonl").open("w", encoding="utf-8") as handle:
        for item in step_records:
            handle.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")

    scored = [item for item in episodes if item["score"] is not None]
    clean = [item for item in scored if item["status"] == "completed"]
    by_domain: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in scored:
        by_domain[item["domain"]].append(item)

    def aggregate(items: list[dict[str, Any]]) -> tuple[int, float, float | None]:
        total = sum(float(item["score"]) for item in items)
        return len(items), total, total / len(items) if items else None

    official_n, official_sum, official_mean = aggregate(scored)
    clean_n, clean_sum, clean_mean = aggregate(clean)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "config_name": config_name,
        "run_id": run_id,
        "expected_tasks": sum(len(ids) for ids in task_map.values()),
        "result_count": len(scored),
        "step_count": len(step_records),
        "official": {
            "denominator": official_n,
            "score_sum": official_sum,
            "mean_score": official_mean,
            "positive_tasks": sum(float(item["score"]) > 0 for item in scored),
            "zero_tasks": sum(float(item["score"]) == 0 for item in scored),
        },
        "clean": {
            "definition": (
                "status == completed; excludes explicitly classified "
                "infrastructure/init/timeout failures"
            ),
            "denominator": clean_n,
            "score_sum": clean_sum,
            "mean_score": clean_mean,
        },
        "by_domain": {
            domain: {"count": n, "score_sum": total, "mean_score": mean}
            for domain, items in sorted(by_domain.items())
            for n, total, mean in [aggregate(items)]
        },
        "status_counts": dict(sorted(Counter(item["status"] for item in episodes).items())),
        "failure_counts": dict(
            sorted(
                Counter(item["failure_type"] for item in episodes if item["failure_type"]).items()
            )
        ),
    }
    _write_json(output / "summary.json", summary)

    started_at, finished_at = _parse_log_times(args.slurm_log)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "benchmark": "osworld",
        "config_name": config_name,
        "run_id": run_id,
        "status": "completed" if len(scored) == summary["expected_tasks"] else "incomplete",
        "expected_tasks": summary["expected_tasks"],
        "started_at": args.started_at or started_at,
        "finished_at": args.finished_at or finished_at,
        "path_base": "qwen-cua-harness repository root",
        "config_file": _rel(output / "config.yaml"),
        "config_sha256": _sha256(output / "config.yaml"),
        "raw_run_path": _rel(raw_run),
        "source_args_path": _rel(source_args_path),
        "task_manifest_path": _rel(task_manifest),
        "slurm_log_path": _rel(args.slurm_log) if args.slurm_log else None,
        "model": profile["model"],
        "inference": profile["inference"],
        "serving": profile["serving"],
        "slurm": {"job_id": args.job_id, "node": args.node, "exit_code": args.exit_code},
        "git": {
            "harness_commit": _git_commit(HARNESS_ROOT),
            "repo_commit": args.repo_commit or _git_commit(osworld_root.parent),
        },
    }
    _write_json(output / "manifest.json", manifest)
    checksum_files = [
        "config.yaml",
        "episodes.jsonl",
        "manifest.json",
        "steps.jsonl",
        "summary.json",
    ]
    _write_json(
        output / "checksums.json",
        {
            "algorithm": "sha256",
            "files": {name: _sha256(output / name) for name in checksum_files},
        },
    )
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw_run", type=Path)
    parser.add_argument("config", type=Path)
    parser.add_argument("--output-root", type=Path, default=HARNESS_ROOT / "results")
    parser.add_argument("--run-id")
    parser.add_argument("--task-manifest", type=Path)
    parser.add_argument("--slurm-log", type=Path)
    parser.add_argument("--job-id", type=int)
    parser.add_argument("--node")
    parser.add_argument("--exit-code", type=int)
    parser.add_argument("--started-at")
    parser.add_argument("--finished-at")
    parser.add_argument("--repo-commit")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    try:
        output = export_run(args)
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    print(_rel(output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
