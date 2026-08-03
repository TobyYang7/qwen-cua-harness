#!/usr/bin/env python3
"""Run OSWorld or CUA-Gym with inference parameters from one model profile."""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from qwen_cua.deploy import load_profile

HARNESS_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EVAL_ROOT = (HARNESS_ROOT / "../../osworld_eval").resolve()
OSWORLD_RUNNER = HARNESS_ROOT / "scripts/eval/osworld_runner.py"
CUAGYM_RUNNER = HARNESS_ROOT / "scripts/eval/cuagym_runner.py"


def _endpoint(host: str, port: int) -> str:
    client_host = "127.0.0.1" if host in {"0.0.0.0", "::"} else host
    return f"http://{client_host}:{port}/v1"


def _common_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path, help="model profile under configs/models")
    parser.add_argument(
        "--eval-root",
        type=Path,
        default=DEFAULT_EVAL_ROOT,
        help="osworld_eval checkout (default: ../../osworld_eval)",
    )
    parser.add_argument(
        "--python",
        type=Path,
        help="evaluation Python (default: <eval-root>/.venv/bin/python or current Python)",
    )
    parser.add_argument("--run-id", help="output run ID; generated when omitted")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--dry-run", action="store_true", help="print the command only")
    subparsers = parser.add_subparsers(dest="benchmark", required=True)

    osworld = subparsers.add_parser("osworld", help="run the OSWorld desktop benchmark")
    osworld.add_argument("--path-to-vm", type=Path)
    osworld.add_argument("--task-manifest", type=Path)
    osworld.add_argument("--domain", default="all")
    osworld.add_argument("--num-workers", type=int, default=20)
    osworld.add_argument("--max-steps", type=int, default=50)
    osworld.add_argument("--disable-proxy", action="store_true")
    osworld.add_argument("--result-dir", type=Path)
    osworld.add_argument("--recording-dir", type=Path)
    osworld.add_argument(
        "--no-export-results",
        action="store_true",
        help="do not create results/{config_name}/{run_id} after evaluation",
    )

    cuagym = subparsers.add_parser("cuagym", help="run the CUA-Gym browser benchmark")
    cuagym.add_argument("--tasks", type=Path, required=True)
    cuagym.add_argument("--urls-json", type=Path, required=True)
    cuagym.add_argument("--result-dir", type=Path)
    cuagym.add_argument("--num-envs", type=int, default=8)
    cuagym.add_argument("--browser-pool", type=int)
    cuagym.add_argument("--max-steps", type=int, default=30)
    cuagym.add_argument("--episode-timeout-seconds", type=float, default=900)
    cuagym.add_argument("--limit", type=int)
    return parser


def _profile_args(profile: dict[str, object]) -> tuple[dict[str, object], dict[str, object]]:
    model = profile["model"]
    inference = profile["inference"]
    serving = profile["serving"]
    assert isinstance(model, dict) and isinstance(inference, dict) and isinstance(serving, dict)
    return model, {**inference, "base_url": _endpoint(str(serving["host"]), int(serving["port"]))}


def _osworld_command(
    args: argparse.Namespace,
    python: Path,
    model: dict[str, object],
    inference: dict[str, object],
    run_id: str,
) -> list[str]:
    result_dir = (
        os.path.relpath(args.result_dir.resolve(), args.eval_root.resolve())
        if args.result_dir
        else str(Path("results/osworld-std") / run_id)
    )
    recording_dir = (
        os.path.relpath(args.recording_dir.resolve(), args.eval_root.resolve())
        if args.recording_dir
        else str(Path("recordings/osworld-std") / run_id)
    )
    task_manifest = (
        os.path.relpath(args.task_manifest.resolve(), args.eval_root.resolve())
        if args.task_manifest
        else "evaluation_examples/test_nogdrive.json"
    )
    command = [
        str(python),
        os.path.relpath(OSWORLD_RUNNER, args.eval_root.resolve()),
        "--model",
        str(model["served_name"]),
        "--agent_type",
        "qwencua",
        "--provider_name",
        "docker",
        "--headless",
        "--action_space",
        "claude_computer_use",
        "--observation_type",
        "screenshot",
        "--screen_width",
        "1920",
        "--screen_height",
        "1080",
        "--image_size",
        "1280",
        "720",
        "--relative_coordinate",
        "--test_config_base_dir",
        "evaluation_examples/examples",
        "--test_all_meta_path",
        task_manifest,
        "--cache_dir",
        "cache",
        "--result_dir",
        result_dir,
        "--recording_dir",
        recording_dir,
        "--domain",
        args.domain,
        "--num_workers",
        str(args.num_workers),
        "--max_steps",
        str(args.max_steps),
        "--temperature",
        str(inference["temperature"]),
        "--top_p",
        str(inference["top_p"]),
        "--top_k",
        str(inference["top_k"]),
        "--max_tokens",
        str(inference["max_tokens"]),
        "--qwencua_history_n",
        str(inference["history_n"]),
        "--only_n_most_recent_images",
        str(inference["image_max"]),
        "--context_max_items",
        str(inference["context_max_items"]),
        "--context_max_chars",
        str(inference["context_max_chars"]),
    ]
    if bool(inference["enable_thinking"]):
        command.append("--enable_thinking")
    if bool(inference["context_memory"]):
        command.append("--context_memory")
    if args.path_to_vm:
        command.extend(
            [
                "--path_to_vm",
                os.path.relpath(args.path_to_vm.resolve(), args.eval_root.resolve()),
            ]
        )
    if args.disable_proxy:
        command.append("--disable_proxy")
    return command


def _cuagym_command(
    args: argparse.Namespace,
    python: Path,
    model: dict[str, object],
    inference: dict[str, object],
    run_id: str,
) -> list[str]:
    result_dir = (
        os.path.relpath(args.result_dir.resolve(), args.eval_root.resolve())
        if args.result_dir
        else str(Path("results_cuagym") / run_id)
    )
    command = [
        str(python),
        os.path.relpath(CUAGYM_RUNNER, args.eval_root.resolve()),
        "--agent_type",
        "qwencua",
        "--model",
        str(model["served_name"]),
        "--base_url",
        str(inference["base_url"]),
        "--api_key",
        args.api_key,
        "--tasks",
        os.path.relpath(args.tasks.resolve(), args.eval_root.resolve()),
        "--urls_json",
        os.path.relpath(args.urls_json.resolve(), args.eval_root.resolve()),
        "--result_dir",
        result_dir,
        "--num_envs",
        str(args.num_envs),
        "--browser_pool",
        str(args.browser_pool or args.num_envs),
        "--max_steps",
        str(args.max_steps),
        "--max_agent_steps",
        str(args.max_steps),
        "--episode_timeout_seconds",
        str(args.episode_timeout_seconds),
        "--temperature",
        str(inference["temperature"]),
        "--top_p",
        str(inference["top_p"]),
        "--top_k",
        str(inference["top_k"]),
        "--max_tokens",
        str(inference["max_tokens"]),
        "--qwencua_history_n",
        str(inference["history_n"]),
        "--max_recent_images",
        str(inference["image_max"]),
        "--context_max_items",
        str(inference["context_max_items"]),
        "--context_max_chars",
        str(inference["context_max_chars"]),
    ]
    if bool(inference["enable_thinking"]):
        command.append("--enable_thinking")
    if bool(inference["context_memory"]):
        command.append("--context_memory")
    if args.limit is not None:
        command.extend(["--limit", str(args.limit)])
    return command


def main() -> int:
    parser = _common_parser()
    args = parser.parse_args()
    eval_root = args.eval_root.resolve()
    if not eval_root.is_dir():
        parser.error(f"eval root does not exist: {eval_root}")
    config = args.config if args.config.is_absolute() else HARNESS_ROOT / args.config
    profile = load_profile(config.resolve())
    model, inference = _profile_args(profile)
    if args.python:
        # Keep a virtualenv's python symlink intact. Path.resolve() follows it
        # to /usr/bin/python and silently drops the environment's dependencies.
        python_path = (
            args.python
            if args.python.is_absolute()
            else Path.cwd() / args.python
        )
        python: str | Path = os.path.relpath(python_path.absolute(), eval_root)
    elif (eval_root / ".venv/bin/python").exists():
        python = Path(".venv/bin/python")
    else:
        python = os.path.relpath(Path(sys.executable).resolve(), eval_root)
    run_id = args.run_id or (
        f"{model['served_name']}_qwencua_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}"
    )
    if args.benchmark == "osworld":
        command = _osworld_command(args, python, model, inference, run_id)
    else:
        command = _cuagym_command(args, python, model, inference, run_id)

    print(f"config={os.path.relpath(config.resolve(), HARNESS_ROOT)}")
    print(f"cwd={os.path.relpath(eval_root, HARNESS_ROOT)}")
    print(shlex.join(command))
    if args.dry_run:
        return 0

    env = os.environ.copy()
    env["OPENAI_BASE_URL"] = str(inference["base_url"])
    env["OPENAI_API_KEY"] = args.api_key
    env["QWEN_CUA_ENV_ROOT"] = str(eval_root.parent)
    python_path = [str(HARNESS_ROOT / "src"), str(eval_root), str(eval_root.parent)]
    if env.get("PYTHONPATH"):
        python_path.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(python_path)
    completed = subprocess.run(command, cwd=eval_root, env=env, check=False)
    if args.benchmark == "osworld" and not args.no_export_results:
        raw_result = (
            args.result_dir.resolve()
            if args.result_dir
            else eval_root / "results/osworld-std" / run_id
        )
        export_command = [
            sys.executable,
            "scripts/export_results.py",
            os.path.relpath(raw_result, HARNESS_ROOT),
            os.path.relpath(config.resolve(), HARNESS_ROOT),
        ]
        export_result = subprocess.run(
            export_command,
            cwd=HARNESS_ROOT,
            env=env,
            check=False,
        )
        if completed.returncode == 0 and export_result.returncode != 0:
            return export_result.returncode
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
