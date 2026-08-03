from __future__ import annotations

import argparse
import json
import os
import shlex
from pathlib import Path
from typing import Any

import yaml

MODEL_KEYS = {"id", "revision", "served_name"}
INFERENCE_KEYS = {
    "enable_thinking",
    "temperature",
    "top_p",
    "top_k",
    "max_tokens",
    "history_n",
    "image_max",
}
SERVING_KEYS = {
    "host",
    "port",
    "dtype",
    "tensor_parallel_size",
    "data_parallel_size",
    "api_server_count",
    "max_model_len",
    "max_num_seqs",
    "gpu_memory_utilization",
    "trust_remote_code",
    "disable_custom_all_reduce",
}


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping")
    return value


def _require_exact_keys(section: dict[str, Any], expected: set[str], name: str) -> None:
    missing = expected - section.keys()
    unknown = section.keys() - expected
    if missing or unknown:
        details = []
        if missing:
            details.append(f"missing: {', '.join(sorted(missing))}")
        if unknown:
            details.append(f"unknown: {', '.join(sorted(unknown))}")
        raise ValueError(f"invalid {name} keys ({'; '.join(details)})")


def _require_string(value: Any, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")


def _require_bool(value: Any, name: str) -> None:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a boolean")


def _require_int(value: Any, name: str, *, minimum: int = 1, maximum: int | None = None) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    if value < minimum or (maximum is not None and value > maximum):
        suffix = f"..{maximum}" if maximum is not None else " or greater"
        raise ValueError(f"{name} must be {minimum}{suffix}")


def _require_number(
    value: Any,
    name: str,
    *,
    minimum: float,
    maximum: float | None = None,
    minimum_inclusive: bool = True,
) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number")
    below = value < minimum if minimum_inclusive else value <= minimum
    if below or (maximum is not None and value > maximum):
        left = "[" if minimum_inclusive else "("
        upper = str(maximum) if maximum is not None else "infinity"
        raise ValueError(f"{name} must be in {left}{minimum}, {upper}]")


def load_profile(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    profile = _mapping(raw, "profile")
    _require_exact_keys(
        profile,
        {"schema_version", "model", "inference", "serving"},
        "top-level",
    )
    if isinstance(profile["schema_version"], bool) or profile["schema_version"] != 1:
        raise ValueError(f"unsupported schema_version: {profile['schema_version']!r}")

    model = _mapping(profile["model"], "model")
    inference = _mapping(profile["inference"], "inference")
    serving = _mapping(profile["serving"], "serving")
    _require_exact_keys(model, MODEL_KEYS, "model")
    _require_exact_keys(inference, INFERENCE_KEYS, "inference")
    _require_exact_keys(serving, SERVING_KEYS, "serving")

    for key in ("id", "revision", "served_name"):
        _require_string(model[key], f"model.{key}")
    _require_bool(inference["enable_thinking"], "inference.enable_thinking")
    _require_number(inference["temperature"], "inference.temperature", minimum=0)
    _require_number(
        inference["top_p"],
        "inference.top_p",
        minimum=0,
        maximum=1,
        minimum_inclusive=False,
    )
    _require_int(inference["top_k"], "inference.top_k", minimum=0)
    for key in ("max_tokens", "history_n", "image_max"):
        _require_int(inference[key], f"inference.{key}")

    for key in ("host", "dtype"):
        _require_string(serving[key], f"serving.{key}")
    _require_int(serving["port"], "serving.port", maximum=65535)
    for key in (
        "tensor_parallel_size",
        "data_parallel_size",
        "api_server_count",
        "max_model_len",
        "max_num_seqs",
    ):
        _require_int(serving[key], f"serving.{key}")
    _require_number(
        serving["gpu_memory_utilization"],
        "serving.gpu_memory_utilization",
        minimum=0,
        maximum=1,
        minimum_inclusive=False,
    )
    for key in ("trust_remote_code", "disable_custom_all_reduce"):
        _require_bool(serving[key], f"serving.{key}")
    return profile


def build_command(profile: dict[str, Any], vllm_bin: str) -> list[str]:
    model = profile["model"]
    inference = profile["inference"]
    serving = profile["serving"]
    command = [
        vllm_bin,
        "serve",
        model["id"],
        "--revision",
        model["revision"],
        "--served-model-name",
        model["served_name"],
        "--host",
        str(serving["host"]),
        "--port",
        str(serving["port"]),
        "--dtype",
        str(serving["dtype"]),
        "--tensor-parallel-size",
        str(serving["tensor_parallel_size"]),
        "--data-parallel-size",
        str(serving["data_parallel_size"]),
        "--api-server-count",
        str(serving["api_server_count"]),
        "--max-model-len",
        str(serving["max_model_len"]),
        "--max-num-seqs",
        str(serving["max_num_seqs"]),
        "--gpu-memory-utilization",
        str(serving["gpu_memory_utilization"]),
        "--limit-mm-per-prompt",
        json.dumps({"image": inference["image_max"]}, separators=(",", ":")),
        "--default-chat-template-kwargs",
        json.dumps({"enable_thinking": inference["enable_thinking"]}, separators=(",", ":")),
    ]
    if serving["trust_remote_code"]:
        command.append("--trust-remote-code")
    if serving["disable_custom_all_reduce"]:
        command.append("--disable-custom-all-reduce")
    return command


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Deploy vLLM using one validated Qwen-CUA model profile."
    )
    parser.add_argument("config", type=Path, help="path to configs/models/*.yaml")
    parser.add_argument(
        "--vllm-bin",
        default=os.environ.get("VLLM_BIN", "vllm"),
        help="vLLM executable (default: VLLM_BIN or vllm)",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="validate and print the command only"
    )
    args = parser.parse_args(argv)

    try:
        profile = load_profile(args.config)
    except (OSError, yaml.YAMLError, ValueError) as exc:
        parser.error(str(exc))
    command = build_command(profile, args.vllm_bin)
    print(shlex.join(command), flush=True)
    if args.dry_run:
        return 0
    os.execvp(command[0], command)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
