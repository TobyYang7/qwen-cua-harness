#!/usr/bin/env python
"""Evaluate osworld_eval CUA agents against the CUA-Gym browser sandbox.

This runner wires the harness agents in ``mm_agents/`` (multiple CUA model
families) onto ``utils/cua_env.py::CUAGymEnv`` -- the in-process asyncio browser
sandbox used by training -- and scores them with the task bundles' reward.py.

Flow per episode (one CUAGymEnv instance == one episode, per its docstring):
  reset(bundle) -> screenshot -> [agent.predict(screenshot) -> sanitize +
  canonicalize pixel pyautogui to the canonical 0..999 grid -> env.step(JSON)]
  loop until DONE/FAIL/done-flag/step-cap -> reward.

Agents are synchronous blocking implementations, so ``predict`` is called via
``asyncio.to_thread``. Concurrency is bounded by a semaphore (``--num_envs``);
episodes share the process-global headless-chromium pool (``CUA_BROWSER_POOL``).

Nothing existing is modified. See ``cuagym_adapters.py`` for the action
sanitiser and per-agent construction. Resumable: a task whose
``result_dir/<task_id>/result.json`` exists is skipped.
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import concurrent.futures
import json
import logging
import os
import sys
import time
import traceback
from typing import Any, Dict, List, Optional, Tuple

# ---- path wiring: CUA-Gym environment code is an external dependency, while
# this runner and its Qwen-CUA adapter live in qwen-cua-harness.
_HERE = os.path.dirname(os.path.abspath(__file__))
_HARNESS_ROOT = os.path.dirname(os.path.dirname(_HERE))
_REPO_ROOT = os.path.abspath(
    os.environ.get("QWEN_CUA_ENV_ROOT", os.path.join(_HARNESS_ROOT, "../.."))
)
for _p in (_REPO_ROOT, _HERE, os.path.join(_HARNESS_ROOT, "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CUA-Gym evaluation runner for osworld_eval agents.")
    p.add_argument(
        "--agent_type",
        required=True,
        choices=["qwencua"],
    )
    p.add_argument("--model", required=True, help="Served model name.")
    p.add_argument("--base_url", default="http://127.0.0.1:8901/v1")
    p.add_argument("--api_key", default="dummy")
    p.add_argument(
        "--tasks",
        default=os.path.join(_REPO_ROOT, "data", "val.jsonl"),
        help="Path to the task jsonl (slime format rows with a metadata bundle).",
    )
    p.add_argument(
        "--urls_json",
        default=os.environ.get("CUA_URLS_JSON"),
        help="Mock app_type -> base_url map JSON (defaults to $CUA_URLS_JSON).",
    )
    p.add_argument("--result_dir", required=True)
    p.add_argument("--max_steps", type=int, default=30, help="Env step cap per episode.")
    p.add_argument(
        "--episode_timeout_seconds",
        type=float,
        default=900.0,
        help="Wall-clock timeout for one episode; <=0 disables the timeout.",
    )
    p.add_argument("--num_envs", type=int, default=8, help="Concurrent episodes.")
    p.add_argument(
        "--max_agent_steps",
        type=int,
        default=None,
        help="Agent turn cap per episode (defaults to --max_steps).",
    )
    p.add_argument("--temperature", type=float, default=None, help="Overrides the agent default.")
    p.add_argument("--top_p", type=float, default=None, help="Overrides the agent default.")
    p.add_argument("--top_k", type=int, default=None, help="Overrides the agent default.")
    p.add_argument("--min_p", type=float, default=None, help="Overrides the agent default.")
    p.add_argument("--presence_penalty", type=float, default=None)
    p.add_argument("--repetition_penalty", type=float, default=None)
    p.add_argument(
        "--enable_thinking",
        action="store_true",
        help="Enable chat-template thinking for agents that support it (qwen35_xml, qwen3vl).",
    )
    p.add_argument(
        "--preserve_thinking",
        action="store_true",
        help="Replay historical Qwen thinking blocks (recommended by Qwen3.6 for agents).",
    )
    p.add_argument("--max_tokens", type=int, default=1500)
    p.add_argument("--history_n", type=int, default=5)
    p.add_argument(
        "--max_recent_images",
        type=int,
        default=5,
        help="Maximum screenshots in one model request, including the current observation.",
    )
    p.add_argument(
        "--qwencua_history_n",
        type=int,
        default=50,
        help=(
            "qwencua only. Turns kept in the message list, upstream default 50. "
            "Distinct from --history_n: the qwencua contract keeps a turn's text "
            "even after its screenshot falls outside --max_recent_images, so this "
            "is a text budget, not an image budget."
        ),
    )
    p.add_argument(
        "--context_memory",
        action="store_true",
        help="Enable task-local evolving context managed by the meta context skill.",
    )
    p.add_argument("--context_max_items", type=int, default=8)
    p.add_argument("--context_max_chars", type=int, default=6000)
    p.add_argument("--limit", type=int, default=None, help="Only run the first N tasks (smoke).")
    p.add_argument("--browser_pool", type=int, default=None, help="Override CUA_BROWSER_POOL.")
    args = p.parse_args()
    if args.max_agent_steps is None:
        args.max_agent_steps = args.max_steps
    return args


def load_bundles(tasks_path: str) -> List[Dict[str, Any]]:
    """Read a jsonl of slime rows and reconstruct task_loader-shaped bundles."""
    path = tasks_path
    if not os.path.isabs(path) and not os.path.exists(path):
        alt = os.path.join(_REPO_ROOT, path)
        if os.path.exists(alt):
            path = alt
    bundles: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            md = row.get("metadata", row)
            bundles.append(
                {
                    "task_id": md["task_id"],
                    "app_type": md["app_type"],
                    "instruction": md["instruction"],
                    "task_dir": md.get("task_dir"),
                    "reward_py_path": md.get("reward_py_path"),
                    "initial_setup_path": md.get("initial_setup_path"),
                }
            )
    return bundles


def _atomic_write_bytes(path: str, payload: bytes) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "wb") as f:
        f.write(payload)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _atomic_write_json(path: str, payload: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, default=str)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _clone_ui_metadata(env: Any) -> Dict[str, Any]:
    """Snapshot verifier-relevant CUA-Gym state associated with a screenshot."""
    try:
        return copy.deepcopy(getattr(env, "last_current_state", {}) or {})
    except Exception:
        return json.loads(json.dumps(getattr(env, "last_current_state", {}) or {}, default=str))


async def _capture_observation(
    env: Any,
    episode_dir: str,
    index: int,
) -> Tuple[Optional[bytes], Optional[str], Dict[str, Any]]:
    screenshot = await env.screenshot()
    if screenshot is None:
        return None, None, {}
    relative_path = f"images/step_{index:03d}.png"
    _atomic_write_bytes(os.path.join(episode_dir, relative_path), screenshot)
    return screenshot, relative_path, _clone_ui_metadata(env)


# --------------------------------------------------------------------------- #
# Episode driver
# --------------------------------------------------------------------------- #
async def _predict_with_retry(agent, instruction: str, screenshot: bytes, retries: int = 2):
    """Call the blocking agent.predict off-thread; retry transient failures."""
    last_err: Optional[Exception] = None
    for attempt in range(retries + 1):
        try:
            return await asyncio.to_thread(agent.predict, instruction, {"screenshot": screenshot})
        except Exception as e:  # noqa: BLE001
            last_err = e
    raise last_err  # type: ignore[misc]


async def _score_current_state(env: Any) -> Tuple[float, bool]:
    """Run the task verifier now and return normalized [0, 1] reward."""
    raw_reward, success = await env._compute_reward()
    return float(raw_reward) / 10.0, bool(success)


async def run_episode(
    bundle: Dict[str, Any],
    args,
    mock_app_urls: Dict[str, str],
    logger: logging.Logger,
) -> Dict[str, Any]:
    """Drive a single episode end to end. Never raises: errors -> score 0 record."""
    # Import here so a broken optional dependency for one agent family never
    # blocks the whole run at import time.
    from utils.cua_env import CUAGymEnv
    from cuagym_adapters import (
        build_agent,
        normalize_actions,
        png_size,
        sanitize_and_canonicalize,
    )

    task_id = bundle["task_id"]
    app_type = bundle["app_type"]
    instruction = bundle["instruction"]
    episode_dir = os.path.join(args.result_dir, task_id)

    result: Dict[str, Any] = {
        "task_id": task_id,
        "app_type": app_type,
        "instruction": instruction,
        "agent_type": args.agent_type,
        "model": args.model,
        "score": 0.0,
        "success": False,
        "steps": 0,
        "invalid_actions": 0,
        "agent_declared_fail": False,
        "step_records": [],
        "trajectory_path": "trajectory.json",
        "error": None,
    }
    trajectory: Dict[str, Any] = {
        "task": instruction,
        "platform": "web",
        "application": app_type,
        "task_id": task_id,
        "model": args.model,
        "steps": [],
        "correct": False,
        "score": 0.0,
        "error": None,
    }
    result["_trajectory"] = trajectory

    env: Any = None
    spec: Any = None
    try:
        spec = build_agent(args.agent_type, args)
        spec.reset(logger)

        env = CUAGymEnv(mock_app_urls=mock_app_urls, max_steps=args.max_steps)
        _obs_text, info = await env.reset(bundle)

        screenshot, observation_path, ui_metadata = await _capture_observation(
            env, episode_dir, 0
        )
        if screenshot is None:
            result["error"] = "initial CUA-Gym screenshot unavailable"
            return result

        done = False
        for turn in range(args.max_agent_steps):
            cw, ch = spec.coord_dims if spec.coord_dims else png_size(screenshot)

            rec: Dict[str, Any] = {
                "turn": turn,
                "agent_response": "",
                "raw_actions": None,
                "canonical_code": [],
                "action_result": "",
                "invalid_action": False,
                "step_reward": 0.0,
                "substep_rewards": [],
                "notes": [],
            }
            try:
                response, raw_actions = await _predict_with_retry(
                    spec.agent, instruction, screenshot
                )
            except Exception as e:  # noqa: BLE001
                rec["invalid_action"] = True
                rec["action_result"] = (
                    f"agent.predict failed: {type(e).__name__}: {e}"
                )
                result["invalid_actions"] += 1
                result["step_records"].append(rec)
                result["error"] = rec["action_result"]
                trajectory["steps"].append(
                    {
                        "observation": observation_path,
                        "ui_metadata": ui_metadata,
                        "action": "(agent.predict failed)",
                        "next_observation": observation_path,
                        "reward": 0.0,
                        "model_response": "",
                        "error": rec["action_result"],
                    }
                )
                break

            # Trajectories are training/debug artifacts: retain the complete
            # thinking + XML response and complete action, not display snippets.
            rec["agent_response"] = str(response)
            try:
                rec["raw_actions"] = [str(a) for a in (raw_actions or [])]
            except Exception:
                rec["raw_actions"] = str(raw_actions)

            norm = normalize_actions(raw_actions)
            wait_seconds = 0.0
            turn_invalid = False
            step_reward: Optional[float] = None
            step_success = False

            if not norm or all(n.kind == "noop" for n in norm):
                rec["invalid_action"] = True
                rec["action_result"] = "agent produced no actionable output"
                result["invalid_actions"] += 1
            else:
                for na in norm:
                    if na.kind == "noop":
                        continue
                    if na.kind == "wait":
                        wait_seconds = min(max(na.seconds, 0.0), 10.0)
                        rec["canonical_code"].append(f"wait({wait_seconds})")
                        rec["action_result"] = "WAIT"
                        break
                    if na.kind in ("done", "fail"):
                        obs_text, raw_reward, d, sinfo = await env.step(
                            '{"type": "done"}'
                        )
                        rec["canonical_code"].append(na.kind.upper())
                        rec["action_result"] = (
                            f"terminate({na.kind}) reward={raw_reward} "
                            f"premature={bool(sinfo.get('premature_done'))}"
                        )
                        if na.kind == "fail":
                            result["agent_declared_fail"] = True
                        step_reward = float(raw_reward) / 10.0
                        step_success = bool(sinfo.get("won", False))
                        rec["substep_rewards"].append(step_reward)
                        done = bool(d)
                        break

                    canonical, notes = sanitize_and_canonicalize(
                        na.code,
                        cw,
                        ch,
                        coordinate_space=getattr(
                            spec, "coordinate_space", "pixels"
                        ),
                    )
                    rec["notes"].extend(notes)
                    if canonical is None:
                        rec["canonical_code"].append(None)
                        turn_invalid = True
                        continue
                    obs_text, raw_reward, d, sinfo = await env.step(
                        json.dumps({"type": "pyautogui", "code": canonical})
                    )
                    rec["canonical_code"].append(canonical)
                    rec["action_result"] = str(obs_text)
                    if d:
                        sub_reward = float(raw_reward) / 10.0
                        sub_success = bool(sinfo.get("won", False))
                    else:
                        # Base CUAGymEnv only returns terminal reward from
                        # step(). Run the same verifier after every action so
                        # the stored trajectory contains genuine step reward.
                        sub_reward, sub_success = await _score_current_state(env)
                    rec["substep_rewards"].append(sub_reward)
                    step_reward, step_success = sub_reward, sub_success
                    if d:
                        done = True
                        break

            if wait_seconds:
                await asyncio.sleep(wait_seconds)

            if step_reward is None:
                step_reward, step_success = await _score_current_state(env)
                rec["substep_rewards"].append(step_reward)
            rec["step_reward"] = step_reward
            result["score"] = step_reward
            result["success"] = step_success

            if turn_invalid:
                rec["invalid_action"] = True
                result["invalid_actions"] += 1
            result["step_records"].append(rec)

            next_screenshot, next_path, next_ui_metadata = (
                await _capture_observation(env, episode_dir, turn + 1)
            )
            action_parts = [
                str(code) for code in rec["canonical_code"] if code is not None
            ]
            rendered_action = "\n".join(action_parts)
            if not rendered_action:
                rendered_action = (
                    "\n".join(rec["raw_actions"])
                    if isinstance(rec["raw_actions"], list)
                    else str(rec["raw_actions"] or "(no action)")
                )
            trajectory["steps"].append(
                {
                    "observation": observation_path,
                    "ui_metadata": ui_metadata,
                    "action": rendered_action,
                    "next_observation": next_path,
                    "reward": step_reward,
                    "model_response": rec["agent_response"],
                    "raw_actions": rec["raw_actions"],
                }
            )

            if next_screenshot is None:
                result["error"] = "CUA-Gym screenshot unavailable after action"
                break
            screenshot = next_screenshot
            observation_path = next_path
            ui_metadata = next_ui_metadata
            if done:
                break

        # Always grade the final live state, including runs that reached the
        # agent-turn cap without emitting terminate.
        final_score, final_success = await _score_current_state(env)
        result["score"] = final_score
        result["success"] = final_success
        result["steps"] = len(result["step_records"])
        return result

    except Exception as e:  # noqa: BLE001
        result["error"] = f"{type(e).__name__}: {e}"
        result["steps"] = len(result["step_records"])
        logger.warning("episode %s crashed: %s\n%s", task_id, e, traceback.format_exc())
        return result
    finally:
        if spec is not None:
            export_context = getattr(spec.agent, "export_context", None)
            if callable(export_context):
                try:
                    context_state = export_context()
                    if context_state is not None:
                        result["context_memory"] = context_state
                        trajectory["context_memory"] = context_state
                except Exception as export_exc:  # noqa: BLE001
                    logger.warning("context export failed for %s: %s", task_id, export_exc)
            export_diagnostics = getattr(
                spec.agent,
                "export_context_diagnostics",
                None,
            )
            if callable(export_diagnostics):
                try:
                    context_diagnostics = export_diagnostics()
                    result["context_diagnostics"] = context_diagnostics
                    trajectory["context_diagnostics"] = context_diagnostics
                except Exception as export_exc:  # noqa: BLE001
                    logger.warning(
                        "context diagnostics export failed for %s: %s",
                        task_id,
                        export_exc,
                    )
        trajectory["correct"] = bool(result.get("success", False))
        trajectory["score"] = float(result.get("score", 0.0) or 0.0)
        trajectory["error"] = result.get("error")
        if env is not None:
            try:
                await env.close()
            except Exception:
                pass


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def _write_result(result_dir: str, result: Dict[str, Any]) -> None:
    task_dir = os.path.join(result_dir, result["task_id"])
    trajectory = result.pop("_trajectory", None)
    if trajectory is not None:
        _atomic_write_json(
            os.path.join(task_dir, "trajectory.json"),
            trajectory,
        )
    _atomic_write_json(os.path.join(task_dir, "result.json"), result)


def _load_existing(result_dir: str, task_id: str) -> Optional[Dict[str, Any]]:
    path = os.path.join(result_dir, task_id, "result.json")
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None
    return None


def build_summary(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(results)
    total_score = sum(r.get("score", 0.0) for r in results)
    n_success = sum(1 for r in results if r.get("success"))
    total_steps = sum(r.get("steps", 0) for r in results)
    total_invalid = sum(r.get("invalid_actions", 0) for r in results)

    by_app: Dict[str, Dict[str, Any]] = {}
    for r in results:
        app = r.get("app_type", "?")
        g = by_app.setdefault(app, {"n": 0, "score_sum": 0.0, "success": 0})
        g["n"] += 1
        g["score_sum"] += r.get("score", 0.0)
        g["success"] += 1 if r.get("success") else 0
    for app, g in by_app.items():
        g["mean_score"] = g["score_sum"] / g["n"] if g["n"] else 0.0
        g["success_rate"] = g["success"] / g["n"] if g["n"] else 0.0

    return {
        "n": n,
        "mean_score": total_score / n if n else 0.0,
        "success_rate": n_success / n if n else 0.0,
        "avg_steps": total_steps / n if n else 0.0,
        "invalid_action_rate": total_invalid / total_steps if total_steps else 0.0,
        "total_steps": total_steps,
        "total_invalid_actions": total_invalid,
        "by_app_type": by_app,
    }


async def main_async(args, logger: logging.Logger) -> None:
    from utils.cua_env import load_mock_app_urls

    # asyncio.to_thread otherwise tops out at Python's small default executor
    # and silently under-utilizes --num_envs. Predict calls and verifier calls
    # can both block, so provide two worker slots per environment.
    loop = asyncio.get_running_loop()
    loop.set_default_executor(
        concurrent.futures.ThreadPoolExecutor(
            max_workers=max(32, args.num_envs * 2),
            thread_name_prefix="cuagym",
        )
    )

    if args.urls_json:
        mock_app_urls = load_mock_app_urls(args.urls_json)
    else:
        raise SystemExit("No mock URL map: pass --urls_json or set CUA_URLS_JSON.")

    bundles = load_bundles(args.tasks)
    if args.limit is not None:
        bundles = bundles[: args.limit]
    os.makedirs(args.result_dir, exist_ok=True)

    total = len(bundles)
    sem = asyncio.Semaphore(args.num_envs)
    results: List[Dict[str, Any]] = [None] * total  # type: ignore[list-item]
    counter = {"done": 0}

    async def worker(idx: int, bundle: Dict[str, Any]) -> None:
        existing = _load_existing(args.result_dir, bundle["task_id"])
        if existing is not None:
            results[idx] = existing
            counter["done"] += 1
            print(
                f"[{counter['done']}/{total}] {bundle['task_id']} {bundle['app_type']} "
                f"score={existing.get('score', 0.0):.2f} steps={existing.get('steps', 0)} (cached)"
            )
            return
        async with sem:
            try:
                episode = run_episode(bundle, args, mock_app_urls, logger)
                if args.episode_timeout_seconds > 0:
                    result = await asyncio.wait_for(
                        episode, timeout=args.episode_timeout_seconds
                    )
                else:
                    result = await episode
            except TimeoutError:
                task_id = bundle["task_id"]
                app_type = bundle["app_type"]
                instruction = bundle["instruction"]
                error = (
                    "episode timed out after "
                    f"{args.episode_timeout_seconds:g} seconds"
                )
                logger.error("%s: %s", task_id, error)
                result = {
                    "task_id": task_id,
                    "app_type": app_type,
                    "instruction": instruction,
                    "agent_type": args.agent_type,
                    "model": args.model,
                    "score": 0.0,
                    "success": False,
                    "steps": 0,
                    "invalid_actions": 0,
                    "agent_declared_fail": False,
                    "step_records": [],
                    "trajectory_path": "trajectory.json",
                    "error": error,
                    "_trajectory": {
                        "task": instruction,
                        "platform": "web",
                        "application": app_type,
                        "task_id": task_id,
                        "model": args.model,
                        "steps": [],
                        "correct": False,
                        "score": 0.0,
                        "error": error,
                    },
                }
        _write_result(args.result_dir, result)
        results[idx] = result
        counter["done"] += 1
        print(
            f"[{counter['done']}/{total}] {result['task_id']} {result['app_type']} "
            f"score={result.get('score', 0.0):.2f} steps={result.get('steps', 0)}"
            + (f" ERROR={result['error']}" if result.get("error") else "")
        )

    await asyncio.gather(*(worker(i, b) for i, b in enumerate(bundles)))

    results = [r for r in results if r is not None]
    summary = build_summary(results)
    with open(os.path.join(args.result_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n=== summary ===")
    print(
        f"n={summary['n']}  mean_score={summary['mean_score']:.3f}  "
        f"success_rate={summary['success_rate']:.3f}  avg_steps={summary['avg_steps']:.1f}  "
        f"invalid_action_rate={summary['invalid_action_rate']:.3f}"
    )
    for app, g in sorted(summary["by_app_type"].items()):
        print(
            f"  {app:24s} n={g['n']:3d}  mean_score={g['mean_score']:.3f}  "
            f"success_rate={g['success_rate']:.3f}"
        )


def main() -> None:
    args = parse_args()

    # These must be set before utils.cua_env is imported (module-level reads).
    os.environ.setdefault("CUA_USE_SCREENSHOTS", "1")
    os.environ["CUA_MAX_STEPS"] = str(args.max_steps)
    pool = args.browser_pool or max(args.num_envs, int(os.environ.get("CUA_BROWSER_POOL", "0") or 0))
    os.environ["CUA_BROWSER_POOL"] = str(max(pool, 1))
    # Endpoint for the qwen/evocua agents, which read these at call time.
    os.environ["OPENAI_BASE_URL"] = args.base_url
    os.environ["OPENAI_API_KEY"] = args.api_key

    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    logger = logging.getLogger("cuagym_runner")
    logger.setLevel(logging.INFO)

    t0 = time.time()
    asyncio.run(main_async(args, logger))
    print(f"\ntotal wall time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
