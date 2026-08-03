"""Script to run end-to-end evaluation on the benchmark.
Utils and basic architecture credit to https://github.com/web-arena-x/webarena/blob/main/run.py.
"""

import argparse
import ast
import copy
import datetime
import importlib
import json
import logging
import math
import os
import re
import sys
import time
from functools import partial
from typing import Optional, Dict, Any
from multiprocessing import Pool

import requests
from openai import APIConnectionError, RateLimitError
from tqdm import tqdm

import lib_run_single
from desktop_env.desktop_env import MAX_RETRIES, DesktopEnv

# This vendored runner is intentionally wired to the Qwen-CUA adapter in this
# repository. OSWorld itself (DesktopEnv, evaluators and task definitions)
# remains an environment dependency supplied through PYTHONPATH.

# Almost deprecated since it's not multi-env, use run_multienv_*.py instead

#  Logger Configs {{{ #
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

datetime_str: str = datetime.datetime.now().strftime("%Y%m%d@%H%M%S")

file_handler = logging.FileHandler(os.path.join("logs", "normal-{:}.log".format(datetime_str)), encoding="utf-8")
debug_handler = logging.FileHandler(os.path.join("logs", "debug-{:}.log".format(datetime_str)), encoding="utf-8")
stdout_handler = logging.StreamHandler(sys.stdout)
sdebug_handler = logging.FileHandler(os.path.join("logs", "sdebug-{:}.log".format(datetime_str)), encoding="utf-8")

file_handler.setLevel(logging.INFO)
debug_handler.setLevel(logging.DEBUG)
stdout_handler.setLevel(logging.INFO)
sdebug_handler.setLevel(logging.DEBUG)

formatter = logging.Formatter(
    fmt="\x1b[1;33m[%(asctime)s \x1b[31m%(levelname)s \x1b[32m%(module)s/%(lineno)d-%(processName)s\x1b[1;33m] \x1b[0m%(message)s"
)
file_handler.setFormatter(formatter)
debug_handler.setFormatter(formatter)
stdout_handler.setFormatter(formatter)
sdebug_handler.setFormatter(formatter)

stdout_handler.addFilter(logging.Filter("desktopenv"))
sdebug_handler.addFilter(logging.Filter("desktopenv"))

logger.addHandler(file_handler)
logger.addHandler(debug_handler)
logger.addHandler(stdout_handler)
logger.addHandler(sdebug_handler)
#  }}} Logger Configs #

logger = logging.getLogger("desktopenv.experiment")


def convert_openai_tools_to_anthropic(tools):
    anthropic_tools = []
    for tool in tools or []:
        if tool.get("type") != "function":
            continue
        function = tool.get("function", {})
        anthropic_tools.append(
            {
                "name": function.get("name", ""),
                "description": function.get("description", ""),
                "input_schema": function.get("parameters", {"type": "object", "properties": {}}),
            }
        )
    return anthropic_tools


def _convert_content_item_to_anthropic(item):
    item_type = item.get("type")
    if item_type == "text":
        text_content = item.get("text", "")
        if text_content is None:
            text_content = ""
        if text_content == "":
            return None
        return {"type": "text", "text": text_content}
    if item_type == "image_url":
        url = item.get("image_url", {}).get("url", "")
        if not url.startswith("data:"):
            raise ValueError("Only support data URL image.")
        header, data = url.split(",", 1)
        media_type = header.split(";")[0].removeprefix("data:") or "image/png"
        return {
            "type": "image",
            "source": {"type": "base64", "media_type": media_type, "data": data},
        }
    if item_type == "tool_use":
        return {
            "type": "tool_use",
            "id": item.get("id", ""),
            "name": item.get("name", "computer"),
            "input": item.get("input", {}),
        }
    raise ValueError(f"Unsupported content type: {item_type}")


def convert_openai_to_anthropic_messages(messages):
    systems = []
    chat_messages = []

    for message in messages:
        role = message.get("role")
        content = message.get("content", "")

        if role == "system":
            if isinstance(content, str) and content:
                systems.append(content)
            elif isinstance(content, list):
                for item in content:
                    block = _convert_content_item_to_anthropic(item)
                    if block and block["type"] == "text":
                        systems.append(block["text"])
            continue

        if role == "tool":
            tool_result_content = []
            if isinstance(content, str):
                if content:
                    tool_result_content.append({"type": "text", "text": content})
            elif isinstance(content, list):
                for item in content:
                    block = _convert_content_item_to_anthropic(item)
                    if block is not None:
                        tool_result_content.append(block)
            elif content is not None:
                tool_result_content.append({"type": "text", "text": str(content)})

            chat_messages.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": message.get("tool_call_id", ""),
                            "content": tool_result_content or [{"type": "text", "text": ""}],
                        }
                    ],
                }
            )
            continue

        parts = []
        if isinstance(content, str):
            if content:
                parts.append({"type": "text", "text": content})
        elif isinstance(content, list):
            for item in content:
                block = _convert_content_item_to_anthropic(item)
                if block is not None:
                    parts.append(block)
        elif content is not None:
            parts.append({"type": "text", "text": str(content)})

        if role == "assistant":
            for tool_call in message.get("tool_calls", []):
                if tool_call.get("type") != "function":
                    continue
                function = tool_call.get("function", {})
                arguments = function.get("arguments", "{}")
                try:
                    parsed_arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    parsed_arguments = {}
                parts.append(
                    {
                        "type": "tool_use",
                        "id": tool_call.get("id", ""),
                        "name": function.get("name", "computer"),
                        "input": parsed_arguments,
                    }
                )

        if not parts:
            continue

        if role not in ("user", "assistant"):
            role = "user"
        chat_messages.append({"role": role, "content": parts})

    payload: Dict[str, Any] = {"messages": chat_messages}
    if systems:
        payload["system"] = "\n".join(systems)
    return payload


def _normalize_base_url(base_url: str) -> str:
    normalized = (base_url or "https://api.openai.com/v1").rstrip("/")
    if normalized.endswith("/chat/completions"):
        normalized = normalized[: -len("/chat/completions")]
    elif normalized.endswith("/messages"):
        normalized = normalized[: -len("/messages")]
    return normalized




def convert_anthropic_response_to_openai_message(result):
    content_blocks = []
    tool_calls = []

    for block in result.get("content", []):
        block_type = block.get("type")
        if block_type == "text":
            content_blocks.append({"type": "text", "text": block.get("text", "")})
        elif block_type == "thinking":
            content_blocks.append({"type": "thinking", "thinking": block.get("thinking", "")})
        elif block_type == "tool_use":
            tool_call_id = block.get("id", "")
            tool_name = block.get("name", "computer")
            tool_input = block.get("input", {})
            content_blocks.append(
                {
                    "type": "tool_use",
                    "id": tool_call_id,
                    "name": tool_name,
                    "input": tool_input,
                }
            )
            tool_calls.append(
                {
                    "id": tool_call_id,
                    "type": "function",
                    "function": {
                        "name": tool_name,
                        "arguments": json.dumps(tool_input, ensure_ascii=True),
                    },
                }
            )

    message = {"role": "assistant", "content": content_blocks}
    if tool_calls:
        message["tool_calls"] = tool_calls
    return message


def config() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run end-to-end evaluation on the benchmark")

    # environment config
    parser.add_argument("--path_to_vm", type=str)
    parser.add_argument(
        "--provider_name",
        type=str,
        default="docker",
        help="Virtualization provider (vmware, docker, aws, azure, gcp, virtualbox)",
    )
    parser.add_argument("--headless", action="store_true", default=True, help="Run in headless machine")
    parser.add_argument("--action_space", type=str, default="claude_computer_use", help="Action type")
    parser.add_argument(
        "--observation_type",
        choices=["screenshot", "a11y_tree", "screenshot_a11y_tree", "som"],
        default="screenshot",
        help="Observation type",
    )
    parser.add_argument("--screen_width", type=int, default=1920)
    parser.add_argument("--screen_height", type=int, default=1080)
    parser.add_argument("--sleep_after_execution", type=float, default=1.0)
    parser.add_argument("--max_steps", type=int, default=50)

    # agent config
    parser.add_argument(
        "--agent_type",
        type=str,
        default="qwencua",
        choices=["qwencua"],
        help="Qwen-CUA agent contract provided by this repository",
    )
    parser.add_argument(
        "--qwencua_history_n",
        type=int,
        default=50,
        help=(
            "qwencua only. Turns kept in the message list, upstream default 50. "
            "A turn outside --only_n_most_recent_images keeps its response and "
            "gets a collapsed-screenshot placeholder, so this is a text budget."
        ),
    )
    parser.add_argument("--history_n", type=int, default=5, help="Number of history steps to replay (qwen35_xml agent)")
    parser.add_argument("--enable_thinking", action="store_true", default=False, help="Enable thinking mode (qwen35_xml agent; default off)")
    parser.add_argument(
        "--disable_proxy",
        action="store_true",
        default=False,
        help=(
            "Run tasks that request a proxy over the VM's direct network instead. "
            "Use only for explicit retries when no valid proxy pool is available."
        ),
    )
    parser.add_argument("--max_trajectory_length", type=int, default=3)
    parser.add_argument("--test_config_base_dir", type=str, default="evaluation_examples/examples")
    parser.add_argument("--cache_dir", type=str, default="cache", help="Local OSWorld cache directory")
    parser.add_argument("--only_n_most_recent_images", type=int, default=5, help="Keep only the N most recent images in conversation history to manage token usage")
    parser.add_argument("--min_removal_threshold", type=int, default=5, help="Remove old screenshots in chunks of this size")

    # lm config
    parser.add_argument("--model", type=str, default="aws:claude-sonnet-4-5-20250929")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--max_tokens", type=int, default=4096)
    parser.add_argument("--stop_token", type=str, default=None)
    parser.add_argument("--relative_coordinate", action="store_true", default=False, help="Use relative coordinate for GUI actions")
    parser.add_argument("--relative_size", type=int, nargs=2, default=(1000, 1000), help="Relative coordinate for GUI actions")
    parser.add_argument("--image_size", type=int, nargs=2, default=(1280, 720), help="Resize image to this size before sending to LLM")

    # example config
    parser.add_argument("--domain", type=str, default="all")
    parser.add_argument("--test_all_meta_path", type=str, default="evaluation_examples/test_nogdrive.json")

    # aws config
    parser.add_argument(
        "--region", type=str, default="us-east-1", help="AWS region for the VM"
    )
    parser.add_argument(
        "--client_password", type=str, default="", help="Client password"
    )

    # logging related
    parser.add_argument("--result_dir", type=str, default="./results")
    parser.add_argument("--recording_dir", type=str, default="./recordings", help="Record training trajectory for each example")
    parser.add_argument("--postfix", type=str, default="", help="Postfix for model name")

    # parallel number
    parser.add_argument("--num_workers", type=int, default=1, help="Number of parallel workers")
    parser.add_argument("--rerun_failed", action="store_true", default=False, help="Rerun tasks whose existing result is <= 0.1")
    args = parser.parse_args()

    return args


def _score_from_result_text(text: str) -> float:
    res = text.strip()
    if res.lower() == "true":
        return 1.0
    if res.lower() == "false":
        return 0.0
    return float(res)


def _example_result_dir(args, domain: str, example_id: str) -> str:
    return os.path.join(
        args.result_dir,
        args.action_space,
        args.observation_type,
        args.model,
        domain,
        example_id,
    )


def _write_failure_result(example_result_dir: str, message: str) -> None:
    os.makedirs(example_result_dir, exist_ok=True)
    with open(os.path.join(example_result_dir, "traj.jsonl"), "a", encoding="utf-8") as f:
        f.write(json.dumps({"Error": message}, ensure_ascii=False) + "\n")
    with open(os.path.join(example_result_dir, "result.txt"), "w", encoding="utf-8") as f:
        f.write("0.0\n")

def strip_ansi_codes(escaped_text: str) -> str:
    """"""
    decoded_text = escaped_text.encode('utf-8').decode('unicode_escape')

    ansi_escape = re.compile(r'\x1B\[[0-9;]*[A-Za-z]')
    clean_text = ansi_escape.sub('', decoded_text)

    return clean_text

def _worker_run(task):
    domain, example_id, args = task
    logger = logging.getLogger("desktopenv.experiment")
    try:
        config_file = os.path.join(args.test_config_base_dir, f"{domain}/{example_id}.json")
        with open(config_file, "r", encoding="utf-8") as f:
            example = json.load(f)
        instruction = example["instruction"]

        env = DesktopEnv(
            provider_name=args.provider_name,
            region=args.region,
            client_password=args.client_password,
            path_to_vm=args.path_to_vm,
            action_space=args.action_space,
            screen_size=(args.screen_width, args.screen_height),
            cache_dir=args.cache_dir,
            headless=args.headless,
            os_type="Ubuntu",
            require_a11y_tree=args.observation_type in ["a11y_tree", "screenshot_a11y_tree", "som"],
            enable_proxy=not args.disable_proxy,
        )
        if args.agent_type == "qwen35_xml":
            # Official-XML (ARM C) Qwen3.5 profile. Self-contained LLM call (no
            # tools/tool_choice); decodes grid999 -> real pixels itself, so the
            # env runs with relative_coordinate=False (no double-scaling).
            Qwen35XMLAgent = importlib.import_module("mm_agents.qwen35_xml_agent").Qwen35XMLAgent
            agent = Qwen35XMLAgent(
                platform="Ubuntu",
                model=args.model,
                action_space=args.action_space,
                observation_type=args.observation_type,
                screen_size=(args.screen_width, args.screen_height),
                history_n=args.history_n,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_p=args.top_p if args.top_p is not None else 0.9,
                enable_thinking=args.enable_thinking,
                relative_coordinate=False,
            )
        elif args.agent_type == "qwencua":
            # Upstream Qwen-CUA reference contract. Like qwen35_xml it decodes
            # grid999 to real pixels itself, so relative_coordinate stays False.
            # surface="desktop" swaps upstream's browser-only tool description,
            # which is the one sentence that is wrong on an OSWorld VM.
            from qwen_cua.eval.osworld import QwenCUAAgent
            agent = QwenCUAAgent(
                model=args.model,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_p=args.top_p if args.top_p is not None else 0.95,
                top_k=args.top_k,
                enable_thinking=args.enable_thinking,
                history_n=args.qwencua_history_n,
                image_max=args.only_n_most_recent_images,
                surface="desktop",
            )
            agent.relative_coordinate = False
        else:
            raise ValueError(
                "this vendored runner supports --agent_type qwencua only; "
                "use the benchmark's own runner for other agent contracts"
            )

        example_result_dir = _example_result_dir(args, domain, example_id)
        os.makedirs(example_result_dir, exist_ok=True)

        if args.recording_dir:
            recording_dir = os.path.join(
                args.recording_dir,
                args.model + args.postfix,
                domain,
                example_id
            )
            os.makedirs(recording_dir, exist_ok=True)
        else:
            recording_dir = None

        local_scores = []
        try:
            lib_run_single.run_single_example_autoglm(
                agent=agent,
                env=env,
                example=example,
                max_steps=args.max_steps,
                instruction=instruction,
                args=args,
                example_result_dir=example_result_dir,
                recording_dir=recording_dir,
                scores=local_scores
            )
        except Exception as e:
            logger.error(f"[并发任务异常] {domain}/{example_id}: {e}")
            if hasattr(env, "controller") and env.controller is not None:
                try:
                    env.controller.end_recording(os.path.join(example_result_dir, "recording.mp4"))
                except Exception:
                    pass
            _write_failure_result(example_result_dir, f"Exception in {domain}/{example_id}: {str(e)}")
        finally:
            try:
                env.close()
            except Exception:
                pass

        score = None
        result_path = os.path.join(example_result_dir, "result.txt")
        if os.path.exists(result_path):
            try:
                with open(result_path, "r") as rf:
                    score = _score_from_result_text(rf.read())
            except Exception:
                score = 0.0
        else:
            score = 0.0
        logger.info(f"[Finish] {domain}/{example_id} score={score}")
        return (domain, example_id, score)
    except Exception as e:
        logger = logging.getLogger("desktopenv.experiment")
        logger.error(f"[Initializing Fail] {domain}/{example_id}: {e}")
        _write_failure_result(
            _example_result_dir(args, domain, example_id),
            f"Initializing Fail in {domain}/{example_id}: {str(e)}",
        )
        return (domain, example_id, 0.0)

def call_llm(messages, args):
    logger.info("Calling LLM...")
    model_lower = (args.model or "").lower()
    is_anthropic_compatible = model_lower.startswith("anthropic") or "claude" in model_lower

    proxy_url = os.environ.get("LAN_PROXY")
    proxies = None
    if proxy_url:
        logger.info("setting lan proxy for model")
        proxies = {"http": proxy_url, "https": proxy_url}

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY', '')}",
    }

    base_url = _normalize_base_url(os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    openai_data = {
        "model": args.model,
        "messages": messages,
        "tools": computer_tools,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "stream": False,
    }
    if args.top_p is not None:
        openai_data["top_p"] = args.top_p
    anthropic_payload = convert_openai_to_anthropic_messages(messages)
    anthropic_data = {
        "model": args.model.removeprefix("anthropic:"),
        **anthropic_payload,
        "tools": convert_openai_tools_to_anthropic(computer_tools),
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "stream": False,
    }
    if args.top_p is not None:
        anthropic_data["top_p"] = args.top_p

    request_plan = []
    if is_anthropic_compatible:
        request_plan.append((f"{base_url}/messages", anthropic_data, "anthropic"))
    request_plan.append((f"{base_url}/chat/completions", openai_data, "openai"))

    last_error: Exception | None = None
    for url, data, mode in request_plan:
        try:
            logger.info(f"Making request to: {url}")
            logger.info(f"Model: {args.model}")
            logger.info(f"Request mode: {mode}")
            response = requests.post(
                url,
                json=data,
                headers=headers,
                proxies=proxies,
                timeout=600.0,
            )
            logger.info(f"Status code: {response.status_code}")
            preview = response.text[:1000] if hasattr(response, "text") else ""
            if preview:
                logger.info(f"Response preview: {preview}")
            response.raise_for_status()

            result = response.json()
            logger.info("LLM called successfully.")
            if mode == "anthropic":
                return convert_anthropic_response_to_openai_message(result)
            return result["choices"][0]["message"]
        except requests.exceptions.RequestException as e:
            last_error = e
            logger.error(f"Request error via {mode}: {e}")
            if mode == request_plan[-1][2]:
                break
        except KeyError as e:
            last_error = e
            logger.error(f"Response parsing error via {mode}: missing key {e}")
            if mode == request_plan[-1][2]:
                break

    if last_error is None:
        last_error = RuntimeError("LLM request failed without a captured exception")

    error_msg = f"Unexpected error: {last_error}"
    logger.error(error_msg)
    print(f"ERROR: {error_msg}")
    raise last_error

def test(args: argparse.Namespace, test_all_meta: dict) -> None:
    # log args
    logger.info("Args: %s", args)
    tasks = []
    for domain in test_all_meta:
        for example_id in test_all_meta[domain]:
            tasks.append((domain, example_id, args))
    if not tasks:
        logger.info("No pending tasks")
        return
    logger.info(f"Starting parallel execution: {args.num_workers} processes, {len(tasks)} tasks total")

    results = []
    with Pool(processes=args.num_workers) as pool:
        for res in tqdm(pool.imap_unordered(_worker_run, tasks), total=len(tasks), desc="Parallel execution"):
            results.append(res)

    scores = [s for (_, _, s) in results if s is not None]
    if scores:
        avg = sum(scores) / len(scores)
        logger.info(f"Parallel execution completed. Average score: {avg}")
    else:
        logger.info("No scores obtained.")


def get_unfinished(action_space, use_model, observation_type, result_dir, total_file_json, rerun_failed=False):
    target_dir = os.path.join(result_dir, action_space, observation_type, use_model)

    if not os.path.exists(target_dir):
        return total_file_json

    finished = {}
    for domain in os.listdir(target_dir):
        finished[domain] = []
        domain_path = os.path.join(target_dir, domain)
        if os.path.isdir(domain_path):
            for example_id in os.listdir(domain_path):
                if example_id == "onboard":
                    continue
                example_path = os.path.join(domain_path, example_id)
                if os.path.isdir(example_path):
                    if "result.txt" not in os.listdir(example_path):
                        # empty all files under example_id
                        for file in os.listdir(example_path):
                            os.remove(os.path.join(example_path, file))

                    else:
                        # Check if result is valid
                        try:
                            result_file = os.path.join(example_path, "result.txt")
                            with open(result_file, "r") as rf:
                                score = _score_from_result_text(rf.read())

                                if score > 0.1 or not rerun_failed:
                                    finished[domain].append(example_id)
                                else:
                                    for stale_file in os.listdir(example_path):
                                        os.remove(os.path.join(example_path, stale_file))
                        except Exception:
                            # If error reading/parsing, treat as unfinished and clean up
                            for file in os.listdir(example_path):
                                os.remove(os.path.join(example_path, file))

    if not finished:
        return total_file_json

    for domain, examples in finished.items():
        if domain in total_file_json:
            total_file_json[domain] = [x for x in total_file_json[domain] if x not in examples]

    return total_file_json


def get_result(action_space, use_model, observation_type, result_dir, total_file_json):
    target_dir = os.path.join(result_dir, action_space, observation_type, use_model)
    if not os.path.exists(target_dir):
        print("New experiment, no result yet.")
        return None

    all_result = []

    for domain, example_ids in total_file_json.items():
        for example_id in example_ids:
            result_path = os.path.join(target_dir, domain, example_id, "result.txt")
            if not os.path.exists(result_path):
                continue
            try:
                with open(result_path, "r") as rf:
                    score = _score_from_result_text(rf.read())
            except Exception:
                score = 0.0
            all_result.append(score)

    if not all_result:
        print("New experiment, no result yet.")
        return None
    else:
        completed = len(all_result)
        total = sum(len(v) for v in total_file_json.values())
        print("Current Success Rate:", sum(all_result) / completed * 100, "%")
        if total:
            print("Benchmark Success Rate (missing=0):", sum(all_result) / total * 100, "%")
            print(f"Completed Tasks: {completed}/{total}")
        return all_result


if __name__ == "__main__":
    ####### The complete version of the list of examples #######
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    args = config()
    if args.client_password == "":
        if args.provider_name == "aws":
            args.client_password = "osworld-public-evaluation"
        else:
            args.client_password = "password"
    else:
        args.client_password = args.client_password

    # save args to json in result_dir/action_space/observation_type/model/args.json
    path_to_args = os.path.join(
        args.result_dir,
        args.action_space,
        args.observation_type,
        args.model,
        "args.json",
    )
    os.makedirs(os.path.dirname(path_to_args), exist_ok=True)
    with open(path_to_args, "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=4)

    with open(args.test_all_meta_path, "r", encoding="utf-8") as f:
        test_all_meta = json.load(f)

    if args.domain != "all":
        test_all_meta = {args.domain: test_all_meta[args.domain]}

    test_file_list = get_unfinished(
        args.action_space,
        args.model,
        args.observation_type,
        args.result_dir,
        copy.deepcopy(test_all_meta),
        args.rerun_failed,
    )
    left_info = ""
    for domain in test_file_list:
        left_info += f"{domain}: {len(test_file_list[domain])}\n"
    logger.info(f"Left tasks:\n{left_info}")

    get_result(
        args.action_space,
        args.model,
        args.observation_type,
        args.result_dir,
        test_all_meta,
    )
    test(args, test_file_list)
