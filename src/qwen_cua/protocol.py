from __future__ import annotations

import json
import re
from datetime import date
from typing import Any

from pydantic import ValidationError

from .actions import ComputerAction, computer_action_adapter
from .context import META_CONTEXT_SKILL

COLLAPSED_SCREENSHOT_TEXT = "This screenshot has been collapsed."
TOOL_CALL_PATTERN = re.compile(
    r"<tool_call>(.*?)</tool_call>",
    flags=re.DOTALL | re.IGNORECASE,
)
PARAMETER_PATTERN = re.compile(
    r"<parameter=([^>]+)>(.*?)</parameter>",
    flags=re.DOTALL | re.IGNORECASE,
)
FUNCTION_PATTERN = re.compile(r"<function=([^>]+)>", flags=re.IGNORECASE)


class ToolCallParseError(ValueError):
    pass


def build_tool_definition(*, context_memory: bool = False) -> dict[str, Any]:
    # Kept as an argument for API compatibility. Context is maintained by a
    # separate harness-owned updater and is deliberately absent from the action
    # schema so it cannot perturb computer-use generation.
    _ = context_memory
    actions = [
        "key",
        "key_down",
        "key_up",
        "left_mouse_down",
        "left_mouse_up",
        "type",
        "mouse_move",
        "left_click",
        "left_click_drag",
        "right_click",
        "middle_click",
        "double_click",
        "triple_click",
        "scroll",
        "hscroll",
        "screenshot",
        "wait",
        "terminate",
        "call_user",
    ]
    definition = {
        "type": "function",
        "function": {
            "name": "computer_use",
            "description": "\n".join(
                [
                    "Use a mouse and keyboard to interact with a browser.",
                    "* The coordinate system is a 1000x1000 relative grid.",
                    "* Inspect each screenshot before choosing coordinates.",
                    "* Use screenshot or wait when the interface needs time to update.",
                    "* Verify the visible result before terminating.",
                ]
            ),
            "parameters": {
                "type": "object",
                "required": ["action"],
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": actions,
                        "description": (
                            "key/key_down/key_up use keys; type/call_user use text; "
                            "mouse_move/left_click_drag use coordinate; click actions may "
                            "use coordinate; scroll/hscroll use pixels; wait uses time; "
                            "terminate uses status."
                        ),
                    },
                    "keys": {"type": "array", "items": {"type": "string"}},
                    "text": {"type": "string"},
                    "coordinate": {
                        "type": "array",
                        "items": {"type": "number"},
                        "minItems": 2,
                        "maxItems": 2,
                    },
                    "pixels": {"type": "number"},
                    "time": {"type": "number"},
                    "duration": {"type": "number"},
                    "status": {
                        "type": "string",
                        "enum": ["success", "failure"],
                    },
                },
            },
        },
    }
    return definition


def build_system_prompt(*, context_memory: bool = False) -> str:
    tool_json = json.dumps(
        build_tool_definition(context_memory=context_memory),
        ensure_ascii=False,
    )
    prompt = (
        "You are a multi-purpose intelligent assistant. Based on the user's request, "
        "use the computer tool to complete browser tasks.\n\n"
        "# Tools\n\n"
        "You have access to the following functions:\n\n"
        f"<tools>\n{tool_json}\n</tools>\n\n"
        "If you choose to call a function, reply in this exact format with no suffix:\n\n"
        "<tool_call>\n"
        "<function=computer_use>\n"
        "<parameter=action>\nleft_click\n</parameter>\n"
        "<parameter=coordinate>\n[500, 500]\n</parameter>\n"
        "</function>\n"
        "</tool_call>\n\n"
        "<IMPORTANT>\n"
        "- Function calls must contain an inner <function=computer_use> block inside "
        "<tool_call> tags.\n"
        "- Include every parameter required by the chosen action.\n"
        "- You may reason before the function call, but never write anything after it.\n"
        f"- The current date is {date.today().isoformat()}.\n"
        f"- Collapsed screenshots appear as text: {COLLAPSED_SCREENSHOT_TEXT}\n"
        "- Interact through the visible GUI only. Do not use page scripts, a terminal, "
        "developer tools, browser automation APIs, or hidden application state.\n"
        "- Verify the visible result before terminating with success.\n"
        "</IMPORTANT>"
    )
    if context_memory:
        prompt += (
            "\n\n# Active Meta Skill\n\n"
            "This stable meta skill applies to every task in this run. Task-specific "
            "state belongs in task context, never in the Skill itself.\n\n"
            f"{META_CONTEXT_SKILL}"
        )
    return prompt


def _parse_value(name: str, raw_value: str) -> Any:
    value = raw_value
    if name == "text":
        if value.startswith("\n"):
            value = value[1:]
        if value.endswith("\n"):
            value = value[:-1]
        return value
    value = value.strip()
    if value.startswith(("[", "{")):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    if name in {"pixels", "time", "duration"}:
        try:
            return float(value)
        except ValueError:
            return value
    return value


def parse_tool_calls(response: str) -> list[ComputerAction]:
    if not response.strip():
        return []
    matches = list(TOOL_CALL_PATTERN.finditer(response))
    if not matches:
        if re.search(r"<tool_call\b|<function=", response, flags=re.IGNORECASE):
            raise ToolCallParseError("incomplete or malformed tool-call markup")
        return []

    actions: list[ComputerAction] = []
    for match in matches:
        body = match.group(1)
        function_match = FUNCTION_PATTERN.search(body)
        if not function_match:
            raise ToolCallParseError("tool call is missing a function name")
        if function_match.group(1).strip() != "computer_use":
            raise ToolCallParseError(
                f"unsupported tool function: {function_match.group(1).strip()}"
            )
        params: dict[str, Any] = {}
        for parameter in PARAMETER_PATTERN.finditer(body):
            name = parameter.group(1).strip()
            params[name] = _parse_value(name, parameter.group(2))
        # `context` is harness metadata attached to computer_use by the active
        # meta skill. It is parsed independently and must not reach the strict
        # per-action Pydantic models.
        params.pop("context", None)
        if not params.get("action"):
            raise ToolCallParseError("computer_use call is missing action")
        try:
            actions.append(computer_action_adapter.validate_python(params))
        except ValidationError as exc:
            raise ToolCallParseError(str(exc)) from exc
    return actions


def redact_tool_text(response: str) -> str:
    return re.sub(
        r"(<parameter=text>).*?(</parameter>)",
        r"\1\n[REDACTED]\n\2",
        response,
        flags=re.DOTALL | re.IGNORECASE,
    )


def repair_instruction(error: str) -> str:
    return (
        "Your previous response could not be executed because its computer_use XML was "
        f"invalid: {error}. Return exactly one complete <tool_call> block using "
        "<function=computer_use> and all parameters required by the action. Do not write "
        "anything after </tool_call>."
    )


def context_action_repair_instruction() -> str:
    return (
        "Your context_update was received, but this turn did not contain a computer_use "
        "action. The task is still active. Return exactly one complete <tool_call> block "
        "using <function=computer_use> and all parameters required by the action. Do not "
        "repeat context_update and do not write anything after </tool_call>."
    )


def context_missing_repair_instruction() -> str:
    return (
        "Your computer_use action is valid, but the active meta skill requires a "
        "<parameter=context> JSON snapshot in every tool call. Repeat the same intended "
        "computer_use action, add the complete context parameter with status, completed, "
        "current_state, facts, failures, and next_steps, and write nothing after "
        "</tool_call>."
    )


def build_context_update_messages(
    messages: list[dict[str, Any]],
    action_response: str,
) -> list[dict[str, Any]]:
    """Build a side-hop request that updates memory without choosing a new action."""

    system = (
        "You are the context-memory updater for a computer-use agent. Review the task, "
        "history, current screenshot, and the action that is planned but has not executed. "
        "Return one JSON object matching the supplied response schema. Keep it "
        "short and factual. Do not include secrets or historical screen coordinates. "
        "Represent the visible state before the planned action; do not claim that action "
        "already succeeded."
    )
    request = (
        "Planned action (not executed yet):\n"
        f"{redact_tool_text(action_response)}\n\n"
        "Now return the replacement task-context snapshot."
    )
    return [
        {"role": "system", "content": [{"type": "text", "text": system}]},
        *messages[1:],
        {"role": "user", "content": [{"type": "text", "text": request}]},
    ]


def context_response_format(
    *,
    max_items: int = 8,
    max_item_chars: int = 240,
) -> dict[str, Any]:
    """OpenAI-compatible strict JSON schema for the memory-controller hop."""

    max_items = max(1, int(max_items))
    max_item_chars = max(32, int(max_item_chars))
    string_list = {
        "type": "array",
        "maxItems": max_items,
        "items": {"type": "string", "maxLength": max_item_chars},
    }
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "task_context_snapshot",
            "strict": True,
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "status": {
                        "type": "string",
                        "enum": ["in_progress", "blocked", "completed", "failed"],
                    },
                    "completed": string_list,
                    "current_state": string_list,
                    "facts": string_list,
                    "failures": string_list,
                    "next_steps": string_list,
                },
                "required": [
                    "status",
                    "completed",
                    "current_state",
                    "facts",
                    "failures",
                    "next_steps",
                ],
            },
        },
    }
