"""Batch-evaluation adapter for OSWorld-style screenshot environments.

The wire contract is not restated here: ``build_system_prompt``,
``parse_tool_calls``, ``repair_instruction`` and the pydantic action models are
imported from this package's canonical modules. This file adds the batch-facing
state machine and compiles validated actions into the pyautogui program strings
and sentinel tokens commonly accepted by screenshot benchmark runners.

Upstream (``qwen_cua/runner.py``) drives a Playwright browser directly and puts
a human in the loop. Three of its behaviours are reproduced here because they
are contract-visible to the model, and are exactly what our own harnesses lack:

  * **Collapsed screenshots.** A turn that falls outside the image budget keeps
    its assistant response and gets the literal text
    ``This screenshot has been collapsed.`` in place of the image, rather than
    being deleted from the message list (``AgentHistory.build_messages``).
  * **Execution feedback.** Each observation after the first is introduced by
    ``<tool_response>\\nExecuted left_click.\\n</tool_response>``, so the model
    is told its action ran instead of having to infer it from the next frame.
  * **One repair round-trip.** A response whose computer_use XML does not parse
    is re-asked once with ``repair_instruction()`` appended before it counts as
    a dead turn.

Deliberate divergences from upstream, all forced by batch evaluation:

  * ``image_max`` defaults to **5**, not upstream's 20, to match the
    ``--limit-mm-per-prompt '{"image":5}'`` our vLLM servers run with.
  * ``call_user`` cannot block on a human. It executes as a no-op and the model
    is told, through the normal feedback channel, that no operator is available.
  * ``screenshot`` compiles to ``WAIT``: our runners capture a fresh frame every
    turn regardless, so re-capturing is already implied.
  * Execution feedback is derived from the actions the model emitted, not from
    what the environment actually accepted -- the agent never sees the runner's
    sanitizer verdict.
  * The safety gate (password / upload / download / form-submit / cross-origin
    approvals) is absent; it only exists to pause for an operator.

``predict(instruction, obs) -> (response_text, [pyautogui_or_token, ...])``
follows the common screenshot-agent convention. The adapter may return
**several** actions for one turn, matching the reference harness.
"""

import base64
import json
import logging
import math
import os
import time
from io import BytesIO
from typing import Any

import openai
from PIL import Image

from ..actions import (
    CallUserAction,
    ClickAction,
    CoordinateAction,
    KeyAction,
    ScreenshotAction,
    ScrollAction,
    TerminateAction,
    TypeAction,
    WaitAction,
    action_to_public_dict,
)
from ..protocol import (
    COLLAPSED_SCREENSHOT_TEXT,
    ToolCallParseError,
    build_system_prompt,
    parse_tool_calls,
    repair_instruction,
)

logger = logging.getLogger("qwen_cua.eval.osworld")

MAX_RETRY_TIMES = 5
COORDINATE_MAX = 999

# Upstream is browser-only: the tool description says "interact with a browser"
# and the opening sentence says "browser tasks". Left verbatim for CUA-Gym,
# which is a browser. OSWorld is a full desktop, so both are actively wrong
# there and get swapped -- see ``_system_prompt``.
_BROWSER_TASKS = "complete browser tasks."
_DESKTOP_TASKS = "complete computer tasks."

_BROWSER_LINE = "Use a mouse and keyboard to interact with a browser."
_DESKTOP_LINE = (
    "Use a mouse and keyboard to interact with a desktop GUI, and take screenshots.\n"
    "* You do not have access to a terminal or an applications menu. "
    "You must click on desktop icons to start applications.\n"
    "* Some applications take time to start or to process an action, so you may "
    "need to wait and take a further screenshot to see the result."
)
# The tool description lives inside the json.dumps'd <tools> blob, so the swap
# has to happen on the ESCAPED form. Substituting the raw multi-line string
# puts literal newlines inside a JSON string and makes the blob unparseable.
_BROWSER_LINE_JSON = json.dumps(_BROWSER_LINE)[1:-1]
_DESKTOP_LINE_JSON = json.dumps(_DESKTOP_LINE)[1:-1]


def _process_image(payload: bytes) -> str:
    """Upstream ``runner._process_image``: cap the pixel count, snap to /32.

    Coordinates are relative to the full frame on both sides, so resizing here
    does not move any grounding target.
    """
    image = Image.open(BytesIO(payload)).convert("RGB")
    width, height = image.size
    max_pixels = 16 * 16 * 4 * 12_800
    if width * height > max_pixels:
        scale = math.sqrt(max_pixels / (width * height))
        width = max(32, int(width * scale))
        height = max(32, int(height * scale))
    width = max(32, round(width / 32) * 32)
    height = max(32, round(height / 32) * 32)
    if image.size != (width, height):
        image = image.resize((width, height), Image.Resampling.LANCZOS)
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


# --------------------------------------------------------------------------- #
# ComputerAction -> pyautogui
# --------------------------------------------------------------------------- #
_PYAUTOGUI_CLICK_FN = {
    "left_click": "click",
    "right_click": "rightClick",
    "middle_click": "middleClick",
    "double_click": "doubleClick",
}

# Upstream normalizes to Playwright key names; pyautogui wants its own.
_KEY_ALIASES = {
    "control": "ctrl",
    "cmd": "command",
    "meta": "command",
    "option": "alt",
    "return": "enter",
    "escape": "esc",
    "pgup": "pageup",
    "pgdn": "pagedown",
    "del": "delete",
    "ins": "insert",
}


def _grid_to_pixel(value: int, size: int) -> int:
    if size <= 1:
        return int(value)
    clamped = min(max(int(value), 0), COORDINATE_MAX)
    return int(round(clamped * (size - 1) / COORDINATE_MAX))


def _py_str(text: str | None) -> str:
    return json.dumps("" if text is None else str(text), ensure_ascii=False)


def _py_key(key: str) -> str:
    lowered = str(key).strip().lower()
    return _KEY_ALIASES.get(lowered, lowered)


def compile_action(action: Any, screen_width: int, screen_height: int) -> str | None:
    """Compile one validated ComputerAction to an env.step argument.

    Returns a special token ("WAIT"/"DONE"/"FAIL"), a pyautogui program string,
    or None when the action has no environment effect (call_user).
    """
    if isinstance(action, TerminateAction):
        return "DONE" if action.status == "success" else "FAIL"
    if isinstance(action, WaitAction):
        return "WAIT"
    if isinstance(action, ScreenshotAction):
        # The runner captures a frame every turn; an explicit re-capture is a
        # no-op that still has to consume the turn.
        return "WAIT"
    if isinstance(action, CallUserAction):
        return None

    if isinstance(action, KeyAction):
        keys = [_py_key(k) for k in action.keys]
        if not keys:
            return None
        if action.action == "key":
            if len(keys) > 1:
                return f"pyautogui.hotkey({', '.join(_py_str(k) for k in keys)})"
            return f"pyautogui.press({_py_str(keys[0])})"
        if action.action == "key_down":
            return "\n".join(f"pyautogui.keyDown({_py_str(k)})" for k in keys)
        return "\n".join(f"pyautogui.keyUp({_py_str(k)})" for k in reversed(keys))

    if isinstance(action, TypeAction):
        return f"pyautogui.write({_py_str(action.text)}, interval=0.01)"

    if isinstance(action, CoordinateAction):
        x = _grid_to_pixel(action.coordinate[0], screen_width)
        y = _grid_to_pixel(action.coordinate[1], screen_height)
        if action.action == "mouse_move":
            return f"pyautogui.moveTo({x}, {y})"
        return f"pyautogui.dragTo({x}, {y}, duration={float(action.duration)}, button='left')"

    if isinstance(action, ClickAction):
        # coordinate is optional upstream: a bare click lands wherever the
        # cursor already is.
        target = ""
        if action.coordinate is not None:
            x = _grid_to_pixel(action.coordinate[0], screen_width)
            y = _grid_to_pixel(action.coordinate[1], screen_height)
            target = f"{x}, {y}"
        if action.action in _PYAUTOGUI_CLICK_FN:
            return f"pyautogui.{_PYAUTOGUI_CLICK_FN[action.action]}({target})"
        if action.action == "triple_click":
            return f"pyautogui.tripleClick({target})"
        if action.action == "left_mouse_down":
            prefix = f"pyautogui.moveTo({target})\n" if target else ""
            return f"{prefix}pyautogui.mouseDown(button='left')"
        if action.action == "left_mouse_up":
            prefix = f"pyautogui.moveTo({target})\n" if target else ""
            return f"{prefix}pyautogui.mouseUp(button='left')"
        return None

    if isinstance(action, ScrollAction):
        # Upstream drives page.mouse.wheel(0, -pixels): positive `pixels` means
        # scroll up, which is also pyautogui.scroll's sign convention.
        if action.action == "scroll":
            return f"pyautogui.scroll({int(action.pixels)})"
        return f"pyautogui.hscroll({int(action.pixels)})"

    return None


def _feedback_for(action: Any) -> str:
    if isinstance(action, CallUserAction):
        return (
            "call_user is unavailable: this session runs unattended with no "
            "operator to answer. Continue on your own."
        )
    return f"Executed {action.action}."


# --------------------------------------------------------------------------- #
# Agent
# --------------------------------------------------------------------------- #
class QwenCUAAgent:
    """Unattended screenshot agent speaking the Qwen-CUA reference contract."""

    def __init__(
        self,
        model: str = "qwen-cua",
        max_tokens: int = 2048,
        temperature: float = 0.6,
        top_p: float = 0.95,
        top_k: int = 20,
        enable_thinking: bool = True,
        history_n: int = 50,
        image_max: int = 5,
        surface: str = "browser",
        repair_malformed: bool = True,
        no_tool_call: str = "finish",
        **kwargs,
    ):
        if surface not in {"browser", "desktop"}:
            raise ValueError(f"surface must be browser|desktop, got {surface!r}")
        if no_tool_call not in {"finish", "invalid"}:
            raise ValueError(f"no_tool_call must be finish|invalid, got {no_tool_call!r}")
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.enable_thinking = enable_thinking
        self.history_n = max(1, int(history_n))
        self.image_max = max(1, int(image_max))
        self.surface = surface
        self.repair_malformed = repair_malformed
        self.no_tool_call = no_tool_call

        # Consumed by the CUA-Gym recorder: this agent decodes grid999 to real
        # pixels itself, so the env must not scale a second time.
        self.relative_coordinate = False

        self.screenshots: list[str] = []
        self.responses: list[str] = []
        self.action_summaries: list[str] = []
        self.feedback: dict[int, str] = {}
        self.actions: list[str] = []
        # Diagnostics. A high no_tool_call_finishes with a low step count is the
        # signature of a contract mismatch, not of an easy task set.
        self.repairs = 0
        self.parse_failures = 0
        self.no_tool_call_finishes = 0

    # ------------------------------- prompt -------------------------------- #
    def _system_prompt(self) -> str:
        prompt = build_system_prompt()
        if self.surface != "desktop":
            return prompt
        for old, new in (
            (_BROWSER_TASKS, _DESKTOP_TASKS),
            (_BROWSER_LINE_JSON, _DESKTOP_LINE_JSON),
        ):
            if prompt.count(old) != 1:
                raise RuntimeError(
                    f"upstream system prompt no longer contains {old!r} exactly once; "
                    "the desktop prompt swap in qwen_cua.eval.osworld needs updating"
                )
            prompt = prompt.replace(old, new, 1)
        return prompt

    def _build_messages(self, instruction: str) -> list[dict[str, Any]]:
        """Verbatim port of upstream ``AgentHistory.build_messages``."""
        total = len(self.screenshots)
        start = max(0, total - self.history_n)
        collapsed_before = max(0, total - self.image_max)
        earlier_actions = self.action_summaries[:start]
        prompt = (
            "Please generate the next move according to the UI screenshot, instruction "
            "and previous actions.\n\n"
            f"Instruction: {instruction}\n\n"
            "Previous actions from omitted turns:\n"
            f"{chr(10).join(earlier_actions) if earlier_actions else 'None'}"
        )
        messages: list[dict[str, Any]] = [
            {
                "role": "system",
                "content": [{"type": "text", "text": self._system_prompt()}],
            }
        ]
        for index in range(start, total):
            content: list[dict[str, Any]] = []
            if index == start:
                content.append({"type": "text", "text": prompt})
            elif self.feedback.get(index):
                content.append(
                    {
                        "type": "text",
                        "text": f"<tool_response>\n{self.feedback[index]}\n</tool_response>",
                    }
                )
            if index < collapsed_before:
                content.append({"type": "text", "text": COLLAPSED_SCREENSHOT_TEXT})
            else:
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{self.screenshots[index]}"},
                    }
                )
            messages.append({"role": "user", "content": content})
            if index < len(self.responses):
                messages.append(
                    {
                        "role": "assistant",
                        "content": [{"type": "text", "text": self.responses[index]}],
                    }
                )
        return messages

    # ------------------------------- predict ------------------------------- #
    def predict(self, instruction: str, obs: dict) -> tuple[str, list]:
        screenshot_bytes = obs["screenshot"]
        width, height = Image.open(BytesIO(screenshot_bytes)).size

        self.screenshots.append(_process_image(screenshot_bytes))
        messages = self._build_messages(instruction)

        response = self._call_llm(messages)
        try:
            parsed = parse_tool_calls(response)
        except ToolCallParseError as exc:
            self.parse_failures += 1
            if not self.repair_malformed:
                parsed = []
                response = f"{response}\n[unparsed: {exc}]"
            else:
                logger.warning("[qwencua] malformed tool call, one repair: %s", exc)
                self.repairs += 1
                repair_messages = [
                    *messages,
                    {"role": "assistant", "content": response},
                    {"role": "user", "content": repair_instruction(str(exc))},
                ]
                response = self._call_llm(repair_messages)
                try:
                    parsed = parse_tool_calls(response)
                except ToolCallParseError as exc2:
                    logger.warning("[qwencua] repair also malformed: %s", exc2)
                    parsed = []

        self.responses.append(response)
        self.action_summaries.append(
            json.dumps(
                [action_to_public_dict(a) for a in parsed],
                ensure_ascii=False,
            )
        )

        actions: list[str] = []
        feedback_parts: list[str] = []
        for action in parsed:
            token = compile_action(action, width, height)
            feedback_parts.append(_feedback_for(action))
            if token is not None:
                actions.append(token)
            if isinstance(action, TerminateAction):
                break

        if not parsed:
            # Upstream ends the run here ("Model returned a final assistant
            # message") and verifies whatever state the page is in.
            self.no_tool_call_finishes += 1
            if self.no_tool_call == "finish":
                actions = ["DONE"]
            feedback_parts.append("No tool call was produced.")

        # The next observation carries this turn's execution result, exactly as
        # upstream feeds `<tool_response>` back in.
        self.feedback[len(self.screenshots)] = "\n".join(feedback_parts)

        rendered = ", ".join(a.action for a in parsed) or "(no action)"
        self.actions.append(rendered)
        logger.info("[qwencua] actions=%s tokens=%s", rendered, actions)
        return response, actions

    # ------------------------------- llm call ------------------------------ #
    def _call_llm(self, messages: list[dict[str, Any]]) -> str:
        base_url = os.environ.get("OPENAI_BASE_URL", "http://127.0.0.1:8000/v1")
        api_key = os.environ.get("OPENAI_API_KEY", "EMPTY")
        client = openai.OpenAI(base_url=base_url, api_key=api_key, timeout=600)

        # vLLM applies Qwen's thinking switch through chat_template_kwargs.
        # A top-level ``enable_thinking`` extra field is accepted but ignored by
        # vLLM 0.24, which silently leaves Qwen3.5 in thinking mode.
        extra_body: dict[str, Any] = {"top_k": self.top_k}
        if not self.enable_thinking:
            extra_body["chat_template_kwargs"] = {"enable_thinking": False}
        params: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "extra_body": extra_body,
        }

        last_err: Exception | None = None
        for attempt in range(1, MAX_RETRY_TIMES + 1):
            try:
                resp = client.chat.completions.create(**params)
                message = resp.choices[0].message
                content = (message.content or "").strip()
                reasoning = (getattr(message, "reasoning_content", None) or "").strip()
                # Upstream reassembles the split reasoning channel into a single
                # `<think>` block and keeps it in history verbatim.
                if reasoning and content:
                    return f"<think>\n{reasoning}\n</think>\n{content}"
                return content or reasoning
            except Exception as e:  # noqa: BLE001
                last_err = e
                logger.error("[qwencua] call failed (%d/%d): %s", attempt, MAX_RETRY_TIMES, e)
                if attempt < MAX_RETRY_TIMES:
                    time.sleep(5)
        assert last_err is not None
        raise last_err

    # ------------------------------- reset --------------------------------- #
    def reset(self, _logger=None):
        global logger
        if _logger is not None:
            logger = _logger
        self.screenshots = []
        self.responses = []
        self.action_summaries = []
        self.feedback = {}
        self.actions = []
        self.repairs = 0
        self.parse_failures = 0
        self.no_tool_call_finishes = 0
