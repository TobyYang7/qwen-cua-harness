"""Adapters that bridge osworld_eval CUA agents to the CUA-Gym browser sandbox.

This module is a helper for ``run_cuagym.py``. It contains two independent
concerns:

1. Action sanitisation + canonicalisation. The harness agents emit *pixel*-space
   pyautogui code that may exceed the CUA-Gym action whitelist (imports,
   ``time.sleep`` lines, ``dragTo`` with a duration, non-literal args, more than
   ``MAX_PROGRAM_OPS`` statements, out-of-range coordinates). CUA-Gym's
   ``env.step`` only accepts a JSON envelope whose ``code`` is a *canonical
   0..999 grid* program parseable by ``computer_use_protocol.parse_pyautogui_code``
   (AST whitelist, literal args only, <= 8 statements). ``sanitize_and_canonicalize``
   drops what cannot survive, clamps coordinates into range, and converts the
   surviving pixel subset to the canonical grid.

2. Agent construction + output normalisation. Each ``--agent_type`` has its own
   constructor contract (see ``build_agent``); their ``predict`` outputs differ
   (list of pyautogui strings with ``WAIT``/``DONE``/``FAIL`` specials, or, for
   ScaleCUA, a list of action dicts). ``normalize_actions`` folds all of them
   into a single ``[NormAction, ...]`` representation the runner steps uniformly.

No existing file is modified; everything here is additive.
"""

from __future__ import annotations

import ast
import importlib
import os
import struct
from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Tuple

# The environment's coordinate contract + AST whitelist are the single source of
# truth; import them rather than re-deriving. Repo root must already be on
# sys.path (the runner inserts it before importing this module).
from computer_use_protocol import (  # noqa: E402
    canonicalize_pyautogui_code,
    _ALLOWED_FUNCTIONS,
    MAX_PROGRAM_OPS,
)

# CUA-Gym renders every episode into a fixed 1280x720 viewport (cua_env.py:118).
DEFAULT_VIEWPORT = (1280, 720)

# Owl is trained/prompted against a fixed nominal 1920x1080 screen and rescales
# its own output with ``convert_point_format`` (x*1920/1932) regardless of the
# image it is shown (owl_agent.py:70-73, get_prompt ignores width/height). Its
# emitted pixels therefore live in ~1920x1080 space, NOT the 1280x720 screenshot
# space. Since 1280x720 and 1920x1080 share the 16:9 aspect ratio, canonicalising
# owl output with 1920x1080 lands on the correct 0..999 grid cell. Every other
# agent decodes coordinates against the actual screenshot dimensions.
OWL_COORD_DIMS = (1920, 1080)

# Positional coordinate layout per pyautogui function.
_XY_FUNCS = {
    "click",
    "rightClick",
    "middleClick",
    "doubleClick",
    "tripleClick",
    "moveTo",
    "dragTo",
}
_SCROLL_FUNCS = {"scroll", "hscroll"}


# --------------------------------------------------------------------------- #
# 1. Sanitise + canonicalise
# --------------------------------------------------------------------------- #
def png_size(data: bytes) -> Tuple[int, int]:
    """Read (width, height) from a PNG byte string without decoding pixels."""
    if (
        isinstance(data, (bytes, bytearray))
        and len(data) >= 24
        and bytes(data[:8]) == b"\x89PNG\r\n\x1a\n"
    ):
        try:
            w, h = struct.unpack(">II", bytes(data[16:24]))
            if w > 0 and h > 0:
                return int(w), int(h)
        except struct.error:
            pass
    return DEFAULT_VIEWPORT


def _is_pyautogui_call(node: ast.stmt) -> bool:
    return (
        isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Attribute)
        and isinstance(node.value.func.value, ast.Name)
        and node.value.func.value.id == "pyautogui"
    )


def _all_literals(call: ast.Call) -> bool:
    for a in call.args:
        if isinstance(a, ast.Starred):
            return False
        try:
            ast.literal_eval(a)
        except Exception:
            return False
    for kw in call.keywords:
        if kw.arg is None:
            return False
        try:
            ast.literal_eval(kw.value)
        except Exception:
            return False
    return True


def _clamped_node(node: ast.expr, lo: float, hi: float) -> ast.expr:
    try:
        val = ast.literal_eval(node)
    except Exception:
        return node
    if isinstance(val, bool) or not isinstance(val, (int, float)):
        return node
    if lo <= val <= hi:
        return node
    return ast.copy_location(ast.Constant(float(max(lo, min(hi, val)))), node)


def _clamp_call_coords(call: ast.Call, sw: int, sh: int) -> None:
    """Clamp x/y coordinates into [0, size-1] so canonicalize never range-errors."""
    fn = call.func.attr  # type: ignore[union-attr]
    if fn in _XY_FUNCS:
        xi, yi = 0, 1
    elif fn in _SCROLL_FUNCS:
        xi, yi = 1, 2
    else:
        xi = yi = None

    if xi is not None and len(call.args) > yi:
        call.args[xi] = _clamped_node(call.args[xi], 0, sw - 1)
        call.args[yi] = _clamped_node(call.args[yi], 0, sh - 1)
    for kw in call.keywords:
        if kw.arg == "x":
            kw.value = _clamped_node(kw.value, 0, sw - 1)
        elif kw.arg == "y":
            kw.value = _clamped_node(kw.value, 0, sh - 1)


def _collect_statements(raw_code: str, notes: List[str]) -> List[ast.stmt]:
    """Parse the whole program; on syntax error, salvage line by line."""
    try:
        return list(ast.parse(raw_code, mode="exec").body)
    except SyntaxError:
        salvaged: List[ast.stmt] = []
        for line in raw_code.splitlines():
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            try:
                salvaged.extend(ast.parse(s, mode="exec").body)
            except SyntaxError:
                notes.append(f"unparsable line dropped: {s[:80]}")
        return salvaged


def _coalesce_char_presses(kept: List[ast.stmt], notes: List[str]) -> List[ast.stmt]:
    """Merge runs of single-printable-char pyautogui.press('x') into one write().

    The ScaleCUA agent compiles `type` into one press() per character; with the
    protocol's 8-statement cap that would truncate any real text. write() is a
    single statement and also preserves case.
    """

    def _press_char(node: ast.stmt) -> Optional[str]:
        call = node.value  # type: ignore[union-attr]
        if call.func.attr != "press" or call.keywords or len(call.args) != 1:
            return None
        try:
            val = ast.literal_eval(call.args[0])
        except Exception:
            return None
        if isinstance(val, str) and len(val) == 1 and val.isprintable():
            return val
        return None

    out: List[ast.stmt] = []
    run: List[str] = []

    def _flush() -> None:
        if not run:
            return
        if len(run) == 1:
            out.append(ast.parse(f"pyautogui.press({run[0]!r})").body[0])
        else:
            text = "".join(run)
            out.append(ast.parse(f"pyautogui.write({text!r})").body[0])
            notes.append(f"coalesced {len(run)} char presses into write()")
        run.clear()

    for node in kept:
        ch = _press_char(node)
        if ch is not None:
            run.append(ch)
        else:
            _flush()
            out.append(node)
    _flush()
    return out


def _scale_call_coords(call: ast.Call, fx: float, fy: float) -> None:
    """Multiply x/y coordinate literals in place (used to undo agent-side rescales)."""
    fn = call.func.attr  # type: ignore[union-attr]
    if fn in _XY_FUNCS:
        xi, yi = 0, 1
    elif fn in _SCROLL_FUNCS:
        xi, yi = 1, 2
    else:
        xi = yi = None

    def _scaled(node: ast.expr, factor: float) -> ast.expr:
        try:
            value = ast.literal_eval(node)
        except Exception:
            return node
        if isinstance(value, (int, float)):
            return ast.Constant(value=float(value) * factor)
        return node

    if xi is not None and len(call.args) > yi:
        call.args[xi] = _scaled(call.args[xi], fx)
        call.args[yi] = _scaled(call.args[yi], fy)
    for kw in call.keywords:
        if kw.arg == "x":
            kw.value = _scaled(kw.value, fx)
        elif kw.arg == "y":
            kw.value = _scaled(kw.value, fy)


def sanitize_and_canonicalize(
    raw_code: str,
    screen_width: int,
    screen_height: int,
    coordinate_space: str = "pixels",
) -> Tuple[Optional[str], List[str]]:
    """Reduce harness pixel pyautogui to a canonical 0..999 grid program.

    Returns ``(canonical_code, notes)``. ``canonical_code`` is None when nothing
    executable survived (the caller records an ``invalid_action``). ``notes``
    lists every dropped/clamped/truncated element for the per-step log.
    """
    notes: List[str] = []
    if not isinstance(raw_code, str) or not raw_code.strip():
        return None, ["empty action code"]

    kept: List[ast.stmt] = []
    for node in _collect_statements(raw_code, notes):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            continue  # imports are noise from the harness; drop silently
        if not _is_pyautogui_call(node):
            try:
                snippet = ast.unparse(node)
            except Exception:
                snippet = type(node).__name__
            notes.append(f"dropped non-pyautogui stmt: {snippet[:80]}")
            continue
        call: ast.Call = node.value  # type: ignore[assignment]
        fn = call.func.attr  # type: ignore[union-attr]
        if fn not in _ALLOWED_FUNCTIONS:
            notes.append(f"dropped disallowed pyautogui.{fn}")
            continue
        if not _all_literals(call):
            notes.append(f"dropped non-literal-arg pyautogui.{fn}")
            continue
        kept.append(node)

    if not kept:
        notes.append("no executable pyautogui statements survived sanitize")
        return None, notes

    kept = _coalesce_char_presses(kept, notes)

    if len(kept) > MAX_PROGRAM_OPS:
        notes.append(f"truncated {len(kept)} ops to {MAX_PROGRAM_OPS}")
        kept = kept[:MAX_PROGRAM_OPS]

    # "owl_grid": GUI-Owl-1.5 (Qwen3-VL base) emits native 0..999 grid coords;
    # owl_agent then rescales them by 1920/1932 (x) and 1080/1092 (y) as if they
    # were 1932x1092 pixels. Undo that rescale and treat them as relative_1000.
    if coordinate_space == "owl_grid":
        for node in kept:
            _scale_call_coords(node.value, 1932 / 1920, 1092 / 1080)  # type: ignore[attr-defined]
        clamp_w = clamp_h = 1000
        canon_space = "relative_1000"
    else:
        clamp_w, clamp_h = screen_width, screen_height
        canon_space = "pixels"

    lines: List[str] = []
    for node in kept:
        _clamp_call_coords(node.value, clamp_w, clamp_h)  # type: ignore[attr-defined]
        lines.append(ast.unparse(node))
    pixel_code = "\n".join(lines)

    try:
        canonical = canonicalize_pyautogui_code(
            pixel_code,
            screen_width=screen_width,
            screen_height=screen_height,
            coordinate_space=canon_space,
        )
    except Exception as e:  # noqa: BLE001
        notes.append(f"canonicalize failed: {type(e).__name__}: {e}")
        return None, notes
    return canonical, notes


# --------------------------------------------------------------------------- #
# 2. Agent output normalisation
# --------------------------------------------------------------------------- #
@dataclass
class NormAction:
    kind: str  # "code" | "done" | "fail" | "wait" | "noop"
    code: Optional[str] = None
    seconds: float = 2.0


_SPECIALS = {"WAIT", "DONE", "FAIL"}


def _norm_string(s: str) -> NormAction:
    stripped = (s or "").strip()
    upper = stripped.upper()
    if upper == "WAIT":
        return NormAction("wait")
    if upper == "DONE":
        return NormAction("done")
    if upper == "FAIL":
        return NormAction("fail")
    if not stripped:
        return NormAction("noop")
    return NormAction("code", code=stripped)


def normalize_actions(raw_actions: Any) -> List[NormAction]:
    """Fold every agent's ``predict`` output into a list of NormAction.

    Handles the two shapes in mm_agents:
      * list of strings (qwen35_xml / qwen3vl / evocua / owl), where an element is
        either pixel pyautogui code or a ``WAIT``/``DONE``/``FAIL`` sentinel;
      * list of dicts (ScaleCUA), each with ``action_type`` in
        {``tool_use``, ``DONE``, ``FAIL``} and a ``command`` pyautogui string.
    """
    if raw_actions is None:
        return []
    if isinstance(raw_actions, str):
        raw_actions = [raw_actions]

    out: List[NormAction] = []
    for elem in raw_actions:
        if isinstance(elem, str):
            out.append(_norm_string(elem))
        elif isinstance(elem, dict):
            at = str(elem.get("action_type", "")).upper()
            if at == "DONE":
                out.append(NormAction("done"))
            elif at == "FAIL":
                out.append(NormAction("fail"))
            elif at == "WAIT":
                out.append(NormAction("wait"))
            else:
                cmd = elem.get("command")
                if isinstance(cmd, str) and cmd.strip():
                    out.append(_norm_string(cmd))
                else:
                    out.append(NormAction("noop"))
        else:
            out.append(NormAction("noop"))
    return out


# --------------------------------------------------------------------------- #
# 3. Agent construction
# --------------------------------------------------------------------------- #
@dataclass
class AgentSpec:
    agent: Any
    coord_dims: Optional[Tuple[int, int]]  # None => use actual screenshot size
    reset: Callable[[Any], None]  # reset(logger)
    coordinate_space: str = "pixels"  # or "owl_grid" (native 0..999 grid)


def _f(val: Optional[float], default: float) -> float:
    return default if val is None else val


def build_agent(agent_type: str, args) -> AgentSpec:
    """Instantiate a harness agent wired for the served OpenAI-compatible model.

    OPENAI_BASE_URL / OPENAI_API_KEY are set by the runner in os.environ, which
    the qwen/evocua agents read at call time. Owl and ScaleCUA take the endpoint
    explicitly.
    """
    viewport = DEFAULT_VIEWPORT

    if agent_type in {"qwen35_xml", "qwen36_xml"}:
        Qwen35XMLAgent = importlib.import_module(
            "mm_agents.qwen35_xml_agent"
        ).Qwen35XMLAgent
        is_qwen36 = agent_type == "qwen36_xml"
        agent = Qwen35XMLAgent(
            platform="Ubuntu",
            model=args.model,
            action_space="pyautogui",
            observation_type="screenshot",
            screen_size=viewport,
            history_n=args.history_n,
            max_tokens=args.max_tokens,
            temperature=_f(args.temperature, 1.0 if is_qwen36 else 0.0),
            top_p=_f(args.top_p, 0.95 if is_qwen36 else 0.9),
            top_k=(
                args.top_k
                if getattr(args, "top_k", None) is not None
                else (20 if is_qwen36 else -1)
            ),
            min_p=(
                args.min_p
                if getattr(args, "min_p", None) is not None
                else 0.0
            ),
            presence_penalty=(
                args.presence_penalty
                if getattr(args, "presence_penalty", None) is not None
                else (1.5 if is_qwen36 else None)
            ),
            repetition_penalty=(
                args.repetition_penalty
                if getattr(args, "repetition_penalty", None) is not None
                else (1.0 if is_qwen36 else None)
            ),
            enable_thinking=bool(getattr(args, "enable_thinking", False)),
            preserve_thinking=bool(getattr(args, "preserve_thinking", False)),
            max_recent_images=getattr(args, "max_recent_images", None),
            coordinate_type="relative",
            relative_coordinate=False,
        )
        return AgentSpec(agent, None, lambda lg: agent.reset(lg))

    if agent_type == "qwencua":
        from qwen_cua.eval.osworld import QwenCUAAgent
        # Upstream defaults (temp 0.6 / top_p 0.95 / top_k 20, thinking on) unless
        # the caller overrides them. image_max is bounded by --max_recent_images
        # so it can never exceed the server's --limit-mm-per-prompt.
        agent = QwenCUAAgent(
            model=args.model,
            max_tokens=args.max_tokens,
            temperature=_f(args.temperature, 0.6),
            top_p=_f(args.top_p, 0.95),
            top_k=(
                args.top_k if getattr(args, "top_k", None) is not None else 20
            ),
            enable_thinking=bool(getattr(args, "enable_thinking", False)),
            # NOT args.history_n: for this contract history_n is a text budget
            # (collapsed turns keep their response), so the guiowl15-style
            # default of 5 would disable the collapse mechanism entirely.
            history_n=getattr(args, "qwencua_history_n", 50),
            image_max=(
                args.max_recent_images
                if getattr(args, "max_recent_images", None)
                else 5
            ),
            surface="browser",
            context_memory=bool(getattr(args, "context_memory", False)),
            context_max_items=getattr(args, "context_max_items", 8),
            context_max_chars=getattr(args, "context_max_chars", 6000),
        )
        return AgentSpec(agent, None, lambda lg: agent.reset(lg))

    if agent_type == "guiowl15":
        GUIOwl15Agent = importlib.import_module(
            "mm_agents.guiowl15_agent"
        ).GUIOwl15Agent
        agent = GUIOwl15Agent(
            model=args.model,
            history_n=min(args.history_n, 4),
            max_recent_images=args.max_recent_images,
            max_tokens=args.max_tokens,
        )
        return AgentSpec(agent, None, lambda lg: agent.reset(lg))

    if agent_type == "qwen3vl":
        Qwen3VLAgent = importlib.import_module("mm_agents.qwen3vl_agent").Qwen3VLAgent
        agent = Qwen3VLAgent(
            platform="ubuntu",
            model=args.model,
            action_space="pyautogui",
            observation_type="screenshot",
            coordinate_type="relative",
            api_backend="openai",
            history_n=args.history_n,
            max_tokens=args.max_tokens,
            temperature=_f(args.temperature, 0.0),
            top_p=_f(args.top_p, 0.9),
            enable_thinking=bool(getattr(args, "enable_thinking", False)),
        )
        return AgentSpec(agent, None, lambda lg: agent.reset(lg))

    if agent_type == "evocua":
        EvoCUAAgent = importlib.import_module(
            "mm_agents.evocua.evocua_agent"
        ).EvoCUAAgent
        agent = EvoCUAAgent(
            model=args.model,
            action_space="pyautogui",
            observation_type="screenshot",
            screen_size=viewport,
            coordinate_type="relative",
            max_steps=args.max_agent_steps,
            max_tokens=args.max_tokens,
            temperature=_f(args.temperature, 0.01),
            top_p=_f(args.top_p, 0.9),
            prompt_style="S2",
            max_history_turns=min(
                4,
                max(int(getattr(args, "max_recent_images", 5)) - 1, 0),
            ),
            resize_factor=32,
        )
        return AgentSpec(agent, None, lambda lg: agent.reset(lg))

    if agent_type == "owl":
        owl_mod = importlib.import_module("mm_agents.owl_agent")
        # Upstream get_image_url pushes screenshots to Aliyun OSS (needs
        # access_key_id etc.); serve them inline as data URLs instead.
        owl_mod.get_image_url = lambda b64: f"data:image/png;base64,{b64}"
        OwlAgent = owl_mod.OwlAgent
        agent = OwlAgent(
            model=args.model,
            api_url=args.base_url,
            api_key=args.api_key,
            platform="ubuntu",
            action_space="pyautogui",
            observation_type="screenshot",
            history_n=3,
            temperature=_f(args.temperature, 0.6),
            top_p=_f(args.top_p, 0.95),
            top_k=1,
            runtime_conf={
                "infer_mode": "fn_call",
                "input_swap": False,
                "screen_height": OWL_COORD_DIMS[1],
                "screen_width": OWL_COORD_DIMS[0],
            },
            engine="openai",
        )
        # Owl.reset(runtime_logger) requires a positional argument.
        return AgentSpec(
            agent,
            OWL_COORD_DIMS,
            lambda lg: agent.reset(lg),
            coordinate_space="owl_grid",
        )

    if agent_type == "scalecua":
        mod = importlib.import_module("mm_agents.scalecua-osworld")
        utils_mod = importlib.import_module("mm_agents.scalecua-osworld.utils")
        ScaleCUAOSWorldAgent = mod.ScaleCUAOSWorldAgent
        computer_tools = utils_mod.computer_tools
        gen_func = _make_scalecua_gen_func(
            base_url=args.base_url,
            api_key=args.api_key,
            model=args.model,
            tools=computer_tools,
            temperature=_f(args.temperature, 0.0),
            top_p=_f(args.top_p, 0.9),
            max_tokens=args.max_tokens,
        )
        agent = ScaleCUAOSWorldAgent(
            platform="Ubuntu",
            action_space="pyautogui",
            observation_type="screenshot",
            screen_size=viewport,
            image_size=viewport,
            relative_coordinate=True,
            only_n_most_recent_images=5,
            min_removal_threshold=5,
            gen_func=gen_func,
        )
        return AgentSpec(agent, None, lambda lg: agent.reset(lg))

    raise ValueError(f"unknown --agent_type: {agent_type}")


def _make_scalecua_gen_func(
    *, base_url, api_key, model, tools, temperature, top_p, max_tokens
):
    """A local OpenAI-compatible chat.completions gen_func for ScaleCUA.

    ScaleCUA injects its LLM call as ``gen_func(messages) -> openai_message_dict``
    (run_scalecua_os.py:413,478-545). We return the assistant message as a plain
    dict so ``ScaleCUAOSWorldAgent.execute`` can read ``tool_calls``/``content``.
    Retries transient failures twice before surfacing an empty message.
    """
    import openai

    client = openai.OpenAI(base_url=base_url, api_key=api_key)

    def gen_func(messages):
        last_err = None
        for attempt in range(3):
            try:
                resp = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    tools=tools,
                    tool_choice="auto",
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_tokens,
                    stream=False,
                )
                return resp.choices[0].message.model_dump()
            except Exception as e:  # noqa: BLE001
                last_err = e
        return {"role": "assistant", "content": f"[gen_func error] {last_err}"}

    return gen_func
