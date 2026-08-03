"""Contract tests for the standalone OSWorld batch-evaluation adapter.

Two things are checked, both of which have silently broken a run before:

  1. Every action in the upstream ``computer_use`` enum compiles to a pyautogui
     program, sentinel token, or the documented unattended ``call_user`` no-op.
  2. The message list matches upstream ``AgentHistory.build_messages`` -- image
     count under the budget, collapsed placeholders for older turns, and the
     ``<tool_response>`` feedback channel.

Run: python -m pytest tests/test_osworld_eval_agent.py -q
"""

import pytest

from qwen_cua.eval.osworld import (
    COLLAPSED_SCREENSHOT_TEXT,
    QwenCUAAgent,
    compile_action,
    parse_tool_calls,
)

W, H = 1280, 720


def _xml(*params: str) -> str:
    body = "\n".join(params)
    return f"<tool_call>\n<function=computer_use>\n{body}\n</function>\n</tool_call>"


def _p(name: str, value: str) -> str:
    return f"<parameter={name}>\n{value}\n</parameter>"


# One sample per action in build_tool_definition()'s enum.
CASES = {
    "key": _xml(_p("action", "key"), _p("keys", '["ctrl", "c"]')),
    "key_down": _xml(_p("action", "key_down"), _p("keys", '["shift"]')),
    "key_up": _xml(_p("action", "key_up"), _p("keys", '["shift"]')),
    "left_mouse_down": _xml(_p("action", "left_mouse_down"), _p("coordinate", "[100, 200]")),
    "left_mouse_up": _xml(_p("action", "left_mouse_up")),
    "type": _xml(_p("action", "type"), _p("text", "hello world")),
    "mouse_move": _xml(_p("action", "mouse_move"), _p("coordinate", "[500, 500]")),
    "left_click": _xml(_p("action", "left_click"), _p("coordinate", "[500, 420]")),
    "left_click_bare": _xml(_p("action", "left_click")),
    "left_click_drag": _xml(_p("action", "left_click_drag"), _p("coordinate", "[10, 20]")),
    "right_click": _xml(_p("action", "right_click"), _p("coordinate", "[1, 2]")),
    "middle_click": _xml(_p("action", "middle_click"), _p("coordinate", "[3, 4]")),
    "double_click": _xml(_p("action", "double_click"), _p("coordinate", "[5, 6]")),
    "triple_click": _xml(_p("action", "triple_click"), _p("coordinate", "[7, 8]")),
    "scroll": _xml(_p("action", "scroll"), _p("pixels", "-300")),
    "hscroll": _xml(_p("action", "hscroll"), _p("pixels", "120")),
    "wait": _xml(_p("action", "wait"), _p("time", "2")),
    "screenshot": _xml(_p("action", "screenshot")),
    "terminate": _xml(_p("action", "terminate"), _p("status", "success")),
    "terminate_fail": _xml(_p("action", "terminate"), _p("status", "failure")),
    "call_user": _xml(_p("action", "call_user"), _p("text", "which account?")),
}


@pytest.mark.parametrize("name", sorted(CASES))
def test_every_action_compiles(name):
    actions = parse_tool_calls(CASES[name])
    assert len(actions) == 1, f"{name} did not parse to exactly one action"
    token = compile_action(actions[0], W, H)

    if name == "call_user":
        assert token is None  # no environment effect by design
    else:
        assert token, f"{name} compiled to nothing"


def test_terminate_maps_to_done_and_fail():
    assert compile_action(parse_tool_calls(CASES["terminate"])[0], W, H) == "DONE"
    assert compile_action(parse_tool_calls(CASES["terminate_fail"])[0], W, H) == "FAIL"


def test_grid999_decodes_against_the_real_screen():
    # 999 is the far edge of the grid and must land on the last pixel.
    token = compile_action(
        parse_tool_calls(_xml(_p("action", "left_click"), _p("coordinate", "[999, 999]")))[0], W, H
    )
    assert token == f"pyautogui.click({W - 1}, {H - 1})"
    token = compile_action(
        parse_tool_calls(_xml(_p("action", "left_click"), _p("coordinate", "[0, 0]")))[0], W, H
    )
    assert token == "pyautogui.click(0, 0)"


def test_multiple_tool_calls_in_one_turn():
    response = CASES["left_click"] + "\n" + CASES["type"]
    actions = parse_tool_calls(response)
    assert [a.action for a in actions] == ["left_click", "type"]


# --------------------------------------------------------------------------- #
# message construction
# --------------------------------------------------------------------------- #
def _agent(**kw) -> QwenCUAAgent:
    return QwenCUAAgent(model="test", image_max=5, history_n=50, **kw)


def _fake_turns(agent: QwenCUAAgent, n: int) -> None:
    for i in range(n):
        agent.screenshots.append(f"b64-{i}")
        agent.responses.append(f"response-{i}")
        agent.action_summaries.append(f'[{{"action": "left_click"}}]  # {i}')
        agent.feedback[i] = "Executed left_click."


def test_image_budget_is_respected():
    agent = _agent()
    _fake_turns(agent, 12)
    messages = agent._build_messages("do the thing")
    images = sum(
        1
        for m in messages
        for part in m["content"]
        if isinstance(part, dict) and part.get("type") == "image_url"
    )
    collapsed = sum(
        1
        for m in messages
        for part in m["content"]
        if isinstance(part, dict) and part.get("text") == COLLAPSED_SCREENSHOT_TEXT
    )
    assert images == 5, "image_max=5 must bound what reaches --limit-mm-per-prompt"
    assert collapsed == 7, "older turns keep their slot as a text placeholder"


def test_older_turns_keep_their_assistant_response():
    agent = _agent()
    _fake_turns(agent, 12)
    messages = agent._build_messages("do the thing")
    assistants = [m for m in messages if m["role"] == "assistant"]
    assert len(assistants) == 12, "collapsing an image must not drop the response"


def test_feedback_channel_and_prompt_placement():
    agent = _agent()
    _fake_turns(agent, 3)
    messages = agent._build_messages("do the thing")
    first_user = messages[1]["content"]
    # Upstream puts the instruction text BEFORE the image on the first turn.
    assert first_user[0]["type"] == "text"
    assert "Instruction: do the thing" in first_user[0]["text"]
    assert first_user[1]["type"] == "image_url"
    second_user = messages[3]["content"]
    assert second_user[0]["text"] == "<tool_response>\nExecuted left_click.\n</tool_response>"


def test_omitted_turns_are_summarised_not_lost():
    agent = QwenCUAAgent(model="test", image_max=5, history_n=4)
    _fake_turns(agent, 10)
    messages = agent._build_messages("do the thing")
    prompt = messages[1]["content"][0]["text"]
    assert "Previous actions from omitted turns:" in prompt
    # history_n=4 keeps the last 4 turns; the other 6 land in the summary.
    assert prompt.count("left_click") == 6


def test_evolving_context_is_injected_and_task_local():
    agent = _agent(context_memory=True)
    _fake_turns(agent, 1)
    agent.task_context.ensure_task("do the thing")
    update = agent.task_context.apply_response(
        (
            '<context_update>{"status":"in_progress","completed":["opened app"],'
            '"current_state":["dialog visible"],"facts":[],"failures":[],'
            '"next_steps":["confirm"]}</context_update>'
        ),
        turn=1,
    )
    assert update.applied

    messages = agent._build_messages("do the thing")
    prompt = messages[1]["content"][0]["text"]

    assert "task-local-memory" in prompt
    assert "opened app" in prompt
    assert "Evolving Task Context" in messages[0]["content"][0]["text"]

    agent._build_messages("a different task")
    assert agent.task_context.revision == 0
    assert agent.task_context.snapshot.completed == []


def test_context_only_response_gets_one_action_repair(monkeypatch):
    agent = _agent(context_memory=True)
    responses = iter(
        [
            (
                '<context_update>{"status":"in_progress","completed":[],'
                '"current_state":["blank screen"],"facts":[],"failures":[],'
                '"next_steps":["wait"]}</context_update>'
            ),
            CASES["wait"],
            (
                '{"status":"in_progress","completed":[], '
                '"current_state":["strict updater state"],"facts":[],'
                '"failures":[],"next_steps":["wait"]}'
            ),
        ]
    )
    monkeypatch.setattr(
        agent,
        "_call_llm",
        lambda _messages, **_kwargs: next(responses),
    )
    from io import BytesIO

    from PIL import Image

    buffer = BytesIO()
    Image.new("RGB", (W, H), "white").save(buffer, format="PNG")

    response, actions = agent.predict("wait for the app", {"screenshot": buffer.getvalue()})

    assert actions == ["WAIT"]
    assert agent.repairs == 1
    assert "context_update" in response
    assert "computer_use" in response
    assert agent.task_context.revision == 1
    assert agent.task_context.snapshot.current_state == ["strict updater state"]
    assert agent.context_policy_updates_ignored == 1


def test_action_without_context_gets_snapshot_repair(monkeypatch):
    agent = _agent(context_memory=True)
    repaired = (
        '{"status":"in_progress","completed":[],'
        '"current_state":["loading"],"facts":[],"failures":[],'
        '"next_steps":["wait"]}'
    )
    responses = iter([CASES["wait"], repaired])
    calls = []

    def fake_call(_messages, **kwargs):
        calls.append(kwargs)
        return next(responses)

    monkeypatch.setattr(
        agent,
        "_call_llm",
        fake_call,
    )
    from io import BytesIO

    from PIL import Image

    buffer = BytesIO()
    Image.new("RGB", (W, H), "white").save(buffer, format="PNG")

    _response, actions = agent.predict(
        "wait for the app",
        {"screenshot": buffer.getvalue()},
    )

    assert actions == ["WAIT"]
    assert agent.context_repairs == 1
    assert agent.context_repair_failures == 0
    assert agent.task_context.revision == 1
    assert calls[1]["temperature"] == 0.0
    assert calls[1]["max_tokens"] == agent.max_tokens
    assert calls[1]["response_format"]["type"] == "json_schema"


def test_context_diagnostics_are_exported_per_task():
    agent = _agent(context_memory=True)
    agent.context_repairs = 3
    agent.context_repair_failures = 1
    agent.repairs = 2
    agent.parse_failures = 1
    agent.no_tool_call_finishes = 4

    assert agent.export_context_diagnostics() == {
        "context_repairs": 3,
        "context_repair_failures": 1,
        "context_policy_updates_ignored": 0,
        "action_repairs": 2,
        "parse_failures": 1,
        "no_tool_call_finishes": 4,
    }


def test_desktop_surface_swaps_every_browser_mention():
    browser = _agent(surface="browser")._system_prompt()
    desktop = _agent(surface="desktop")._system_prompt()
    assert "interact with a browser." in browser
    assert "complete browser tasks." in browser
    # Both mentions must go, not just the one in the tool description.
    assert "interact with a browser." not in desktop
    assert "complete browser tasks." not in desktop
    assert "desktop GUI" in desktop
    # everything else must be untouched
    assert browser.count("<tool_call>") == desktop.count("<tool_call>")


@pytest.mark.parametrize("surface", ["browser", "desktop"])
def test_tools_block_stays_valid_json(surface):
    """The desktop swap edits a string *inside* the json.dumps'd <tools> blob.

    Substituting the raw multi-line description puts literal newlines inside a
    JSON string, which is not parseable -- and nothing downstream would say so.
    """
    import json
    import re

    prompt = _agent(surface=surface)._system_prompt()
    blob = re.search(r"<tools>\n(.*?)\n</tools>", prompt, re.S)
    assert blob, f"{surface}: no <tools> block"
    tools = json.loads(blob.group(1))
    assert tools["function"]["name"] == "computer_use"


@pytest.mark.parametrize(
    ("enable_thinking", "expected_template_kwargs"),
    [(False, {"enable_thinking": False}), (True, None)],
)
def test_vllm_thinking_switch_uses_chat_template_kwargs(
    monkeypatch, enable_thinking, expected_template_kwargs
):
    """vLLM ignores a top-level enable_thinking field in extra_body."""
    captured = {}

    class _Completions:
        def create(self, **kwargs):
            captured.update(kwargs)
            message = type("Message", (), {"content": "OK", "reasoning_content": None})()
            choice = type("Choice", (), {"message": message})()
            return type("Response", (), {"choices": [choice]})()

    class _Client:
        def __init__(self, **_kwargs):
            self.chat = type("Chat", (), {"completions": _Completions()})()

    monkeypatch.setattr("qwen_cua.eval.osworld.openai.OpenAI", _Client)
    agent = _agent(enable_thinking=enable_thinking)
    assert agent._call_llm([]) == "OK"

    extra_body = captured["extra_body"]
    assert "enable_thinking" not in extra_body
    assert extra_body.get("chat_template_kwargs") == expected_template_kwargs
