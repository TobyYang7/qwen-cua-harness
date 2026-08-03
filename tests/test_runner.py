from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from io import BytesIO
from pathlib import Path
from typing import Any

from PIL import Image

from qwen_cua.actions import ComputerAction
from qwen_cua.computer import Screenshot, TargetInspection
from qwen_cua.config import Settings
from qwen_cua.models import (
    ApprovalRequest,
    BrowserMode,
    RunOutcome,
    RunStatus,
    StartRunRequest,
    VerificationStatus,
)
from qwen_cua.runner import RunnerManager


class FakeModelClient:
    def __init__(self, responses: list[str]):
        self.responses = responses
        self.calls: list[dict[str, Any]] = []

    async def generate(self, **kwargs: Any) -> str:
        self.calls.append(kwargs)
        return self.responses.pop(0)

    async def close(self) -> None:
        return None


class FakeComputer:
    instances: list[FakeComputer] = []

    def __init__(self, **_: Any):
        self.url = "about:blank"
        self.executed: list[ComputerAction] = []
        self.file: Path | None = None
        self.__class__.instances.append(self)

    async def start(self) -> None:
        return None

    async def close(self) -> None:
        return None

    async def navigate(self, url: str) -> None:
        self.url = url

    def current_url(self) -> str:
        return self.url

    async def capture(self) -> Screenshot:
        image = Image.new("RGB", (1920, 1080), "#f5f5f5")
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        return Screenshot(
            payload=buffer.getvalue(),
            page_url=self.url,
            page_title="Fake browser",
            width=1920,
            height=1080,
        )

    async def execute(self, action: ComputerAction) -> None:
        self.executed.append(action)

    async def inspect_target(
        self,
        action: ComputerAction,
    ) -> TargetInspection | None:
        if action.action == "type":
            return TargetInspection(
                element_ref="password-ref",
                tag="input",
                input_type="password",
                text="Password",
                href="",
                form_action="",
                download=False,
                is_password=True,
                is_file=False,
                is_submit=False,
            )
        return None

    async def set_input_file(self, _: str, path: Path) -> None:
        self.file = path

    async def read_lab_state(self) -> dict[str, Any]:
        return {
            "columns": {
                "backlog": ["release-notes"],
                "progress": ["polish-demo", "browser-tests"],
                "done": ["tag-release"],
            }
        }


def settings(tmp_path: Path) -> Settings:
    return replace(
        Settings.from_env(),
        data_root=tmp_path / "data",
        labs_root=tmp_path / "labs",
        models=("test-model",),
        default_model="test-model",
        default_max_turns=7,
        allow_private_urls=True,
    )


async def wait_for_status(
    manager: RunnerManager,
    run_id: str,
    expected: RunStatus,
) -> None:
    for _ in range(200):
        if (await manager.get_run(run_id)).status is expected:
            return
        await asyncio.sleep(0.01)
    raise AssertionError(f"run did not reach {expected}")


async def test_verified_scenario_run_persists_replay(tmp_path: Path) -> None:
    model = FakeModelClient(
        [
            (
                "<tool_call><function=computer_use>"
                "<parameter=action>terminate</parameter>"
                "<parameter=status>success</parameter>"
                "</function></tool_call>"
            )
        ]
    )
    manager = RunnerManager(
        settings(tmp_path),
        model_client=model,  # type: ignore[arg-type]
        computer_factory=FakeComputer,
    )
    started = await manager.start_run(
        StartRunRequest(
            prompt="Complete the board.",
            scenario_id="kanban-reprioritize",
            model="test-model",
            browser_mode=BrowserMode.HEADLESS,
        )
    )
    await wait_for_status(manager, started.run_id, RunStatus.COMPLETED)
    detail = await manager.get_run(started.run_id)
    assert detail.max_turns == 7
    assert detail.summary is not None
    assert detail.summary.outcome is RunOutcome.SUCCESS
    assert detail.summary.verification is VerificationStatus.PASSED
    assert (tmp_path / "data" / "runs" / started.run_id / "replay.json").is_file()
    await manager.close()


async def test_sensitive_custom_url_action_waits_for_approval(tmp_path: Path) -> None:
    model = FakeModelClient(
        [
            (
                "<tool_call><function=computer_use>"
                "<parameter=action>type</parameter>"
                "<parameter=text>secret</parameter>"
                "</function></tool_call>"
            ),
            (
                "<tool_call><function=computer_use>"
                "<parameter=action>terminate</parameter>"
                "<parameter=status>success</parameter>"
                "</function></tool_call>"
            ),
        ]
    )
    manager = RunnerManager(
        settings(tmp_path),
        model_client=model,  # type: ignore[arg-type]
        computer_factory=FakeComputer,
    )
    started = await manager.start_run(
        StartRunRequest(
            prompt="Sign in.",
            target_url="http://127.0.0.1:8000",
            custom_url_risk_acknowledged=True,
            model="test-model",
        )
    )
    await wait_for_status(
        manager,
        started.run_id,
        RunStatus.WAITING_FOR_APPROVAL,
    )
    detail = await manager.get_run(started.run_id)
    assert detail.pending_intervention is not None
    assert detail.pending_intervention.kind == "password"
    assert detail.pending_intervention.action["text"] == "[REDACTED]"
    await manager.resolve_approval(
        started.run_id,
        detail.pending_intervention.id,
        ApprovalRequest(decision="approve"),
    )
    await wait_for_status(manager, started.run_id, RunStatus.COMPLETED)
    final = await manager.get_run(started.run_id)
    assert final.summary is not None
    assert final.summary.outcome is RunOutcome.UNVERIFIED
    assert FakeComputer.instances[-1].executed[0].action == "type"
    await manager.close()


async def test_local_runner_uses_strict_task_context_controller(tmp_path: Path) -> None:
    legacy = (
        '<context_update>{"status":"in_progress","completed":[],'
        '"current_state":["legacy policy state"],"facts":[],"failures":[],'
        '"next_steps":[]}</context_update>'
    )
    wait_action = (
        "<tool_call><function=computer_use>"
        "<parameter=action>wait</parameter>"
        "</function></tool_call>"
    )
    terminate_action = (
        "<tool_call><function=computer_use>"
        "<parameter=action>terminate</parameter>"
        "<parameter=status>success</parameter>"
        "</function></tool_call>"
    )
    strict_first = (
        '{"status":"in_progress","completed":[],'
        '"current_state":["strict controller state"],"facts":[],"failures":[],'
        '"next_steps":["finish"]}'
    )
    strict_final = (
        '{"status":"completed","completed":["finished"],'
        '"current_state":[],"facts":[],"failures":[],"next_steps":[]}'
    )
    model = FakeModelClient(
        [f"{legacy}\n{wait_action}", strict_first, terminate_action, strict_final]
    )
    configured = replace(settings(tmp_path), context_memory=True)
    manager = RunnerManager(
        configured,
        model_client=model,  # type: ignore[arg-type]
        computer_factory=FakeComputer,
    )
    started = await manager.start_run(
        StartRunRequest(
            prompt="Wait, then finish.",
            target_url="http://127.0.0.1:8000",
            custom_url_risk_acknowledged=True,
            model="test-model",
        )
    )

    await wait_for_status(manager, started.run_id, RunStatus.COMPLETED)

    context_path = (
        tmp_path / "data" / "runs" / started.run_id / "context" / "state.json"
    )
    for _ in range(100):
        if context_path.is_file():
            break
        await asyncio.sleep(0.01)
    assert context_path.is_file()
    exported = json.loads(context_path.read_text(encoding="utf-8"))
    assert exported["revision"] == 2
    assert exported["snapshot"]["completed"] == ["finished"]
    assert "legacy policy state" not in context_path.read_text(encoding="utf-8")
    assert model.calls[1]["response_format"]["type"] == "json_schema"
    assert model.calls[3]["response_format"]["type"] == "json_schema"
    detail = await manager.get_run(started.run_id)
    events = await manager.get_events(started.run_id)
    assert detail.summary is not None
    assert any(event.type == "context_policy_update_ignored" for event in events)
    await manager.close()
