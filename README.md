# Qwen-CUA Harness

This repository contains the standalone browser-first reference harness for
running Qwen-CUA, or a compatible computer-use model, through an
OpenAI-compatible multimodal endpoint.

[Upstream project](https://github.com/xlang-ai/Qwen-CUA) ·
[Technical report](https://github.com/xlang-ai/Qwen-CUA/blob/main/paper/Qwen-CUA.pdf) ·
[Original demo](https://github.com/xlang-ai/Qwen-CUA/tree/main/demo)

> [!NOTE]
> This is a standalone extraction of the reference demo from
> [`xlang-ai/Qwen-CUA`](https://github.com/xlang-ai/Qwen-CUA), based on upstream
> commit [`668fd21`](https://github.com/xlang-ai/Qwen-CUA/commit/668fd213b6d84cab1ad3d2d18cbb1a0f197ce3ad).
> It contains the harness, operator console, local labs, and tests. It does not
> include Qwen-CUA model weights, training code, or a hosted inference endpoint.

The demo provides:

- a Next.js operator console for starting runs and reviewing screenshots,
  actions, approvals, and replay artifacts;
- a FastAPI runner that owns model sessions and isolated Playwright browsers;
- a typed `computer_use` protocol compatible with the XML tool calls used by
  the Qwen-CUA evaluation harness;
- two deterministic local labs plus an explicitly acknowledged custom-URL mode;
- a CLI and Docker Compose setup.

The Python package also includes an unattended OSWorld-style evaluation
adapter at `qwen_cua.eval.osworld`. It preserves the reference prompt,
collapsed-screenshot history, execution feedback, malformed-call repair, and
the normalized `0..999` coordinate contract while compiling actions to the
`pyautogui` strings and sentinel tokens expected by screenshot benchmark
runners.

```python
from qwen_cua.eval.osworld import QwenCUAAgent

agent = QwenCUAAgent(
    model="qwen-cua",
    surface="desktop",
    enable_thinking=False,
    image_max=5,
)
response, actions = agent.predict(instruction, {"screenshot": png_bytes})
```

This module is intentionally environment-agnostic: OSWorld, CUA-Gym, or a
similar runner owns VM lifecycle, screenshots, action sanitization, stepping,
and scoring. The adapter only owns model messages, protocol parsing, action
compilation, repair, and per-episode history.

> [!CAUTION]
> Computer use can make mistakes and websites can contain prompt injection.
> Use fresh browser contexts, avoid authenticated or high-stakes workflows, and
> review every requested approval.
>
> The runner is a local development service and has no authentication. Keep it
> bound to loopback unless you add an authenticated reverse proxy.

## Architecture

```text
Operator Console / CLI
          |
          v
FastAPI Runner -----> OpenAI-compatible Qwen endpoint
     |                         |
     |                  XML computer_use
     v                         |
Typed Safety Gate <------------+
     |
     v
Isolated Playwright Chromium
     |
     +---- screenshots / events / replay
```

The model never receives a DOM, accessibility tree, terminal, or browser
automation API. It sees screenshots and emits mouse/keyboard actions on a
normalized `0..999` coordinate grid. The runner validates those actions and
executes them through a restricted Playwright adapter.

## Prerequisites

- Python 3.10+
- Node.js 22+
- Corepack
- an OpenAI-compatible endpoint serving Qwen-CUA or a compatible model

## Native quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
python -m playwright install chromium

corepack enable
corepack prepare pnpm@10.26.0 --activate
pnpm install

cp .env.example .env
```

Edit `.env`:

```dotenv
QWEN_CUA_BASE_URL=http://127.0.0.1:8000/v1
QWEN_CUA_API_KEY=dummy
QWEN_CUA_MODELS=your-model-name
QWEN_CUA_DEFAULT_MODEL=your-model-name
```

Start both services:

```bash
set -a
source .env
set +a
pnpm dev
```

Open <http://127.0.0.1:3000>, select a safe lab, and start a run.

On Linux, if `qwen-cua doctor` reports missing browser libraries, install them
once with `sudo python -m playwright install-deps chromium`.

Run the services separately when debugging:

```bash
qwen-cua serve
pnpm dev:web
```

## Docker Compose

Docker runs Chromium headlessly; every captured frame remains visible in the
operator console.

```bash
cp .env.example .env
# Edit the endpoint and model values in .env.
docker compose up --build
```

When the model endpoint runs on the Docker host, set
`QWEN_CUA_BASE_URL=http://host.docker.internal:<port>/v1`. Linux installations
may need to add an `extra_hosts` mapping for `host.docker.internal`.

## CLI

Check the endpoint and browser installation:

```bash
qwen-cua doctor
```

Run a safe lab without the web console:

```bash
qwen-cua run \
  --scenario kanban-reprioritize \
  --prompt "Move the cards to the target state and verify the result."
```

Run a custom URL:

```bash
qwen-cua run \
  --scenario "" \
  --url https://example.com \
  --prompt "Describe the page and terminate."
```

## Model protocol

The system prompt advertises one `computer_use` function. A model action looks
like this:

```xml
<tool_call>
<function=computer_use>
<parameter=action>
left_click
</parameter>
<parameter=coordinate>
[500, 420]
</parameter>
</function>
</tool_call>
```

Supported actions include clicks, drag, mouse movement, key presses, typing,
vertical/horizontal scroll, wait, screenshot, `call_user`, and
`terminate(success|failure)`. Multiple complete tool-call blocks may be
returned in one model turn.

## Safety model

- Built-in labs are local, reset in a fresh browser context, and verified
  against explicit state.
- Custom URLs require an acknowledgement before the run starts.
- Non-HTTP(S), credential-bearing, private, loopback, link-local, reserved, and
  multicast targets are blocked by default.
- Password entry, file upload, downloads, form submission, and cross-origin
  navigation pause for operator review.
- Uploads are selected by the operator and limited to 10 MB by default.
- API keys and the endpoint URL are runner-only configuration. The browser
  receives an allowlisted set of model names, never credentials.
- Raw model responses are local replay data. Type parameters are redacted from
  UI events to reduce accidental secret exposure.

Custom URL runs are marked `unverified`. A model claiming success is not proof
that a real-world task succeeded.

## Replay artifacts

Each run is stored under `data/runs/<run-id>/`:

```text
run.json
events.jsonl
replay.json
screenshots/
downloads/
uploads/
```

The console supports SSE reconnection, screenshot scrubbing, raw response
inspection, and replay download.

## Configuration

The complete local template is in [`.env.example`](./.env.example). Notable
settings:

- `QWEN_CUA_MODELS`: comma-separated model allowlist shown in the UI.
- `QWEN_CUA_ENABLE_THINKING`: model-side thinking toggle when supported.
- `QWEN_CUA_HISTORY_N`: maximum retained textual turns.
- `QWEN_CUA_IMAGE_MAX`: recent screenshots retained as images.
- `QWEN_CUA_ALLOW_PRIVATE_URLS`: opt-in private-network browsing.
- `QWEN_CUA_MAX_CONCURRENT_RUNS`: local runner concurrency limit.

## Development checks

```bash
pnpm generate:api
pnpm lint
pnpm typecheck
pnpm test
pnpm build
# or all checks:
pnpm check
```

Live model calls are intentionally not part of the default test suite. Unit and
integration tests use deterministic fake model responses.

## Scope and limitations

- The first release supports Chromium browser workflows, not a full Linux,
  Windows, or macOS desktop.
- It supports OpenAI-compatible Chat Completions endpoints, not in-process
  Transformers inference.
- File uploads require operator selection; the agent cannot browse arbitrary
  host files.
- Browser safety inspection reduces risk but cannot eliminate prompt injection
  or infer the consequence of every website action.

## Upstream and license

The harness was extracted from the Apache-2.0-licensed
[`xlang-ai/Qwen-CUA`](https://github.com/xlang-ai/Qwen-CUA) repository. The
upstream license and notice are preserved in [LICENSE](./LICENSE) and
[NOTICE](./NOTICE).
