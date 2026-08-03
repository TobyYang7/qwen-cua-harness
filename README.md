# Qwen-CUA Harness

Qwen3.5 computer-use 的独立 harness，包含：

- 基于截图和 XML `computer_use` 工具调用的 Qwen-CUA agent；
- 可接入 OSWorld / CUA-Gym 的评测 adapter；
- 本地 FastAPI + Playwright runner 与 Web 控制台；
- 由单一 YAML profile 驱动的 vLLM 部署脚本。

本仓库不包含模型权重、训练代码或托管推理服务。

## 模型配置

`configs/models/` 提供四个可直接部署的 profile：

| Profile | 模型 | Thinking | `max_tokens` |
|---|---|---:|---:|
| `qwen3.5_9b_nothink.yaml` | Qwen3.5-9B | 关闭 | 2048 |
| `qwen3.5_9b_think.yaml` | Qwen3.5-9B | 开启 | 8192 |
| `qwen3.5_35b_nothink.yaml` | Qwen3.5-35B-A3B | 关闭 | 2048 |
| `qwen3.5_35b_think.yaml` | Qwen3.5-35B-A3B | 开启 | 8192 |

每个 profile 只保留三类信息：

- `model`：Hugging Face 模型 ID、固定 revision、served name；
- `inference`：agent 请求使用的采样、文本历史和图像预算；
- `serving`：vLLM 启动参数。

历史 run ID、Slurm job ID、某次补跑并发数等运行记录不属于部署配置，不再放进
profile。脚本使用严格 schema：缺字段、多字段、类型错误或数值越界时会直接报错，
避免配置项被静默忽略。vLLM 的图像上限由 `inference.image_max` 派生，不维护重复字段。

## 部署 vLLM

安装 Python 包：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

先检查将执行的完整命令：

```bash
python scripts/deploy_vllm.py \
  configs/models/qwen3.5_9b_nothink.yaml \
  --vllm-bin /data/toby/vllm-env/bin/vllm \
  --dry-run
```

确认后启动服务（去掉 `--dry-run`）：

```bash
python scripts/deploy_vllm.py \
  configs/models/qwen3.5_9b_nothink.yaml \
  --vllm-bin /data/toby/vllm-env/bin/vllm
```

也可以通过 `VLLM_BIN` 指定二进制：

```bash
VLLM_BIN=/data/toby/vllm-env/bin/vllm \
  python scripts/deploy_vllm.py configs/models/qwen3.5_35b_think.yaml
```

除 vLLM 可执行文件路径外，脚本不接受模型、端口或并行度的命令行覆盖；需要调整时请
修改或复制 YAML。这样 dry-run、正式部署和后续复现实验使用的是同一份参数。

部署脚本会显式传入所有 `serving` 字段，从 `inference.image_max` 派生服务端图像上限，
并把 `inference.enable_thinking` 转成：

```text
--default-chat-template-kwargs '{"enable_thinking":true|false}'
```

因此 think/nothink 不依赖 vLLM 或模型模板的隐式默认值。Qwen-CUA 的 XML 工具由
agent 自己解析，不需要 `--enable-auto-tool-choice` 或 `--tool-call-parser`。

> 35B-A3B profile 使用 DP=8、每张 GPU 一份完整模型副本；部署前请确认八张可见 GPU
> 都有足够显存。端口、DP、上下文长度等资源相关参数都可以在复制出的 profile 中修改。

## 调用参数

服务端的 thinking 默认值由 profile 固定；评测端仍应从同一 profile 读取
`inference`，并传给 `QwenCUAAgent`：

```python
from pathlib import Path

from qwen_cua.deploy import load_profile
from qwen_cua.eval.osworld import QwenCUAAgent

profile = load_profile(Path("configs/models/qwen3.5_9b_nothink.yaml"))
model = profile["model"]
inference = profile["inference"]

agent = QwenCUAAgent(
    model=model["served_name"],
    **inference,
    surface="desktop",
)
response, actions = agent.predict(instruction, {"screenshot": png_bytes})
```

vLLM endpoint 使用 OpenAI Chat Completions 协议，默认地址由 profile 的
`serving.host` / `serving.port` 决定，例如：

```bash
export OPENAI_BASE_URL=http://127.0.0.1:8953/v1
export OPENAI_API_KEY=EMPTY
```

OSWorld / CUA-Gym runner 负责 VM 生命周期、截图、动作执行和评分；adapter 只负责消息
历史、工具协议解析、坐标转换与 malformed-call repair。

## 本地交互式 runner

如需运行浏览器 demo：

```bash
python -m playwright install chromium
corepack enable
corepack prepare pnpm@10.26.0 --activate
pnpm install
cp .env.example .env
```

在 `.env` 中设置 endpoint、API key 和 served model name，然后启动：

```bash
set -a
source .env
set +a
pnpm dev
```

Web 控制台位于 <http://127.0.0.1:3000>，runner 默认位于
<http://127.0.0.1:4001>。runner 没有认证，只应绑定 loopback，除非前面已有认证代理。

## 协议与安全

模型只看到截图，通过归一化 `0..999` 坐标输出 XML 工具调用：

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

支持点击、拖拽、移动、按键、输入、滚动、等待、截图、`call_user` 和
`terminate(success|failure)`。computer use 可能误操作，网页也可能包含 prompt
injection；请使用隔离环境，不要直接接入已登录账号或高风险工作流。

## 验证

```bash
PYTHONPATH=src python -m pytest -q
ruff check src tests scripts
python scripts/deploy_vllm.py configs/models/qwen3.5_9b_think.yaml --dry-run
```

默认测试不访问真实模型服务。

## License

本项目从 Apache-2.0 许可的
[`xlang-ai/Qwen-CUA`](https://github.com/xlang-ai/Qwen-CUA) 抽取并独立维护；见
`LICENSE` 与 `NOTICE`。
