# Results 数据格式

本目录用于保存 `qwen-cua-harness` 的结构化评测结果。每次评测按下面的层级单独存放：

```text
results/
└── {config_name}/
    └── {run_id}/
        ├── config.yaml
        ├── manifest.json
        ├── summary.json
        ├── episodes.jsonl
        ├── steps.jsonl
        └── checksums.json
```

- `config_name`：本次评测使用的配置名称，例如 `qwen3.5_35b_nothink`。
- `run_id`：一次实际运行的唯一标识，通常包含模型名和启动时间。
- 同一个 config 可以保存多次运行，彼此不会覆盖。

## 如何生成

结构化结果由本仓库的 `scripts/export_results.py` 生成。脚本读取 OSWorld 的原始 run、
模型 YAML、任务 manifest、每题的 `result.txt`、`traj.jsonl`、`runtime.log`、截图和录屏，
然后完成以下处理：

1. 按任务 manifest 固定 domain 和 task 顺序。
2. 读取 `result.txt` 得到每题 score。
3. 逐行解析 `traj.jsonl`，生成任务级 `episodes.jsonl` 和步骤级 `steps.jsonl`。
4. 使用本仓库的 Qwen-CUA XML parser 生成 `parsed_actions` 和解析诊断。
5. 从错误文本识别 proxy、Docker、初始化和超时等基础设施失败。
6. 分别计算 official 与 clean 口径，并生成 domain 聚合。
7. 保存 config 快照、运行 manifest 和 SHA-256。

通常不需要手动调用：`scripts/run_eval.py ... osworld` 在评测结束后会自动导出。如果要
把一个已经存在的原始 run 补充导出，可以执行：

```bash
python scripts/export_results.py \
  ../../osworld_eval/results/osworld-std/{run_id} \
  configs/models/{config_name}.yaml
```

附加 Slurm 元数据：

```bash
python scripts/export_results.py \
  ../../osworld_eval/results/osworld-std/{run_id} \
  configs/models/{config_name}.yaml \
  --slurm-log ../../logs/slurm_job.log \
  --job-id 1234 \
  --node node89 \
  --exit-code 0
```

默认拒绝覆盖已有的 `results/{config_name}/{run_id}`。确认需要重新生成时显式传入
`--overwrite`。脚本只读取原始 run，不修改或移动原始截图、轨迹和录屏。

## 路径约定

结构化数据中的 `*_path`、`config_file` 等文件引用均为相对路径，统一以 **`qwen-cua-harness` 仓库根目录** 为基准解析，不以当前 run 目录为基准。

例如：

```json
{
  "raw_run_path": "../../osworld_eval/results/osworld-std/example_run"
}
```

在仓库根目录中可按以下方式访问：

```bash
cd /path/to/qwen-cua-harness
ls ../../osworld_eval/results/osworld-std/example_run
```

`model_response`、`executed_action` 等原始轨迹文本中可能出现 `/home/user/...` 或 `/tmp/...`。这些是 OSWorld 虚拟机内部的路径，属于模型输出或执行动作的一部分，不是评测产物的文件引用，因此保留原文。

## 文件说明

### `config.yaml`

保存本次运行实际采用的关键模型、推理和服务参数，用于复现实验。主要分为：

- `model`：模型 ID、revision 和服务名称。
- `inference`：thinking 开关、采样参数、最大输出长度、历史轮数和图片数量。
- `serving`：服务地址、端口、精度、并行方式、显存比例和上下文长度等部署参数。
- `schema_version`：该文件所使用的数据结构版本。

### `manifest.json`

记录一次运行的身份、来源和运行环境，是读取整个 run 时的入口文件。

| 字段 | 含义 |
| --- | --- |
| `schema_version` | 数据结构版本。当前为 `1`。 |
| `benchmark` | 评测集名称，例如 `osworld`。 |
| `config_name` | 配置名称，对应上一级目录名。 |
| `run_id` | 本次运行 ID，对应当前目录名。 |
| `status` | 整次运行的状态。 |
| `expected_tasks` | 预期任务总数。 |
| `started_at` / `finished_at` | UTC 格式的运行起止时间。 |
| `path_base` | 相对路径的解析基准。 |
| `config_file` | 本次运行保存的配置快照。 |
| `config_sha256` | 配置快照的 SHA-256。 |
| `raw_run_path` | OSWorld 原始运行结果目录。 |
| `source_args_path` | 原始评测参数文件。 |
| `task_manifest_path` | 评测任务清单。 |
| `slurm_log_path` | Slurm 作业日志。 |
| `model` | 模型 ID、revision 和服务名称。 |
| `inference` | 本次推理参数。 |
| `serving` | 本次部署参数。 |
| `slurm` | job ID、节点、退出码等作业信息。 |
| `git` | harness 仓库及上层仓库的 commit，用于代码版本追踪。 |

### `summary.json`

保存本次运行的聚合统计，适合直接用于报表和不同 config 之间的比较。

- `result_count`：实际记录的任务数。
- `step_count`：所有任务的步骤总数。
- `expected_tasks`：预期任务数。
- `official`：官方统计口径。
  - `denominator`：所有已记录任务数，包括基础设施失败任务。
  - `score_sum`：所有任务得分之和。
  - `mean_score`：`score_sum / denominator`。
  - `positive_tasks` / `zero_tasks`：得分大于 0 和等于 0 的任务数量。
- `clean`：排除明确分类为基础设施、初始化或超时失败后的统计口径。
  - `definition`：当前 clean 集合的筛选规则。
  - `denominator`、`score_sum`、`mean_score`：clean 集合的任务数、总分和平均分。
- `by_domain`：按应用或任务域统计的 `count`、`score_sum` 和 `mean_score`。
- `status_counts`：按任务状态汇总的数量。
- `failure_counts`：按失败类型汇总的数量，例如 `proxy_pool_empty`、`docker_404`。

OSWorld 的单题 `score` 可能是 `0`、`1`，也可能是二者之间的部分得分。因此总体正确率使用平均 score，不应只用“得分大于 0 的任务数 / 总任务数”。

### `episodes.jsonl`

任务级数据。每一行是一个独立 JSON 对象，对应一个评测任务。JSONL 可以逐行流式读取，无需一次把整个结果加载到内存。

| 字段 | 含义 |
| --- | --- |
| `schema_version` | 数据结构版本。 |
| `config_name` / `run_id` | 所属配置和运行。 |
| `task_id` | OSWorld 任务的唯一 ID。 |
| `domain` | 任务所属应用或任务域。 |
| `status` | `completed` 或 `infra_failed` 等任务状态。 |
| `score` | 该任务的最终得分。 |
| `steps` | 该任务实际记录的步骤数。 |
| `duration_seconds_observed` | 从轨迹中观测到的持续时间。 |
| `failure_type` / `failure_message` | 失败分类及原始错误信息；成功时为 `null`。 |
| `repairs` / `repair_failures` | 动作解析修复次数及修复失败次数。 |
| `no_action_turns` | 没有生成有效动作的轮数。 |
| `trajectory_path` | 原始 `traj.jsonl` 的相对路径。 |
| `result_path` | 原始 `result.txt` 的相对路径。 |
| `runtime_log_path` | 任务运行日志的相对路径。 |
| `recording_path` | 任务录屏的相对路径。 |

任务的主键是 `(run_id, task_id)`。可使用该组合与 `steps.jsonl` 关联。

### `steps.jsonl`

步骤级数据。每一行对应模型与环境的一次交互，使用 `(run_id, task_id, step_num)` 唯一定位。

| 字段 | 含义 |
| --- | --- |
| `schema_version` | 数据结构版本。 |
| `config_name` / `run_id` | 所属配置和运行。 |
| `task_id` / `domain` | 所属任务及任务域。 |
| `step_num` | 任务内的步骤序号。 |
| `action_timestamp` | 动作记录时间。 |
| `model_response` | 模型原始输出，完整保留。 |
| `parsed_actions` | 从模型输出中解析出的结构化动作列表。 |
| `executed_action` | 环境实际执行的动作。 |
| `tool_call_valid` | 模型工具调用是否成功解析。 |
| `parse_error` | 解析错误信息；无错误时为 `null`。 |
| `thinking_chars` | 检测到的 thinking 文本字符数。 |
| `reward` | 当前步骤记录的 reward。 |
| `done` | 执行该步骤后任务是否结束。 |
| `info` | 环境返回的附加信息，内容可能随任务变化。 |
| `screenshot_path` | 当前步骤截图的相对路径。 |

截图、录屏和原始轨迹不会复制进 `results/`，而是通过相对路径指向原始 OSWorld 结果，避免重复占用大量存储空间。

### `checksums.json`

记录当前 run 中核心结构化文件的 SHA-256，用于检查文件是否被意外修改或传输损坏。

```json
{
  "algorithm": "sha256",
  "files": {
    "config.yaml": "...",
    "episodes.jsonl": "...",
    "manifest.json": "...",
    "steps.jsonl": "...",
    "summary.json": "..."
  }
}
```

校验方式：

```bash
cd results/{config_name}/{run_id}
sha256sum config.yaml episodes.jsonl manifest.json steps.jsonl summary.json
```

## 数据之间的关系

```text
manifest.json
  ├── config.yaml                 本次运行参数快照
  ├── summary.json                run 级聚合统计
  ├── episodes.jsonl              每个 task 一行
  │     └── (run_id, task_id)
  └── steps.jsonl                 每个 step 一行
        └── (run_id, task_id, step_num)
```

推荐读取顺序：

1. 读取 `manifest.json` 确认 run 身份、模型版本、代码版本和原始数据位置。
2. 读取 `summary.json` 查看整体结果。
3. 读取 `episodes.jsonl` 分析单题得分和失败类型。
4. 需要排查具体行为时，再按 `run_id + task_id` 查询 `steps.jsonl`，并结合截图、录屏和原始轨迹分析。

## 简单读取示例

Python：

```python
import json
from pathlib import Path

run_dir = Path("results/qwen3.5_35b_nothink/<run_id>")

summary = json.loads((run_dir / "summary.json").read_text())

with (run_dir / "episodes.jsonl").open() as f:
    episodes = [json.loads(line) for line in f]

failed = [item for item in episodes if item["status"] != "completed"]
print(summary["official"]["mean_score"])
print(len(failed))
```

命令行：

```bash
jq '.official, .clean, .failure_counts' \
  results/{config_name}/{run_id}/summary.json

jq -c 'select(.status != "completed")' \
  results/{config_name}/{run_id}/episodes.jsonl
```

## 兼容性约定

- 读取程序应先检查 `schema_version`，避免把不同版本的数据结构混用。
- 新增可选字段时，读取程序应允许字段缺失或值为 `null`。
- 不应根据目录位置推断模型参数；模型和部署参数应以 `manifest.json` 与 `config.yaml` 为准。
- 分析时应明确使用 `official` 还是 `clean` 口径，不能将两种分母混在一起比较。

## OSWorld 当前评测结果

下表汇总当前已经完成的 OSWorld 361 题结果。`Overall` 和各类别准确率均为对应任务
`score` 的平均值；类别名称与 OSWorld domain 映射如下：`Calc`（libreoffice_calc）、
`Impress`（libreoffice_impress）、`Writer`（libreoffice_writer）、`Multi-apps`
（multi_apps），其余列使用 domain 的常用名称。

本轮异常任务已经重新评测并合并回原始 result，因此最终表中的四组结果均为
`361/361 completed`、无剩余基础设施错误。修复任务采用 VM 直连网络以绕过空 proxy
pool，与正式 proxy 网络条件不完全等价。

| Model | Profile | Thinking | Tasks | Overall | Chrome | GIMP | Calc | Impress | Writer | Multi-apps | OS | Thunderbird | VLC | VS Code | Result run |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Qwen3.5-35B-A3B | `qwen3.5_35b_nothink.yaml` | off | 361/361 | **41.63%** | 47.74% | 69.23% | 23.40% | 40.22% | 30.42% | 25.29% | 54.17% | 66.67% | 52.32% | 78.26% | `qwen35-35b-a3b_qwencua_20260803_101359` |
| Qwen3.5-35B-A3B | `qwen3.5_35b_think.yaml` | on | 361/361 | **42.70%** | 56.43% | 57.69% | 21.28% | 31.71% | 56.51% | 29.33% | 58.33% | 73.33% | 41.18% | 69.57% | `qwen35-35b-a3b-think_qwencua_20260803_103643` |
| Qwen3.5-9B | `qwen3.5_9b_nothink.yaml` | off | 361/361 | **37.29%** | 41.22% | 46.15% | 27.66% | 38.09% | 43.47% | 19.11% | 62.50% | 60.00% | 47.06% | 56.52% | `qwen35-9b_qwencua_20260803_110049` |
| Qwen3.5-9B | `qwen3.5_9b_think.yaml` | on | 361/361 | **33.11%** | 34.69% | 46.15% | 31.91% | 31.91% | 52.16% | 18.91% | 41.67% | 46.67% | 41.18% | 34.78% | `qwen35-9b-think_qwencua_20260803_115245` |

结构化结果位于本目录下的 `structured/<run-id>/summary.json`；原始异常任务的备份位于
`repair-backups/`。Qwen3.5-4B 的 nothink/think 评测当前正在 node05 上各使用 4 张
GPU，完成后再追加到本表。
