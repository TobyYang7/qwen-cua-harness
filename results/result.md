# OSWorld 当前评测结果

下表汇总当前已经完成的 OSWorld 361 题结果。`Overall` 和各类别准确率均为对应任务
`score` 的平均值；类别名称与 OSWorld domain 映射如下：`Calc`（libreoffice_calc）、
`Impress`（libreoffice_impress）、`Writer`（libreoffice_writer）、`Multi-apps`
（multi_apps），其余列使用 domain 的常用名称。`Avg steps` 为 361 个任务的
`steps` 字段算术平均值。

本轮异常任务已经重新评测并合并回原始 result，因此最终表中的四组结果均为
`361/361 completed`、无剩余基础设施错误。修复任务采用 VM 直连网络以绕过空 proxy
pool，与正式 proxy 网络条件不完全等价。

| Model | Profile | Thinking | Tasks | Overall | Avg steps | Chrome | GIMP | Calc | Impress | Writer | Multi-apps | OS | Thunderbird | VLC | VS Code | Result run |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Qwen3.5-35B-A3B | `qwen3.5_35b_nothink.yaml` | off | 361/361 | **41.63%** | 28.76 | 47.74% | 69.23% | 23.40% | 40.22% | 30.42% | 25.29% | 54.17% | 66.67% | 52.32% | 78.26% | `qwen35-35b-a3b_qwencua_20260803_101359` |
| Qwen3.5-35B-A3B | `qwen3.5_35b_think.yaml` | on | 361/361 | **42.70%** | 28.45 | 56.43% | 57.69% | 21.28% | 31.71% | 56.51% | 29.33% | 58.33% | 73.33% | 41.18% | 69.57% | `qwen35-35b-a3b-think_qwencua_20260803_103643` |
| Qwen3.5-9B | `qwen3.5_9b_nothink.yaml` | off | 361/361 | **37.29%** | 32.80 | 41.22% | 46.15% | 27.66% | 38.09% | 43.47% | 19.11% | 62.50% | 60.00% | 47.06% | 56.52% | `qwen35-9b_qwencua_20260803_110049` |
| Qwen3.5-9B | `qwen3.5_9b_think.yaml` | on | 361/361 | **33.11%** | 20.45 | 34.69% | 46.15% | 31.91% | 31.91% | 52.16% | 18.91% | 41.67% | 46.67% | 41.18% | 34.78% | `qwen35-9b-think_qwencua_20260803_115245` |

结构化结果位于本目录下的 `structured/<run-id>/summary.json`；原始异常任务的备份位于
`repair-backups/`。Qwen3.5-4B 的 nothink/think 评测当前正在 node05 上各使用 4 张
GPU，完成后再追加到本表。
