# Qwen3.5 Context Memory V2 / multi_apps Binary Acc

- 日期：2026-08-03
- 分支：`codex/context-skill-memory`
- 数据：OSWorld `multi_apps` 全部 93 tasks
- 指标：只有 evaluator `result == 1.0` 才算 pass；fractional reward 一律按失败
- 采样：temperature 0.6，top_p 0.95，top_k 20，单次 run

## 结论

V2 采用 harness-level meta skill 和 task-local evolving context。Action policy 只生成
GUI action；每个有效 action 后由独立的 strict JSON Schema updater 产生 full-replacement
快照。它不是每个 task 一份特殊 Skill，也不会在 task 间共享动态 memory。

9B 的两种模式都比各自 baseline 增加 4 个 exact pass：nothink 从 17/93 提升到
21/93，think 从 16/93 提升到 20/93，均为 +4.30 pp。9B nothink V1 的自由格式
context 协议只有 48.83% 语义覆盖率，Acc 反而从 17/93 降到 16/93；V2 的有效
action updater 覆盖率为 100%，且两个 9B run 都是 0 updater failure。

35B nothink 的 node05 全量结果尚在运行，完成后补入同一报告。

## Binary Acc

| 模型 / 模式 | 协议 | Exact pass | Acc | 对对应 baseline |
|---|---|---:|---:|---:|
| Qwen3.5-9B nothink | baseline | 17/93 | 18.28% | — |
| Qwen3.5-9B nothink | V1 free-format | 16/93 | 17.20% | -1 pass / -1.08 pp |
| Qwen3.5-9B nothink | V2 strict updater | 21/93 | 22.58% | +4 pass / +4.30 pp |
| Qwen3.5-9B think | baseline | 16/93 | 17.20% | — |
| Qwen3.5-9B think | V2 strict updater | 20/93 | 21.51% | +4 pass / +4.30 pp |
| Qwen3.5-35B-A3B nothink | baseline | 22/93 | 23.66% | — |
| Qwen3.5-35B-A3B nothink | V2 strict updater | 待完成 | 待完成 | 待完成 |

逐题 paired transition：

| V2 run | Gains | Losses | Unchanged |
|---|---:|---:|---:|
| 9B nothink vs nothink baseline | 10 | 6 | 77 |
| 9B think vs think baseline | 10 | 6 | 77 |

9B think V2 比 9B nothink V2 少 1 个 pass（-1.08 pp）；这不是同一模式的
baseline 对照，不能单独归因于 thinking。

## Context 协议审计

| Run | Valid-action updater | 成功 | 失败 | 覆盖率 |
|---|---:|---:|---:|---:|
| 9B nothink V2 | 3,328 | 3,328 | 0 | 100% |
| 9B think V2 | 1,248 | 1,248 | 0 | 100% |
| 35B nothink V2 | 待完成 | 待完成 | 待完成 | 待完成 |

9B think 有 166 次 action repair / parse failure、61 次 no-tool finish；这些轮没有
有效 action 时不会伪造 context updater 成功。两条 Docker 初始化 404 样本已真实重跑，
覆盖原始失败 artifact 后再统计 93-task Acc。

## 运行资源

- node89 / 9B：GPU0–3，TP=1、DP=4、API servers=4；全量完成并释放四张卡。
- node05 / 35B-A3B：GPU0–7，TP=1、DP=8、API servers=8；从 32 workers
  断点提升到 64 workers，已完成 task 保留，未完成 task 清理后重跑。
- max steps=50，history=50，最多保留 5 张近期图片，context 最多 8 项 / 6000 字符。

## 可比性限制

- 这是 temperature 0.6 的单次 run，不是多 seed 置信区间；+4 pass 需要更多 seed
  才能判断是否稳定。
- baseline 与 V2 使用同一模型 revision 和推理超参数，但基础设施 repair、并行度和
  direct-network 路由不能保证逐 task 完全相同。
- V2 会为每个有效 action 增加一次 updater 推理请求；本报告只比较 Acc，不比较
  token 成本、延迟或 reward 总分。

## 结果目录

- 9B nothink V2：`qwen35-9b-context-v2-multiapps_20260803_165847`
- 9B think V2：`qwen35-9b-think-context-v2-multiapps_20260803_174839`
- 35B nothink V2：`qwen35-35b-a3b-context-v2-multiapps_20260803_174500`
