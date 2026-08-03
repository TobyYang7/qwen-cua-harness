# Qwen3.5-9B Context Memory / multi_apps 评测

- 日期：2026-08-03
- 分支：`codex/context-skill-memory`
- 模型：`Qwen/Qwen3.5-9B@c202236235762e1c871ad0ccb60c8ee5ba337b9a`
- 模式：nonthink，temperature 0.6，top_p 0.95，top_k 20
- 数据：OSWorld `multi_apps` 全部 93 tasks，max steps 50
- 实验 run：`qwen35-9b-context-multiapps_20260803_155400`
- baseline：`qwen35-9b_qwencua_20260803_110049`

## 结论

当前原型验证了“harness-level meta skill + task-local evolving context”可以在
OSWorld 全链路运行，但这个自由格式协议对 Qwen3.5-9B 没有产生 Acc 收益。
按 `result == 1` 的唯一口径，baseline 为 17/93 = 18.28%，V1 为
16/93 = 17.20%，减少 1 个 pass / 1.08 pp；逐题是 7 gains、8 losses、
78 unchanged。fractional reward 一律不算 pass。

主要瓶颈不是 context 存储，而是模型对输出协议的遵循：2,814 个有效决策轮中，
主动作只有 22 次直接带 `context` parameter。harness 因此几乎每轮都需要
side-hop，而 side-hop 只成功产生 1,352 个语义快照。最终语义 context 覆盖率为
1,374 / 2,814 = 48.83%。确定性 recent-action tail 仍在有效动作后演化，
但它无法替代稳定的语义快照。

## 历史 reward 与行为诊断（不作为 Acc）

以下 reward 仅保留用于复现实验行为，不用于模型效果对比。

| 指标 | Baseline | Context prototype | 差值 |
|---|---:|---:|---:|
| 总分 | 17.7718 | 16.7103 | -1.0614 |
| 平均成功率 | 19.1094% | 17.9681% | -1.1413 pp |
| 得分 > 0 的 task | 18 | 17 | -1 |
| 平均步数 | 40.89 | 30.26 | -10.63 |
| 中位步数 | 50 | 43 | -7 |
| 跑满 50 步的 task | 60 | 46 | -14 |

更少的步数不能直接解读为效率提升：多个 baseline 成功、context 回退的 task
在 2–5 步就结束，表明 meta/context prompt 可能增加了过早终止。

## Context 协议指标

| 指标 | 值 |
|---|---:|
| 去重后的模型决策轮 | 2,814 |
| 主动作直接带 context | 22 (0.78%) |
| side-hop 成功快照 | 1,352 |
| 语义快照总数 | 1,374 (48.83%) |
| 有过至少一次语义快照的 task | 86 / 93 |
| 有动作但没有语义 context 的决策轮 | 1,420 |

按有效 action 轮估算，side-hop 尝试了 2,772 次，成功率约 48.77%。
这意味着当前实现接近翻倍了模型请求数，却没有保证每轮语义 memory。

## 分数发生变化的 tasks

| Task ID | Baseline | Context | Delta |
|---|---:|---:|---:|
| `09a37c51-e625-49f4-a514-20a773797a8a` | 0.7718 | 0 | -0.7718 |
| `0e5303d4-8820-42f6-b18d-daf7e633de21` | 0 | 1 | +1 |
| `3a93cae4-ad3e-403e-8c12-65303b271818` | 1 | 0 | -1 |
| `47f7c0ce-a5fb-4100-a5e6-65cd0e7429e5` | 0 | 0.7103 | +0.7103 |
| `48d05431-6cd5-4e76-82eb-12b60d823f7d` | 0 | 1 | +1 |
| `510f64c8-9bcc-4be1-8d30-638705850618` | 0 | 1 | +1 |
| `58565672-7bfe-48ab-b828-db349231de6b` | 1 | 0 | -1 |
| `68a25bd4-59c7-4f4d-975e-da0c8509c848` | 1 | 0 | -1 |
| `6d72aad6-187a-4392-a4c4-ed87269c51cf` | 1 | 0 | -1 |
| `716a6079-22da-47f1-ba73-c9d58f986a38` | 0 | 1 | +1 |
| `8df7e444-8e06-4f93-8a1a-c5c974269d82` | 1 | 0 | -1 |
| `9219480b-3aed-47fc-8bac-d2cffc5849f7` | 0 | 1 | +1 |
| `a503b07f-9119-456b-b75d-f5146737d24f` | 1 | 0 | -1 |
| `a74b607e-6bb5-4ea8-8a7c-5d97c7bbcd2a` | 1 | 0 | -1 |
| `acb0f96b-e27c-44d8-b55f-7cb76609dfcd` | 0 | 1 | +1 |
| `c867c42d-a52d-4a24-8ae3-f75d256b5618` | 0 | 1 | +1 |
| `ce2b64a2-ddc1-4f91-8c7d-a88be7121aac` | 1 | 0 | -1 |

## 运行配置与可比性限制

- 两边使用同一模型 revision、nothink、sampling、history/image window 和 max steps。
- 前 17 个 context task 在 GPU0 单 replica 上完成；剩余 76 个在 GPU0–3 上以
  TP=1 / DP=4 / API=4 断点续跑。并行度只改变吞吐，不改变单请求推理参数。
- 当前 run 对所有 task 使用 direct network；baseline 是修复后的 structured 结果，
  包含重跑/合并的基础设施失败 task，不能保证每个 task 的网络路由完全相同。
- 93 个 multi_apps task 中有 18 个在 baseline metadata 中标记 `requires_proxy`。
- temperature 为 0.6，且只跑了一个 seed/run。-1.14 pp 不应解读为统计显著的
  memory 负收益，但足以说明当前协议没有证明正收益。

## 下一步

1. 使用 vLLM structured JSON / JSON schema 约束 side-hop，不再依赖 XML 自由生成。
2. 不再要求主 action 携带 context；把 memory updater 明确作为 harness-owned controller，
   避免 meta prompt 干扰 action policy。
3. 将 updater 输入缩到 task、上一快照、当前观察和当前 action，并评估更小的
   专用 memory controller，降低近 2x 的推理开销。
4. 增加 `context off` / `deterministic action tail only` / `semantic snapshot` 三组消融，
   使用 GPU0–3、TP=1 / DP=4 / API=4 以相同网络配置至少跑 3 个 seed。
