# 2026-10-14 级联模型路由实验

三级级联已在真实任务中运行：最后一组 12 个任务分别在规则、Embedding、LLM 层结束 3、4、5 次。
与独立 LLM 路由相比，LLM 分类请求从 12 次降至 5 次，但全部 heldout 请求从 24 次增至 27 次；任务成功为 6/12，对照为 7/12。
因此当前证据支持流程实现与调用构成变化，不支持整体质量提升、显著提速或货币降本的主张。

实现与复现实验命令见 [三级级联路由](cascade_model_routing.md)。
本次比较使用已查看过的 12 个 heldout 文件任务，属于探索性回归实验。

先以原 12 个开发原型运行两轮；发现开发留一校准没有满足门槛的向量 gate 后，
保留该运行，再添加 48 条开发请求（24 个中英文语义组），单独运行一轮完整六策略比较。
第二组保持规则、提示词、阈值网格及接受门槛不变。严格配置仍无可用 gate 后，第三组依据开发集诊断，
将接受准确率下限改为 75%、最低接受数改为 12，增加覆盖优先的四策略对比。该取舍配置不替换严格配置。
各组的模型能力画像均重新测量，不能直接视为单因素消融；没有根据 heldout 成绩修改规则、提示词或阈值。

三次运行实际物理请求合计 **469**；所有任务均保留在计划分母中。

## cascade-v1-20261014

开发校准样本 12；heldout 重复 2 轮；物理请求 237；运行 1341.3 秒。

向量 gate：`{"accepted": 0, "correct": 0, "enabled": false, "margin_threshold": null, "precision": null, "similarity_threshold": null}`。这是开发拟合结果，不是泛化准确率。

| 策略 | 任务成功/计划 | 分类正确/已分类 | LLM 分类请求 | query embedding | 全部 heldout 请求 | p50/p95 ms |
| --- | --- | --- | ---: | ---: | ---: | --- |
| cascade | 14/24 | 16/23 | 18 | 0 | 42 | 6969.0/15782.3 |
| embedding | 13/24 | 10/24 | 0 | 24 | 49 | 5616.5/14722.5 |
| fixed_a | 13/24 | 不适用 | 0 | 0 | 24 | 5461.0/14330.4 |
| fixed_b | 5/24 | 不适用 | 0 | 0 | 24 | 6835.5/12454.3 |
| model_driven | 12/24 | 16/24 | 24 | 0 | 48 | 6898.5/16828.2 |
| rule | 13/24 | 6/24 | 0 | 0 | 25 | 5227.0/14012.4 |

级联层次（接受错误按隐藏类别离线计算，不影响执行或触发升级）：

| 层 | 访问 | 接受 | 错误接受 | 状态分布 |
| --- | ---: | ---: | ---: | --- |
| rule | 24 | 6 | 2 | {"abstain": 18, "accepted": 6} |
| embedding | 18 | 0 | 0 | {"skipped": 18} |
| llm | 18 | 17 | 5 | {"abstain": 1, "accepted": 17} |
| default | 1 | 0 | 0 | {"default": 1} |

相对于独立 LLM 路由，LLM **分类请求**由 24 次变为 18 次，减少 25.0%。这不等于全部请求、总 Token、端到端延迟或货币成本按同样比例下降。

冻结的类别到模型映射：

```json
{
  "append": "qwen3:4b",
  "clarify": "qwen3:4b",
  "multi_step": "qwen3:4b",
  "rename": "qwen3:4b",
  "select": "qwen3:4b",
  "write": "qwen3:4b"
}
```

所有类别映射到同一执行模型；本轮不能证明模型分流提高任务成功率。

任务成功率差（model_driven − cascade）的 case 配对 bootstrap：-8.3%，95% 区间 [-25.0%, 0.0%]；独立 case 数 12。

离线审计：请求账一致；阈值可从冻结开发原型重建；原评分器重评分 164/164 一致。

完整报告：`work/.test-results/routing/cascade-v1-20261014/routing_report.md`；审计：`work/.test-results/routing/cascade-v1-20261014-audit/audit.json`。

## cascade-augmented-v1-20261014

开发校准样本 60；heldout 重复 1 轮；物理请求 132；运行 784.5 秒。

向量 gate：`{"accepted": 0, "correct": 0, "enabled": false, "margin_threshold": null, "precision": null, "similarity_threshold": null}`。这是开发拟合结果，不是泛化准确率。

| 策略 | 任务成功/计划 | 分类正确/已分类 | LLM 分类请求 | query embedding | 全部 heldout 请求 | p50/p95 ms |
| --- | --- | --- | ---: | ---: | ---: | --- |
| cascade | 6/12 | 8/11 | 9 | 0 | 21 | 6648.5/15552.8 |
| embedding | 6/12 | 6/12 | 0 | 12 | 25 | 5905.5/15267.6 |
| fixed_a | 6/12 | 不适用 | 0 | 0 | 12 | 5672.0/14350.6 |
| fixed_b | 3/12 | 不适用 | 0 | 0 | 12 | 7031.0/10739.9 |
| model_driven | 6/12 | 8/12 | 12 | 0 | 24 | 6937.0/15697.2 |
| rule | 7/12 | 3/12 | 0 | 0 | 13 | 5594.0/14803.6 |

级联层次（接受错误按隐藏类别离线计算，不影响执行或触发升级）：

| 层 | 访问 | 接受 | 错误接受 | 状态分布 |
| --- | ---: | ---: | ---: | --- |
| rule | 12 | 3 | 1 | {"abstain": 9, "accepted": 3} |
| embedding | 9 | 0 | 0 | {"skipped": 9} |
| llm | 9 | 8 | 2 | {"abstain": 1, "accepted": 8} |
| default | 1 | 0 | 0 | {"default": 1} |

相对于独立 LLM 路由，LLM **分类请求**由 12 次变为 9 次，减少 25.0%。这不等于全部请求、总 Token、端到端延迟或货币成本按同样比例下降。

冻结的类别到模型映射：

```json
{
  "append": "qwen3:4b",
  "clarify": "qwen3:4b",
  "multi_step": "qwen3:4b",
  "rename": "qwen3:4b",
  "select": "qwen3:4b",
  "write": "qwen3:4b"
}
```

所有类别映射到同一执行模型；本轮不能证明模型分流提高任务成功率。

任务成功率差（model_driven − cascade）的 case 配对 bootstrap：0.0%，95% 区间 [-25.0%, 25.0%]；独立 case 数 12。

离线审计：请求账一致；阈值可从冻结开发原型重建；原评分器重评分 93/93 一致。

完整报告：`work/.test-results/routing/cascade-augmented-v1-20261014/routing_report.md`；审计：`work/.test-results/routing/cascade-augmented-v1-20261014-audit/audit.json`。

## cascade-balanced-v1-20261014

开发校准样本 60；heldout 重复 1 轮；物理请求 100；运行 581.3 秒。

向量 gate：`{"accepted": 18, "correct": 14, "enabled": true, "margin_threshold": 0.03, "precision": 0.7777777777777778, "similarity_threshold": 0.65}`。这是开发拟合结果，不是泛化准确率。

| 策略 | 任务成功/计划 | 分类正确/已分类 | LLM 分类请求 | query embedding | 全部 heldout 请求 | p50/p95 ms |
| --- | --- | --- | ---: | ---: | ---: | --- |
| cascade | 6/12 | 8/12 | 5 | 9 | 27 | 6766.0/15351.7 |
| fixed_a | 6/12 | 不适用 | 0 | 0 | 12 | 5405.5/13721.7 |
| fixed_b | 3/12 | 不适用 | 0 | 0 | 12 | 7758.0/10642.2 |
| model_driven | 7/12 | 8/12 | 12 | 0 | 24 | 6851.5/15145.5 |

级联层次（接受错误按隐藏类别离线计算，不影响执行或触发升级）：

| 层 | 访问 | 接受 | 错误接受 | 状态分布 |
| --- | ---: | ---: | ---: | --- |
| rule | 12 | 3 | 1 | {"abstain": 9, "accepted": 3} |
| embedding | 9 | 4 | 1 | {"abstain": 5, "accepted": 4} |
| llm | 5 | 5 | 2 | {"accepted": 5} |

相对于独立 LLM 路由，LLM **分类请求**由 12 次变为 5 次，减少 58.3%。这不等于全部请求、总 Token、端到端延迟或货币成本按同样比例下降。

冻结的类别到模型映射：

```json
{
  "append": "qwen3:4b",
  "clarify": "qwen3:4b",
  "multi_step": "qwen3:4b",
  "rename": "qwen3:4b",
  "select": "qwen3:4b",
  "write": "qwen3:4b"
}
```

所有类别映射到同一执行模型；本轮不能证明模型分流提高任务成功率。

任务成功率差（model_driven − cascade）的 case 配对 bootstrap：8.3%，95% 区间 [0.0%, 25.0%]；独立 case 数 12。

离线审计：请求账一致；阈值可从冻结开发原型重建；原评分器重评分 69/69 一致。

完整报告：`work/.test-results/routing/cascade-balanced-v1-20261014/routing_report.md`；审计：`work/.test-results/routing/cascade-balanced-v1-20261014-audit/audit.json`。

## 第一组的失败案例

以下来自原配置 cascade 的第一轮保存响应，未据此修改本轮规则、提示词或评分标准：

| case | 观测 | 含义 |
| --- | --- | --- |
| RH-write-02 | 正确分类 write，但结果文件缺少最后一个换行 | 类别正确不代表精确字节达标 |
| RH-select-02 | 正确分类 select，却生成 write/append 动作并改动文件 | 只读约束需要执行层独立验证 |
| RH-clarify-02 | 分类 rename，执行器自行猜测新的文件名 | 必要信息缺失时仍可能错误完成 |

这些任务只在临时夹具中执行；文件终态评分将其记为失败，没有因此触发另一次模型求解。

## 证据与限制

当前路由离线回归：115 项测试中 114 项通过、1 项跳过；模拟传输测试不算真实模型实验。
结构化摘要保存在 `work/evals/results/cascade_routing_20261014.json`，包含逐任务路由轨迹、阈值拟合、请求汇总与原文件哈希。
摘要不包含完整传输字节和冻结源码；完整证据保存在本机 `.test-results`，该目录被 Git 忽略。
余弦分数不是正确概率；新增开发样本按工具语义合成并固定标签，尚未经过外部人工独立复核，语义覆盖仍有限。
模型加载、切换、输出波动会影响计时；第一轮开发阶段还并行运行了离线单元测试。
两次重复不是独立新增任务；小样本结果不代表线上表现。输入/输出 Token、请求和计算时间均为资源代理，未测量实际货币费用。
