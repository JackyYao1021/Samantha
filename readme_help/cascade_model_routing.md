# 规则、Embedding、LLM 三级级联路由

Samantha 的独立模型路由实验新增 `cascade` 策略，按规则、Embedding、LLM 顺序分类。
某层接受类别后立即停止路由，再查询开发集冻结的 `family_models` 选择任务执行模型。
原有五种策略保留为对照；文件 Agent 的 CLI/Web 默认模型与审批流程不受此实验改动影响。

实际运行与限制见 [2026-10-14 三组实验结果](cascade_model_routing_results_20261014.md)。

## 各层设计

1. **规则层**：复用公开请求及文件清单的规则，增加 `abstain`。
   未匹配操作、否定操作、条件/备选表达或无明确顺序的多类操作交给后层。
   引号中的操作词按字面数据处理。原单层 `rule` 的兜底行为保持原样，确保对照定义稳定。
   规则是保守启发式，不能保证没有误判；错误接受会阻止后层纠正，单独计入报告。
2. **Embedding 层**：BGE-M3 对请求编码，与 development 原型计算余弦相似度。
   每类以最近原型代表；只有最高分不低于相似度阈值、且与最佳不同类别的分差不低于 margin 时接受。
   两个最近样本若属于同一类别，不作为前两类计算分差。缺少不同类别原型不能接受。
3. **LLM 层**：最多一次 Qwen3 结构化分类，输出六种任务类别之一或 `unknown`。
   JSON Schema 和返回值校验限定分类契约；格式正确不等于语义正确。
   分类超时、非法输出或 `unknown` 时采用冻结的默认执行模型，类别保留 `null`，不伪造为 `multi_step`。

路由器只接收公开的 `request/files/directories`；标签、目标文件、成功分数不进入路由输入。
各层保留 `status/family/reason/duration_ms`，向量层额外保留候选类别、最近原型、相似度与类别分差。
向量层普通传输错误继续到 LLM；预算耗尽、证据持久化失败或用户中断立即停止，不能触发新请求。

三级升级属于**分类阶段**。之后执行模型仅在传输或输出契约失败时最多切换一次候选模型；
文件终态不符合金标准不会触发切换。默认回退、分类升级、执行失败切换分别记录。
最坏情况下一个级联任务有 1 次 query embedding、1 次 LLM 分类、2 次求解请求。

## 开发集校准

配置文件为 `work/evals/experiments/routing_cascade_local.json`：

- 相似度候选：`0.45, 0.55, 0.65, 0.75, 0.85, 0.95`。
- 类别分差候选：`0.03, 0.05, 0.10, 0.15, 0.20`。
- 接受至少 3 个开发案例，并要求被接受开发案例的经验准确率为 100%。
- 满足条件时最大化开发集接受数；同分依次选更高经验准确率、更高相似度阈值、更大分差。
- 无满足条件的组合时禁用向量层接受，规则拒绝后直接交给 LLM，避免没有校准依据的强行分类。

每个开发样本作为 query 时从原型库移除自身；原型含 `group_id` 时移除整个语义组。
当前 v1 数据没有语义组字段，采用 leave-one-case-out；不能宣称进行了语义组隔离验证。
为避免开发集规则全命中而没有向量校准样本，使用全部开发案例校准向量层，含规则命中的案例。
这是阈值拟合过程，不是对级联残余样本的独立泛化估计。余弦相似度也不是正确概率。

开发集建立的模型映射、原型、全部阈值候选及逐案例留一预测在 heldout 开始前保存并冻结。
原型与校准文件保存哈希；离线审计使用冻结源码重新拟合阈值，并核对原型是否来自实际开发 embedding 响应。
heldout 分数不参与选阈值、改映射、接受路由或触发模型切换。

补充配置 `routing_cascade_augmented_local.json` 通过 `prototype_dataset` 加载
`routing_prototypes_v2.json` 的 48 条路由开发请求（24 个语义组，每组中英文各一条）。
它们与原 12 个开发任务共同组成 60 个向量原型；新增请求不参与执行模型能力画像或任务成功率评分。
新增组按组排除、原 12 项仍按 case 排除。语料只包含公开请求及开发标签，不包含 heldout 目标或成绩。
新增样本按工具语义合成，标签由场景定义确定，尚未经过外部人工独立复核。
加载器拒绝非 development 数据、重复 ID/请求、组内冲突标签以及与任务原文完全重复的条目。
规则、阈值候选、接受门槛和提示词保持不变；独立 `embedding` 对照与级联共用同一扩充原型库。

由于严格配置仍无法拟合可用门槛，另提供 `routing_cascade_balanced_local.json` 作为覆盖优先敏感性实验。
它使用相同 60 个原型和相同阈值网格，将 `min_precision` 设为 0.75、`min_accepted` 设为 12。
这是依据开发集分数新增的质量/覆盖取舍，不是高精度保证，也不替换原先的保守配置。
测试前冻结选出的阈值；比较两种固定模型、独立 LLM 分类和级联四种策略各一轮，48 次 heldout 尝试。
最低 85、理论最多 133 个物理请求；预算 150 次请求、1200 秒。该配置用于观察实际向量早退与误判的关系。

## 实验协议与指标

延用 24 个受限文件工具任务：12 development、12 heldout；六类各 2 个 development、2 个 heldout。
比较 `fixed_a/fixed_b/rule/embedding/model_driven/cascade` 六种策略，heldout 重复 2 轮，共 144 次任务尝试。
两个生成模型为 `qwen3:4b` 和 `qwen2.5vl:7b`，向量模型为 `bge-m3:latest`。
每轮按固定 seed 打乱 case/policy 顺序，所有策略均实际调用求解模型，不用固定策略结果替代路由结果。

最低计划 217 个物理请求，包含 24 个开发求解请求、144 个 heldout 求解请求、24 个独立 LLM 路由请求、
1 个开发原型批量 embedding 请求与 24 个独立向量路由请求。
级联最多额外增加 48 个请求；执行失败切换最多额外增加 96 个请求，总理论上界为 361。
运行限制为 380 个物理请求、2400 秒墙钟、每请求 45 秒、每次求解最多 768 个输出 Token。
客户端超时不保证服务端立即停止已提交计算。

补充配置在相同 12 个 heldout 上运行六策略各一轮，共 72 次任务尝试，最低 121、理论最多 193 个请求；
上限设为 220 个请求、1800 秒，仍使用 45 秒单请求超时。它是另一次完整探索性比较，
不能把两次运行之间的差异单独归因于扩充原型，因为重新进行了开发能力测量，且运行时冷热条件也不同。

报告同时给出：

- 任务成功数/全部计划数、完整覆盖、端到端 p50/p95、执行失败切换次数。
- 分类准确率及分类覆盖；每层进入、接受、拒绝、错误、默认回退和错误接受数量。
- LLM 分类调用、query embedding 调用、全部实际请求及 Token/计算耗时覆盖。
- 同 case/repeat 配对、先在 case 内平均重复的 bootstrap 差异区间。

当前 heldout 已在先前迭代中被查看，因此本轮是探索性回归比较，不是新的盲测。
12 个独立任务及 2 次重复不足以证明生产收益；重复不能视为 24 个独立样本。
本地模型加载/切换计入延迟，请求及 Token 数不等于货币费用。
若开发集把全部类别映射到同一个模型，即使分类层节省调用，也不能解释为成功实现了有收益的模型分流。

## 运行与审计

从仓库根目录进入 `work` 后运行；每次实际实验、审计使用新的输出目录：

```powershell
Set-Location work
..\.venv\Scripts\python.exe -B -m evals.routing_runner plan --config evals/experiments/routing_cascade_local.json
..\.venv\Scripts\python.exe -B -m evals.routing_runner run --config evals/experiments/routing_cascade_local.json --allow-local-model --output .test-results/routing/cascade-v1-20261014
..\.venv\Scripts\python.exe -B -m evals.routing_audit --run .test-results/routing/cascade-v1-20261014 --output .test-results/routing/cascade-v1-20261014-audit
```

`--case-limit` 只用于协议 smoke，不代表完整模型能力成绩。
不指定 `--config` 时仍使用旧五策略配置，避免历史入口悄然改变预算。

扩充原型后的完整运行使用：

```powershell
..\.venv\Scripts\python.exe -B -m evals.routing_runner run --config evals/experiments/routing_cascade_augmented_local.json --allow-local-model --output .test-results/routing/cascade-augmented-v1-20261014
```

覆盖优先实验使用 `--config evals/experiments/routing_cascade_balanced_local.json`
和新的输出目录 `.test-results/routing/cascade-balanced-v1-20261014`。

离线测试从仓库根目录运行，不调用模型：

```powershell
.\.venv\Scripts\python.exe -B -m unittest discover -s work/tests -p 'test_routing*.py' -q
```

关键文件：`work/evals/routing_cascade.py` 实现级联与校准；`routing_runner.py` 负责有界调用和证据冻结；
`routing_analysis.py` 分开汇总分类与任务指标；`routing_audit.py` 核验原型来源并重建阈值；
`work/tests/test_routing_cascade.py` 覆盖早退、类别分差、数据泄漏及异常停止。
