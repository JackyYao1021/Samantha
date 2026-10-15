# 本地模型路由实验

本实验补充模型与 Agent 评测、任务分类、模型能力画像、规则/Embedding/模型分类路由及请求级观测。
它是 Samantha 仓库中的独立实验入口，不改变生产 CLI 的默认 Qwen 模型或审批流程。

首轮完整运行见 [2026-10-12 结果与审计](model_routing_results_20261012.md)。
新增三级级联及六策略对比见 [规则 → Embedding → LLM 级联路由](cascade_model_routing.md)。
三组实际运行见 [2026-10-14 级联结果与审计](cascade_model_routing_results_20261014.md)。
下文旧五策略配置继续保留；级联实验使用独立配置，阈值只在开发集校准。

## 任务与评分

`work/evals/datasets/routing_tasks_v1.jsonl` 是 24 项自建、中英文各半的受限文件工具任务。
write、rename、append、select、multi_step、clarify 六类各含 2 项 development 和 2 项 heldout。
两个 split 使用不同请求模板、文件名与约束。参考目标、任务类别和 split 不进入模型或分类器输入。
这不是 Terminal-Bench、SWE-bench 或公开标准 benchmark。

模型返回结构化 `status/actions/answer/question`。四种工具为 write、append、rename、mkdir，
最多 12 个动作，每条写入内容最多 64 KiB。程序先验证整批形状、相对路径及顺序前置条件，
再在新建临时目录执行 Python 文件操作，禁止生成 Bash、任意 Python 代码和外部工具。
评分器独立核对所有最终文件字节、目录、回答路径及必要澄清，保留无关文件约束。
终态通过不表示每一步操作都匹配某个金标准轨迹，也不证明任意终端命令的执行能力。

## 事先确定的协议

当前配置使用宿主机已安装的 `qwen3:4b`、`qwen2.5vl:7b` 与 `bge-m3:latest`。
记录实际模型 digest、量化元数据和每次请求的参数，不下载新模型，不读取 `.env`，不启用 Azure。
模型家族不同，参数更大不等于能力更强；所谓 economical 仅使用开发集观测计算时间的代理。

1. 两个生成模型分别完成 12 个开发任务，形成逐类成功率和指标覆盖率画像。
2. 仅据开发集选择总体成功率最高的 reference，以及计时覆盖完整、平均 provider compute 较低的 economical。
   每类优先选择开发成功率较高的模型；同分选择 economical，最后按模型 ID 打破平局。
   无完整 compute 数据时依次回退完整请求计时、显式说明无经济证据的稳定默认。
3. 冻结源文件、数据集、配置和提示词；开发结果形成 selection 与 development embedding 原型。
   后续不依据 heldout 成绩修改规则、类别映射或阈值。
4. 在同一 12 个留出任务上比较以下五种策略，所有成绩都来自真实模型调用：
   `fixed_a` 固定 Qwen3；`fixed_b` 固定 Qwen2.5-VL；`rule` 公开请求规则分类；
   `embedding` BGE-M3 请求向量与开发原型的 cosine 最近邻；`model_driven` Qwen3 结构化类别分类。
   分类结果通过被冻结的逐类能力映射选择生成模型。
5. 受限降级只针对 transport 或输出契约失败，最多换另一个候选模型一次。
   金标准文件差异、`task_success` 和隐藏类别不触发降级。持久化失败停止实验，不重放模型请求。
   原型初始化不可用时，Embedding 策略保留 incomplete，不用默认模型成绩冒充路由成绩。

固定模型控制始终报告 A/B 的原始名称。reference/economical 是开发结果，不能预先把 B 称作强模型。
使用 seed 打乱开发 case/model 与每轮留出 case/policy 的调度顺序，避免人为按容易程度排队。

## 指标与限制

主指标是 `成功任务 / 全部计划 heldout attempts`，未启动、超时或证据不完整的任务不从分母移除。
另报 started/complete 覆盖率、逐类成功率、Wilson 区间、实际请求数、输入/输出 Token 覆盖、
端到端 p50/p95、路由耗时与降级。请求计时包含加载；原始响应单独保留 provider load 和 compute，
非流式实验不测 TTFT。Embedding 没有报告输出 Token 时记为 unknown，不擅自补零。

路由误选诊断需要同一 case/repeat 两个固定对照的完整证据：初始所选模型的固定对照失败、
另一个固定对照成功时标记 identifiable_misroute。缺证据为 unknown，两者均失败不算错路由。
这是不同真实调用之间的描述性诊断，不能证明模型波动下的反事实。最终路由成功率始终保留真实执行结果。
重复先在 case 内平均，再做 case 配对 bootstrap，不能把 repeats 当作独立任务。
不同策略的缺失配对显式列出，不用跨 repeat 配对。

API 费用始终为 unknown：本地 Token、计算时间和请求次数都是资源代理，未测显存占用、能耗和货币成本。
开发校准调用保留在 development 结果、原型建立调用保留在 manifest；审计报告另按阶段汇总，
不混入单条 heldout 策略延迟。
12 个独立留出任务和默认单轮仅能支持 exploratory 结论；此实验不证明线上路由、灰度、
生产反馈闭环、通用长程 Agent 成功率或人工 judge 已校准。

## 运行

从仓库根目录检查计划，不调用模型：

```powershell
Set-Location work
..\.venv\Scripts\python.exe -B -m evals.routing_runner plan
```

默认一轮包含 24 个校准生成请求、60 个留出求解请求、12 个模型分类请求、
1 个开发原型 embedding 请求与 12 个留出 query embedding 请求，最低 109 个物理请求。
降级额外计数；上限 130 个物理请求、900 秒墙钟预算、每请求 35 秒与 768 个输出 Token。
墙钟预算和超时是客户端协作式停止，不保证服务器立即取消已提交的计算。
原生接口与兼容接口不是两个独立模型，本实验固定原生 Ollama transport。

显式启用本地模型，输出必须是新的目录：

```powershell
..\.venv\Scripts\python.exe -B -m evals.routing_runner run --allow-local-model --output .test-results/routing/local-routing-v1-20261012
```

`--case-limit 2` 可用于接口/报告 smoke，但报告会注明 subset，不可作为完整模型能力结果。
正式结果中包括 manifest、冻结源码与数据、capabilities、selection、prototypes、计划表、
逐任务结果、每次请求/响应/遥测、summary、comparison 和中文 `routing_report.md`。
`.test-results` 是 Git 忽略目录；公开仓库不能仅依赖这些本地文件作为可下载证据。

运行完成后使用保存的原版本源码重评分、核对请求账，并单列校准/setup/heldout 成本与加载时间：

```powershell
..\.venv\Scripts\python.exe -B -m evals.routing_audit --run .test-results/routing/local-routing-v1-20261012 --output .test-results/routing/local-routing-v1-20261012-audit
```

离线审计不会调用模型，原报告保持原貌；新输出包含 `audit.json` 与 `audit_report.md`。
已冻结实验之后对严格 JSON 校验或报告兼容性的修复不会替换原评分器，审计另行披露原始输出问题。

离线回归（不调用模型）：

```powershell
Set-Location ..
.\.venv\Scripts\python.exe -B -m unittest discover -s work/tests -p 'test_routing*.py' -q
```

若继续扩大实验，应事先新增数据版本或 heldout repeats 配置，保留旧报告，避免依据当前 heldout
反复修改路由。公开 benchmark、人工校准和实际线上实验应作为独立后续验证。
