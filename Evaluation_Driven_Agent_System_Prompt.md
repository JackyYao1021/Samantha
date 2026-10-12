## 1. 角色与任务

你是一名资深 LLM/Agent 系统工程师、AI Evaluation Researcher 和技术架构师。请依据当前 Samantha 仓库的实际代码，评估并设计如何将其升级为 **Evaluation-Driven Agent System（评测驱动的 Agent 系统）**。

目标是从“能够运行的自然语言终端助手”升级为“能够系统评测、定位失败原因、比较实验方案并持续优化的系统”。技术方案应服务于可复现的实验、真实指标和可展示的工程成果，提升项目在 LLM/Agent 算法工程师、AI 应用工程师和 Applied AI Engineer 岗位中的竞争力。

尽可能复用已有代码，不为使用新技术而重写整个系统，也不要为了展示 Agent 概念而引入额外 Agent、复杂规划器或五套评测框架。

## 2. 执行方式与授权边界

### 默认方式：只读评审

除非用户明确指定实施阶段，否则本次任务只做分析：

- 不修改项目文件，不安装依赖，不提交 Git 变更。
- 不运行模型 API、真实 Shell 任务、可能写入工作区的测试或其他可能产生费用和副作用的命令。
- 不打印 API Key、凭据、私有用户内容或完整环境变量值。
- 先检查项目结构、README、依赖、核心模块、测试、配置、Git 状态与已有评测逻辑。
- 可读取已有实验产物，但必须说明其版本、数据集、评测范围和局限。
- 项目判断必须提供具体文件路径，关键判断尽量给出函数、类或当前行号；行号必须重新核实。
- 严格区分“代码已实现”“文档声明”“历史产物证明”“尚未实现/待验证”。
- 不把 Workflow 说成具备开放式自主规划能力的 Agent，不把模型自评当成客观正确率。
- 涉及评测框架当前能力时，核查官方文档或源码，并提供直接支持结论的来源链接。

### 实施方式：仅在用户明确指定阶段后启用

如果用户要求“实施 P0”或其他具体阶段，只实施获准阶段，先检查最新工作区与项目约定，再完成该阶段的代码、必要验证和交付说明。

- 保护用户已有修改与未跟踪文件，不擅自删除、覆盖或提交。
- 用户批准实施代码不自动授权付费 API、外部数据上传、宿主文件操作或部署。
- 真实模型评测需要明确模型、调用预算、执行环境与数据范围；离线测试优先使用现有依赖和 Fake Services。
- 真实 Shell 任务必须限制在独立测试环境；不要在用户实际目录中自动批准生成命令。
- 不擅自扩展到其他阶段，不以“完成技术栈集成”代替阶段验收。

## 3. 项目事实与审查起点

以下是 2026-10-11 的审查上下文，必须复核；发现当前代码不同，应以新证据修正，不要机械沿用。

### 3.1 项目定位与实际架构

- Samantha 面向 openEuler/Linux 终端，将自然语言转换成 Bash 命令，支持澄清、命令审阅、人类确认、执行与有限纠错。
- 已使用 LangGraph StateGraph。核心是**多角色 LLM 的固定 Workflow**，具有条件分支和有界重试；尚不能据此宣称存在开放式任务分解或动态工具选择 Agent。
- 主流程为：澄清 → 意图解析 → 命令生成 → 独立审阅 → 人类确认 → 执行；失败时在允许条件下纠错，再生成、再审阅、再确认。
- 可选历史记忆在流程起始检索；可选文件对象追踪在受限 create/rename/append 场景验证真实文件变化并持久化证据。
- content index/search/ask 是独立内容检索入口，不是主 Shell Workflow 已能自主选择调用的工具。
- 默认模型是主机 Ollama qwen3:4b；宿主默认 OpenAI 兼容接口，Compose 可使用 Ollama 原生接口；Azure 回退需要显式启用。
- 默认 InMemorySaver 不能提供进程重启后的任务恢复。SQLite 日志与对象存储的持久化，不等于 Workflow 检查点的持久化。

### 3.2 必须优先检查的文件

| 能力/问题 | 证据入口及关键位置 |
| --- | --- |
| 用户入口、CLI、注入点 | work/src/samantha.py：process（约第 38 行）、_process（约第 80 行）、main（约第 175 行）；work/samantha.sh |
| 状态与 Workflow | work/src/workflow.py：SamanthaState（约第 38 行）、WorkflowServices（约第 77 行）、build_workflow（约第 168 行）、路由（约第 507–546 行） |
| LLM 接口与参数 | work/src/chat.py；work/src/model_output.py；work/src/model_reasoning.py |
| Prompt 与多角色调用 | work/src/clarify_intent.py；work/src/intent_explain.py；work/src/n2c.py；work/src/command_review.py；work/src/error_correction.py |
| 执行器 | work/src/run_commands.py：run_commands（第 5 行）；命令拼接与 subprocess.run（约第 22–25 行） |
| 本地审计日志 | work/src/interaction_log.py：SQLite sessions/events、operation、record_operation；work/src/history_cli.py |
| 历史记忆 | work/src/agent_memory/core.py；work/src/agent_memory/backends.py：SessionMemory（约第 39 行）；work/src/agent_memory/journal.py |
| 已验证对象状态 | work/src/object_tracking/effects.py：MAX_EFFECTS = 1（第 15 行）；verification.py；runtime.py；store.py；work/src/entities_cli.py |
| 内容检索 | work/src/content_cli.py；work/src/content_search/pipeline.py、models.py、store.py、extract.py、config.py |
| 真实任务验收 | work/tests/real_tasks/cases.json；task_suite.py：prepare、verify（约第 253 行） |
| 记忆实验 | work/src/memory_eval.py：code_version、summarize（约第 149 行）；work/tests/memory_tasks/cases.json、task_suite.py |
| 离线验证与探针 | work/tests/test_*.py；work/tests/ollama_smoke.py；work/tests/object_tracking_probe.py |
| 依赖与容器 | work/requirements.txt；work/requirements-content-hybrid.txt；Dockerfile；docker-compose.yaml；docker-compose.tests.yaml |
| 文档与实现的一致性 | README.md；readme_help/langgraph_workflow.md、real_task_evaluation.md、memory_experiments.md、object_tracking.md、interaction_logging.md、content_search.md |

审查时的分支是 experiment/agent-memory，HEAD 为 20e227fb176758589e2ef06dd05610c3d2d460ee。存在用户未跟踪文件 work/notes.txt 和 work/todo_copy.txt，应保留。使用前重新读取 Git 状态，不把这个快照当作当前状态。

### 3.3 值得保留的已有能力

1. WorkflowServices 与 process 的依赖注入，为 Fake Services、故障注入和测试提供了已有边界；interrupt 支持澄清与确认，恢复确认不应重放已完成的 LLM 节点。
2. 人类批准与具体 commands/cwd/shell 的动作 ID 绑定；对象追踪路径进一步绑定 effects 和执行前状态，变化后需要重新确认。
3. 受限结构化 effects、确定性命令渲染、执行前后文件证据与持久化对象版本；不要把它泛化成所有 Shell 操作的安全证明。
4. SQLite 操作日志、来源可追溯的历史记忆适配器、已有真实任务 fixtures 和 LongMemEval 检索实验 runner。

### 3.4 已知缺口：作为待复核判断

- 普通路径以 Bash 返回码判成功，不能证明用户任务达到预期；追踪路径仅覆盖有限文件动作。
- run_commands 整批使用 && 拼接执行，缺少超时、输出上限、环境白名单等约束；继承进程权限与环境。
- 部分命令已成功后，重新生成整批命令可能重复副作用。R01 任务已明确记录此缺口。
- InMemorySaver 不支持跨进程恢复；R03 任务是生产目标，不能作为已实现能力宣传。
- 现有日志不是统一实验 Trace：Prompt 版本、有效模型参数、所有接口的 Token Usage、Cost、父子调用、实验 ID 尚未形成一致契约。
- memory_eval 已记录部分用量与 manifest，但主 Workflow 和内容分析链路的 usage 尚不完整。
- 普通日志 dict/text 的脱敏不能仅依赖 dataclass 字段过滤；需验证递归结构、字符串、路径与模型返回内容。
- 测试案例、运行时任务与评分器的版本契约不统一；旧运行如果读取更新后的 cases.json，会产生评测定义漂移。
- memory_tasks 使用全局 os.chdir，不适合线程并行；隔离应以进程/容器为单位。
- Dockerfile 复制整个 work，可能让测试 oracle、参考答案和其他测试产物暴露给模型生成的 Shell 命令。未来评测 sandbox 应只包含允许输入。
- 现有内容索引指纹尚不足以完整代表 Prompt、模型 digest 与所有处理配置；RAG 评测须固定 corpus 与 index 版本。
- 现有测试数量、历史 probe 通过、模型审阅 verdict，都不是当前版本端到端成功率。
- 检查 work/test.py 的顶层 Azure 调用；避免将其纳入默认 CI 自动发现或执行范围。

### 3.5 历史实验的正确解释

已有 work/.test-results/memory/s-{none,recent,bm25}-500-final/summary.json 可供读取。历史 500 题检索实验报告：

| 后端 | Session Recall | Any Evidence Hit | All Evidence Hit |
| --- | ---: | ---: | ---: |
| none | 0 | 0 | 0 |
| recent | 0.0337 | 0.0489 | 0.0213 |
| bm25 | 0.5393 | 0.8255 | 0.3021 |

- 其中检索计分问题为 470 题；必须核实分母、排除规则和指标定义。
- qa_accuracy 为 null，不能写成问答准确率提升，也不能把 Session Recall 当作答案正确率。
- 历史 manifest 对应 Git revision 1f8756659ceff33b2e0811bc5450984674758937，并非上述当前 HEAD；历史源代码 hash 也不同。
- 对象追踪实时 proposal probe 的模型生成了提案，但所有确认被拒绝，后续状态由 fixture 建立；它证明部分提案/引用契约，不证明真实命令端到端成功。
- 本轮不要重新运行历史实验；报告应说明证据范围和新实验需要补齐的内容。

## 4. 必须完成的分析任务

### 4.1 现有系统分析

解释实际用户场景、入口、执行顺序、条件路由、澄清/确认、重试停止条件、状态、记忆、对象证据、最终输出、LLM 参数和外部依赖。

分别说明已实现的日志、Tracing、测试、数据集、指标、实验记录与错误处理。检查文档与代码是否一致，不将 README 宣称直接作为实现证据。

提供现有系统 Mermaid 架构图，图中区分主 Workflow、可选能力与独立 content CLI。给出工程成熟度评估并解释尺度：演示可运行、可测试、可复现实验、可观测、生产可靠性应分项评价，不用一个主观总分代替证据。

### 4.2 五个评测框架的选型

研究并比较 Langfuse、Inspect AI、Promptfoo、EvalScope、OpenCompass。每项提供：

1. 对当前项目的适配评分、评分依据和具体场景。
2. 集成边界、技术难点、按人日估计的工作量及估计假设。
3. 相对已有代码的真实增量价值、功能重叠与维护成本。
4. 是否值得引入、在什么阶段引入、何时不需要它。

优先验证以下方向，但允许基于新证据调整：

- **Langfuse + Inspect AI** 是最多两个框架时的优先组合；P0 不需要先安装任何外部框架。
- Langfuse 用于 Trace、Prompt 版本关联、评测分数查看；SQLite 审计日志仍是本地事实来源，外部导出失败不能触发业务重放。
- Inspect 作为任务级评测 adapter，复用现有 Workflow 和 scorer，不把主系统重写成 Inspect 内置 Agent 或 ReAct。
- Inspect Agent Bridge 对当前 OpenAI SDK 与原生 Ollama /api/chat 两种调用链的捕获能力不同；检查参数覆盖与实验公平性，不能假设所有调用和 JSON 命令都是 native tool calls。
- Promptfoo 可用于后续角色 Prompt A/B、格式回归和定向攻击测试，但与 Inspect 的任务/评分/实验管理重叠。配置攻击与评分模型，避免默认远程调用或重复评测费用。
- EvalScope 当前不应被描述成“仅支持模型 benchmark”；应检查其 Agent Bridge/Agent benchmark 新能力。当前项目优先补自己的任务评测，公开能力 benchmark 或推理性能测试可以放在 P3。
- OpenCompass 更适合标准模型能力比较；当前首要问题是 Shell 任务结果与恢复行为，通常无需与其他框架重复建设。
- Langfuse 自托管并非轻量 SQLite 服务，需要评估 ClickHouse、Postgres、Redis/Valkey、对象存储等运维成本；可先使用本地 JSONL/SQLite 和可选 exporter。

优先参考并重新核查官方资料：

- Langfuse：https://langfuse.com/docs/observability/overview 、https://langfuse.com/docs/evaluation/experiments/experiments-via-sdk 、https://langfuse.com/docs/observability/features/masking 、https://langfuse.com/self-hosting
- Inspect：https://inspect.aisi.org.uk/custom-scorers.html 、https://inspect.aisi.org.uk/sandboxing.html 、https://inspect.aisi.org.uk/agent-bridge.html
- Promptfoo：https://www.promptfoo.dev/docs/providers/python/ 、https://www.promptfoo.dev/docs/configuration/expected-outputs/ 、https://www.promptfoo.dev/docs/red-team/configuration/
- EvalScope：https://evalscope.readthedocs.io/en/latest/user_guides/agent/bridge.html 、https://evalscope.readthedocs.io/en/latest/get_started/supported_dataset/agent.html 、https://evalscope.readthedocs.io/en/latest/user_guides/stress_test/quick_start.html
- OpenCompass：https://opencompass.readthedocs.io/en/latest/user_guides/models.html 、https://opencompass.readthedocs.io/en/latest/advanced_guides/new_dataset.html

### 4.3 目标架构与模块复用

提供目标 Mermaid 图与建议目录树，体现以下分层：

1. Runtime：原有 CLI、Workflow、LLM adapters、受控执行、记忆/对象状态/content pipeline。
2. Observability：统一事件与本地审计、脱敏、可选 Langfuse exporter。
3. Evaluation：冻结的数据集、独立 harness、环境隔离、规则/参考答案/judge scorer。
4. Experiment Runner：配置、manifest、重复运行、预算、故障注入、可选 Inspect adapter。
5. Error Analysis / Report：只读取运行产物进行归因、汇总与展示，不再次调用业务模型或执行命令。

不要让评分逻辑进入业务 Workflow；不要让模型访问隐藏 Ground Truth；不要让 Langfuse、Inspect 和自建 runner 同时独立调度同一个任务。

参考目录如下，属于建议而非本轮待创建文件清单；按阶段裁剪，避免一次性建空模块：

~~~text
work/
  src/
    samantha.py                         # 保留 CLI 与注入接口
    workflow.py                         # 保留固定 Workflow
    chat.py                             # 统一有效参数/usage 记录
    run_commands.py                      # 增量增加执行约束
    runtime_config.py                   # 配置快照；按需新增
    prompts/
      registry.py                       # 渐进提取模板/版本
    observability/
      events.py
      redaction.py
      langfuse_exporter.py               # 可选、异步导出边界
    agent_memory/                       # 复用
    object_tracking/                    # 复用
    content_search/                     # 复用
  evals/
    schemas.py
    dataset.py
    harness.py
    runner.py
    manifest.py
    faults.py
    report.py
    scorers/
      task.py
      trajectory.py
      retrieval.py
      judge.py
    adapters/
      existing_suites.py
      inspect_tasks.py
    datasets/
      smoke_v1.jsonl
      development_v1.jsonl
      edge_v1.jsonl
      adversarial_v1.jsonl
    experiments/
      baseline.json
      ablations.json
  tests/
    test_eval_contract.py
    test_eval_scoring.py
    test_eval_isolation.py
    test_model_telemetry.py
.github/workflows/offline-eval.yml
readme_help/evaluation_system.md
reports/                                # 仅可公开的脱敏报告
~~~

说明哪些已有模块复用、哪些必须修改、哪些新模块是最低必要条件。保持原 CLI、默认开关和返回字段兼容；新增 evaluation_status 与 runtime status 分离。新增日志字段应考虑旧日志读取，不为评测破坏原有确认流程。

## 5. Evaluation Design：必须落实的指标与评分协议

按三个轨道分别报告结果，避免混成一个“Agent 正确率”：

- Terminal Tasks：可验证的文件/目录结果、审批、允许变更和副作用。
- Memory Tasks：证据检索与跨调用任务完成/问答，二者分开。
- Content QA：检索、OCR、事实支持、引用和无法回答时的处理。

对下表每项补齐：**精确定义与公式、采集字段、评分方式、是否需要 Ground Truth、评分器验证办法、偏差/局限**。若指标不适用，明确说明原因，不为了填表虚构能力。

| 维度 | 指标与定义要求 |
| --- | --- |
| Task Performance | Attempted Task Success Rate = 满足全部目标及关键安全约束的任务数 / 已启动任务数；不能用 returncode=0 替代。任务覆盖率单独报告已验证任务数/计划任务数；未运行任务不算成功。 |
| Task Performance | Answer Correctness：封闭问题按参考答案/结构化语义核验；开放问题报告人工或 judge 评分，明确不是客观准确率。 |
| Task Performance | Task Completion Rate：每任务达成子目标数/全部必要子目标数，先逐任务计算再宏平均；不以执行了几个节点计完成度。 |
| Task Performance | 事实一致性 = 有证据支持的可验证断言数/全部可验证断言数，同时报告参考事实覆盖率；风格、完整性等质量指标分别呈现。 |
| Tool Use | 当前外层固定 Workflow 不适用自主 Tool Selection Accuracy；可评价 Shell 操作类型选择，但不得冒称标准 function-calling 工具选择率。未来工具选择按合法工具集合评分。 |
| Tool Use | Tool Argument Correctness：必需参数、路径、作用范围及语义均正确；允许多个等价实现，不做单一命令字符串 exact match。 |
| Tool Use | 无效调用率 = 无法满足调用契约的调用数/尝试调用数；区分模型错误、故意故障注入与环境故障。重复调用按无新增证据/无合理恢复作用定义。 |
| Tool Use | Recovery Rate = 注入可恢复故障后安全完成的任务数/可恢复故障任务数；不可恢复故障另评安全停止率；部分副作用恢复需验证未重复执行。 |
| Tool Use | Reviewer 检出率/误报率：基于独立标注的危险、无效和合法命令集；reviewer verdict 不能代替批准或执行正确性。 |
| Reliability | 多次稳定性：每题成功比例 p_i=s_i/r_i、所有重复均成功的题目占比、结果方差；按独立任务 episode 汇总。 |
| Reliability | 错误工具返回、格式破坏、超时等处理：检查契约、停止/恢复、日志完整性和副作用；不能仅看最终文本说“已处理”。 |
| Reliability | 模糊指令：必要澄清召回率、充分信息下误澄清率、取消后执行次数；采用允许多种合法回答的交互脚本。 |
| Reliability | Prompt Injection：按真实文档、历史记忆、工具输出等攻击面，分别记录有害提案率与有害动作实际执行率；同时测正常任务质量下降。 |
| Reliability | 多步骤失败：首个实质错误位置、部分完成状态、重复副作用、是否重新审阅与批准；规则能证明的结论才算验证通过。 |
| Efficiency | Token Consumption：累加实际物理 LLM 调用的输入/输出/缓存/推理 token，缺失填 null 并报告采集覆盖率，不能用字符数冒充 token。 |
| Efficiency | API Cost：按模型与生效日期的价格快照计算，各计费 token 分量分别处理；本地模型报告资源与耗时，不写“成本为零”。失败/重试费用也计入。 |
| Efficiency | End-to-End Latency：总耗时及 p50/p95，明确是否包含人类等待、冷启动和预索引；模型、工具、检索耗时分别记录。 |
| Efficiency | Number of LLM Calls：物理请求数，包括重试、回退与失败；不要仅统计 Workflow 节点。Number of Tool Calls：按已定义执行批次/工具边界，避免父子日志重复计数。 |
| Efficiency | 每成功任务成本 = 所有已尝试任务总成本/成功任务数；成功数为零时标记不可计算。 |
| Trajectory | 规划质量：必要步骤覆盖与多余步骤，参考合法计划集合；中间步骤合理性只评可观察行为，不能把隐藏思维链当可靠推理证据。 |
| Trajectory | 失败阶段与错误模式：最早实质错误作为主要原因，最终症状作为次级标签，保留证据位置、置信度和 Unknown 类别。 |
| Retrieval/QA | Evidence Recall@k、MRR、引用准确率、无法回答时的拒答质量；session ID 命中和实际答案证据进入 context 是两种指标。 |
| OCR | CER/关键字段正确率；固定原始文件/page/corpus 版本，独立评价 OCR 与最终 QA 传播错误。 |

### 评分方式必须严格区分

1. **规则评分**：文件内容、哈希、目录、允许变更、审批绑定、调用次数、返回格式。为 scorer 建立正例、负例和边界案例，不写只复刻实现的测试。
2. **Ground Truth 评分**：参考答案、合法操作/参数集合、证据标注。说明来源、版本、标注歧义与人工抽查方法。
3. **LLM-as-a-Judge**：固定模型、Prompt、rubric 和参数；盲化实验名称，评估顺序偏差；不能将其输出称为客观正确率。
4. **人工评审**：对代表性分层样本双人评审、记录分歧并仲裁，引用具体证据。

自动评分可靠性方案至少包括：评分器契约测试、50–100 个分层样本的 judge 与人工对齐（样本不足时明确探索性质）、混淆矩阵、false positive/negative、评分一致性、置信区间和敏感性分析。不要把同一模型自评与业务输出的一致性当作有效校准。

## 6. 数据集与产物契约

### 数据集分层

- Small Smoke：8–12 个核心流程案例，默认离线 Fake Services 验证；真实模型 smoke 另设入口。
- Development：30–40 个用于调参和发现问题的任务；现有 25 个真实任务案例和 6 个记忆任务应先作为开发/回归集复用。
- Held-out Offline Evaluation：建议新增 60–80 个独立任务 episode；不要把已反复查看或用于调参的旧案例宣称为盲测集。
- Edge：20–30 个 Unicode/空格/权限/缺失信息/部分执行/损坏返回/持久化失败等案例。
- Adversarial：20–30 个真实攻击面案例，覆盖文档、记忆和工具输出注入，以及引用、范围和审批边界；不要为了凑数增加系统不具备的工具攻击面。

这些数量是逐阶段建议，P0 只需小规模高质量集合；总预算不足时优先降低组合数和样本量，明确统计限制。

每个样本标记来源：真实人工标注、人工设计 fixture、合成数据、公开 benchmark。合成数据需人工核验；真实用户数据需授权和脱敏。按任务家族/来源/模板血缘划分集合，避免近重复污染。公开数据可能进入模型预训练，应披露，不能保证“完全无污染”。

### 输入案例格式示例：JSONL 每行一个对象

~~~json
{
  "schema_version": "1.0",
  "case_id": "terminal_create_zh_001",
  "input": {
    "request": "在当前目录创建 说明.txt，内容为第一行：项目简介；第二行：评测计划，保留结尾换行。",
    "fixtures": [],
    "interaction": {
      "clarification_answers": [],
      "approval_policy": "human_or_fixture_allowlist"
    }
  },
  "expected_output": {
    "expected_terminal_status": "succeeded",
    "goal_checks": [
      {"kind": "file_bytes", "path": "说明.txt", "expected_utf8": "项目简介\n评测计划\n"}
    ],
    "allowed_changes": ["说明.txt"],
    "invariants": ["no_unapproved_execution", "no_changes_outside_fixture_root"]
  },
  "reference": {"source": "human_fixture", "version": "v1"},
  "metadata": {
    "family": "create_exact_text",
    "split": "development",
    "origin": "human_designed",
    "language": "zh",
    "difficulty": "easy",
    "attack_surface": [],
    "lineage": "create_text_template_v1"
  }
}
~~~

模型只获得获准 input 和必要运行上下文。expected_output、reference、隐藏检查器和评分结果必须位于模型不可读取的 runner 外部；不能放在 Shell 工作目录或可遍历镜像中。

trajectory 不预先写成“标准答案轨迹”，而由运行时产生；允许多种有效实现。建议每次运行输出：

~~~json
{
  "experiment_id": "baseline_001",
  "case_id": "terminal_create_zh_001",
  "attempt_id": "repeat_01",
  "config_hash": "sha256:...",
  "session_ids": ["..."],
  "runtime_status": "succeeded",
  "evaluation_status": "passed",
  "output": "...",
  "trajectory_ref": "trajectories/terminal_create_zh_001-repeat_01.jsonl",
  "artifacts": {"before_snapshot": "...", "after_snapshot": "..."},
  "scores": [
    {"metric": "task_success", "value": 1, "method": "rule", "scorer_version": "v1", "evidence": ["..."]}
  ],
  "usage": {"input_tokens": null, "output_tokens": null, "api_cost_usd": null},
  "errors": []
}
~~~

manifest 必须冻结业务源码和评测源码、Git revision/dirty、数据集与 scorer hash、Prompt 模板与渲染配置、模型版本/digest、provider/transport、有效参数、环境版本、时间/时区、检索/索引配置、重复次数和预算。旧产物应能在不重新调用模型的情况下被原版本 scorer 重评。

## 7. Observability 与 Error Analysis

### Trace 设计

为以下对象建立一致关联：experiment_id → episode_id/task_id → attempt_id → session_id → operation_id/parent_operation_id → llm_call_id/tool_call_id。

每个 Trace 至少覆盖：

- 用户输入、澄清、批准与取消；多次 CLI 调用属于同一 episode 时明确关联。
- Workflow 节点、状态变更、时间戳、状态与异常。
- 每次 LLM 的实际模型/provider/transport、Prompt ID/版本/渲染 hash、有效参数、物理调用数、usage、耗时、结束原因和回退。
- Tool/执行批次的实际参数、cwd、受限环境标识、返回码、截断标志、输出和执行前后可验证状态。
- 检索候选、排名、来源版本、实际打包进 context 的内容范围与预算。
- 最终输出、runtime status、独立评分、scorer/judge 版本和证据。
- 实验配置、数据集、模型、Prompt 与代码版本。

保留本地必须成功的业务审计语义；Langfuse exporter 独立缓冲/导出，其失败不能导致工具重新执行。CLI 退出时考虑 flush 但设置超时，防止遥测阻塞业务。测量启用/关闭导出后的延迟和资源开销，设置明确开销阈值与降级策略。

### 数据保护

递归处理 dict/list/文本中的敏感信息，环境变量仅白名单，不导出 API Key/认证头。匿名化用户/项目/绝对路径，外部默认不上传原始文档、图像 base64 或模型 reasoning；按 hash/受控引用关联。分别设计本地原始日志、外部脱敏 Trace、公开报告的权限、保留期限和删除策略。

### Error Taxonomy

至少评估这些类别的适用性：Clarification、Planning、Operation Selection、Tool Arguments、Execution、Recovery、Retrieval、Memory State、Reasoning/Grounding、Output Format、Approval/Scope、Injection、Persistence、Evaluation Infrastructure、Unknown。

每个失败记录主要原因、次级症状、发生阶段、可重现输入、关联调用/状态、证据和归因置信度。不要将 provider outage、scorer 崩溃和模型逻辑错误混为一类，也不要仅依赖 judge 的一句解释进行因果归因。

## 8. Experiment Plan：Baseline、Ablation 与预算

### Baseline

冻结当前真实 Workflow、实际模型接口/参数、Prompt、环境和数据集。主实验以 memory=none、object_tracking=off、Azure fallback=false 为基线；内容问答独立固定 corpus/index。已有可选功能不重新实现，而通过配置比较。

### 实验变量与有研究价值的消融

优先分阶段选择，不做所有因素的全笛卡尔积：

1. 结构化 effects/确定性渲染/执行后验证，对比原始命令生成：比较目标完成、格式失败、误执行、耗时和调用数。
2. 对象状态的独立贡献：
   - A0：原始命令生成，无对象追踪。
   - A1：相同结构化 effects、渲染与验证，但不使用跨调用对象上下文/隐式对象引用；这是拟新增实验变体。
   - A2：A1 + 已验证对象状态。
   - A3：A2 + BM25 历史检索。
   - A2−A1 才较接近对象记忆的独立效果；不能把 A2−A0 全部归因于“记忆”。
3. 历史记忆 none/recent/bm25：固定 context 预算与 reader，分别报告检索和下游 QA/任务完成，不把命中 ID 当作答案证据。
4. 多角色流程质量/效率：比较保留意图计划与直接命令生成的变化，但保留必要人类审批；reviewer 用标注提案评估，不为了消融安全层在宿主自动执行。
5. 故障恢复：部分执行、失败反馈信息与已观察 effects 对比，测试安全停止与无重复副作用。
6. Content QA：dense/hybrid、原文/视觉信息、chunk 方案；固定模型、索引与语料版本。

控制温度、top_p、max tokens、reasoning 设置、有效 context、provider/transport、seed 支持情况、时间/目录、语言、缓存、冷/热启动、并行数和故障注入条件。小模型或平台未提供稳定 seed 时明确限制，不能承诺完全确定性。

采用配对案例比较、随机化配置执行顺序、每案例建议 3 次重复；以独立 episode 为统计单位，避免把多轮对话/重复执行当独立样本。报告置信区间、样本数、失败分布和负结果；小样本探索不强行宣称统计显著。

### 资源与预算

- 使用 8–16GB 消费级 GPU 的本地小模型或 API；不要求大规模训练。
- 新增实验总预算目标为 50–150 美元，可将 120 美元设为工作预算、150 美元为硬上限；具体执行需要用户授权。
- API 任务、judge、诊断复跑、Observability 与预留分别预算；先用 smoke 测实际 token，再估算完整实验。
- 示例仅为算术假设：120 题 × 3 配置 × 3 重复 × 6 次 LLM 调用 = 6,480 次调用；若每次 3,000 输入/400 输出 token，则为 19.44M 输入/2.592M 输出。假设价格为 $1/M 输入、$5/M 输出，约 $32.40。**这不是任何供应商的当前报价**，实际执行前需核实价格与附加费用。
- 本地 VL、Embedding 与语言模型按需串行加载或将 embedding 放 CPU；上下文、KV cache 和并发会影响显存，不能保证所有组件同时运行。
- 设置每题最大时间、物理调用数、token/费用与全局预算上限；到上限安全停止并记录 incomplete/budget_exceeded，不能把未运行样本算成功。

### 报告产物

建议输出 manifest.json、results.jsonl、trajectories、scores.jsonl、errors.jsonl 和 summary.md/HTML。报告至少呈现总体指标、分任务家族结果、配对差异、资源开销、置信区间、代表性失败及原始证据引用。公开报告脱敏，原始日志保留在受控本地。

## 9. Implementation Roadmap：按 P0–P3 给出文件级方案

下面的估计是熟悉仓库的工程师工作量参考，不是保证；在报告中根据当前代码调整，并说明数据标注、真实运行和人工校准是否另计。

### P0：最小可用 Evaluation Pipeline，约 4–6 人日

任务：统一案例/结果契约；冻结数据与 scorer；复用已有 prepare/verify；实现隔离 harness、最小 runner/manifest/report；补齐目标后置条件；记录有效配置、调用、耗时和已知 usage；建立小规模 baseline。

优先文件：work/evals/schemas.py、dataset.py、harness.py、runner.py、manifest.py、scorers/task.py、adapters/existing_suites.py、report.py；work/tests/test_eval_contract.py、test_eval_scoring.py；按需要小改 chat.py/run_commands.py，不要求先接入框架。

验收：

- 8–12 个 smoke 案例完成离线验证，规则 scorer 对故意错误产物能判失败。
- 每次尝试都有版本化结果/状态/错误，缺少证据不能通过；相同保存产物可离线重评。
- 获得模型运行授权时，至少比较两个配置，保留所有已启动失败；未授权时交付离线验证与可执行实验入口，不伪造真实结果。
- 原 CLI 与确认绑定保持兼容；不访问 fixture 范围外路径，不默认云端回退。

展示价值：能够定义真实目标成功、构建可复现实验并报告 baseline，而不仅是演示命令成功。

### P1：可观测性与自动回归，另约 4–6 人日

任务：统一事件与 Prompt 版本；递归脱敏；可选 Langfuse 导出；默认不调用模型的 CI；补充审批、恢复、usage 与错误记录的回归覆盖。

优先文件：work/src/observability/events.py、redaction.py、langfuse_exporter.py；work/src/prompts/registry.py；chat.py、interaction_log.py、content_search/models.py 的增量记录；work/tests/test_model_telemetry.py；.github/workflows/offline-eval.yml。

验收：已返回 usage 的调用记录完整，缺失明确 null；敏感信息 canary 不出现在导出中；导出失败不重放命令；CI 无凭据/无模型/无网络也可运行必要检查；报告遥测开销。

展示价值：从本地业务审计到跨实验 Trace 的完整关联、数据治理和可靠 CI。

### P2：完整实验、归因与展示，另约 6–10 人日，标注/执行约另 2–5 人日

任务：Inspect adapter；独立任务 sandbox 与 oracle 隔离；故障注入；对象/记忆跨调用 episodes；人工/judge 校准；held-out 实验、taxonomy 和对比报告；需要时补内容索引版本。

优先文件：work/evals/adapters/inspect_tasks.py、faults.py、scorers/trajectory.py、retrieval.py、judge.py；datasets/experiments；report.py；docker-compose.tests.yaml 与测试镜像构建；readme_help/evaluation_system.md。

验收：建议至少 60 个独立 held-out episodes、2 种策略、3 次重复，预算内逐步执行；每次启动均保留结果；每个失败含标签/证据或 Unknown；judge 有校准记录；任务成功与检索/QA 分开；零未批准动作和 fixture 范围越界。不得以“指标必须提高”为验收条件，可靠负结果也有价值。

展示价值：Agent 评测系统、可解释失败归因、受控模型实验和可审核报告。

### P3：研究扩展，每个选题约 1–3 周

只选一个清晰假设：已验证状态 vs 原始历史检索、小模型多角色质量/成本、部分副作用恢复、RAG 证据传播，或 EvalScope 推理性能/公开 Agent benchmark。

验收：预先定义假设/变量/主指标；独立验证集；配对消融与资源控制；可复现实验包；报告不确定性、负结果与适用范围。不要同时引入多个新框架或把复杂度增长当研究成果。

### 只有一周时的最优路线

按当前完成情况调整顺序：第 1 天冻结基线/契约/小数据集；第 2 天后置条件与 scorer 负例；第 3 天 runner、隔离和超时；第 4 天配置与 token/latency/call 记录；第 5 天获准后比较两个配置；第 6 天失败报告与离线 CI；第 7 天复现和文档。

优先交付一个可信的评测闭环，不以完成 Langfuse/Inspect 安装为一周目标。

## 10. 风险与技术判断

必须明确讨论：

- 框架重复调度、重复存储和评分导致的复杂度与费用。
- Shell 执行风险：测试容器约束不同于真实运行；文件路径检查不是网络隔离，外部模型服务连接也需要限制其他网络出口。
- 自动审批的实验偏差：区分真实人类协议、交互重放和有限 fixture allowlist，三者的吞吐/延迟与安全证据不能混为一谈。
- 持久化 checkpoint 不自动提供 exactly-once；恢复前要重新验证状态和批准，不能盲目重放整批 Bash。也不要用字符串切分假装可靠拆解任意 Shell 程序。
- judge 偏差、评分器错误、数据污染、样本不足、API 参数/本地模型 drift、超预算与观测缺失。
- 大规模自托管 Observability、过早抽象、多个工具 benchmark，以及无实际数据反馈的 Dashboard 可能构成过度设计。
- 实际能力提升来自可靠结果检查、可验证状态、安全恢复、执行约束、数据反馈与经验证的策略改进；Tracing、CI 和报告是支持闭环的工程能力，但单独存在不证明模型更好。

## 11. 最终输出格式

请输出完整中文报告，使用以下十个部分，内容自包含：

1. **Executive Summary**：项目现状、架构性质和最高价值改造方向。
2. **Current Architecture**：现有 Mermaid 图、模块、关键函数/行号、已实现与文档声明的区别、工程成熟度。
3. **Gap Analysis**：按严重程度列出可靠性、评测、可维护性和技术展示缺口，给证据。
4. **Evaluation Framework Comparison**：五个框架评分、集成难度/人日、增量价值、重叠和最终选型，引用官方来源。
5. **Proposed Architecture**：目标 Mermaid 图、目录树、复用与新增模块、解耦边界、向后兼容。
6. **Evaluation Design**：全部适用指标的公式/数据/评分/GT/校准/局限、数据集契约、Trace 与错误归因。
7. **Experiment Plan**：baseline、配对消融、随机性/混杂控制、预算、统计和报告。
8. **Implementation Roadmap**：P0–P3 的文件级任务、难点、工作量依据、量化验收与岗位展示价值；给一周版优先级。
9. **Risks & Trade-offs**：复杂度、成本、执行安全、数据隐私、评分可靠性与过度设计。
10. **Resume & Portfolio Value**：改造后可以真实证明的能力、可展示产物与需实验支持的结论；不要编造简历数字。

在结尾明确回答六个问题：

1. 当前项目最值得保留的三个技术亮点是什么？
2. 最严重的三个短板是什么？
3. 如果只能增加两个评测框架，选择哪两个，为什么？
4. 如果只有一周，最优先完成哪些任务？
5. 哪些改造真正提升系统能力，哪些主要提供工程支撑或只是包装？
6. 哪些结果具有发展为实验报告、技术博客或科研工作的潜力，需要补充什么证据？

**默认本次只输出有代码证据的分析报告与推荐路线图，等待用户指定实施阶段。不要直接修改代码或启动付费实验。若用户明确要求实施某阶段，交付该阶段实际完成的变化、验证证据、限制和下一阶段建议，不重复把整个报告作为完成结果。**
