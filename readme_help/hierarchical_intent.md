# 分级意图识别与不确定性回退

文件任务 Agent 可以在逐步规划前，用一次结构化模型请求共同提取任务域、操作和参数。默认仍是 `baseline`：直接由现有 `ModelPlanner` 选择下一工具。`hierarchical` 是显式启用的实验模式；目前没有足够的正式对照证据支持更改默认值。

这项实验识别“用户要做什么”，不选择“使用哪个模型”。[既有模型路由实验](model_routing_experiments.md)的结果不属于本实验。原 `clarify_intent.py` → `intent_explain.py` → LangGraph/Bash 工作流保持独立，不迁移其状态、不自动转入 Bash。

## 启用与恢复

在仓库根目录，使用已安装的依赖与本地模型：

```powershell
# Web：仅新任务使用指定模式；历史任务保留创建时的模式。
.\.venv\Scripts\python.exe work/src/samantha.py serve --intent-mode hierarchical

# CLI：使用已有工作区，每次变更仍单独审批。
.\.venv\Scripts\python.exe work/src/samantha.py agent run "找到文件名含 contract 的文件，保留文件名复制到 archive，保留源文件" --workspace work/.samantha/workspace --intent-mode hierarchical --max-model-calls 16 --interactive

# 也可设置共用默认值；显式 CLI 参数优先。
$env:SAMANTHA_INTENT_MODE = "hierarchical"
.\.venv\Scripts\python.exe work/src/samantha.py agent resume TASK_ID --interactive
.\.venv\Scripts\python.exe work/src/samantha.py agent answer TASK_ID "叫 notes.txt"
.\.venv\Scripts\python.exe work/src/samantha.py agent revise TASK_ID "请复制并保留源文件，不要移动"

# 后续新任务恢复基线；不会更改已有任务。
$env:SAMANTHA_INTENT_MODE = "baseline"
```

`answer` 解释为正在等待的澄清问题的回答；`resume` 恢复同一任务，不把“继续”当成新目标，也不重复分类。完成后的上下文指代通过 `agent run ... --parent-id TASK_ID` 或 Web 的“继续这个话题”关联同一工作区父任务。

父任务目前只继承已验证的变更动作，不继承普通 read/list 的 observed 历史；“刚才读取的文件”可能仍需要工具定位或澄清。文件名出现在证据中也不能单独证明选中了正确指代。

`revise` 仅接受 ready、awaiting_input、awaiting_approval 的未完成任务。它保留原请求和已验证操作、追加纠正文本，并使所有尚未执行的动作审批失效。Web 的待确认操作内也有“修正需求后重新规划”。HTTP 对应 `POST /api/tasks/TASK_ID/revise`，body 为 `{"text":"新的要求"}`，沿用现有 CSRF/工作区检查。`POST /api/tasks` 可传 `max_model_calls`；新任务模式由服务器配置决定，`GET /api/config` 返回该配置。

## 接口与生命周期

实现入口是 [`intent.py`](../work/src/task_agent/intent.py)、[`runtime.py`](../work/src/task_agent/runtime.py)、[`planner.py`](../work/src/task_agent/planner.py)。`ModelIntentClassifier(mode)(state, tools)` 返回 JSON 对象；`intent_schema(mode)` 用于结构化输出，`validate_intent` 用项目现有的显式 Python 校验方式校验结构与跨字段关系，无新依赖。

新任务或用户补充／纠正需求后，运行时最多尝试分类一次。两个语义层在同一调用输出；没有内部重试、Embedding 分类或第二层串行请求。随后规划器仍根据每次真实观察选择下一步。工具执行后不机械重新分类。

SQLite 现有任务 JSON 原子保存 `intent_mode`、`intent`、`intent_history`、`intent_dirty`、`intent_update_reason` 和中断中的 `intent_attempt`。历史版本记录 revision、schema_version、mode、reason、status、错误及时间。旧快照按 baseline 恢复，不需要变更 SQLite 表。当前 `TaskStore` 的状态与事件原子提交机制保持不变。

解析失败、未知标签、域和操作不一致、伪造上下文路径或分类异常会记录 fallback，然后交回同一规划器。分类中断在恢复时记为 interrupted，不自动重试。预算耗尽则任务失败，不会以“回退”掩盖。所有 9 个工具始终可用：错误任务域不会永久屏蔽正确操作；观察工具也不会被删掉。

意图只是可能错误的提示，不能替代 `request` 和 clarification/correction 文本。原始要求不会被摘要覆盖；上下文超长先丢弃冗余意图，再裁剪旧观察，显式约束仍放不下则停止而非静默删减。模型报告 finish 依然不等于完整任务成功。

## 版本化契约与标签

当前契约 `schema_version="1.0"`，分类提示版本为 `1.1`。`domains` 是候选标签数组；`mixed` 表示明确跨域任务，多个其他标签也可表达候选。域定义和边界如下：

| 域 | 操作范围 | 示例 |
|---|---|---|
| `file_management` | 文件与目录的创建、追加、覆盖、复制、移动、列目录、文件名搜索 | 复制 a.txt 到 b.txt 并保留源文件 |
| `content_query` | 文件阅读和内容搜索／据内容回答 | 读取 code.txt 并报告代码 |
| `system` | 系统、服务、软件包命令；当前文件工具不支持 | 重启电脑 |
| `mixed` | 请求确有多个域，不可丢弃其中操作 | 读日志，再创建摘要文件 |
| `unknown` | 不能确定要执行的任务 | “处理一下”，且无可用任务上下文 |

第二层操作：`create`、`overwrite`、`append`、`read`、`copy`、`move`、`list`、`search_name`、`search_text`、`mkdir`、`unsupported`、`unknown`。删除和网络上传也是 unsupported；它们不必属于 system。`read` 属于 content_query，单纯列目录属于 file_management。工具 `search_files` 同时搜索文件名和 UTF-8 文本，所以两个搜索意图可用同一工具；`search_content` 只查询已有内容索引，不自动建立索引。

| 字段 | 含义 |
|---|---|
| `operations[]` | 每项含唯一 `id`、`operation`、`parameters`、`depends_on`；保留重复操作，依赖只指向前面已有 ID |
| `parameters.<name>` | `{value,source,evidence}`；名称是 path/content/source/destination/query/limit，未知／不支持操作使用 description |
| `constraints[]` | `{kind,value,source,evidence}`，例如 preserve_source、no_overwrite、preserve_existing、exact_content、order、scope |
| `unresolved_references[]` | `{text,operation_id,parameter,candidates}`，保留歧义候选，不猜路径 |
| `missing_information[]` | `{operation_id,parameter,reason,resolution}`，resolution 是 environment/context/user |
| `ambiguities[]` | `{text,operation_ids,resolution}`，resolution 也可为 planner |
| `support` | supported / partial / unsupported / unknown；与缺参数不同 |
| `fallback` | none / planner / clarify；clarify 必须有只能由用户提供的信息依据，仍由规划器决定下一步 |

来源枚举是 `request`、`clarification`、`verified_context`、`environment`、`unresolved`；unresolved 的 value 必须为 null。必需参数缺失须显式列入 missing_information 或 unresolved_references，不能用空参数假装已完整识别。额外字段、未知标签、重复 ID、循环／前向依赖、非法类型会被拒绝。

规范化约束：用户明确要求精确字节时 `exact_content=true`，字节本身仍放 content 参数；保留源、禁止覆盖、保留现有内容、保留文件名分别使用布尔值；`destination_directory` 保存目标目录；只读为 `scope="read_only"`。同项不重复。复制／移动或追加／覆盖的否定条件由正确的 operation 表达，顺序用 depends_on；评分忽略冗余 order 约束，不因此惩罚同一语义的重复表达。

来源和证据字段也是模型输出，不是事实证明。路径自称来自 verified_context/environment 时，校验其确实出现在对应已验证证据／成功观察中；这不能证明所选指代一定正确或文件仍未变化。读取内容由工具重新读取，变更仍执行现有前置指纹检查。无法唯一定位的引用应保留 unresolved；相关评测覆盖两个候选文件的情况。

环境可查的信息（文件是否存在、目录中有什么、搜索命中和文件内容）交给工具；文件名偏好、精确追加文本、多个候选中用户究竟指哪个，才需要澄清。引用文本和文件内的指令性文字作为数据。复制保留源，移动移除源；追加保留原字节，覆盖替换内容。分类不能授予写权限，也不能扩大工具能力。

## 预算、审批与崩溃恢复

`max_steps` 仍限制规划决策数；`max_model_calls` 单独限制包括分类、规划、失败请求和工具模型在内的物理请求。默认 `max_steps + 4`，可指定 1–256。任务原有活动时间上限为 600 秒，审批和等待回答的时间不计入。生产模式不隐式启用云模型、下载模型或传输重试。

通过 [`model_telemetry.py`](../work/src/model_telemetry.py) 的请求边界，在网络发送前持久化 reservation：`model_requests_reserved` 为预算消耗上界，`model_calls` 为已有完整遥测的实际请求数。请求结束保存结果、phase、耗时和可获得的 Token；缺失 usage 保持未知。外层实验预算先检查，再执行任务预算准入，两者同时有效，拒绝准入不算物理请求。

进程中断留下的 reservation 恢复为 uncertain，仍消耗预算，不伪造 Token 或成功记录。中断活动阶段按保存的时间到恢复时间保守收费，上限为当时剩余活动预算；这可能包含停机时间，并可能使长时间后恢复的任务安全停止。普通审批等待不受影响。

审批继续绑定具体 prepared action 的 identity；修改需求撤销未执行审批。现有执行前指纹、执行后独立验证和 executing 动作的重启对账保留。已经追加的内容不会因恢复而再追加。意图正确、高置信或模型自述完成均不能跳过这些步骤。本版不使用自报置信度、相似度阈值或正确概率。

## 标注与比较协议

当前 [`intent_tasks_v2.json`](../work/evals/datasets/intent_tasks_v2.json) 包含 64 条、31 个语义组：开发集 38 条／18 组，留出集 26 条／13 组。翻译、改写及同模板小变体同组同 split；标准答案根据用户请求和真实工具语义独立编写，未通过模型输出反推。契约校验仅检查形状和一致性。

冻结文件 [`intent_freeze_v2.json`](../work/evals/experiments/intent_freeze_v2.json) 记录数据与 [`intent_protocol_v2.json`](../work/evals/experiments/intent_protocol_v2.json) 的 SHA-256。v1 仍保留；v2 在任何真实模型调用前，根据独立审查补齐目录、文件名保留、依赖和精确内容约束，明确规范化标注。正式比较前验证哈希并保存源代码、模型 digest 和配置；修改数据或协议需新版本与可追踪记录。

| 组 | 方法 | 共用条件 |
|---|---|---|
| A | 原 ModelPlanner 直接选择工具 | 相同本地模型、任务、真实 ToolRegistry、逐动作审批和终态评分 |
| B | 单层操作／参数识别，不输出 domains | 相同规划器及回退机制；仅评测适配器 |
| C | 一次请求输出两级语义，加同一规划器与回退 | 全部观察和纠错工具保留 |

同时提供 same_total（各组均 8 次物理请求）与 extra_classifier（A 为 8，B/C 显式增加首次分类及一次用户补充分类额度）两种条件。额外预算不隐藏在规划器预算中；总延迟、分类开销、实际请求和 Token 缺失覆盖分别报告。正式计划是留出集三次重复、两种预算、三种模式，共 468 次任务运行，物理请求上限 3924，尚未运行。

分类指标包括域和操作 multilabel Macro-F1、多操作次数完整性、操作一对一匹配后的参数值／来源、约束精确提取、关键混淆、unknown 和 fallback。A 无分类指标，B 无域指标；无效预测计为缺失，初始与用户补充后的分类分开评分。自由描述和证据措辞不作逐字评分。Macro-F1 排除金标与预测均未出现的标签，同时报告标签支持数。

端到端评分独立比较完整文件／目录集合、精确字节与工作区外哨兵，并单独检查审批和恢复协议。问答还需预定答案事实及真实观察，澄清还需匹配缺失槽位，不能只看文件没变或 runtime.succeeded。关键词评分不等于通用语义判断。所有计划任务保留在分母，预算／基础设施中断不被删除。错误标签允许并存，只表示可观察的偏差，不宣称找到了模型错误的因果原因。

默认切换须完整留出配对结果支持：至少三次重复，C 相比 A 完整成功率提高至少 5 个百分点且语义组配对 bootstrap 95% 下界大于零；无审批绕过、禁止覆盖或重复追加；相同预算不退步；请求数中位数和延迟 p95 均不超过 A 的 1.25 倍。这些是预先定义的产品验收门槛，不是概率校准阈值。当前保持 baseline。

## 可复现命令

先在仓库根目录跑离线测试，再进入 `work` 执行对照工具；输出目录必须尚不存在。

```powershell
.\.venv\Scripts\python.exe -B -m unittest discover -s work/tests -p 'test_*.py' -q
Set-Location work

# 只查看计划及预算；不调用模型。
..\.venv\Scripts\python.exe -B -m evals.intent_eval --plan
..\.venv\Scripts\python.exe -B -m evals.intent_eval --plan --formal

# 64条 × A/B/C × 两种预算，真实文件工具，注入oracle规划/分类。
..\.venv\Scripts\python.exe -B -m evals.intent_eval --split all --conditions same_total extra_classifier --output .test-results/intent-offline

# 有界开发冒烟，所有模式共享30次总请求；预留总时长余量。
$env:QWEN_MODEL = "qwen3:4b"
$env:QWEN_API_MODE = "openai"
$env:QWEN_BASE_URL = "http://127.0.0.1:11434/v1"
$env:QWEN_TIMEOUT = "30"
..\.venv\Scripts\python.exe -B -m evals.intent_eval --live --case create_exact_en --case copy_preserve_zh --case missing_name_zh --max-calls 30 --max-seconds 480 --output .test-results/intent-smoke

# 更大规模正式对照：本次尚未运行。需自行预留本机运行时间。
..\.venv\Scripts\python.exe -B -m evals.intent_eval --live --formal --max-calls 3924 --max-seconds 86400 --output .test-results/intent-formal

# 校验原始产物哈希后，用新评分器重新评分；不再调用模型或文件工具。
..\.venv\Scripts\python.exe -B -m evals.intent_eval --regrade .test-results/intent-smoke --output .test-results/intent-smoke-regraded
```

正式计划的 3924 是请求预算上界，不是预计实耗；尚无完整数据支持可靠时长估计。24 小时总截止可能使该计划不完整，报告会保留未启动项。每个生成请求的超时同时受全局剩余时间约束。上述冒烟限定开发集，不允许据此宣称总体提升；正式比较只提供命令，不会自动启动。

评分器版本 `2.1` 补充按真实观察时序检查 gold 的操作依赖。该修复来自离线“先追加再读取”的反例，不修改模型、提示词或冻结数据。旧运行产物保持不变；重评分报告单列旧代码哈希、新评分器哈希和零新增请求数。

## 验证记录

本次新增功能的实测结果由 [`intent_validation_20261013.json`](../work/evals/results/intent_validation_20261013.json) 记录。修改前全量离线测试为 625 项、3 项跳过，全部通过；历史测试数量不作为当前数量。脚本规划器与分类器注入只验证协议，不证明模型总体准确率提高。真实模型小样本只作为开发冒烟；正式留出对照尚未运行。

2026-10-13 集成后实际验证：

| 验证 | 实测结果 | 能支持的结论 |
|---|---|---|
| 全量离线单元／集成测试 | 694 项，3 项跳过，0 失败，86.432 秒 | 含原 LangGraph、文件工具、HTTP、预算和恢复回归 |
| A/B/C × 两种预算 × 64 条脚本任务 | 384/384 完整任务、文件终态、协议通过；scorer 2.1 重评分相同 | 真实文件工具和独立评分器接线通过，非模型能力收益 |
| 本地 qwen3:4b 开发冒烟 | 30 次物理请求，372.188 秒；9 次计划中启动 8、完成 7、完整成功 3 | 全局预算与错误回退生效；覆盖不完整 |
| 分类请求 | 5/5 在单请求 30 秒处超时，全部回退；有效分类结果 0 | 本轮未验证真实模型成功提取语义，不能报告分类准确率提升 |

冒烟三组对同一复制请求均成功；创建请求三组均把 LF 写成字面量 `\n`，被精确审批拒绝。缺文件名请求中 C 组猜测文件名 `memo` 而未澄清，也被拒绝；A 组在总请求预算耗尽时中断，B 组未启动。B/C 分类额外消耗约 150 秒，没有获得可用分类结果。本轮没有显示分级识别收益，保持 baseline 默认。

遥测报告 42,482 个已知 Token，覆盖 25 次请求；另外 5 次超时请求的 Token 未知。部分离线验证与冒烟并行进行，因此本轮耗时仅用于运行诊断，不是受控性能比较。单次 30 秒冒烟超时不代表默认 120 秒配置下必然同样失败；后者本轮未验证。

原始冒烟位于 `work/.test-results/intent-smoke-v2-20261013/`，评分器 2.1 的无模型重评分位于 `work/.test-results/intent-smoke-v2-20261013-regrade21/`；离线重评分在 `work/.test-results/intent-offline-v2-regrade21/`。这些目录被 Git 忽略，机器可读摘要保留哈希、版本、配置、逐例结果与限制。独立审查记录见 [`intent_review_20261013.json`](../work/evals/results/intent_review_20261013.json)。未运行浏览器点击／视觉验收或远端 Linux CI。
