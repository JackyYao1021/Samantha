# Agent memory experiments

实验分支：`experiment/agent-memory`。基于 `1f87566`，保留现有 LangGraph
命令审查、人工确认、SQLite 日志和真实任务验收流程。

## Benchmark 决策

| 评测 | 用途 | 当前支持 |
| --- | --- | --- |
| [LongMemEval cleaned S](https://github.com/xiaowu0162/LongMemEval) | 主公共 benchmark：信息提取、跨会话推理、知识更新、时间推理、无答案判断 | 历史导入、memory replay、检索诊断、固定 reader QA、官方 judge 格式输出 |
| LongMemEval oracle | 检查 reader 和导入协议，区分理解失败和检索失败 | 同一 runner；只包含证据会话，不能与 S 的分数混报 |
| Samantha Memory Tasks v1 | 实际多次 CLI 调用、审批和文件结果 | 6 个中文场景、13 次调用、独立临时目录、逐步日志与文件检查 |
| 现有 25 个真实任务 | 检查 memory 接入后的功能回归 | 沿用现有 real-task suite；R01/R03 仍是尚未完成的生产目标 |

选择 LongMemEval 是因为有公开数据、明确的知识更新/时间/无答案测试，以及官方
QA judge。它评测的是“历史记忆 + reader”，不包含 Samantha 的命令执行。因此，
公共 QA 分数和终端任务成功率分别报告，不合并成一个分数。

[LongMemEval-V2](https://github.com/xiaowu0162/LongMemEval-V2) 已有 agent 轨迹、
工作流知识、动态状态和环境陷阱测试，适合作为下一阶段扩展；当前分支未实现其
多模态轨迹协议，不宣称兼容 V2。LoCoMo 暂不列入首批，以避免两个聊天 QA
benchmark 重复投入；之后可新增数据适配器。

## 当前解耦边界

`work/src/agent_memory/core.py` 定义稳定接口：

```python
class MemoryBackend(Protocol):
    def ingest(self, session: MemorySession) -> None: ...
    def retrieve(self, query: MemoryQuery) -> list[MemoryHit]: ...
    def describe(self) -> dict: ...
    def close(self) -> None: ...
```

- `MemorySession`：会话 ID、用户/项目范围、发生时间、消息和来源 metadata。
  导入 ID 与原始来源 ID 不同时，`source_session_id` 属性提供原始来源；adapter
  返回的 `MemoryHit.session_id` 应采用此属性，不能采用合成的存储 ID。
- `MemoryQuery`：当前问题、范围、截止时间、返回数量和上下文字符预算。
- `MemoryHit`：内容、来源会话、时间和可选分数。提炼事实/合并摘要可用
  `source_session_ids` 保留多个原始来源；时间采用这些来源中最晚的证据时间。
- Backend 内部可以使用不同 extractor、consolidator、store、retriever。
  工作流和评测 runner 都只依赖上述接口。
- `describe()` 只返回可公开的实现名称、版本和实验参数，不能包含凭据。
- `ingest()` 应在同一范围/会话 ID 下幂等；`retrieve()` 必须遵守用户、项目和时间
  范围。返回原始来源会话 ID，便于追踪和计算召回率。
- 调用方再次检查时间和统一打包预算，保存来源和截断标记。自定义 adapter
  仍需自行保证范围隔离；不能把两个用户的索引混用。
- 生命周期由调用方管理；graph 不关闭 backend。暂采用同步接口，适配器可在内部
  管理批量处理和队列；新增异步 runner 时另行扩展协议。

已实现的基线：

| Backend | 行为 |
| --- | --- |
| `none` | 不保存、不返回历史 |
| `recent` | SQLite 保存原始会话，按发生时间取最近的会话 |
| `bm25` | 同一原始会话存储，BM25 按当前查询排序，时间用于同分排序 |

BM25 是无 embedding 的词法基线，英文标识符/路径、中文单字和双字切分。
每次检索计算当前范围的语料统计，适合基线实验，尚未提供大型在线索引。
这些基线不提炼事实，不自动判定哪条矛盾事实有效，也不代表 Mem0/SimpleMem。

外部实现用显式 `package.module:factory` 注册：

```python
# my_memory_adapter.py（示意；替换内部实现即可）
def create(path=None, options=None):
    return MyMemoryBackend(path=path, options=options or {})
```

Factory 必须返回实现四个方法的对象。Mem0、LangMem、A-MEM 的库依赖、配置和
数据转换放在各自 adapter 中，不进入各 agent 或 workflow。LangGraph Store
也可以由 adapter 持有。尚未安装这些 SDK，也未实现这些 adapter。

## 工作流使用

默认 `SAMANTHA_MEMORY_BACKEND=none`；不启用时保持原来的模型调用和审批路径。
启用后增加一个 `retrieve_memory` 节点，在澄清前检索；澄清、计划、命令生成、
审查、纠错共享同一份上下文。原始 `user_input` 和审批动作标识保持独立。
记忆作为有来源的历史参考，当前请求优先，历史批准不能授权当前执行。
检索失败在模型/执行前终止任务。

终端在 `session_finished` 提交后，将真实用户输入、澄清、执行结果和最终状态
交给 backend。取消/失败、计划命令和实际执行结果分开标记，不把计划宣称为完成。
命令结果仍对应整批 Bash，尚未增加每一步产物验证或自动路径移动同步。
写 memory 失败会单独记录 `memory_error` 并提示，不重标已完成任务、不重放命令。
评测将这种失败作为失败检查项。

```bash
export SAMANTHA_MEMORY_BACKEND=recent
export SAMANTHA_MEMORY_PATH=/work/.samantha/memory.sqlite3
export SAMANTHA_MEMORY_USER=local
export SAMANTHA_MEMORY_PROJECT=my-project
export SAMANTHA_MEMORY_LIMIT=5
export SAMANTHA_MEMORY_BUDGET=6000
samantha '在这里创建 notes.txt，内容为 alpha'
samantha '给刚才的文件追加 beta'
```

项目 ID 应在目录切换后保持稳定。默认项目 ID 为调用时的绝对工作目录，优先隔离
不同目录。每个任务仍使用独立日志会话/`thread_id`，避免复用旧任务的审批和命令。
Graph 检查点继续使用 `InMemorySaver`；memory 实验没有顺带实现跨进程任务恢复。
在只读 Docker task-lab 中，应将 `SAMANTHA_MEMORY_PATH` 改为
`/test-results/memory.sqlite3`；下面的专用终端 runner 会自行使用可写的结果目录。

Python 调用方可显式注入 backend、`MemoryScope`、预算，或直接调用
`build_workflow(..., memory=..., memory_scope=...)`。图直接调用只检索，写入由调用方
负责；`samantha.process()` 自动从最终日志写入。

## 公共评测运行

在仓库根目录，Windows 示例：

```powershell
.\.venv\Scripts\python.exe work/src/memory_eval.py fetch --variant oracle --output work/.test-results/memory/data/oracle.json
.\.venv\Scripts\python.exe work/src/memory_eval.py fetch --variant s --output work/.test-results/memory/data/s.json
.\.venv\Scripts\python.exe work/src/memory_eval.py validate --data work/.test-results/memory/data/oracle.json
.\.venv\Scripts\python.exe work/src/memory_eval.py run --data work/.test-results/memory/data/oracle.json --variant oracle --backend bm25 --limit 10 --output work/.test-results/memory/oracle-bm25-smoke
```

`fetch` 先将上游 revision 解析成不可变 commit，再下载并验证格式，保存
`.source.json` 中的 commit、URL 和 SHA-256。可用 `--revision <commit>` 固定下载。
数据和实验输出都放在已忽略的 `work/.test-results/`，不提交数据集到仓库。
输出目录必须是新目录，避免覆盖以前的运行。

下面进行真实模型 QA（Linux/Docker 用 `python3` 替换 Python 路径）：

```powershell
.\.venv\Scripts\python.exe work/src/memory_eval.py run --data work/.test-results/memory/data/s.json --variant s --backend recent --mode qa --top-k 5 --context-chars 6000 --limit 20 --output work/.test-results/memory/s-recent-qa20
```

CLI 也支持 `samantha memory-eval ...`。Reader 使用项目的 Qwen/Ollama 配置，
固定 temperature=0，禁用 Azure fallback。`--backend-options options.json` 用于
自定义 adapter；配置文件不会原样写入报告。

Runner 对每题新建 backend 存储与用户范围，按时间排序导入所有历史；
构建时不传问题、答案、题型或证据标签，消息只保留 role/content。
检索时才传问题和 question_date。`answer_session_ids` 只用于后续评分。
官方 cleaned S 的重复会话 ID 以独立 occurrence ID 存储，原始 ID 留在 metadata；
即使内容相同也保留各自日期，不覆盖或改变上游时间证据。

输出：

- `manifest.json`：数据 SHA-256、variant、题目 ID、完整/子集标记、backend factory、
  Git revision/dirty 标记、实际 Python 源码 SHA-256、上下文预算、reader 配置、运行状态。
- `results.jsonl`：逐题结果、构建/检索/reader 延迟、实际来源、截断、上下文长度、
  接口返回的 token 用量和错误；失败题保留在分母中。
- `summary.json`：总体/题型分组的来源会话召回率、any/all evidence coverage、延迟。
- `hypotheses.jsonl`：QA 模式输出官方格式 `question_id` / `hypothesis`。

字符预算衡量的是完整序列化 memory context，包含来源字段，**不是 token 预算**。
接口没有返回 token 计数时写 null，不用字符数伪装 token 数。
来源会话召回表示来源 ID 覆盖，不证明截断后的片段含答案。Oracle 召回尤其不能
代表真实检索能力。QA 正确率需要官方 judge，runner 不用 substring 命中替代它。

按 [官方说明](https://github.com/xiaowu0162/LongMemEval#testing-your-system)，
另行安装其 lite evaluator 后运行：

```bash
python3 /path/to/LongMemEval/src/evaluation/evaluate_qa.py gpt-4o /absolute/path/to/hypotheses.jsonl /absolute/path/to/s.json
```

这里明确是外部 judge，会需要其模型凭据；本地 runner 不会自动调用它。
正式公共成绩使用完整 500 题和相同 judge。部分题仅报告为 smoke/subset，
S/oracle、不同 reader、预算或数据 revision 的结果分别报告。

## 已运行的检索诊断

当前完整跑过 cleaned S 和 oracle 各 500 题、三个 backend，均无运行错误。
没有调用 QA reader 或官方 judge，也没有完成真实模型终端任务。

- 官方数据 revision：`98d7416c24c778c2fee6e6f3006e7a073259d48f`。
- S SHA-256：`d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442`。
- S 参数：`top_k=5`，完整序列化上下文预算 `6000` 字符，逐题隔离并按日期 replay。
- 以下为 S 的 470 道非 abstention 题按题平均的来源 ID 覆盖；30 道 abstention
  没有正例召回分母，仍保留在总题数和错误统计中。

| Backend | 来源会话召回 | 至少一个来源覆盖 | 所有来源覆盖 | 完成/总数 |
| --- | ---: | ---: | ---: | ---: |
| `none` | 0.00% | 0.00% | 0.00% | 500/500 |
| `recent` | 3.37% | 4.89% | 2.13% | 500/500 |
| `bm25` | 53.93% | 82.55% | 30.21% | 500/500 |

最终代码的原始结果在 `work/.test-results/memory/s-{none,recent,bm25}-500-final/`，每组保留
manifest、逐题结果、summary 和 SQLite 存储。这些是基线诊断，**不是 QA 正确率**。
三个过程同时运行，延迟只作排错记录；性能比较需要单独运行。

本次整段会话的贪心装包常被第一段占满，在知识更新/多会话题上遗漏其余来源。
即使来源 ID 已覆盖，截断也可能丢掉实际答案。后续优先比较分块检索和有来源的
事实提取，再用固定 reader 的真实 QA 检查是否改善；不根据召回指标直接宣称
memory 系统优劣。

## 终端系统评测

```powershell
.\.venv\Scripts\python.exe work/tests/memory_tasks/task_suite.py list
```

真实执行需要 Linux/Bash，可使用现有 Docker lab：

```bash
docker compose -f docker-compose.tests.yaml --profile real-tasks run --rm --build task-lab
# 在容器中：
python3 /work/tests/memory_tasks/task_suite.py run MT01 --backend recent --output /test-results/memory/mt01-recent-1
```

按每步提示人工审阅并回答；涉及指代时不要补充提示禁止提供的文件名。
没有自动批准模型命令的代码。每次调用后关闭并重新打开 backend，检查跨调用持久化。
每步验证文件字节、禁止出现的路径、无关文件、审批日志、澄清次数和 memory 写入。
前一步失败即停止，后续依赖步骤不计为通过。每组 backend 使用新的 fixture。

MT01 最近产物、MT02 重命名更新、MT03 中文/空格文件名、MT04 取消操作、
MT05 当前明确请求优先、MT06 项目隔离。它是手工设计的项目回归集，不能当作
公开标准 benchmark 或大规模泛化证据。下一批增加失败恢复、外部路径变化和
重复错误经验；那些测试需要更细的执行/产物证据。

## 实验协议与扩展顺序

首批比较 `none`、`recent`、`bm25`。固定数据 SHA、题目 ID、reader、top-k、
上下文预算、reasoning 配置和 judge。使用单独开发集选择参数，测试集结果不反向
用于整理策略训练；小子集调试完成后运行完整 S。当前完整 QA 尚未运行。

在固定基线上逐步接入事实提取，再添加知识更新、时间过滤、经历记忆，分别做
消融实验。之后试验 A-MEM 关联或技能策略。Adapter 接入测试至少覆盖幂等、
范围隔离、时间截止、来源追踪、重启、失败处理和预算约束。

当 SQLite/BM25 成为瓶颈时，替换 adapter 内部的 store/index，而不是修改各 agent。
保持数据格式版本、提取策略版本和 embedding 标识；向量索引作为派生数据可重建。
后台提取需要持久任务队列，不能依赖 CLI 退出后的内存线程。

## 离线验证

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s work/tests -v
```

`test_agent_memory.py` 使用真实 SQLite、LangGraph 和临时文件，模拟模型/工具，
验证范围隔离、时间和预算、拒绝/写入失败、跨调用记忆、benchmark 标签隔离、
chronological replay、官方输出格式和错误分母。离线通过表示实现协议通过，
不代表真实模型 QA 或终端 benchmark 已通过。
