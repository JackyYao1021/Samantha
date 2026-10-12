# 可恢复的文件任务 Agent 与本地操作界面

本模式把模型逐步选择工具、用户确认、原生文件操作、产物验证和任务恢复接到同一个持久任务上。CLI 与网页使用相同的 `AgentRuntime` 和 SQLite 数据库。它是新增的 `samantha agent` 模式；原有 `samantha <request>` 的 Bash / LangGraph 工作流继续使用原接口，其历史会话不能转换成新模式的可恢复任务。

## 快速启动

使用已有基础依赖，不需要 Node、前端构建或额外 Python 包。启动前准备现有 Ollama `qwen3:4b`，仓库根目录的 `.env` 配置仍生效；新规划器禁止 Azure fallback。

Windows，在仓库根目录执行：

```powershell
.\.venv\Scripts\python.exe work/src/samantha.py serve
```

Linux / openEuler：

```bash
python3 work/src/samantha.py serve
# 或 source work/samantha.sh 后：samantha serve
```

浏览器打开 `http://127.0.0.1:8765`。默认操作目录是 `work/.samantha/workspace`，数据库是 `work/.samantha/tasks.sqlite3`。两者均位于已有 Git 忽略目录。指定自己的目录：

```powershell
.\.venv\Scripts\python.exe work/src/samantha.py serve --workspace "C:/path/to/task-files" --port 8765
```

服务默认只绑定本机回环地址。网页可查看任务、填写请求、回答澄清、审阅每次变更、拒绝或停止任务、查看工具结果及文件验证，并在刷新或服务重启后继续同一任务。界面中的模型名称代表配置，不代表已经验证模型连接。

### Docker / openEuler

在仓库根目录执行：

```powershell
docker compose up -d --build oe
```

浏览器打开 `http://127.0.0.1:8765`。Compose 自动运行 `serve --host 0.0.0.0 --port 8765`，宿主机端口仅映射到 `127.0.0.1:8765`。`--host 0.0.0.0` 用于容器内监听，直接在宿主机运行时保留默认监听地址即可；本控制台没有用户登录认证，不应发布到公网或局域网。

已有的 `./work:/work` 挂载会保留任务数据库和工作区文件。模型仍使用宿主机 Ollama，容器默认地址为 `http://host.docker.internal:11434/v1`；若 `.env` 设置了 `QWEN_BASE_URL`，请确保该地址可从容器访问，容器内的 `127.0.0.1` 指向容器自身。

查看日志用 `docker compose logs -f oe`，进入容器终端用 `docker compose exec oe bash`，停止服务用 `docker compose stop oe`。

可以先输入：

> 创建 notes.txt，内容为 alpha，最后加一个换行。

确认文件写入并完成后，使用界面的后续任务功能：

> 在刚才的文件后面追加 beta，最后加一个换行。

后续任务显式关联父任务 ID，仅继承同一工作区中已验证的操作证据；授权不会继承。有关文件的事实仍需重新读取或在执行前核验。

## CLI

```bash
# 新建并交互式完成；每次写入单独确认。
python3 work/src/samantha.py agent run "创建 notes.txt，写入 hello" --workspace ./documents --interactive

# 不使用 --interactive：在需要确认或澄清时返回持久化任务 JSON。
python3 work/src/samantha.py agent run "列出目录中的文件" --workspace ./documents
python3 work/src/samantha.py agent list
python3 work/src/samantha.py agent show TASK_ID
python3 work/src/samantha.py agent resume TASK_ID --interactive
python3 work/src/samantha.py resume TASK_ID --interactive
python3 work/src/samantha.py agent approve TASK_ID ACTION_ID
python3 work/src/samantha.py agent approve TASK_ID ACTION_ID --reject
python3 work/src/samantha.py agent answer TASK_ID "文件名是 notes.txt"
python3 work/src/samantha.py agent cancel TASK_ID

# 关联已结束的同一工作区任务。
python3 work/src/samantha.py agent run "给刚才的文件追加一行" --workspace ./documents --parent-id TASK_ID --interactive

# 数据库参数放在子命令前，或设置 SAMANTHA_TASK_DB。
python3 work/src/samantha.py agent --db ./task-state.sqlite3 list
```

## 工具选择与状态

模型每次只能返回一个经过结构校验的选择：调用工具、提出必要澄清、报告结束。工具的返回值会进入下一次决策，操作名称和参数不由固定脚本事先选定。

模型的输出 JSON Schema 根据实际工具注册表生成，每个工具分支限定名称和参数，同时提供澄清/结束分支；运行器仍独立验证返回值。接入方式参考 [Ollama 官方结构化输出文档](https://docs.ollama.com/capabilities/structured-outputs)。这可约束输出格式，不能单独保证工具选择或用户目标正确。

| 工具 | 行为 |
| --- | --- |
| `list_files` | 列出一层目录 |
| `read_file` | 读取有界 UTF-8 文本及文件哈希 |
| `search_files` | 搜索文件名和文本内容，无需模型或索引 |
| `search_content` | 查询已存在的 Ollama dense 内容索引，不自动下载模型或入库 |
| `write_file` | 创建或明确请求覆盖一个文件 |
| `append_file` | 保留原始字节并追加精确 UTF-8 内容 |
| `copy_file` / `move_file` | 复制或移动文件，不覆盖现有目标 |
| `make_directory` | 创建单层目录，父目录须已存在 |

后五类写操作必须确认。目录及文件路径局限在任务工作区内，拒绝隐藏路径、路径穿越、符号链接、reparse point、硬链接及设备路径。本模式没有通用 Bash 工具。文件上限为 1 MiB，读取输出上限 32 KiB，搜索还有扫描数、字节数和时间上限；截断会明确返回。

内容索引须先通过已有 `content index` 入口创建，并设置 `CONTENT_EMBEDDING_BACKEND=ollama` 及已安装的 `bge-m3`。命中会经过工作区筛选；可用哈希用于排除已失效内容，无法核验的索引结果会标为新鲜度未验证。检索到的文档是数据，不能赋予执行权限。

任务保存请求、澄清问答、逐次观察、已验证操作、待确认动作、审批身份、决策次数、物理请求遥测及事件序列。默认每个任务最多 12 次模型决策，可配置 1–32；调用前先保存决策计数，进程退出不会重置该预算。每次模型响应最多 2048 tokens，没有 SDK 自动重试。600 秒活跃预算累计规划、准备、执行和验证时间，不包含等待用户确认；达到预算后不再接受新的模型决策或启动写操作，已发生效果的恢复核对仍可进行。模型请求超时取 `QWEN_TIMEOUT` 与剩余时间的较小值；它是传输超时，不是强制中断文件操作的计时器。进程突然退出时，尚未写入的耗时和请求遥测可能缺失，持久决策计数仍保留。

模型上下文最多 24000 个序列化字符，先剔除较旧的工具观察和父任务证据，并明确记录省略数量。当前请求、澄清答案和本任务已验证动作不会静默丢弃；这些必要信息自身超限时停止并提示缩小任务。字符限制不冒充精确 token 限制。

模型本身的有效窗口仍由服务端决定。此次本机 `/api/ps` 报告 Qwen3 使用 4096 上下文、CPU 推理；模型元数据宣称的最大窗口不能替代实际运行配置。较长任务须核对服务端窗口并评测上下文选择，不能由 24000 字符上限推断模型一定能完整接收。规划器 v2 用行动/返回对话强调已经完成的操作，保留完整审计数据但不把 inode、mtime 等重复细节全部塞入模型提示。

状态为 `ready → running → awaiting_approval / awaiting_input → ready`，最终进入 `succeeded`、`failed` 或 `cancelled`。`succeeded` 表示模型报告结束且没有未解决的工具错误；已执行写操作均经过文件验证，但这不等于独立证明用户的完整语义目标，后者由任务级评分检查。

## 审批与恢复

SQLite 以同步事务保存任务快照及事件，使用进程级文件锁串行运行同一任务数据库的操作。网页的任务队列是内存调度，数据库是权威状态；服务重启不会自动重放任务，需要点击继续或使用 resume。

每个写操作保存 `before` 文件指纹、精确预期字节/哈希、预览和动作身份。审批绑定工作区及完整动作；工具执行前再次检查前置条件，执行后独立比较实际文件字节、路径存在性和目标状态。

| 中断点 | 恢复行为 |
| --- | --- |
| 等待确认 | 保留同一动作和任务 ID，继续等待 |
| 已确认、尚未开始执行 | 重新核验前置条件后执行原动作 |
| 已登记执行，目标结果已存在 | 核验并登记恢复成功，禁止再次追加 |
| 已登记执行，文件仍完全符合前置状态 | 要求重新确认后才重试 |
| 部分文件变化或与前后条件都不符 | 标记冲突并停止，不重放整批操作 |

copy/move 的部分效果不会被假定为成功。例如移动时目标已写入、源仍存在，会保留证据并停止，由用户检查后发起新任务。已经验证完成的步骤不因后续失败而回滚。

停止请求独立持久化，运行器在当前模型调用结束或当前工具操作完成后的安全边界响应。它不强行终止文件写入，也不撤销已经完成的操作。交互 CLI 的 Ctrl+C 保存现场，后续可恢复。

这里的边界是应用级文件工具约束，不能替代 OS 沙箱。它不防止恶意外部进程在最终系统调用间竞态修改文件，不承诺断电下的跨文件事务，也不支持多个独立数据库同时写同一工作区。推荐同一工作区通过一个服务和数据库操作。

## 任务级评测

新增 runner 使用真实受限文件工具，独立检查最终完整文件树和文件字节。默认脚本规划器验证运行协议；显式 `--live` 才调用模型。公开开发集包含创建、恰好一次追加、复制/移动、检索后生成、澄清、等待确认时重启、拒绝、外部变更冲突、路径越界和建目录后写入。

```powershell
# 仓库根目录；每次重复使用新的文件夹和任务数据库。
Set-Location work
..\.venv\Scripts\python.exe -m evals.task_agent_eval --repeats 2

# 小规模真实模型 + 实际文件执行，最多 24 次请求。
..\.venv\Scripts\python.exe -m evals.task_agent_eval --live --max-cases 3 --max-calls 24 --max-steps 8 --max-seconds 180
```

Linux 将解释器换成 `python3`。评测仅在新建的隔离 fixture 中按固定的“一次性精确工具与参数许可”确认操作；意外写操作会被拒绝。评分器不会把 `expected_files` 或参考步骤交给真实模型。脚本模式、真实模型开发集和独立留出质量实验必须分开报告。

输出位于新的 `work/.test-results/task-agent-*`：manifest、冻结数据、任务事件、执行前后快照、动作/验证记录、逐题结果、汇总 JSON 和 Markdown。`task_success` 同时要求文件结果正确、协议通过、运行器正常结束和评测完整通过；`file_outcome_success` 单独记录文件结果。取消、预算不足或超时的任务，即使留下正确文件，也不算端到端成功。拒绝、冲突等协议场景单独统计，并记录物理请求、延迟、代码/数据/证据哈希。脚本测试通过不代表模型成功率；少量真实开发任务也不是泛化成绩。

可在 `work` 目录只读取已有证据重新评分，输出必须使用新目录：

```powershell
..\.venv\Scripts\python.exe -m evals.task_agent_eval --regrade .test-results/原报告目录 --output .test-results/新评分目录
```

重评分先校验原报告的证据哈希，不执行文件工具或模型请求，不覆盖原报告；报告中的调用量和耗时仍属于原始运行。

```powershell
# 在仓库根目录运行新增测试。
.\.venv\Scripts\python.exe -m unittest discover -s work/tests -p "test_task_agent*.py" -v
```

恢复测试包含真实子进程在追加后 `os._exit` 的情况，重新启动后验证没有重复追加。网页 HTTP 测试检查真实 API 协议、工作区限制、CSRF、跨来源拒绝和任务交互。

离线 CI 仍使用独立网络命名空间，只启用该命名空间内的 loopback，供 HTTP 集成测试使用；不接通宿主模型服务或外网。本机 Windows 测试与远端 Linux CI 的实际运行记录须分别核验。

## 代码入口

- `work/src/task_agent/runtime.py`：持久任务状态机、审批、恢复与工具选择循环。
- `work/src/task_agent/store.py`：SQLite 快照、事件、停止请求和进程锁。
- `work/src/task_agent/tools.py`：工具契约、文件操作与验证。
- `work/src/task_agent/planner.py`：现有 Qwen transport 上的结构化下一步决策。
- `work/src/task_agent/web.py`、`static/`：本地 HTTP 服务与无需构建的界面。
- `work/evals/task_agent_eval.py`：任务级文件产物评分。

新模式采用显式的应用状态机和 SQLite checkpoint。原有 LangGraph 流程不被隐式替换，后续如统一为一个图执行器，应保持这里的动作持久化、审批身份和恢复语义。

本次实测与证据边界见 [2026-10-13 验证记录](task_agent_validation_20261013.md)。
