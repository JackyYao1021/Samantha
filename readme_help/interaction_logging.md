# 持久化交互日志

Samantha 默认开启 SQLite 日志，无需额外安装依赖。每次自然语言请求或
`samantha content ...` 调用都有独立的 `session_id`，用户输入、输出和操作按
`sequence` 保存，并带有 UTC 时间戳和 `turn` 轮次。程序重启后仍可查看。

## 查看和导出

```bash
samantha history list
samantha history list --limit 50 --json
samantha history show latest
samantha history show <session_id> --json
samantha history export latest --output session.json
samantha history export <session_id> --format jsonl --output session.jsonl
```

导出省略 `--output` 时写入 stdout。导出到文件时必须使用新文件名，不会覆盖
已有文件或日志数据库。`latest` 指最近开始的调用；并发使用时可指定完整 ID。
历史查询不会生成新会话，因此查看日志后 `latest` 仍指原调用。

未加载 Bash 函数时，也可以直接运行：

```bash
python3 work/src/samantha.py history list
python3 work/src/samantha.py history show latest --json
```

Windows 项目虚拟环境示例：

```powershell
.\.venv\Scripts\python.exe work/src/samantha.py history list
```

## 保存位置

默认数据库是仓库下的 `work/.samantha/interactions.sqlite3`，不受当前工作目录
变化影响。可用环境变量或仓库 `.env` 中的 `SAMANTHA_LOG_PATH` 指定路径。
环境变量优先，`process(..., log_file=...)` 也支持调用方指定路径。

```bash
export SAMANTHA_LOG_PATH=/absolute/path/interactions.sqlite3
samantha history --log-path /absolute/path/interactions.sqlite3 list
```

Docker Compose 默认使用 `/work/.samantha/interactions.sqlite3`，对应宿主机的
`work/.samantha/interactions.sqlite3`。已有的 `./work:/work` 挂载使容器重建后
仍能读取日志；自定义路径时应选择持久化挂载中的位置。日志目录已加入
`.gitignore` 和 `.dockerignore`。

## 记录范围

| 事件 | 保存内容 |
| --- | --- |
| `session_started` | 初始目录、传入的历史上下文或 content 参数列表 |
| `user_input` | 原始请求、每次澄清回答、每次确认或拒绝；保留空白、中文和换行 |
| `assistant_output` / `prompt` | 终端响应、进度、提问、确认提示、输出及 stdout/stderr 来源 |
| `operation_started` | 操作 ID、名称及传入参数，在调用之前提交 |
| `operation_finished` | 相同操作 ID、返回结果或异常类型/信息，以及执行耗时 |
| `workflow_update` | 各节点更新，包括意图、命令、说明、批准结果、重试次数和状态 |
| `awaiting_input` | 等待澄清或确认的完整内容 |
| `interrupted` / `workflow_error` | EOF、Ctrl+C 或运行异常 |
| `session_finished` | 最终状态、错误和最终工作流状态；content 调用保存退出码 |

一次调用的初始请求是第 1 轮，之后每个用户回答递增轮次；后续输出和操作归属
最新一轮。传入的 `terminal_history` 保存为上下文来源，不伪装成本次新输入。

操作记录覆盖澄清、意图解析、命令生成、解释、纠错、目录交接、命令执行，
以及 content 初始化、索引、每个文件的索引、搜索、问答、诊断和关闭。
`shell.execute` 额外保存实际 Bash 参数列表、起始目录、原始 stdout、stderr
和退出码；失败后纠错不会覆盖之前的命令或结果。

普通函数返回时 `operation_finished.status` 为 `completed`；执行器返回
`success` 或 Bash 返回退出码时标为 `succeeded` / `failed`。结构校验失败会
出现在后续 `workflow_update` 和最终会话状态中。命令生成记录代表计划，
只有执行操作记录才表示调用了执行器。拒绝确认不会生成执行操作。

## 持久化和边界

每个事件独立提交，SQLite 使用 WAL、`synchronous=FULL` 和 30 秒锁等待。
会话与事件通过外键关联；每个会话的事件序号在写事务中分配，可同时运行
多个 Samantha 进程。会话结束状态与最后事件在同一个事务内提交。

如果日志无法打开或写入，Samantha 会报错并停止后续操作。如果某个操作已经
发生后写入失败，之前的开始记录仍然保留，但不能据此推断操作是否完成。
强制终止或掉电时，已提交的事件保留，会话可能仍标为 `running`，开始记录
可能没有对应的结束记录。日志不会自动重放命令。

当前执行器将命令列表拼接成一个 Bash 脚本，所以退出码和原始输出对应整批
执行；没有逐条命令的退出码。失败时后续 `&&` 命令可能未执行，执行日志
不会将命令列表中的每一步宣称为已完成。LangGraph 检查点仍在内存中，
持久化日志不提供跨进程继续执行或自动把历史会话送入模型的功能。

日志保存用户输入和工具返回的完整文本，不做截断或自动清理。模型角色记录
保存其调用参数和最终返回内容，不采集模型内部推理、环境变量或 API 密钥配置。
用户文本、命令输出和搜索来源本身可能包含敏感内容，应按本地数据管理日志
和导出文件。SQLite 数据库不加密；运行中备份应使用 SQLite backup API，
避免只复制主数据库而漏掉 WAL 中已提交的记录。

## 验证

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s work/tests -v
```

`test_interaction_log.py` 使用真实 SQLite 和 LangGraph、模拟模型/执行器，
验证完整多轮记录、确认前提交、取消、错误与重试、原始 shell 输出、日志
写入失败时停止执行、并发连接、子进程强制退出、历史查询与 JSON/JSONL 导出，
以及 content 的输出和配置密钥排除。测试日志使用临时目录。
