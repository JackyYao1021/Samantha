# P0：跨调用文件对象追踪

P0 在显式启用后，为单个文件的创建、重命名和追加保存对象身份、当前绝对路径及执行证据。后续独立 CLI 调用可以读取这些证据，用于理解“刚才的文件”。模型只提出预期变化；只有程序验证了实际文件结果，才更新对象记录。

实施范围对应 [small_model_multiturn_plan.md](small_model_multiturn_plan.md) 的 P0。P1–P4 的任务槽位、上下文重组、分块检索及通用失败记忆尚未实施。对象存储独立于历史记忆，`SAMANTHA_MEMORY_BACKEND=none` 时也可使用。

## 启用与关闭

CLI 默认关闭对象追踪。启用时，必须提供稳定的项目 ID；在同一项目内切换目录后，仍使用同一个 ID。

| 环境变量 | 行为 |
| --- | --- |
| `SAMANTHA_OBJECT_TRACKING` | `1`、`true` 或 `yes` 启用，大小写不敏感；其他值关闭 |
| `SAMANTHA_OBJECT_PROJECT` | 推荐显式配置的稳定项目 ID；缺省时使用 `SAMANTHA_MEMORY_PROJECT`，两者均为空则启动失败 |
| `SAMANTHA_OBJECT_USER` | 用户 ID；缺省使用 `SAMANTHA_MEMORY_USER`，最终缺省为 `local` |
| `SAMANTHA_OBJECT_PATH` | SQLite 对象数据库路径；缺省为仓库的 `work/.samantha/objects.sqlite3` |
| `SAMANTHA_MEMORY_BACKEND` | 原有历史记忆开关，默认 `none`；与对象追踪分别配置 |
| `SAMANTHA_BASH` | 原有执行器的 Bash 路径；默认 `/bin/bash` |

对象数据按 `(user_id, project_id)` 隔离。项目 ID 表示存储范围，不会自动限制用户明确请求的文件位置；实际操作仍需独立审查及当次人工确认。数据库路径本身不会替代用户、项目隔离。

设置 `SAMANTHA_OBJECT_TRACKING=false` 即可关闭，数据库保留。关闭后使用原有工作流和命令格式，不要求生成 `Effects`。再次启用时会重新检查已有文件状态，不恢复历史审批。

## Linux 三轮示例

需要已安装 Bash、Python 3.10+，并按 [使用指南](how_to_use_Samantha.md) 配置现有模型服务。以下命令从仓库根目录运行；示例使用新的临时目录，避免已有同名文件干扰创建前置检查。

```bash
python3 -m pip install -r work/requirements.txt
source work/samantha.sh

export SAMANTHA_OBJECT_TRACKING=true
export SAMANTHA_OBJECT_PROJECT=samantha-p0-demo
export SAMANTHA_OBJECT_USER=local
export SAMANTHA_MEMORY_BACKEND=none
export SAMANTHA_OBJECT_PATH="$PWD/work/.samantha/objects.sqlite3"

demo_dir="$(mktemp -d)"
cd "$demo_dir"

samantha "创建 notes.txt，内容为 alpha 后接一个换行。"
samantha "把刚才的文件重命名为 todo.txt。"
samantha "给刚才的文件追加 beta 后接一个换行。"
```

每次调用都会建立新的交互日志会话，并展示审查结果、命令和预期文件变化，由用户分别确认。预期最终只有 `todo.txt`，内容字节为 `alpha\nbeta\n`。重命名和追加应沿用同一对象 ID，版本依次为 1、2、3。

可以用以下命令检查示例结果：

```bash
test ! -e notes.txt
python3 -c 'from pathlib import Path; assert Path("todo.txt").read_bytes() == b"alpha\nbeta\n"'
```

如果模型生成了不符合 P0 格式的回答，工作流会在执行前停止；如果明确返回 `Effects: []`，命令仍走原有审批流程，但该次操作不建立对象追踪证据。以上示例是人工验收步骤，尚不代表真实模型已稳定通过三轮场景。

## Python 接口

`work/src` 可导入时，可以向原有 `process` 注入对象存储和范围：

```python
from agent_memory import MemoryScope
from object_tracking.store import ObjectStore
from samantha import process

scope = MemoryScope(user_id="local", project_id="my-stable-project")
with ObjectStore("/absolute/path/objects.sqlite3") as store:
    state = process(
        "创建 notes.txt，内容为 alpha 后接一个换行。",
        object_store=store,
        object_scope=scope,
        log_file="/absolute/path/interactions.sqlite3",
    )
```

接口参数是 `object_scope`，其类型复用 `MemoryScope`。提供 `object_store` 时必须提供 `object_scope` 或已有的 `memory_scope`；对象存储生命周期由调用方管理。省略 `object_store` 时保留原有调用方式。图层也可通过 `build_workflow(..., object_tracker=ObjectTracker(store, scope))` 注入相同能力。

测试可继续注入 `WorkflowServices`。启用追踪的命令生成服务应向工作流返回 `Commands`、`Explanation`、`Effects` 的 JSON 字符串；原有 `parse_commands` 入口保留。

默认生成器开启追踪时，真实模型只返回 `Explanation` 和 `Effects`，不生成受支持操作的 Shell 字符串。程序先校验效果，再用 `render_effects` 生成规范 Bash，供独立审查及当次确认。原始模型提议另记在 `objects.render_commands` 日志参数中，与最后的命令提议区分；意外包含命令字段会失败。这样避免模型重复描述同一内容时复制错路径或混淆实际换行与字面量 `\\n`。

原始创建／追加提议用 `content: {"lines": ["alpha"], "ends_with_newline": true}` 表示文本，而不是让模型猜测换行转义层数。程序用换行连接各行，并按布尔值添加末尾换行，得到最终效果中的 UTF-8 字符串 `alpha\n`。空行用空字符串表示；行内不允许实际换行。用户要求的字面量反斜杠文本保持原样，程序不会把它猜测转换成换行。重命名的 `content` 仍为 `null`。

模型明确返回 `Effects: []` 时，默认生成器额外调用一次原有命令生成服务，将结果补上空效果；不会因为格式错误自动退回旧生成器。该分支仍走原有审查、确认和执行流程，不产生已验证对象。关闭开关时没有这次效果提议或额外调用。

启用时，默认意图解析服务也使用严格 JSON Schema，要求返回唯一且非空的 `text` 字符串，再把其中的自然语言计划交给命令生成服务。例如 `{"text":"创建 /tmp/notes.txt，准确写入 alpha 和一个换行。"}`。缺少、空白或类型错误的 `text` 会停止流程。关闭追踪后，两个 agent 均保留原有默认入口及提示词；不会默认改变模型、采样或 reasoning 配置。

## 受支持的命令与效果

一个批次最多追踪一个效果，且必须只有一条匹配的受支持命令。下面是程序交给工作流的最终提议；默认模型的原始提议只有 `Explanation`、`Effects` 两个字段：

```json
{
  "Commands": ["printf '%s' 'alpha' > 'notes.txt'"],
  "Explanation": "创建文件并写入指定内容。",
  "Effects": [{
    "kind": "create",
    "path": "notes.txt",
    "destination": null,
    "content": "alpha",
    "entity_id": null,
    "entity_version": null
  }]
}
```

推荐使用以下字面量模板：

```bash
printf '%s' 'EXACT CONTENT' > 'PATH'
printf '%s' 'EXACT CONTENT' >> 'PATH'
mv -- 'SOURCE PATH' 'DESTINATION PATH'
```

`content` 是准确写入的 UTF-8 文本，包括所有换行。`printf '%s'` 不解释内容参数中的 `\n`；需要换行时，应把真实换行放入 JSON 解码后的内容和 Shell 引号内。`render_effects` 提供规范的 Shell 引用及命令渲染，支持中文、空格、单引号和字面量 Shell 符号。

绑定器也接受 `printf '%s\n' 'TEXT'`、不带选项和反斜杠的简单 `echo 'TEXT'`，以及不带选项的 `mv SOURCE DESTINATION`，但最终内容和路径仍必须准确匹配效果。输出重定向符号前必须有空白。命令中的路径不允许 `..` 组件；Windows Bash 路径使用正斜杠。

创建必须使用新对象身份，`entity_id` 和 `entity_version` 为 `null`。重命名要求 `destination` 为新路径、`content` 为 `null`；创建和追加要求 `destination` 为 `null`、`content` 为字符串。追加内容不能为空。引用已有对象时 ID 和版本必须同时提供；显式路径匹配当前有效对象时，运行时也会补全这两个字段，并检查范围、路径和版本。

相对路径按当次调用的起始目录转为绝对路径；已存路径不会按下一次调用的目录重新解释。命令与效果在审查前、执行前分别绑定校验。Shell 展开、命令替换、通配符、管道、分隔符、额外命令、任意 `mv` 选项等不能用于受追踪效果。

目录、复制、删除、复杂批处理及其他 Shell 操作需要返回 `Effects: []`。它们保留原有行为，但不会建立已验证对象事件；后续读取对象证据时会重新检查文件，发现外部变化则使旧对象失效。启用后缺失 `Effects` 或格式错误会在审批前失败，不会默默退回其他格式。

## 验证及身份规则

提议内容最多为 **64 KiB UTF-8 字节**。受验证的源文件、目标文件及追加后的预期内容，最多为 **1 MiB**。验证器只进行程序定义的只读文件检查，不运行模型给出的验证 Shell。

| 操作 | 执行前 | 执行后 |
| --- | --- | --- |
| 创建 | 目标不存在 | 目标为可读普通文件，大小及 SHA-256 与指定内容一致 |
| 重命名 | 源文件可读，目标不存在；已引用对象的指纹和版本匹配 | 旧路径消失，新路径大小及内容一致，设备号和 inode 与原文件一致 |
| 追加 | 源文件可读；已引用对象的指纹和版本匹配 | 同一设备号及 inode，完整内容等于原内容加指定追加内容 |

符号链接、路径祖先中的符号链接、Windows reparse point、非普通文件、不可读文件、超出大小限制的文件均不进入受验证闭环。创建覆盖和重命名覆盖不受支持。跨设备 `mv` 无法保持设备号及 inode，因此不能被确认为受支持的重命名。P0 不自动回滚已执行的文件变化。

对象指纹保存设备号、inode、大小、mtime、ctime 和 SHA-256。指纹是当前观察证据，并非永久身份保证。读取候选前会重新检查文件；缺失会留下 `tombstone`，内容或身份改变则标为 `stale`，不会继续作为有效候选。外部移动不会仅凭相似文件名自动关联。

已验证的重命名及追加保持对象 ID 并递增版本。在同一路径删除后重新创建的文件采用新的对象 ID。复制和主动删除没有 P0 效果接口；主动删除本身不会被记为已验证删除事件，但后续缺失检查仍可使旧对象失效。

存储使用 SQLite 事务及版本条件更新。较旧版本不能覆盖较新的路径；提交前会再次只读验证产物。事件去重键包含用户、项目、日志会话、执行操作 ID 和效果编号；重复提交同一来源及同一效果不会重复更新，复用来源但修改效果会报冲突。

事件保存旧/新路径、执行状态、验证状态、检查条件、前后指纹和来源 ID，并保留当次动作标识。执行原始结果先持久化到交互日志，再验证产物、提交对象更新。可通过 `store.candidates(scope)`、`store.lookup(scope, entity_id)` 和 `store.events(scope, entity_id=...)` 读取记录；这些接口不会恢复可执行命令或审批。

对象上下文包含项目候选、最近关注对象及失效信息，预算为 6000 字符、最多 20 个候选。有候选或失效信息时，提示词先放标记为证据的 JSON，再放当前请求/计划；空证据不会额外传给 agent。最近关注只是指代线索，不能覆盖显式用户路径。

P0 增加了一个保守的指代检查：常见“刚才的文件”等表达没有有效最近关注对象时，会先要求明确文件路径，避免仅因旧候选仍在就误选旧文件。显式源文件名可绕过这个缺失关注检查；当前规则识别中文追加/重命名的源或目标子句，以及部分英文 rename/append/add-to 形式。规则先屏蔽引号内容，再定位目标子句，避免把追加文本或重命名的新名称误当成源文件；可识别独立路径、带引号的名字及 README 等无扩展名 ASCII 名字。

这个检查只判断这些有限语句中是否明确指定了源/目标，不是完整的自然语言解析器，也不能证明生成效果满足用户的所有要求。未加引号的中文描述、嵌套引号、复杂复合请求及未覆盖的句式仍可能需要澄清；真正执行的路径继续依赖模型计划、效果校验、独立审查及当次确认。多候选、复杂指代及用户改口仍需要真实模型或后续 P1 消歧机制。

## 审批及失败行为

审批继续绑定实际命令、起始目录和 Shell；新增追踪标识还绑定效果和审查前快照。修改任何这些信息都会阻止沿用旧审批。等待确认期间文件发生变化时，执行前再次检查会停止该次动作，需要按当前事实发起新请求及审查。

用户拒绝或取消不会提交“已完成”对象。取消、已执行但不受支持的操作，以及没有提交已验证对象的失败，会清除本次读取的旧最近关注记录；文件操作成功但对象保存失败时也清除旧关注，避免下一轮误指向旧文件。已有对象及历史事件保留。清除时检查来源操作 ID，不会删除其他并发调用刚提交的新关注记录。随后模糊的“刚才的文件”需要重新指定目标。

受追踪命令返回失败后，P0 停止自动纠错重试，避免重复追加等副作用；验证器仍检查实际结果，如果预期效果确实已发生，可以保留该事实及新的关注对象，同时记录原执行状态为失败。Shell 返回成功但产物不满足条件时，原始执行结果保留，任务标为失败，且不会登记未经验证的完成事实。

对象数据库写入失败或版本冲突放入单独的 `tracking_error`。已成功执行的命令结果不会被改写或重放；终端会报告追踪需要处理，调用方可查看 `state["tracking_error"]`。P0 未实现通用日志恢复导入工具，修复派生状态时仍应仅重建记录或进行只读检查，禁止重放文件命令。

对象数据库可跨进程复用，每次 CLI 调用仍创建新的日志会话和 LangGraph 线程。现有 `InMemorySaver` 只用于当前进程；本功能不提供跨进程图恢复，也不保存旧的 `approved` 状态以继续执行。

## 离线验证

从仓库根目录运行完整离线测试：

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s work/tests -v
```

Linux 对应命令：

```bash
python3 -m unittest discover -s work/tests -v
bash work/tests/shell_smoke.sh
```

聚焦运行 P0 集成测试：

```powershell
$env:SAMANTHA_BASH = 'C:\Program Files\Git\bin\bash.exe'
.\.venv\Scripts\python.exe -m unittest discover -s work/tests -p test_object_tracking.py -v
```

```bash
python3 -m unittest discover -s work/tests -p test_object_tracking.py -v
```

集成测试使用临时目录和真实 SQLite、LangGraph，注入 fake agents，并通过真实 Bash 在三个独立 Python 子进程中完成创建→重命名→追加。它检查同一对象身份、版本递增、最新路径、三次独立审批/日志会话、最终字节及无关文件。Windows 可使用 Git Bash；Linux 使用 `SAMANTHA_BASH` 或可用的 `bash`。没有 Bash 时相关子进程测试会明确跳过。

其他离线测试位于 `test_object_effects.py`、`test_object_store.py`、`test_object_generation.py`、`test_object_references.py` 和 `test_object_focus.py`，覆盖命令绑定、精确 UTF-8 内容、恶意 Shell 形式、路径字面量、存储范围、对象失效、来源去重、版本冲突、指代检查及启用/关闭时的模型接口契约。2026-10-11 完整离线回归为 233 项，跳过 1 项 Windows 符号链接权限测试，其余通过；Git Bash 包装器 smoke 测试通过。离线通过只能验证工程协议，不代表已测得真实 Qwen 多轮质量提升；真实模型的三轮指代、歧义澄清及旧路径冲突仍按计划单独验收。

## 真实 Qwen 的提议探针

`work/tests/object_tracking_probe.py` 是单独运行的真实模型诊断工具，不属于离线测试，也不自动批准模型命令。它对三轮请求调用真实默认 agent，每次确认均拒绝，澄清请求则取消；执行服务另外设置为一旦调用就报错，确保模型生成的命令不会执行。

为了让下一轮读取真实对象证据，探针在调用之间使用确定性的 Python fixture 创建并重命名临时文件，分别保存执行/验证来源。因此它检查的是提议格式、当前路径引用、审查结果和不必要澄清，并不证明真实模型已经执行三轮任务。报告中应区分 `steps` 和 `fixture_seeds`；最终 fixture 的 `todo.txt` 只有 `alpha\n`，没有模型追加的 beta。

在现有模型服务已启动时，从仓库根目录运行；每次使用新的报告文件名，工具会拒绝覆盖已有报告及 `.artifacts` 目录：

```powershell
.\.venv\Scripts\python.exe work/tests/object_tracking_probe.py --base-url http://127.0.0.1:11434/v1 --api-mode ollama --report work/.test-results/object-tracking/live-contract-probe-example.json
```

```bash
python3 work/tests/object_tracking_probe.py --base-url http://127.0.0.1:11434/v1 --api-mode ollama --report work/.test-results/object-tracking/live-contract-probe-example.json
```

报告保存公开模型配置、可读到的 digest、代码版本、逐步状态和日志/对象数据库位置。v2 报告的 `raw_model_proposals` 是原始效果提议，`generated_command_proposals` 是程序渲染后的命令提议；两者分别保留。Azure fallback 被关闭。缺失的模型元信息保持未知，不推断实际运行上下文大小。

2026-10-11 最终探针 `work/.test-results/object-tracking/p0-qwen-contract-lines-20261011.json` 的创建、重命名、追加三项全部通过。使用本机 `qwen3:4b`（digest `359d7dd4bcda…`）、temperature 0.2、top_p 1、reasoning `none`；三轮均没有多余澄清，路径、对象 ID／版本、精确内容和审查结果符合预期。报告的源码树 hash 与最终实现一致。它仍是拒绝执行的提议探针；真实模型命令执行的三轮成功率及相对基线提升尚未测量。

早期失败报告也保留在同一目录：缺失 `text`、多余澄清、生成超时、路径复制错误和换行转义错误分别推动了严格意图 schema、简化上下文、程序渲染命令及结构化行内容。中断的诊断报告标为 `interrupted`，不计作完成的探针。不要把这些诊断或离线三进程测试混报成真实终端任务成绩。

## 代码入口

| 文件 | 职责 |
| --- | --- |
| `work/src/object_tracking/effects.py` | 效果 schema、生成约束、路径及内容校验、命令绑定与渲染 |
| `work/src/object_tracking/verification.py` | 有界只读快照、创建/重命名/追加的前后条件验证 |
| `work/src/object_tracking/store.py` | 分范围对象、事件、失效检查、事务及幂等版本更新 |
| `work/src/object_tracking/runtime.py` | 上下文、对象引用检查、审批快照及执行证据提交协调 |
| `work/src/workflow.py` | 显式启用的图接入、审查/审批绑定、执行后验证及失败处理 |
| `work/src/samantha.py` | `process` 参数、CLI 环境配置及独立日志会话 |
| `work/src/n2c.py` | 仅在启用时扩展命令生成 JSON schema |
| `work/src/intent_explain.py`、`work/src/chat.py` | 启用时要求严格且非空的意图 `text` 字段 |
| `work/tests/object_tracking_probe.py` | 拒绝执行的真实模型提议/指代诊断及独立 fixture 证据 |
