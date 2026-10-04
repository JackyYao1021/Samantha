# 文件和图片内容搜索

实现链路：原文提取 / Qwen2.5-VL 视觉理解 → 分块、摘要、标签 → BGE-M3 编码
→ Qdrant 索引 → 文件搜索 → 可选的带来源问答。

## 本机运行

Python 3.10+。默认使用 FlagEmbedding 的 BGE-M3 dense+sparse 混合检索：

```powershell
.venv/Scripts/python.exe -m pip install -r work/requirements-content-hybrid.txt
ollama pull qwen2.5vl:7b
.venv/Scripts/python.exe work/src/content_cli.py doctor
.venv/Scripts/python.exe work/src/content_cli.py index ./documents --json
.venv/Scripts/python.exe work/src/content_cli.py search "登录失败的截图" --ext png
.venv/Scripts/python.exe work/src/content_cli.py search "合同付款安排" --tag 合同 --limit 5
.venv/Scripts/python.exe work/src/content_cli.py ask "合同约定什么时候付款？" --json
```

默认首次编码会从 Hugging Face 下载 `BAAI/bge-m3`；后续使用本机模型缓存。
视觉模型使用 Ollama 的 `qwen2.5vl:7b`，需要提前下载。也可以配置已有的
OpenAI-compatible Qwen2.5-VL 服务，使用其 `/v1` 地址和实际模型名称。
模型下载和推理所需的内存、磁盘、GPU 资源取决于模型及运行方式。

如希望使用 Ollama 运行两个模型，安装基础依赖并启用纯 dense 检索：

```powershell
.venv/Scripts/python.exe -m pip install -r work/requirements.txt
ollama pull qwen2.5vl:7b
ollama pull bge-m3
$env:CONTENT_EMBEDDING_BACKEND = "ollama"
$env:CONTENT_EMBEDDING_MODEL = "bge-m3"
.venv/Scripts/python.exe work/src/content_cli.py doctor
.venv/Scripts/python.exe work/src/content_cli.py index ./documents
.venv/Scripts/python.exe work/src/content_cli.py search "合同付款安排"
```

Ollama `/api/embed` 只提供 dense 输出；此模式不执行 sparse 混合检索。
切换后端会使用另一个默认 collection，须重新入库。两种后端均使用 1024 维
BGE-M3 文本向量，模型和后端身份不同的向量禁止混入同一 collection。

Linux / openEuler 中可以使用原有 shell 入口：

```bash
source work/samantha.sh
samantha content doctor
samantha content index "/path/to/my documents"
samantha content search "合同付款安排" --tag 合同 --ext pdf
samantha content ask "合同约定什么时候付款？"
```

独立 CLI 与 `samantha content` 使用相同的索引和配置。原有自然语言执行命令
仍使用原来的 LangGraph 工作流。路径包含空格时必须加引号。

## 配置

配置项见根目录的 `.env.example`；CLI 自动读取根目录 `.env`，已设置的环境变量
优先。内容理解使用单独的 `CONTENT_VL_MODEL`，不会自动将 `qwen3:4b` 当作视觉模型。

| 配置 | 默认值 / 含义 |
| --- | --- |
| `CONTENT_VL_BASE_URL` | `QWEN_BASE_URL` 或 `http://127.0.0.1:11434/v1` |
| `CONTENT_VL_MODEL` | `qwen2.5vl:7b` |
| `CONTENT_VL_API_KEY` | `ollama` |
| `CONTENT_EMBEDDING_BACKEND` | `flagembedding`；另选 `ollama` |
| `CONTENT_EMBEDDING_MODEL` | FlagEmbedding 默认 `BAAI/bge-m3`；Ollama 默认 `bge-m3` |
| `CONTENT_EMBEDDING_DEVICE` | `cpu`；FlagEmbedding 可设置 `cuda:0` |
| `CONTENT_OLLAMA_URL` | `http://127.0.0.1:11434`，用于 Ollama embedding |
| `CONTENT_QDRANT_URL` | 空值使用嵌入式持久化；例如 `http://127.0.0.1:6333` |
| `CONTENT_QDRANT_API_KEY` | 可选，供远程 Qdrant 使用 |
| `CONTENT_QDRANT_PATH` | 默认 `work/.content-index/qdrant`，与当前执行目录无关 |
| `CONTENT_COLLECTION` | 默认按 embedding 后端分别命名 |
| `CONTENT_PDF_VISION` | `auto` / `always` / `never` |
| `CONTENT_CHUNK_SIZE` / `CONTENT_CHUNK_OVERLAP` | 1200 / 150 个字符，非 token |
| `CONTENT_BATCH_SIZE` | 每批 8 个 embedding / Qdrant points |
| `CONTENT_TIMEOUT` | 模型请求超时 180 秒 |

`doctor` 检查模型服务、已安装模型和 Qdrant 可访问性，不下载或加载权重。
FlagEmbedding 的诊断仅检查 Python 包是否存在，完整权重是否可用在首次编码时验证。

## 文件处理与搜索行为

- 文本：TXT、MD、RST、CSV、TSV、JSON、LOG、HTML；保留文本，支持 UTF-8、
  带 BOM 的 UTF-16 和 GB18030。HTML 当前保留源码，CSV 保留文本行。
- DOCX：按正文顺序提取段落和表格，并分析内嵌图片。内嵌图片须能被 Pillow 解码；
  页眉、页脚、文本框和旧版 DOC 文件暂不支持。
- PDF：按页保留可提取的原文；`auto` 对扫描页、有位图或矢量图形的页面调用 VL。
  `always` 分析所有页面；`never` 仅提取原文，扫描页会报错，防止静默遗漏。
  密码保护的 PDF 会报错。表格文本不保证保留原始二维结构。
- 图片：PNG、JPG、JPEG、WEBP、BMP、TIF、TIFF；分别索引视觉描述、OCR 文本和
  摘要，多页 TIFF 按帧处理。输入模型的图像最长边限制为 2048 像素，小字可能丢失。
- 每个文本块原文均保留；文件摘要单独入库。长文档通过分组摘要逐级合并。
  标签规范化为空白归一、小写，最多 24 个，作为 payload 过滤字段，也加入编码文本。
- `--tag` 可重复，含义是同时包含这些标签；`--ext` 按文件后缀筛选。
  标签不会自动从查询中变成硬筛选条件，避免误判标签导致漏检。
- FlagEmbedding dense+sparse 通过 Qdrant RRF 融合；Ollama 使用 cosine dense 检索。
  sparse 是 BGE-M3 学习得到的词项权重，并非 BM25 或严格字符串匹配。
- 搜索按文件合并，每个文件展示最多三个命中块及页码。候选量会逐步增加到最多
  2000，因此极长文件占据候选时可能返回少于 `--limit` 个文件。
- `ask` 基于命中片段回答，返回 `[1]` 等来源编号、路径、页码、内容类型；输入正文
  最多 18000 个字符。无命中时不调用回答模型。生成的描述、OCR 和摘要可能出错，
  应结合原始文件核对。当前不支持上传图片查找外观相似图片。

## 重复入库和失败恢复

同一绝对路径对应同一 `file_id`。内容 SHA-256、模型身份、VL 地址、分块设置或
PDF 视觉策略变化时重新入库；`--force` 可强制重新处理。文件修改过程中入库会
报错，防止生成不一致的索引。模型权重原地替换时请使用 `--force`。

新版本先以 `ready=false` 上传，全部上传成功后标记可搜索，再删除旧版本。
模型或分批上传失败时，旧版本仍可搜索，未完成的新版本不会出现在结果中。
下一次成功更新会清理同一文件的旧版本和未完成版本。发布和删除不是跨请求事务，
发布后、清理前进程中断可能短暂留下两个可搜索版本，可用 `--force` 修复。

目录批量处理会继续处理其他文件，并汇报单个文件错误；部分失败时退出码为 1。
自动跳过隐藏目录、虚拟环境、Git 和 node_modules。文件移动、改名会作为新文件
入库；当前不自动同步删除或清理已移走文件的索引。

嵌入式 Qdrant 适合单进程个人使用，同一存储目录不能被多个进程同时打开；需要并发
搜索时使用 Qdrant server。当前同一文件的入库采用单写入者模式。
原始文件不会被移动或修改；摘要和 tags 存在 Qdrant 中，不写回原文件。

## Docker

Compose 默认 Ollama dense 模式，嵌入式索引存于挂载的 `work` 目录。
宿主机须先准备 `qwen2.5vl:7b` 和 `bge-m3` 两个模型。

```bash
docker compose up -d oe
docker compose exec oe python3 /work/src/content_cli.py doctor
docker compose exec oe python3 /work/src/content_cli.py index /work/documents
```

如启用独立 Qdrant 服务：

```bash
docker compose --profile content-search up -d qdrant
# Container client: CONTENT_QDRANT_URL=http://qdrant:6333
# Host client:      CONTENT_QDRANT_URL=http://127.0.0.1:6333
```

容器中使用 FlagEmbedding hybrid 时，设置 `CONTENT_HYBRID=true` 以安装额外依赖，
设置 `CONTENT_EMBEDDING_BACKEND=flagembedding` 和 `CONTENT_EMBEDDING_MODEL=BAAI/bge-m3`，
然后 `docker compose build oe`。`.env` 中用于本机的 127.0.0.1 模型地址在容器内须
改为 `host.docker.internal`。远程 Qdrant 的 tags、文件类型、ready 等字段会建立 payload 索引。

## 验证

```powershell
.venv/Scripts/python.exe -m unittest discover -s work/tests -p "test_*.py"
```

新增测试使用真实嵌入式 Qdrant，模型输出采用固定 fixture，覆盖 dense/sparse 查询、
标签筛选、文件合并、持久化、更新、失败恢复、图片 OCR、PDF 页码、DOCX 表格和图片、
来源问答以及适配器契约。检索质量、OCR 准确率和实际推理资源仍须用真实模型及文件评估。

实现依据：[Qwen2.5-VL](https://qwenlm.github.io/blog/qwen2.5-vl/)、
[BGE-M3](https://huggingface.co/BAAI/bge-m3)、
[Qdrant Hybrid Query](https://qdrant.tech/documentation/search/hybrid-queries/)、
[Ollama Embedding API](https://docs.ollama.com/api/embed)。
