# 🪄 Samantha Assistant

Follow the steps below to set up Samantha in your new environment.

Samantha requires **Linux with Bash and Python 3.10 or newer**. Model access can
use a Qwen-compatible service, or Azure OpenAI as a fallback.


## 🚀 Getting Started

### Step 1. Clone this repository
```bash
git clone https://github.com/compsoc-oe-week/track1_nottingDuck.git
```
### Step 2. Enter the `work` directory
```bash
cd track1_nottingDuck/work
```

### Step 3. Configure model access

Samantha now defaults to the host's **Ollama `qwen3:4b`** model at
`http://127.0.0.1:11434/v1`. Start Ollama and confirm `ollama list` includes
`qwen3:4b`. No credentials or manual exports are required when Samantha and
Ollama run on the same host. To override the defaults:

```bash
export QWEN_BASE_URL='http://127.0.0.1:11434/v1'
export QWEN_MODEL='qwen3:4b'
export QWEN_API_KEY='ollama'
export QWEN_API_MODE='ollama'
```

For the project's openEuler Docker container, `docker-compose.yaml` supplies
`http://host.docker.internal:11434/v1`, `qwen3:4b`, and the placeholder key
automatically. Environment values in your shell or a Compose `.env` can override
these settings. From the repository root:

```bash
docker compose up -d --build
docker exec -it oe bash
# The following commands run inside the container:
samantha "Create hello.txt in the current directory and write hello world to it"
```

Compose selects `QWEN_API_MODE=ollama`, using the native `/api/chat` endpoint
with `think=false` to request non-thinking output separately from the JSON format.
Protocol reasoning fields are kept out of the displayed answer and history.
The confirmation agent uses a compact prompt and a 768-token output budget;
truncated output is rejected before approval or execution.
Small models can still produce verbose or inaccurate explanations; review the
exact commands shown before approving. The `/v1` suffix in `QWEN_BASE_URL` is
removed for native calls.
Use `QWEN_API_MODE=openai` for vLLM or other OpenAI-compatible services; direct
host execution defaults to this mode unless `QWEN_API_MODE` is set.

The image installs Python dependencies during the build and loads the `samantha`
function in interactive Bash sessions. `setup.sh` is only needed for a native
Linux installation. The `work` directory is mounted into the container, so files
created under `/work` are also visible on the host.

On Windows, Docker Desktop needs WSL 2 and hardware virtualization enabled in
the BIOS. Restart Windows after installing WSL or enabling its Windows features,
then start Docker Desktop and wait until its engine is running. See the
[Docker Desktop Windows requirements](https://docs.docker.com/desktop/setup/install/windows-install/).

From PowerShell at the repository root, run the complete container test:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\work\tests\docker_smoke.ps1
```

This allows the test script to run in that process without changing the system's
PowerShell execution policy.

The script builds and starts the openEuler container, runs the offline workflow
tests and Bash wrapper tests, then calls the host's real `qwen3:4b` model to test
file creation, reading, cancellation, and follow-up clarification. Its report is
saved to `work/.test-results/docker-ollama-smoke.json`. Use `-SkipBuild` to reuse
an existing image. Docker and Ollama must both be running.

If Docker Hub times out while pulling the base image, download the same version
from the [official openEuler repository](https://www.openeuler.org/en/wiki/install/image/)
and give it the tag used by the Dockerfile before retrying:

```powershell
docker pull hub.oepkgs.net/openeuler/openeuler:24.03-lts-sp3
docker tag hub.oepkgs.net/openeuler/openeuler:24.03-lts-sp3 openeuler/openeuler:24.03-lts-sp3
```

If package downloads are slow, put the following build setting in the repository's
local `.env` file, or set it in the PowerShell session before running the test:

```powershell
$env:OPENEULER_REPO_URL = 'https://repo.huaweicloud.com/openeuler'
```

The default is `https://repo.openeuler.org`. The build uses runtime package
repositories and retains openEuler's package signature verification.

Check access from inside the container with:

```bash
curl http://host.docker.internal:11434/v1/models
```

If the host service cannot be reached from Docker, configure Ollama's
`OLLAMA_HOST` bind address and restart it, following the
[Ollama server configuration guide](https://docs.ollama.com/faq#how-do-i-configure-ollama-server).
`0.0.0.0:11434` allows connections beyond localhost; use a bind address and
firewall appropriate to the host network.

For structured output, all agents use JSON mode with a default temperature of
0.2; the clarification agent additionally uses a fixed JSON schema and a compact
prompt suited to the 4B model. Text agents return Markdown inside a JSON text field, which Samantha unwraps
before displaying it. It requests no thinking
with `QWEN_REASONING_EFFORT=none`. In OpenAI mode, for the `qwen3` model family it also appends
the [Qwen3 `/no_think` soft switch](https://qwen.readthedocs.io/en/v3.0/getting_started/quickstart.html)
to the latest user message sent to Qwen, since some Ollama templates ignore the
API setting. Conversation history and Azure fallback input keep the original
request. Other model families receive no soft switch. It also filters thinking
tags from responses that still include them. Request timeout defaults to 120 seconds,
overridable with `QWEN_TIMEOUT`. Set `QWEN_REASONING_EFFORT` to an empty string
when a different backend does not support that request field.

Azure fallback is **disabled by default**, even if a key exists. To explicitly
enable it, configure your own deployment:

```bash
export SAMANTHA_AZURE_FALLBACK='true'
export AZURE_OPENAI_API_KEY='your-key'
export AZURE_OPENAI_ENDPOINT='https://your-resource.openai.azure.com/'
export AZURE_OPENAI_DEPLOYMENT='your-deployment'
export AZURE_OPENAI_API_VERSION='2024-12-01-preview'
```

These variables must be available in the terminal/container running Samantha.

To test model calls on Windows without a container, use the project's virtual
environment. For file-command execution through Git Bash, explicitly configure
its executable:

```powershell
$env:SAMANTHA_BASH = 'C:/Program Files/Git/bin/bash.exe'
.\.venv\Scripts\python.exe work/src/samantha.py "Create hello.txt in the current directory"
```

Direct Python invocation cannot change the directory of the parent PowerShell.

### Step 4. Run the setup script
```bash
source setup.sh
```
#### The setup script will:

- ✅ Ensure **python3** and **pip3** are installed on your system  

- ✅ Install all required Python dependencies listed in `requirements.txt`  

- ✅ Configure the shell so you can use `samantha` as a global command 

If dependencies are already installed, you can load only the terminal function:

```bash
source samantha.sh
```

## 💻 Usage

Once installed, simply run:
```bash
samantha <your natural language command>
```

For example:
```bash
samantha create a file named test.txt in the current directory and write hello world to it
```

Samantha asks follow-up questions when needed. Before execution it displays the
operation summary and the exact commands; reply `y` or `yes` to execute, or `n`
to cancel. Corrected commands require a new confirmation. There are at most
three correction attempts after the initial execution.

The Bash wrapper uses a separate temporary directory-result file for each call.
Only successful operations with a requested directory change move your terminal
to the destination; cancellation keeps the original directory.

For architecture and offline tests, see [the LangGraph workflow guide](langgraph_workflow.md).

An opt-in test uses the actual Ollama model, all main graph stages, and Bash:

```bash
python3 work/tests/ollama_smoke.py
```

Run it from the repository root. It approves only a small allowlist of fixture
commands and checks creation, output, rejection, and clarification in a temporary
directory. It does not use cloud fallback. On Windows:

```powershell
.\.venv\Scripts\python.exe work/tests/ollama_smoke.py --bash 'C:/Program Files/Git/bin/bash.exe' --report work/.test-results/ollama-smoke.json
```
