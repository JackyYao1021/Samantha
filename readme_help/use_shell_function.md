# Terminal shell function

Python runs in a child process, so it cannot change the working directory of
its calling terminal. Samantha uses a Bash function to invoke Python and apply
the resulting directory in the parent shell.

Install dependencies and register the function using the
[setup guide](how_to_use_Samantha.md). If dependencies are already installed,
load the existing wrapper directly:

```bash
source /absolute/path/to/Samantha/work/samantha.sh
samantha go to my home directory
```

The setup script adds an absolute source path to `~/.bashrc`, so future Bash
sessions load the function automatically. There is no need to copy a separate
function implementation into `.bashrc`.

The wrapper creates a unique temporary file for each call and passes its path
through `SAMANTHA_STATE_FILE`. Python initializes it with the starting directory
and writes the final directory after the LangGraph workflow completes. Bash
reads the JSON, changes directories after a successful invocation when needed,
and removes the file. Cancellation retains the starting directory.

See [the LangGraph workflow guide](langgraph_workflow.md) for graph routing,
confirmation, correction, and checkpoint behavior.
