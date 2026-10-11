"""Optional Inspect adapter: one existing runner, or read-only saved results.

Official API references checked 2026-10-11:
https://inspect.aisi.org.uk/tasks.html
https://inspect.aisi.org.uk/solvers.html#custom-solvers
https://inspect.aisi.org.uk/custom-scorers.html
https://inspect.aisi.org.uk/datasets.html#dataset-samples

Inspect is deliberately not a required dependency. This module never uses its
generate callback, Agent Bridge, agents, or shell tools.
"""

import asyncio
from copy import deepcopy
import json
from pathlib import Path


class SingleDispatch:
    """Cache both success and failure so framework epochs/retries never rerun work."""

    def __init__(self, function):
        self.function, self.started, self.result, self.error = function, False, None, None
        self._lock = asyncio.Lock()

    async def run(self):
        async with self._lock:
            if not self.started:
                self.started = True
                try:
                    self.result = await asyncio.to_thread(self.function)
                except BaseException as exc:
                    self.error = exc
            if self.error is not None:
                raise self.error
            return deepcopy(self.result)


def _inspect_api():
    try:
        from inspect_ai import Task
        from inspect_ai.dataset import Sample
        from inspect_ai.scorer import Score, accuracy, scorer
        from inspect_ai.solver import solver
    except ImportError as exc:
        raise ImportError("Inspect adapter requires optional inspect_ai; offline runner does not. No dependency was installed.") from exc
    return Task, Sample, Score, accuracy, scorer, solver


def make_inspect_task(dataset_path=None, config_path=None, output_root=None, *, name="samantha", saved_run=None, run_fn=None):
    """A singleton experiment Sample prevents Inspect from scheduling each episode.

    Repeats, cases, approvals, budgets and all task scoring remain in Samantha's
    runner. A repeated solver invocation returns the cached artifact summary.
    ``saved_run`` loads a prior summary without dispatching any runtime work.
    """
    Task, Sample, Score, accuracy, scorer, solver = _inspect_api()
    if saved_run is not None:
        if any(value is not None for value in (dataset_path, config_path, output_root, run_fn)):
            raise ValueError("Saved-result and dispatch modes are mutually exclusive")
        saved = Path(saved_run).resolve()
        def operation():
            summary = json.loads((saved / "summary.json").read_text(encoding="utf-8"))
            if not isinstance(summary, dict):
                raise ValueError("Saved summary must be an object")
            return summary
        mode = "saved_artifacts_only"
    else:
        if dataset_path is None or config_path is None or output_root is None:
            raise ValueError("Dispatch mode requires dataset, configuration and output root")
        from ..schemas import relative_path
        if relative_path(name).name != name:
            raise ValueError("Inspect experiment name must be one directory name")
        dataset, config = Path(dataset_path).resolve(), Path(config_path).resolve()
        output = Path(output_root).resolve() / name
        def operation():
            if run_fn is None:
                from ..runner import run_experiment
                return run_experiment(dataset, config, output)
            return run_fn(dataset, config, output)
        mode = "single_existing_runner_dispatch"
    dispatch = SingleDispatch(operation)

    @solver
    def samantha_saved_or_runner():
        async def solve(state, generate):
            await dispatch.run()
            return state
        return solve

    @scorer(metrics=[accuracy()])
    def samantha_protocol_summary():
        async def score(state, target):
            if not dispatch.started or dispatch.error is not None:
                raise ValueError("Samantha runner has no completed saved summary")
            summary = dispatch.result
            value = summary.get("protocol_pass_rate")
            if type(value) not in (int, float) or not 0 <= value <= 1:
                raise ValueError("Saved summary is missing a valid protocol pass rate")
            return Score(value=value, explanation="Protocol rate from Samantha artifacts; no Inspect model invocation.",
                         metadata={"mode": mode, "source_experiment_id": summary.get("experiment_id"),
                                   "started_episodes": summary.get("started_episodes"),
                                   "task_success_rate": summary.get("attempted_task_success_rate"),
                                   "framework_sample_is_experiment": True})
        return score

    return Task(dataset=[Sample(id="samantha_experiment", input="Read Samantha experiment artifacts; never call generate.")],
                solver=samantha_saved_or_runner(), scorer=samantha_protocol_summary(), name=name,
                epochs=1, model="mockllm/model", metadata={"adapter_version": "inspect-existing-runner-v1", "mode": mode,
                                                        "sampling_unit": "experiment wrapper; Samantha owns episode statistics"})
