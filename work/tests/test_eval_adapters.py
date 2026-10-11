"""Optional Inspect API mocks: no install, model, generation or double dispatch."""

import asyncio
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch
import json

WORK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORK))

from evals.adapters.inspect_tasks import SingleDispatch, make_inspect_task


def mock_inspect_modules():
    root, dataset, scorer, solver = (types.ModuleType(name) for name in ("inspect_ai", "inspect_ai.dataset", "inspect_ai.scorer", "inspect_ai.solver"))
    root.Task = lambda **kwargs: types.SimpleNamespace(**kwargs)
    dataset.Sample = lambda **kwargs: types.SimpleNamespace(**kwargs)
    scorer.Score = lambda **kwargs: types.SimpleNamespace(**kwargs)
    scorer.accuracy = lambda: "test-accuracy-metric"
    scorer.scorer = lambda **kwargs: lambda function: function
    solver.solver = lambda function: function
    return {module.__name__: module for module in (root, dataset, scorer, solver)}


class InspectAdapterTests(unittest.IsolatedAsyncioTestCase):
    async def test_simultaneous_epochs_dispatch_one_shared_runner(self):
        runner = Mock(return_value={"protocol_pass_rate": .75, "experiment_id": "cached", "started_episodes": 4})
        with tempfile.TemporaryDirectory() as temp, patch.dict(sys.modules, mock_inspect_modules()):
            root = Path(temp)
            task = make_inspect_task(root / "dataset.jsonl", root / "config.json", root / "output", run_fn=runner)
            generate = Mock(side_effect=AssertionError("Inspect generate must never be called"))
            state = types.SimpleNamespace()
            await asyncio.gather(task.solver(state, generate), task.solver(state, generate), task.solver(state, generate))
            score = await task.scorer(state, None)
            self.assertEqual(runner.call_count, 1)
            generate.assert_not_called()
            self.assertEqual(len(task.dataset), 1)
            self.assertEqual(task.model, "mockllm/model")
            self.assertEqual(score.value, .75)
            self.assertTrue(score.metadata["framework_sample_is_experiment"])

    async def test_existing_saved_artifacts_never_dispatch_runner(self):
        with tempfile.TemporaryDirectory() as temp, patch.dict(sys.modules, mock_inspect_modules()):
            root = Path(temp)
            (root / "summary.json").write_text(json.dumps({"protocol_pass_rate": 1, "experiment_id": "already_saved"}), encoding="utf-8")
            task = make_inspect_task(saved_run=root)
            await task.solver(object(), Mock(side_effect=AssertionError()))
            score = await task.scorer(object(), None)
            self.assertEqual(score.value, 1)
            self.assertEqual(score.metadata["mode"], "saved_artifacts_only")

    async def test_failed_dispatch_is_cached_and_never_replays_side_effects(self):
        operation = Mock(side_effect=RuntimeError("injected runner failure"))
        cache = SingleDispatch(operation)
        for _ in range(2):
            with self.assertRaisesRegex(RuntimeError, "injected"):
                await cache.run()
        self.assertEqual(operation.call_count, 1)

    async def test_cached_result_is_independent(self):
        cache = SingleDispatch(lambda: {"values": [1]})
        first = await cache.run()
        first["values"].append(2)
        self.assertEqual(await cache.run(), {"values": [1]})

    async def test_mixed_modes_invalid_summary_and_missing_optional_sdk_are_explicit(self):
        with tempfile.TemporaryDirectory() as temp, patch.dict(sys.modules, mock_inspect_modules()):
            with self.assertRaises(ValueError):
                make_inspect_task(saved_run=temp, output_root=temp)
            task = make_inspect_task(temp, temp, temp, run_fn=lambda *args: {})
            await task.solver(object(), Mock())
            with self.assertRaises(ValueError):
                await task.scorer(object(), None)
        with patch.dict(sys.modules, {"inspect_ai": None}):
            with self.assertRaisesRegex(ImportError, "optional inspect_ai"):
                make_inspect_task(saved_run=temp)


if __name__ == "__main__":
    unittest.main()
