"""Offline per-event telemetry microbenchmark, without Workflow or model calls.

Run from work: python -m evals.telemetry_benchmark --repeats 2000
    --output .test-results/evaluation/p1-telemetry-benchmark.json

These stages are measured independently. They do not measure end-to-end
Workflow latency, SDK/export network overhead, model latency, or CPU/RSS usage.
Capture retains a whole batch so its Python allocation peak includes storage.
"""

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import platform
import statistics
import sys
import time
import tracemalloc

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from observability.events import capture_events, emit_event
from observability.langfuse_exporter import LangfuseExporter
from observability.redaction import redact

from .harness import write_json


SYNTHETIC_EVENT = {
    "kind": "model_call", "role": "model",
    "trace": {"experiment_id": "synthetic", "episode_id": "fixture", "attempt_id": "repeat_01"},
    "data": {"model": "qwen3:4b", "status": "succeeded", "call_type": "chat",
             "usage": {"input_tokens": 12, "output_tokens": 7, "reasoning_tokens": None},
             "effective_parameters": {"temperature": 0.2, "prompt_sha256": "a" * 64},
             "headers": {"Authorization": "Bearer synthetic_benchmark_credential"},
             "nested": [{"freeform": "synthetic fixture text"}, {"path": "/fixture/example.txt"}]},
}


def _percentile(values, fraction):
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower, upper = math.floor(position), math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


@contextmanager
def _stage(name):
    if name == "noop":
        yield lambda: None
    elif name == "capture_events":
        with capture_events():
            yield lambda: emit_event(SYNTHETIC_EVENT["kind"], SYNTHETIC_EVENT["data"],
                                    role=SYNTHETIC_EVENT["role"], trace=SYNTHETIC_EVENT["trace"])
    elif name == "recursive_redact":
        yield lambda: redact(SYNTHETIC_EVENT)
    elif name == "disabled_exporter":
        exporter = LangfuseExporter()  # Disabled: no thread, import, SDK, or network.
        yield lambda: exporter.submit(SYNTHETIC_EVENT)
    else:
        raise ValueError("Unknown benchmark stage")


def benchmark(*, repeats=2000, p95_budget_us=250):
    """Record isolated median/p95 durations and separate Python allocation peaks.

    The local per-stage p95 budget is an explicit microbenchmark budget. It is
    not an assertion about the percentage overhead of a real Workflow.
    """
    if type(repeats) is not int or not 1 <= repeats <= 100000:
        raise ValueError("repeats must be an integer from 1 to 100000")
    if (type(p95_budget_us) not in (int, float) or not math.isfinite(p95_budget_us)
            or p95_budget_us <= 0):
        raise ValueError("p95_budget_us must be positive and finite")
    stages = {}
    for name in ("noop", "capture_events", "recursive_redact", "disabled_exporter"):
        # Warm each stage independently; warm-up captures are discarded.
        with _stage(name) as invoke:
            for _ in range(100):
                invoke()
        samples = []
        with _stage(name) as invoke:
            for _ in range(repeats):
                started = time.perf_counter()
                invoke()
                samples.append((time.perf_counter() - started) * 1_000_000)
        # Allocation tracing perturbs timing, so measure it in a separate batch.
        tracemalloc.start()
        try:
            with _stage(name) as invoke:
                for _ in range(repeats):
                    invoke()
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        stages[name] = {"samples": repeats, "median_us": round(statistics.median(samples), 3),
                        "p95_us": round(_percentile(samples, .95), 3),
                        "python_allocation_peak_bytes": peak}
    baseline = stages["noop"]
    for stage in stages.values():
        stage["median_over_noop_us"] = round(max(0, stage["median_us"] - baseline["median_us"]), 3)
        stage["p95_over_noop_us"] = round(max(0, stage["p95_us"] - baseline["p95_us"]), 3)
        stage["within_microbenchmark_p95_budget"] = stage["p95_over_noop_us"] <= p95_budget_us
    return {"schema_version": "1.0", "kind": "offline_telemetry_microbenchmark",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "python": platform.python_version(), "platform": platform.system(),
            "clock": "time.perf_counter", "repeats": repeats, "warmup_per_stage": 100,
            "synthetic_event_bytes": len(json.dumps(SYNTHETIC_EVENT, sort_keys=True).encode("utf-8")),
            "microbenchmark_p95_budget_us": p95_budget_us, "stages": stages,
            "degradation_policy": "External queue saturation drops events; export/flush failures never replay business calls.",
            "limitations": [
                "Small synthetic event stages are independent, not an end-to-end Workflow overhead measurement.",
                "No model, SDK delivery, network, raw document, image, or production prompt is used.",
                "Python allocation peaks are not process RSS or CPU utilization; capture includes retained batch events.",
                "Per-event clock overhead and host scheduling affect microsecond measurements.",
                "The microbenchmark p95 budget does not enforce a real-Workflow latency threshold."]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=2000)
    parser.add_argument("--p95-budget-us", type=float, default=250)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args(argv)
    if arguments.output.exists():
        raise ValueError("Refusing to overwrite an existing benchmark artifact")
    result = benchmark(repeats=arguments.repeats, p95_budget_us=arguments.p95_budget_us)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(arguments.output, result)
    print(f"Saved telemetry microbenchmark: {arguments.output}")
    return result


if __name__ == "__main__":
    main()
