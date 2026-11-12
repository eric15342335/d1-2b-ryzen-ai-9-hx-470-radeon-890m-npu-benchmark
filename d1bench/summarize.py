"""Latency statistics recomputed from the raw samples every runner preserves."""

import math
import sqlite3
import statistics
from pathlib import Path
from typing import Any

from d1bench.common import RESULTS, Benchmark, read_json


def latency_stats(samples_ms: list[float]) -> dict[str, float]:
    """Mean, median, nearest-rank P95 and requests per second."""
    mean = statistics.mean(samples_ms)
    return {
        "samples": len(samples_ms),
        "mean_ms": mean,
        "median_ms": statistics.median(samples_ms),
        "p95_ms": sorted(samples_ms)[math.ceil(0.95 * len(samples_ms)) - 1],
        "states_per_second": 1000 / mean,
    }


def prometheus(path: Path) -> dict[str, float]:
    """Counters from a llama-server /metrics snapshot."""
    pairs = (line.split() for line in path.read_text(encoding="utf-8").splitlines() if not line.startswith("#"))
    return {p[0]: float(p[1]) for p in pairs if len(p) == 2}  # noqa: PLR2004


def native_rows(bench: Benchmark, layout: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Rows for the in-process runners (ONNX Runtime and Tomoali); one ORT.Run per request."""
    rows = []
    for index, row in enumerate(read_json(bench.result)["workloads"]):
        if bench.kind == "npu":
            forwards = [sum(ms) for ms in row["forward_timings_ms"]]
            tokens = sum(row["packed_tokens"])
        else:
            forwards = [sum(r["ms"] for r in records) for records in row["forward_records_per_request"]]
            tokens = row["output"]["usage"]["input_tokens"]
            assert all(
                sum(r["sequence_length"] for r in records) == tokens for records in row["forward_records_per_request"]
            )
        assert tokens == layout[index]["packed_forward_tokens"], "input differs from the recorded layout"
        mean_forward = statistics.mean(forwards)
        rows.append(
            {
                "id": bench.id,
                "backend": bench.label,
                "workload": layout[index]["name"],
                **latency_stats(row["timings_ms"]),
                "mean_input_forward_ms": mean_forward,
                "input_forward_tokens_per_second": tokens * 1000 / mean_forward,
                "input_tokens": tokens,
                "forward_calls_per_request": 1,
                "generated_tokens": 0,
                "timing_source": bench.result,
            },
        )
    return rows


def llama_rows(bench: Benchmark, layout: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Rows for llama.cpp: oha's per-request durations plus /metrics counter deltas."""
    rows = []
    for entry in layout:
        prefix = RESULTS / f"{bench.result}-{entry['name']}"
        with sqlite3.connect(prefix.with_suffix(".sqlite")) as database:
            # Older files hold several oha runs; the newest is the measured one.
            samples = database.execute(
                "SELECT duration, status FROM oha WHERE run = (SELECT max(run) FROM oha)"
            ).fetchall()
        assert all(status == 200 for _, status in samples), "oha recorded a failed request"  # noqa: PLR2004
        n = len(samples)
        before = prometheus(prefix.with_name(prefix.name + "-metrics-before.txt"))
        after = prometheus(prefix.with_name(prefix.name + "-metrics-after.txt"))
        deltas = {k: after[k] - before[k] for k in after if k in before}
        tokens = entry["separate_total_tokens"]
        assert deltas["llamacpp:prompt_tokens_total"] == tokens * n, "prompt token count differs from the layout"
        assert deltas["llamacpp:prompt_tokens_cached_total"] == 0, "prompt cache was reused"
        assert deltas["llamacpp:tokens_predicted_total"] == 0, "server generated tokens"
        prompt_ms = deltas["llamacpp:prompt_seconds_total"] * 1000 / n
        rows.append(
            {
                "id": bench.id,
                "backend": bench.label,
                "workload": entry["name"],
                **latency_stats([seconds * 1000 for seconds, _ in samples]),
                "mean_input_forward_ms": prompt_ms,
                "input_forward_tokens_per_second": tokens * 1000 / prompt_ms,
                "input_tokens": tokens,
                "forward_calls_per_request": len(entry["separate_forward_tokens"]),
                "generated_tokens": 0,
                "cached_tokens": 0,
                "metrics_deltas": deltas,
                "timing_source": prefix.with_suffix(".sqlite").name,
            },
        )
    return rows


def summarize(benchmarks: list[Benchmark]) -> list[dict[str, Any]]:
    """Comparison rows for every timed benchmark that has results."""
    layout = read_json("input-layout.json")
    rows = []
    for bench in benchmarks:
        if bench.kind in {"onnx", "npu"}:
            rows += native_rows(bench, layout)
        elif bench.kind == "llama":
            rows += llama_rows(bench, layout)
    return rows
