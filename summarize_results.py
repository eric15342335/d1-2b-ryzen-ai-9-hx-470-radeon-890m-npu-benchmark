"""Derive comparable latency statistics from the preserved raw samples."""
import json
import math
import sqlite3
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results"


def read(name):
    return json.loads((OUT / name).read_text(encoding="utf-8-sig"))


def stats(samples):
    assert len(samples) == 20 and all(math.isfinite(x) and x > 0 for x in samples)
    mean = statistics.mean(samples)
    return {"samples": len(samples), "mean_ms": mean, "median_ms": statistics.median(samples),
            "p95_ms": sorted(samples)[math.ceil(.95 * len(samples)) - 1],
            "states_per_second": 1000 / mean, "questions_per_second": 3000 / mean}


def metrics(name):
    return {p[0]: float(p[1]) for line in (OUT / name).read_text().splitlines()
            if line and not line.startswith("#") and len(p := line.split()) == 2}


def main():
    rows = []
    layout = read("input-layout.json")
    for backend, name in [
        ("CPU ONNX q4", "cpu-q4-ticket-common-forward.json"),
        ("iGPU WebGPU ONNX q4", "webgpu-q4-ticket-common-forward.json"),
        ("CPU+NPU Tomoali W4", "tomoali-npu.json"),
    ]:
        data = read(name)
        for index, row in enumerate(data["workloads"]):
            if "forward_timings_ms" in row:
                forwards = [sum(v) for v in row["forward_timings_ms"]]
                tokens = sum(row["packed_tokens"])
            else:
                forwards = [sum(v["ms"] for v in records) for records in row["forward_records_per_request"]]
                tokens = row["output"]["usage"]["input_tokens"]
                assert all(sum(v["sequence_length"] for v in records) == tokens
                           for records in row["forward_records_per_request"])
            assert tokens == layout[index]["packed_forward_tokens"]
            mean_forward = statistics.mean(forwards)
            rows.append({"backend": backend, "workload": layout[index]["name"],
                         **stats(row["timings_ms"]), "mean_input_forward_ms": mean_forward,
                         "input_forward_tokens_per_second": tokens * 1000 / mean_forward,
                         "input_tokens": tokens, "forward_calls_per_request": 1,
                         "max_sequence_tokens": tokens, "generated_tokens": 0,
                         "timing_source": name})
    for tag in ("q4km", "q8"):
        for index, fixture in enumerate(("ticket", "long-ticket")):
            prefix = f"llama-{tag}-{fixture}"
            with sqlite3.connect(OUT / (prefix + ".sqlite")) as database:
                samples = database.execute(
                    "SELECT duration,status FROM oha WHERE run=(SELECT max(run) FROM oha)").fetchall()
            assert len(samples) == 20 and all(s == 200 for _, s in samples)
            before, after = metrics(prefix + "-metrics-before.txt"), metrics(prefix + "-metrics-after.txt")
            deltas = {k: after[k] - before[k] for k in after if k in before}
            tokens = layout[index]["separate_total_tokens"]
            assert deltas["llamacpp:prompt_tokens_total"] == tokens * 20
            assert deltas["llamacpp:prompt_tokens_cached_total"] == 0
            assert deltas["llamacpp:tokens_predicted_total"] == 0
            prompt_ms = deltas["llamacpp:prompt_seconds_total"] * 1000 / 20
            rows.append({"backend": "iGPU llama.cpp " + tag, "workload": fixture,
                         **stats([t * 1000 for t, _ in samples]),
                         "mean_input_forward_ms": prompt_ms,
                         "input_forward_tokens_per_second": tokens * 1000 / prompt_ms,
                         "input_tokens": tokens, "forward_calls_per_request": 3,
                         "max_sequence_tokens": max(layout[index]["separate_forward_tokens"].values()),
                         "generated_tokens": 0, "cached_tokens": 0,
                         "metrics_deltas": deltas, "timing_source": prefix + ".sqlite"})
    (OUT / "comparison.json").write_text(json.dumps({"percentile": "nearest rank", "rows": rows}, indent=2))
    for row in rows:
        print(row["backend"], row["workload"], *(round(row[k], 3) for k in
              ("mean_ms", "median_ms", "p95_ms", "mean_input_forward_ms", "states_per_second")))


if __name__ == "__main__":
    main()
