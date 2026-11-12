"""Measure the official d1 host runtime; only session/provider selection is adapted."""
import argparse
import hashlib
import json
import os
import pathlib
import statistics
import subprocess
import sys
import time
from collections import Counter

import onnxruntime as ort

ROOT = pathlib.Path(__file__).resolve().parent
UPSTREAM = ROOT / ".cache/liquidonnx"
sys.path.insert(0, str(UPSTREAM / "src"))
from liquidonnx.d1 import session as d1  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", choices=["cpu", "dml", "webgpu", "npu"], required=True)
    parser.add_argument("--npu-dll", type=pathlib.Path)
    parser.add_argument("--request", type=pathlib.Path)
    parser.add_argument("--set-size", type=int)
    parser.add_argument("--measure-forward", action="store_true")
    parser.add_argument("--precision", default="q4")
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.repeats < 1 or args.warmup < 1 or args.threads < 1:
        parser.error("repeats, warmup and threads must be positive")
    results = ROOT / "results"
    results.mkdir(exist_ok=True)
    runtime_revision = subprocess.check_output(
        ["git", "-C", str(UPSTREAM), "rev-parse", "HEAD"], text=True).strip()
    if runtime_revision != "26857c2e4d12381c6c630e7f7daf00bb3589f721":
        raise RuntimeError("Upstream checkout differs from the reviewed revision")
    label = f"{args.provider}-{args.precision}" + ("-profile" if args.profile else "")
    if args.request:
        label += "-" + args.request.stem
    if args.set_size:
        label += f"-set{args.set_size}"
    if args.measure_forward:
        label += "-forward"
    sessions = {}
    forward_records = []
    webgpu_device = None
    if args.provider == "webgpu":
        import onnxruntime_ep_webgpu as webgpu
        ort.register_execution_provider_library("webgpu_benchmark", webgpu.get_library_path())
        devices = [d for d in ort.get_ep_devices()
                   if d.ep_name == webgpu.get_ep_name() and d.device.vendor_id == 0x1002]
        if len(devices) != 1:
            raise RuntimeError(f"Expected one AMD WebGPU device, found {len(devices)}")
        webgpu_device = devices[0]
        print(f"Using AMD WebGPU device: {webgpu_device.device.metadata}", flush=True)
    if args.provider == "npu":
        if args.npu_dll is None:
            parser.error("--npu-dll is required for NPU execution")
        dll_directory = os.add_dll_directory(str(args.npu_dll.parent))
        ort.register_execution_provider_library("VitisAIExecutionProvider", str(args.npu_dll))
        devices = [d for d in ort.get_ep_devices() if d.ep_name == "VitisAIExecutionProvider"
                   and d.device.type == ort.OrtHardwareDeviceType.NPU]
        if len(devices) != 1:
            raise RuntimeError(f"Expected one VitisAI NPU, found {len(devices)}")
        webgpu_device = devices[0]
        print(f"Using NPU device: {webgpu_device.device.metadata}", flush=True)

    def load_session(path, ep="cpu", tf32=True):
        options = ort.SessionOptions()
        options.intra_op_num_threads = args.threads
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.enable_mem_pattern = False  # Required by DirectML; same settings on CPU.
        options.enable_profiling = args.profile
        options.profile_file_prefix = str(results / (label + "-" + path.stem))
        providers = ["CPUExecutionProvider"]
        if args.provider == "dml":
            providers.insert(0, ("DmlExecutionProvider", {"device_id": 0}))
        expected = "DmlExecutionProvider" if args.provider == "dml" else "CPUExecutionProvider"
        if webgpu_device is not None:
            provider_options = ({"powerPreference": "low-power"} if args.provider == "webgpu"
                                else {"cache_dir": str(ROOT / ".cache/winml/vitisai-modern")})
            options.add_provider_for_devices([webgpu_device], provider_options)
            value = ort.InferenceSession(str(path), sess_options=options)
            expected = webgpu_device.ep_name
        else:
            value = ort.InferenceSession(str(path), sess_options=options, providers=providers)
        if expected not in value.get_providers():
            raise RuntimeError(f"Requested {expected} missing: {value.get_providers()}")
        value.disable_fallback()  # Do not retry a failed run on a different provider.
        if args.measure_forward:
            original_run = value.run
            def timed_run(*positional, **keywords):
                feed = positional[1] if len(positional) > 1 else keywords["input_feed"]
                started = time.perf_counter()
                result = original_run(*positional, **keywords)
                forward_records.append({"session": path.stem,
                    "ms": (time.perf_counter() - started) * 1000,
                    "sequence_length": int(feed["input_ids"].shape[-1]) if "input_ids" in feed else None})
                return result
            value.run = timed_run
        sessions[path.stem] = value
        return value

    d1.load_session = load_session
    start = time.perf_counter()
    model = d1.load(ROOT / "models/d1-2B-ONNX", args.precision, "cpu", set_size=args.set_size)
    output = {
        "provider": args.provider, "precision": args.precision,
        "ort_version": ort.__version__, "threads": args.threads,
        "model_revision": "2fd61486312020c30a1e3aa5aa677424672b2b37",
        "runtime_revision": runtime_revision,
        "load_seconds": time.perf_counter() - start,
        "profiling": args.profile, "warmup": args.warmup, "workloads": [],
        "set_size": args.set_size,
        "forward_timing_enabled": args.measure_forward,
        "decode_tokens": 0,
        "context": {"checkpoint_max_position_embeddings": 128000, "sliding_window": 256,
                    "runtime_input": "dynamic; no fixed context override", "kv_cache": False},
    }
    fixture = args.request or UPSTREAM / "samples/d1/ticket.json"
    request = json.loads(fixture.read_text())
    for name, state in [("ticket", request["state"]), ("long-ticket", (request["state"] + " ") * 32)]:
        print(f"Running {name} on {args.provider}", flush=True)
        start = time.perf_counter()
        forward_start = len(forward_records)
        answer = model.system_one(state, request["questions"])
        first_ms = (time.perf_counter() - start) * 1000
        first_forward_records = forward_records[forward_start:]
        timings = []
        timed_forwards = []
        if not args.profile:
            for _ in range(args.warmup):
                model.system_one(state, request["questions"])
            for _ in range(args.repeats):
                start = time.perf_counter()
                forward_start = len(forward_records)
                current = model.system_one(state, request["questions"])
                timings.append((time.perf_counter() - start) * 1000)
                timed_forwards.append(forward_records[forward_start:])
                if current["answers"] != answer["answers"]:
                    raise RuntimeError("Repeated deterministic request changed answers")
        row = {"name": name, "state": state, "questions": request["questions"],
               "fixture_sha256": hashlib.sha256(fixture.read_bytes()).hexdigest(),
               "first_ms": first_ms, "timings_ms": timings, "output": answer}
        if args.measure_forward:
            row["first_forward_records"] = first_forward_records
            row["forward_records_per_request"] = timed_forwards
            row["input_forward_definition"] = "sum of synchronous ORT.Run calls; includes transfers and CPU partitions"
            if timed_forwards:
                row["mean_input_forward_ms"] = statistics.mean(sum(r["ms"] for r in request_records)
                                                              for request_records in timed_forwards)
        # first_ms is the first call at this input shape, not a new process cold start.
        if timings:
            mean = statistics.mean(timings)
            row.update(mean_ms=mean, median_ms=statistics.median(timings),
                       p95_ms=sorted(timings)[max(0, int(len(timings) * .95 + .999999) - 1)],
                       states_per_second=1000 / mean,
                       questions_per_second=len(request["questions"]) * 1000 / mean)
        output["workloads"].append(row)
    if args.profile:
        output["profiles"] = {}
        for name, value in sessions.items():
            path = value.end_profiling()
            events = json.loads(pathlib.Path(path).read_text())
            counts = Counter((e.get("args", {}).get("provider"), e.get("args", {}).get("op_name"))
                             for e in events if e.get("cat") == "Node" and e.get("args", {}).get("provider"))
            output["profiles"][name] = {"path": path, "executed_nodes": [
                {"provider": p, "op": op, "events": n} for (p, op), n in counts.items()]}
    (results / f"{label}.json").write_text(json.dumps(output, indent=2))
    print(f"Saved results/{label}.json", flush=True)


if __name__ == "__main__":
    main()
