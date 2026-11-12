"""Benchmark the pinned Tomoali d1-2B runtime on the Ryzen AI NPU."""
import argparse
import hashlib
import json
import os
import pathlib
import statistics
import subprocess
import sys
import time
import traceback


ROOT = pathlib.Path(__file__).resolve().parent
RUNTIME = ROOT / ".cache/tomoali-runtime"
MODEL = ROOT / "models/d1-2b-ryzen-ai-npu"
FIXTURE = ROOT / "fixtures/ticket-common.json"
MODEL_INFO = ROOT / "results/tomoali-model-info.json"
EXPECTED_RUNTIME_REVISION = "e94cf1b50db06eac3aa388c7e55f7610bb27c446"
EXPECTED_MODEL_REVISION = "04c583fcf56ecfaa6d387d03aa8330c896b27f7c"
NATIVE_STARTED = False


def main():
    global NATIVE_STARTED
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=3)
    args = parser.parse_args()
    if args.repeats < 1 or args.warmup < 1:
        parser.error("repeats and warmup must be positive")

    model_info = json.loads(MODEL_INFO.read_text(encoding="utf-8"))
    if model_info["revision"] != EXPECTED_MODEL_REVISION:
        raise RuntimeError("Pinned Tomoali model revision does not match results/tomoali-model-info.json")
    runtime_revision = subprocess.check_output(
        ["git", "-C", str(RUNTIME), "rev-parse", "HEAD"], text=True
    ).strip()
    if runtime_revision != EXPECTED_RUNTIME_REVISION:
        raise RuntimeError("Tomoali runtime checkout differs from the reviewed revision")
    manifest_path = MODEL / "model-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for item in manifest["files"]:
        path = (MODEL / item["path"]).resolve()
        if not path.is_relative_to(MODEL.resolve()) or path.stat().st_size != item["bytes"]:
            raise RuntimeError(f"Model manifest path/size mismatch: {item['path']}")
        with path.open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != item["sha256"]:
                raise RuntimeError(f"Model checksum mismatch: {item['path']}")
    print(f"Verified {len(manifest['files'])} model files against publisher checksums", flush=True)

    sys.path.insert(0, str(RUNTIME))
    import d1npu2
    import onnxruntime as ort

    profile_dir = ROOT / "results/tomoali-npu-profiles" if args.profile else None
    print("Initializing Tomoali d1-2B NPU runtime...", flush=True)
    NATIVE_STARTED = True
    load_started = time.perf_counter()
    model = d1npu2.D1(
        MODEL,
        ryzen_ai_dir=ROOT / ".cache/tomoali-sdk",
        cache_dir=ROOT / ".cache/tomoali-npu-cache",
        threads=4,
        profile_dir=profile_dir,
    )
    load_seconds = time.perf_counter() - load_started

    devices = [
        device for device in ort.get_ep_devices()
        if device.ep_name == "RyzenAILightExecutionProvider"
        and device.device.type == ort.OrtHardwareDeviceType.NPU
    ]
    if len(devices) != 1:
        raise RuntimeError(f"Expected one RyzenAILight NPU device, found {len(devices)}")
    hardware = [
        {"provider": device.ep_name, "type": str(device.device.type),
         "vendor": device.device.vendor, "vendor_id": device.device.vendor_id,
         "device_id": device.device.device_id, "metadata": device.device.metadata}
        for device in devices
    ]

    fixture_bytes = FIXTURE.read_bytes()
    request = json.loads(fixture_bytes)
    result = {
        "backend": "tomoali-ryzen-ai-npu",
        "model_id": model_info["id"],
        "model_revision": model_info["revision"],
        "runtime_revision": runtime_revision,
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "runtime_version": d1npu2.__version__,
        "onnxruntime_version": ort.__version__,
        "hardware": hardware,
        "threads": 4,
        "context_length": 4096,
        "question_count": len(request["questions"]),
        "fixture": str(FIXTURE),
        "fixture_sha256": __import__("hashlib").sha256(fixture_bytes).hexdigest(),
        "load_seconds": load_seconds,
        "profile": args.profile,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "workloads": [],
    }

    original_run = model.language.run
    forward_times = []

    def timed_run(*call_args, **call_kwargs):
        started = time.perf_counter()
        try:
            return original_run(*call_args, **call_kwargs)
        finally:
            forward_times.append((time.perf_counter() - started) * 1000)

    model.language.run = timed_run
    states = (("short", request["state"]), ("state-times-32", (request["state"] + " ") * 32))
    for name, state in states:
        print(f"Running {name} on Ryzen AI NPU...", flush=True)
        forward_times.clear()
        started = time.perf_counter()
        answer = model.decide(state, request["questions"])
        first_ms = (time.perf_counter() - started) * 1000
        first_forward_ms = list(forward_times)
        row = {
            "name": name,
            "state": state,
            "first_ms": first_ms,
            "first_forward_ms": first_forward_ms,
            "packed_tokens": model.last_timing["packed_tokens"],
            "output": answer,
        }
        if not args.profile:
            for _ in range(args.warmup):
                model.decide(state, request["questions"])
            timings_ms, forward_timings_ms = [], []
            for _ in range(args.repeats):
                forward_times.clear()
                started = time.perf_counter()
                current = model.decide(state, request["questions"])
                timings_ms.append((time.perf_counter() - started) * 1000)
                forward_timings_ms.append(list(forward_times))
                if current != answer:
                    raise RuntimeError(f"Repeated {name} decision changed its answer")
            mean_ms = statistics.mean(timings_ms)
            row.update(
                timings_ms=timings_ms,
                forward_timings_ms=forward_timings_ms,
                mean_ms=mean_ms,
                median_ms=statistics.median(timings_ms),
                p95_ms=sorted(timings_ms)[max(0, int(len(timings_ms) * 0.95 + 0.999999) - 1)],
                states_per_second=1000 / mean_ms,
                questions_per_second=len(request["questions"]) * 1000 / mean_ms,
            )
        result["workloads"].append(row)

    if args.profile:
        result["profiles"] = model.profiles()
    output = ROOT / ("results/tomoali-npu-profile.json" if args.profile else "results/tomoali-npu.json")
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                      encoding="utf-8")
    print(f"Saved {output.relative_to(ROOT)}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        traceback.print_exc(file=sys.stderr)
        sys.stdout.flush()
        sys.stderr.flush()
        if NATIVE_STARTED:
            os._exit(1)
        raise
    sys.stdout.flush()
    sys.stderr.flush()
    if NATIVE_STARTED:
        os._exit(0)
