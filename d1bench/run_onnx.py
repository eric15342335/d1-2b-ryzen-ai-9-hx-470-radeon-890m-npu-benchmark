"""Time the official d1 ONNX host runtime (LiquidONNX); only session and provider selection change.

Run through `d1bench run`, which puts the pinned LiquidONNX checkout and the right ONNX Runtime on
PYTHONPATH.
"""

import json
import time
from collections import Counter
from functools import partial
from pathlib import Path
from typing import Annotated, Any

import onnxruntime as ort
import typer
from liquidonnx.d1 import session as d1
from loguru import logger

from d1bench.common import ROOT, WORKLOADS, git_head, load_request, measure, write_json

AMD_VENDOR_ID = 0x1002


def webgpu_device() -> Any:  # noqa: ANN401
    """Register the WebGPU plugin EP and return the single AMD (Radeon 890M) device."""
    # Only the WebGPU overlay has this package.
    import onnxruntime_ep_webgpu as webgpu  # noqa: PLC0415  # pylint: disable=import-outside-toplevel

    ort.register_execution_provider_library("webgpu_benchmark", webgpu.get_library_path())
    devices = [
        d for d in ort.get_ep_devices() if d.ep_name == webgpu.get_ep_name() and d.device.vendor_id == AMD_VENDOR_ID
    ]
    if len(devices) != 1:
        msg = f"expected one AMD WebGPU device, found {len(devices)}"
        raise RuntimeError(msg)
    logger.info("WebGPU device: {}", devices[0].device.metadata)
    return devices[0]


def executed_nodes(session: ort.InferenceSession) -> dict[str, Any]:
    """End profiling and count executed graph nodes per (provider, op): the placement evidence."""
    path = session.end_profiling()
    events = json.loads(Path(path).read_text(encoding="utf-8"))
    nodes = Counter(
        (e["args"]["provider"], e["args"]["op_name"])
        for e in events
        if e.get("cat") == "Node" and e.get("args", {}).get("provider")
    )
    return {"path": path, "executed_nodes": [{"provider": p, "op": op, "events": n} for (p, op), n in nodes.items()]}


def main(
    output: Path,
    provider: Annotated[str, typer.Option(help="cpu, webgpu or dml")],
    precision: str = "q4",
    warmup: int = 3,
    repeats: int = 20,
    threads: int = 4,
    profile: Annotated[bool, typer.Option(help="Collect an ORT node profile instead of timing")] = False,
) -> None:
    """Run both workloads and write raw per-request and per-forward timings to OUTPUT."""
    device = webgpu_device() if provider == "webgpu" else None
    forwards: list[dict[str, Any]] = []
    sessions: dict[str, ort.InferenceSession] = {}

    def load_session(path: Path, ep: str = "cpu", tf32: bool = True) -> ort.InferenceSession:  # noqa: ARG001  # pylint: disable=unused-argument
        """Open one graph on the requested provider, refuse CPU fallback, and time every run."""
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.enable_mem_pattern = False  # DirectML requires it; kept identical on every provider
        options.enable_profiling = profile
        options.profile_file_prefix = str(output.with_suffix("")) + "-" + path.stem
        if device is not None:
            options.add_provider_for_devices([device], {"powerPreference": "low-power"})
            session = ort.InferenceSession(str(path), sess_options=options)
            expected = device.ep_name
        else:
            expected = "DmlExecutionProvider" if provider == "dml" else "CPUExecutionProvider"
            providers: list[Any] = ["CPUExecutionProvider"]
            if provider == "dml":
                providers.insert(0, (expected, {"device_id": 0}))
            session = ort.InferenceSession(str(path), sess_options=options, providers=providers)
        if expected not in session.get_providers():
            msg = f"requested {expected} missing: {session.get_providers()}"
            raise RuntimeError(msg)
        session.disable_fallback()  # a failed run must not silently retry on CPU

        run = session.run

        def timed_run(output_names: Any, input_feed: dict[str, Any], *rest: Any, **kwargs: Any) -> Any:  # noqa: ANN401
            started = time.perf_counter()
            result = run(output_names, input_feed, *rest, **kwargs)
            ms = (time.perf_counter() - started) * 1000
            forwards.append({"session": path.stem, "ms": ms, "sequence_length": int(input_feed["input_ids"].shape[-1])})
            return result

        session.run = timed_run
        sessions[path.stem] = session
        return session

    d1.load_session = load_session
    started = time.perf_counter()
    model = d1.load(ROOT / "models/d1-2B-ONNX", precision, "cpu")
    result: dict[str, Any] = {
        "provider": provider,
        "precision": precision,
        "ort_version": ort.__version__,
        "threads": threads,
        "runtime_revision": git_head(ROOT / ".cache/liquidonnx"),
        "load_seconds": time.perf_counter() - started,
        "profiling": profile,
        "warmup": warmup,
        "workloads": [],
    }
    for name in WORKLOADS:
        request = load_request(name)
        logger.info("running {} on {}", name, provider)
        row = measure(
            partial(model.system_one, request["state"], request["questions"]),
            forwards,
            warmup=0 if profile else warmup,
            repeats=0 if profile else repeats,
        )
        row["first_forward_records"] = row.pop("first_forwards")
        row["forward_records_per_request"] = row.pop("forwards_per_request")
        result["workloads"].append({"name": name, "state": request["state"], "questions": request["questions"], **row})
    if profile:
        result["profiles"] = {name: executed_nodes(session) for name, session in sessions.items()}
    write_json(output, result)
    logger.success("saved {}", output)


if __name__ == "__main__":
    typer.run(main)
