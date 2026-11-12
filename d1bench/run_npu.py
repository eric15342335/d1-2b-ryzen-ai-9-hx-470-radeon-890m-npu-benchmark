"""Time the pinned Tomoali d1-2B runtime (d1npu2) on the Ryzen AI NPU.

Run through `d1bench run npu-tomoali`, which puts the runtime checkout and ORT 1.31.1 on PYTHONPATH.
Model checksums are checked by `d1bench status --verify`, not here, so hashing never precedes timing.
"""

import os
import sys
import time
import traceback
from functools import partial
from pathlib import Path
from typing import Annotated, Any

import d1npu2
import onnxruntime as ort
import typer
from loguru import logger

from d1bench.common import ROOT, WORKLOADS, git_head, load_request, measure, write_json


def main(
    output: Path,
    warmup: int = 3,
    repeats: int = 20,
    profile: Annotated[bool, typer.Option(help="Collect d1npu2 profiles instead of timing")] = False,
) -> None:
    """Run both workloads and write raw per-request and per-forward timings to OUTPUT."""
    logger.info("initializing Tomoali d1-2B NPU runtime")
    started = time.perf_counter()
    model = d1npu2.D1(
        ROOT / "models/d1-2b-ryzen-ai-npu",
        ryzen_ai_dir=ROOT / ".cache/tomoali-sdk",
        cache_dir=ROOT / ".cache/tomoali-npu-cache",
        threads=4,
        profile_dir=output.with_suffix("") if profile else None,
    )
    load_seconds = time.perf_counter() - started
    devices = [
        d
        for d in ort.get_ep_devices()
        if d.ep_name == "RyzenAILightExecutionProvider" and d.device.type == ort.OrtHardwareDeviceType.NPU
    ]
    if len(devices) != 1:
        msg = f"expected one RyzenAILight NPU device, found {len(devices)}"
        raise RuntimeError(msg)

    forwards: list[float] = []
    run = model.language.run

    def timed_run(*args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
        started = time.perf_counter()
        try:
            return run(*args, **kwargs)
        finally:
            forwards.append((time.perf_counter() - started) * 1000)

    model.language.run = timed_run
    result: dict[str, Any] = {
        "backend": "tomoali-ryzen-ai-npu",
        "runtime_revision": git_head(ROOT / ".cache/tomoali-runtime"),
        "runtime_version": d1npu2.__version__,
        "onnxruntime_version": ort.__version__,
        "hardware": [
            {
                "provider": d.ep_name,
                "type": str(d.device.type),
                "vendor": d.device.vendor,
                "vendor_id": d.device.vendor_id,
                "device_id": d.device.device_id,
                "metadata": d.device.metadata,
            }
            for d in devices
        ],
        "threads": 4,
        "context_length": 4096,
        "load_seconds": load_seconds,
        "profile": profile,
        "warmup": warmup,
        "repeats": repeats,
        "workloads": [],
    }
    for name in WORKLOADS:
        request = load_request(name)
        logger.info("running {} on Ryzen AI NPU", name)
        row = measure(
            partial(model.decide, request["state"], request["questions"]),
            forwards,
            warmup=0 if profile else warmup,
            repeats=0 if profile else repeats,
        )
        row["first_forward_ms"] = row.pop("first_forwards")
        row["forward_timings_ms"] = row.pop("forwards_per_request")
        row["packed_tokens"] = model.last_timing["packed_tokens"]
        result["workloads"].append({"name": name, "state": request["state"], **row})
    if profile:
        result["profiles"] = model.profiles()
    write_json(output, result)
    logger.success("saved {}", output)


if __name__ == "__main__":
    # The native NPU runtime can hang during interpreter teardown; upstream's CLI exits the same way.
    code = 0
    try:
        typer.run(main)
    except SystemExit as stop:
        code = stop.code if isinstance(stop.code, int) else 1
    except BaseException:  # noqa: BLE001  # pylint: disable=broad-exception-caught
        traceback.print_exc()
        code = 1
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)
