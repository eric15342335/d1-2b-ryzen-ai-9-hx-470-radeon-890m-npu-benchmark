"""Float32 PyTorch outputs of the original d1-2B checkpoint, the baseline for `d1bench report`.

Each workload runs with default question packing and with set_size=1 (one question per forward),
since packing changes the long input's answers and llama.cpp evaluates questions separately.
"""

import platform
import time
from pathlib import Path
from typing import Any

import torch
import transformers
import typer
from loguru import logger
from transformers import AutoModel

from d1bench.common import ROOT, WORKLOADS, load_request, write_json


def main(output: Path) -> None:
    """Run every workload at both packings and write the answers to OUTPUT."""
    torch.set_num_threads(4)
    torch.set_num_interop_threads(4)
    logger.info("loading reference model on CPU (float32, 4 threads)")
    started = time.perf_counter()
    model = AutoModel.from_pretrained(
        str(ROOT / "models/d1-2B-reference"),
        trust_remote_code=True,
        dtype=torch.float32,
    ).eval()
    result: dict[str, Any] = {
        "dtype": "float32",
        "threads": torch.get_num_threads(),
        "load_seconds": time.perf_counter() - started,
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "workloads": [],
    }
    for name in WORKLOADS:
        request = load_request(name)
        for set_size in (None, 1):
            logger.info("running {}, set_size={}", name, set_size)
            started = time.perf_counter()
            with torch.inference_mode():
                answer = model.system_one(request["state"], request["questions"], set_size=set_size)
            result["workloads"].append(
                {
                    "name": name,
                    "state": request["state"],
                    "set_size": set_size,
                    "elapsed_ms": (time.perf_counter() - started) * 1000,
                    "output": answer,
                },
            )
    write_json(output, result)
    logger.success("saved {}", output)


if __name__ == "__main__":
    typer.run(main)
