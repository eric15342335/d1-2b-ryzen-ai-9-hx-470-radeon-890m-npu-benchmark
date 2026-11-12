"""Run small CPU System One reference checks against the pinned HF checkpoint."""
import hashlib
import json
import pathlib
import platform
import sys
import time

import torch
import transformers
from transformers import AutoModel


ROOT = pathlib.Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models/d1-2B-reference"
MODEL_INFO = ROOT / "results/reference-model-info.json"
FIXTURE = ROOT / "fixtures/ticket-common.json"
OUTPUT = ROOT / "results/pytorch-reference.json"


def main():
    torch.set_num_threads(4)
    torch.set_num_interop_threads(4)
    model_info = json.loads(MODEL_INFO.read_text())
    fixture_bytes = FIXTURE.read_bytes()
    request = json.loads(fixture_bytes)

    print("Loading pinned reference model on CPU (float32, 4 threads)...", flush=True)
    started = time.perf_counter()
    model = AutoModel.from_pretrained(
        str(MODEL_DIR), trust_remote_code=True, dtype=torch.float32
    ).eval()
    load_seconds = time.perf_counter() - started

    result = {
        "model": str(MODEL_DIR),
        "model_revision": model_info["revision"],
        "fixture": str(FIXTURE),
        "fixture_sha256": hashlib.sha256(fixture_bytes).hexdigest(),
        "backend": "transformers-cpu",
        "dtype": "float32",
        "threads": torch.get_num_threads(),
        "load_seconds": load_seconds,
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "workloads": [],
    }
    for name, state in (("short", request["state"]),
                        ("state-times-32", (request["state"] + " ") * 32)):
        for set_size in (None, 1):
            print(f"Running {name}, set_size={set_size}...", flush=True)
            started = time.perf_counter()
            with torch.inference_mode():
                output = model.system_one(state, request["questions"], set_size=set_size)
            result["workloads"].append({
                "name": name,
                "state": state,
                "set_size": set_size,
                "elapsed_ms": (time.perf_counter() - started) * 1000,
                "output": output,
            })

    OUTPUT.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(f"Saved {OUTPUT.relative_to(ROOT)}", flush=True)


if __name__ == "__main__":
    main()
