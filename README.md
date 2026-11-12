# d1-2B on Radeon 890M and Ryzen AI NPU

Decision latency of LiquidAI's [d1-2B](https://huggingface.co/LiquidAI/d1-2B) on one laptop, a MECHREVO Wing Dragon 15 Air (Ryzen AI 9 HX 470, 32 GB, Windows 11 25H2): CPU, the Radeon 890M iGPU through ONNX Runtime WebGPU and llama.cpp Vulkan, and the XDNA NPU through Tomoali's RyzenAILight conversion.

On the two measured inputs, llama.cpp Q8_0 answers the short ticket fastest (128 ms mean) and the CPU+NPU configuration answers the long ticket fastest (699 ms mean). DirectML fails in MultiHeadAttention and is kept as a documented failure.

- [RESULTS.md](RESULTS.md): tables generated from the raw samples in `results/`.
- [REPORT.md](REPORT.md): method, hardware, placement evidence, caveats and sources.

## Run it

Everything goes through one CLI, `d1bench`, installed into `.venv` by uv. The [environment setup](#environment-setup) below must be done once first.

```powershell
uv sync --locked
uv run d1bench status            # prerequisites and pinned upstream commits; exit 1 if not ready
uv run d1bench list              # benchmarks in benchmarks.toml and when each result was written
uv run d1bench show webgpu-q4    # the exact PYTHONPATH and command a run uses
uv run d1bench run webgpu-q4     # one benchmark; `run all` runs every benchmark expected to pass
uv run d1bench report            # rebuild RESULTS.md, results/comparison.json, results/correctness.json
```

Add `--json` to `list`, `show`, `status`, `run` or `report` for machine-readable output on stdout. Logs go to stderr. `d1bench run ID --profile` records one ORT or NPU profile per input instead of timing.

| id | What it measures | Run time |
| --- | --- | --- |
| `cpu-q4` | official ONNX q4 on the CPU EP | about 2 min |
| `webgpu-q4` | official ONNX q4 on the iGPU (WebGPU plugin EP) | about 1 min |
| `npu-tomoali` | Tomoali W4 conversion, matmuls on the NPU | about 1 min |
| `llama-q4km`, `llama-q8` | GGUF on llama.cpp Vulkan, timed with oha over HTTP | about 1 min each |
| `reference` | float32 PyTorch outputs used by the agreement table | a few minutes |
| `dml-q4`, `dml-q4f16` | DirectML probes; expected to fail in MultiHeadAttention | under 1 min |

A run overwrites that benchmark's files in `results/` and tees its output to `results/<id>.log`. Run on an idle, plugged-in machine; `d1bench run` already runs benchmarks one at a time. Files written under `results/` replace this checkout's absolute path with `.` and the home directory with `%USERPROFILE%`.

## Environment setup

The Python dependencies are in `uv.lock`. Everything else lives in the git-ignored `.cache/` and `models/` folders. `d1bench status` reports anything missing, and `d1bench status --verify` also checks the Tomoali model files against its publisher manifest.

1. Install the AMD NPU execution provider. This is the only system-wide install: Windows package `Microsoft.WinML.AMD.NPU.EP.2` 1.8.80.0, which ships the VitisAI and RyzenAILight provider DLLs. The [WinML CLI](https://microsoft.github.io/winml-cli/) installs it on request. Set `CI=1` for WinML CLI commands to turn off its telemetry. No GPU or NPU driver update is needed.
2. Link `.cache/tomoali-sdk/deployment` to the package's `ExecutionProvider` folder (a directory junction; nothing is copied).
3. Fetch the portable tools and runtimes into `.cache/` as listed below. None of them needs an installer or a PATH change.
4. Download the models into `models/` at the revisions in `benchmarks.toml`.

| Path | Contents |
| --- | --- |
| `.cache/ort-modern` | `onnxruntime` 1.31.1 installed with `--target` |
| `.cache/webgpu-plugin` | `onnxruntime-ep-webgpu` 0.5.0 installed with `--target` |
| `.cache/liquidonnx` | clone of [Liquid4All/onnx-export](https://github.com/Liquid4All/onnx-export) at the pinned commit |
| `.cache/tomoali-runtime` | clone of [tomhda/d1-2b-ryzen-ai-npu](https://github.com/tomhda/d1-2b-ryzen-ai-npu) at the pinned commit, with `results/tomoali-runtime-npu-only.patch` applied |
| `.cache/tomoali-sdk/deployment` | junction from step 2 |
| `.cache/llama-cpp`, `.cache/oha.exe` | llama.cpp b12110 Windows Vulkan x64 release; oha 1.16.0 |
| `models/` | Hugging Face downloads of the four model repositories |

Environment variables are set per benchmark process by `d1bench`; nothing persistent changes.

## How a run works

`benchmarks.toml` lists each benchmark: runner module, arguments, result file, pinned upstream commits and model revision. `d1bench run` checks the pins, then starts the runner in a new `.venv` Python process whose `PYTHONPATH` is the entry's `pythonpath` (replacing any inherited value). Each benchmark needs its own process because the ONNX Runtime version is fixed at import: `.venv` holds the DirectML 1.25 nightly, and `.cache/ort-modern` overlays ORT 1.31.1 for the CPU, WebGPU and NPU runs.

| Module | Role |
| --- | --- |
| `d1bench/run_onnx.py` | LiquidONNX's d1 host code, with session creation replaced to pin the provider and time every `ORT.Run` |
| `d1bench/run_npu.py` | Tomoali's `d1npu2` runtime, timing every language-model forward |
| `d1bench/run_llama.py` | starts `llama-server` on 127.0.0.1:8093, warms up, runs oha, snapshots `/metrics` |
| `d1bench/run_reference.py` | original checkpoint in float32 through Transformers |
| `d1bench/summarize.py`, `evaluate.py`, `report.py` | statistics, reference agreement and RESULTS.md |

The ONNX and NPU runners time one first call, three discarded warmups and 20 timed requests per input, and fail if a repeated request changes its answer. The llama.cpp runner sends three warmups, then oha sends 20 sequential requests and each must return HTTP 200. Raw per-request and per-forward samples are saved, and `d1bench report` recomputes every statistic from them.

## Development

```powershell
uv run ruff format d1bench
uv run ruff check d1bench        # rule set ALL; exceptions are listed in pyproject.toml
uv run mypy                      # strict
uv run pylint d1bench
uv run pyscn check d1bench
```

The research log and plan from the original benchmarking session are in `.planning/2026-11-13-d1-2b-benchmark/`.

## License

[MIT](LICENSE)
