# Progress log

## Session: 2026-11-13, benchmarks

- Read the installer, planning-with-files, ponytail, caveman and OpenAI docs instructions.
- Initialized `pyproject.toml` and `.venv` with `uv init --bare --python 3.13 --vcs none` and `uv venv`.
- Installed five skills from their upstream repositories with `bunx --bun skills add` (project scope, copy mode). Bun needed an unsandboxed run because its temp-directory calls failed in the sandbox.
- A verification pass confirmed the five SKILL.md names and lockfile entries. `uv lock` and `uv sync --locked` passed on Python 3.13.15.
- Cloned LiquidONNX into `.cache/liquidonnx` at 26857c2e4d12381c6c630e7f7daf00bb3589f721 and read its d1 and session code before choosing an integration.
- Downloaded the official q4 decoder and q8 vision weights. The upstream CPU smoke inference passed.
- Installed CPU-only Torch, DirectML 1.24.4 and Transformers 5.22.2. Added minimal session-selection and timing glue (`benchmark.py`) around the upstream host code.
- Completed CPU and native WebGPU timings and a separate WebGPU placement profile. Core compute ran on the Radeon 890M; control and index ops stayed on CPU.
- The DirectML q4f16 probe first inherited the WebGPU PYTHONPATH; the provider check rejected it. Rerun with the DirectML environment: q4f16 also fails MultiHeadAttention with 80070057.
- WinML CLI installed AMD NPU EP 1.8.80.0. The generic VitisAI path emitted Invalid NodeIndex and crashed with 0xC0000005, under both the older core and ORT 1.31.1.
- vLLM: the native Windows binary-only resolver finds no wheels. WSL2 Debian exists but no d1-2B decision backend.
- Downloaded llama.cpp b12110 Windows Vulkan and the official Q4_K_M GGUF. Vulkan0 is the Radeon 890M. Timed with oha 1.16 against a loopback-only server on port 8093.
- Stalled Xet downloads: confirmed zero I/O on the owned processes, stopped them and resumed with `HF_HUB_DISABLE_XET=1`.
- Pinned and downloaded the Tomoali model and runtime. Registration works on the HX 470; a one-line local patch excludes GPU devices. Four upstream runtime tests passed (not hardware tests).
- Audited installs outside the workspace through AppX events: only the AMD WinML NPU EP 1.8.80.0 package.
- Added separate ORT.Run timing and the 128,000-position / 256-window metadata. All d1 output-token counts are zero.

## Session: 2026-11-13, CLI and publishing

- Replaced the scripts with the `d1bench` package (Typer, Rich, loguru) and `benchmarks.toml`. Old scripts kept in `.cache/legacy-scripts/`.
- `d1bench report` reproduces `comparison.json` and `correctness.json` exactly.
- Ran the WebGPU, NPU and llama Q8 runners on hardware (5 samples, scratch output). DirectML q4 fails as expected through the CLI.
- Redacted the checkout path and home directory from `results/`, and added `redact()` for future run outputs.
- Moved planning files here, folded `ENVIRONMENT_CHANGES.md` into README setup steps, initialized git.

## Test Results

| Test | Expected | Actual | Status |
|------|----------|--------|--------|
| `d1bench report` vs previous JSON | identical statistics | identical | pass |
| `ruff check` (ALL), `mypy --strict`, `pylint`, `pyscn` | clean | clean, pylint 10.00 | pass |
| `d1bench run dml-q4` | expected failure | MultiHeadAttention 80070057, exit 0 | pass |
