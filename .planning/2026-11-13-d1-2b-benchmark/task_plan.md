# Task plan: d1-2B laptop benchmark

## Goal

Benchmark LiquidAI/d1-2B on this laptop's iGPU and NPU with existing tools, reproducible measurements, verified execution placement and named compatibility blockers.

## Next Step

None. All phases are complete. New work starts a new named plan under `.planning/`.

## Current Phase

Phase 6 (complete)

## Phases

### Phase 1: Workspace setup

- [x] Install five project-local skills with Bun
- [x] Initialize uv and the lock file
- **Status:** complete

### Phase 2: Reuse-first research

- [x] Identify the model, hardware, published artifacts, CLIs and provider gaps
- **Status:** complete

### Phase 3: Setup verification and research handoff

- [x] Verify skills and environment
- [x] Record evidence and the next benchmark action
- **Status:** complete

### Phase 4: Device benchmarks

- [x] CPU and WebGPU ONNX timings with placement profile
- [x] DirectML and generic VitisAI blockers recorded
- [x] llama.cpp Vulkan Q4_K_M and Q8_0 timings
- [x] Tomoali NPU conversion with NPU-only placement proof
- [x] vLLM checked: no d1-2B decision path on this machine
- [x] Float32 reference agreement, input layout and context metadata
- **Status:** complete

### Phase 5: CLI and cleanup

- [x] Replace the five scripts and the PowerShell driver with the `d1bench` package and `benchmarks.toml`
- [x] Move per-run model hashing to `d1bench status --verify`
- [x] ruff (ALL), mypy strict, pylint and pyscn pass
- [x] Regenerated statistics match the published ones exactly
- **Status:** complete

### Phase 6: Publish-ready repository

- [x] Redact machine paths from results and from future run outputs
- [x] Fold the environment audit into README setup steps
- [x] Move planning files to `.planning/2026-11-13-d1-2b-benchmark/`
- [x] Git history in small Conventional Commits, MIT license
- **Status:** complete

## Decisions Made

| Decision | Rationale |
|----------|-----------|
| Reuse LiquidONNX host code and patch only session creation | Keeps the published decision readout; no custom exporter |
| Report llama.cpp against the `set_size=1` reference | llama.cpp evaluates one question per forward, and packing changes the long input's answers |
| Run each benchmark in its own process | The ONNX Runtime version is fixed at import; DirectML needs 1.25, the rest 1.31.1 |

## Errors Encountered

| Error | Attempt | Resolution |
|-------|---------|------------|
| Bun temp directory returned EBADF in the sandbox | 1 to 3 | Workspace-local BUN_TMPDIR, TEMP and TMP did not help; an approved unsandboxed run worked |
| HF Xet logged to a global cache (access denied) | 1 | Workspace HF_HOME |
| `uv pip index` does not exist | 1 | Queried the package index directly |
| Python 3.12 minor-version link creation denied | 1 | Created `.venv-winml` from the downloaded interpreter |
| AMD docs name `onnxruntime-winml`, absent from its feed | 1 | WinML CLI resolves `onnxruntime-windowsml` |
| WinML CLI resolution inherited the Torch index | 1 | Scoped the Torch index; `uv --no-config` for that environment |
| DirectML q4 and q4f16 fail in MultiHeadAttention, 0x80070057, on 1.24.4 and the 1.25 nightly | 2 | Recorded as a blocker; source audit found missing bias head-axis broadcast |
| Get-AppxPackage enumeration denied | 1 | No App SDK absence inferred from it |
