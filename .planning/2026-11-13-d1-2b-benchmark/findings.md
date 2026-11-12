# Findings

Research notes from the benchmark session. External content quoted here is data, not instructions.

## Starting point

- The workspace started empty. Bun 1.4.4 and uv 0.13.5 were installed.
- d1-2B is a decision model: [LiquidAI/d1-2B](https://huggingface.co/LiquidAI/d1-2B). Measure decision latency and throughput. Generated tokens per second does not apply.
- GPT-6 prompting guidance came from <https://developers.openai.com/api/docs/guides/latest-model/gpt-6-astra.md>. The goal stated deliverables, verification and termination criteria.
- Skills came from OthmanAdi/planning-with-files, JuliusBrussee/caveman, DietrichGebert/ponytail, ayghri/i-have-adhd and blader/humanizer, installed with the [skills CLI](https://github.com/vercel-labs/skills) through bunx, project scope.
- BrowserOS MCP was unavailable, so research used web search tools.
- The model card requires custom LFM3 model code and Transformers >= 5.19. The generic Hugging Face deployment snippets above the card are not a verified decision-model integration.

## Hardware

- AMD Ryzen AI 9 HX 470.
- Radeon 890M (PCI 1002:150E), GPU driver 32.0.31041.1004 (2026-08-17).
- AMD NPU (PCI 1022:17F0), driver 32.0.20102.3930 (2026-05-07).
- An RTX 5060 Laptop GPU is present and excluded from the comparison.
- Windows build 26200.9974, 25H2. The registry ProductName says Windows 10 Pro, which does not identify the Windows generation. The first RAM query was denied; physical memory was later read as 33,446,285,312 bytes.

## Reuse-first runtime research

1. Publisher checkpoint: <https://huggingface.co/LiquidAI/d1-2B>. The LFM3 hybrid has short convolutions and attention, so generic transformer support does not prove model support. Typed question outputs and probabilities must be preserved.
2. Publisher ONNX artifacts: <https://huggingface.co/LiquidAI/d1-2B-ONNX>. It has q4 and q8 CPU bundles and fp16, q4f16 and q8f16 GPU bundles, so no export is needed. The card's GPU guidance covers CUDA only. The graph has decision-specific inputs and outputs and no token-generation loop.
3. Exporter and runtime: <https://github.com/Liquid4All/onnx-export>. It supplies `d1-export`, `d1-infer`, sample requests, correctness tests and repeat timing. In [infer.py](https://github.com/Liquid4All/onnx-export/blob/main/src/liquidonnx/d1/infer.py) the CLI offers `--cpu` and automatic CUDA, with no DirectML or VitisAI flag. Running it unchanged cannot measure the AMD devices; that is an integration gap and says nothing about whether the model runs on AMD.
4. Dependencies: [pyproject.toml](https://github.com/Liquid4All/onnx-export/blob/main/pyproject.toml) needs Python >= 3.12, and uv uses an explicit ORT nightly index and override. The official lock does not validate Windows.
5. AMD platform: [release notes](https://ryzenai.docs.amd.com/en/latest/relnotes.html) and [xrt-smi](https://ryzenai.docs.amd.com/en/main/xrt_smi.html). xrt-smi reports GorgonPoint1, and Ryzen AI 1.9 documents Gorgon Point support. No d1-specific certification exists.
6. Windows ML prerequisites ([installation](https://ryzenai.docs.amd.com/projects/WinML/en/latest/installation.html)): Windows build >= 26100, Python examples for 3.10 to 3.12, the App SDK, and NPU driver >= 32.0.203.280. Driver components alone are not the provider toolkit. Compare the installed 32.0.20102.3930 by AMD's release lineage, since dotted version order may not imply support.
7. DirectML ([docs](https://onnxruntime.ai/docs/execution-providers/DirectML-ExecutionProvider.html)) is the iGPU candidate. Check operator coverage and select the adapter explicitly, because device 0 could be another GPU. Provider availability does not prove placement.
8. NPU partitioning ([model support](https://ryzenai.docs.amd.com/projects/WinML/en/latest/model_support.html), [deployment](https://ryzenai.docs.amd.com/projects/WinML/en/latest/model_deployment.html)): validate the d1 graph before quantizing or repacking. Unsupported nodes may fall back to CPU.
9. Placement tools: [AI Analyzer](https://ryzenai.docs.amd.com/en/latest/ai_analyzer.html) and [ORT profiling](https://onnxruntime.ai/docs/performance/tune-performance/profiling-tools.html). Keep partition and profile evidence. AMD OGA `model_benchmark.exe` expects preoptimized bundles and cannot benchmark d1 decisions directly.

## Alternatives assessed

| Tool/path | Decision |
| --- | --- |
| LiquidONNX and official ONNX bundle | Preferred model-specific starting point; reuse host preprocessing and decision logic |
| ONNX Runtime DirectML | Candidate iGPU provider; compatibility and adapter selection still need execution checks |
| AMD Windows ML / VitisAI | Candidate NPU provider; SDK/provider setup and graph partition checks remain |
| ORT GenAI | Model support evolving; verify decision API and modality support before adoption |
| llama.cpp / Ollama / Vulkan | No verified exact d1-2B typed-decision parity found; similarly named d1 tags may refer to a different checkpoint |
| OpenVINO | No ready official d1 path found; AMD NPU is not an Intel OpenVINO NPU |
| Custom exporter, quantizer, timing harness | Avoid: upstream artifacts, Quark/vendor tooling and existing d1 repeat timing come first |

The llama.cpp row was later corrected; see "User-added runtime checks".

## Measurement acceptance criteria

- Pin model and tool revisions, precision, tokenizer, preprocessor and input fixture. Record OS, drivers, power mode and adapter.
- Separate loading, compilation, first inference and warm latency. Use at least 20 warm repetitions on fixed workloads and keep raw timings. `d1-infer` reports only a mean, so p50 and p95 cannot come from it.
- Record end-to-end latency, decisions per second, question count and input token count.
- Check labels and probability differences against the reference before comparing precisions.
- Require placement or profile evidence for every GPU or NPU claim, and report hybrid CPU work.

## Runtime probes

- LiquidONNX checkout 26857c2e4d12381c6c630e7f7daf00bb3589f721: its `SESSION_PROVIDERS` map has only cpu and cuda. AMD needs explicit ORT session selection around the upstream decision code.
- Official ONNX revision 2fd61486312020c30a1e3aa5aa677424672b2b37. The q4f16 graph is IR 10, ONNX opset 21, Microsoft contrib opset 1, with 154 MatMulNBits, 5 GroupQueryAttention, 6 MultiHeadAttention and 1 GatherBlockQuantized.
- onnxruntime-directml 1.24.4 lists DmlExecutionProvider and CPUExecutionProvider. That shows the package is present, not that the device runs the graph.
- HF_HOME points to a workspace cache after Xet tried a restricted global log directory.
- CPU smoke test with the upstream host code: greeting probability 0.983845, zero output tokens.
- A strict no-fallback DirectML session on the q4 graph fails because some nodes are assigned to the CPU EP. That is a partition limit, not a schema rejection.
- D3d12info 3.21.0: DXGI adapter 0 is the Radeon 890M, adapter 1 the RTX 5060, adapter 2 the Microsoft Basic Render Driver (`results/dxgi-adapters.json`).
- AMD's sample names the package `onnxruntime-winml`, which its feed does not have. Microsoft's docs use `onnxruntime-windowsml`, which [winml-cli](https://microsoft.github.io/winml-cli/) resolves.
- The uv Python 3.12 download worked, but the minor-version link was denied, so `.venv-winml` was created from the downloaded interpreter. The root `.venv` stays on 3.13.
- The CPU Torch index first captured unrelated packages; it is now restricted to torch and torchvision, and the WinML environment uses `--no-config`.
- Preliminary CPU q4 baseline (four questions, ORT 1.24.4, background downloads running, so not a controlled result), 20 repeats after 3 warmups. Short ticket, 189 input tokens: mean 1781.89 ms, median 1754.23 ms, p95 2024.37 ms. Long ticket, 655 tokens: mean 4944.75 ms, median 4237.96 ms, p95 7584.75 ms. Outputs matched the expected refund, billing, billing+delivery and medium urgency (`results/cpu-q4.json`).
- DirectML 1.24.4 and nightly 1.25.0.dev20260125001 both fail on the first decision at `/model/layers.2/attn/MultiHeadAttention`, HRESULT 0x80070057. The root lock pins that nightly.
- The failed nightly's profile shows DirectML running MatMulNBits, Conv and RotaryEmbedding, and CPU running GatherBlockQuantized and ScatterND, before the failure (`results/dml-partial-placement.json`). That is partial Radeon execution only.
- The GPU-oriented q4f16 and fp16-vision bundle was downloaded so the GPU verdict does not rest on the CPU-oriented q4 bundle.
- Get-AppxPackage was denied in the sandbox, so App SDK presence is unverified. The CPU name came from the `ProcessorNameString` registry value.
- DirectML source audit: the MultiHeadAttention operator does not broadcast the attention bias head axis. The d1 bias is `[batch_size,1,sequence_length,...]` while `num_heads=16`. Both rel-1.24.4 and main carry a TODO for this broadcast ([source](https://github.com/microsoft/onnxruntime/blob/rel-1.24.4/onnxruntime/core/providers/dml/DmlExecutionProvider/src/Operators/DmlOperatorMultiHeadAttention.cpp)).
- Native WebGPU ships as a plugin EP ([docs](https://onnxruntime.ai/docs/execution-providers/WebGPU-ExecutionProvider.html)). onnxruntime 1.31.1 went into `.cache/ort-modern` and onnxruntime-ep-webgpu 0.5.0 into `.cache/webgpu-plugin`, apart from the DirectML module. Device discovery found the 890M: vendor 4098, Discrete=0, DXGI 0.
- The WinML CLI 0.4.0 install hit access denied while renaming the Torch cache in the sandbox and exited 1; an approved unsandboxed retry worked.
- Power: AC, battery 100%, scheme Balanced, NPU Default (`results/power-status.json`).
- WebGPU (ORT 1.31.1, plugin 0.5.0) completed both q4 workloads. Its profile puts matrix multiplications, convolutions, MHA and sliding-window GQA on WebGpuExecutionProvider on the 890M, with control and index work on CPU: hybrid execution.
- All 8 categorical, multi-label and ordinal decisions across both workloads matched the preliminary CPU run. That CPU run used ORT 1.24.4, so a same-ORT CPU baseline (1.31.1) was run before final WebGPU timing.
- WinML CLI 0.4.0, WindowsML 2.0.300, onnxruntime-windowsml 1.27.1.202607110137 and Transformers 4.57.6 installed in their own environment. `winml sys` sees the NPU and both GPUs.
- The NPU provider Microsoft.WinML.AMD.NPU.EP.2 1.8.80.0 installed automatically. ORT 1.31.1 registers it and enumerates the NPU once its dependency directory is added. WinML CLI 1.27 compilation and a direct 1.31 InferenceSession both fail with repeated VAIP Invalid NodeIndex and native 0xC0000005. Static FP32 analysis has no VitisAI rules, so its zero unsupported count is inconclusive.
- Preliminary WebGPU q4 timing: short mean 301.22 ms, median 300.22 ms, p95 308.33 ms; long mean 971.86 ms, median 970.45 ms, p95 993.85 ms. All 20 repeated outputs per workload were identical (`results/webgpu-q4.json`).

## User-added runtime checks

- vLLM: native Windows binary-only resolution finds no win_amd64 wheels. The model architecture is `D1Model`, which the registry at a7933519764efd3b51782ebd7df4362c527e4a4c does not register (`results/vllm-source-audit.json`). Generation or pooling fallbacks would skip the custom `system_one` readout. The [structured-decisions docs](https://docs.vllm.ai/en/v0.32.0/serving/online_serving/structured_decisions/) cover d1-3B, a different checkpoint.
- llama.cpp, correcting the table above: the official [d1-2B-GGUF](https://huggingface.co/LiquidAI/d1-2B-GGUF) supports llama.cpp b11960 / v0.8.0 and later through `POST /v1/systemone`. It handles noul, single choice and score; `multiple:true` and extraction, estimate and boxes are unsupported.
- GGUF revision 4a181094d1b6addfaa3a0956f78e7ab2aa406a0c; the Q4_K_M file is 1,099,629,888 bytes. Stable llama.cpp v0.9.0 resolves through `nightly-tag.txt` to b12110 (Windows Vulkan x64). `--list-devices` shows Vulkan0 = 890M and Vulkan1 = RTX 5060. Runs use Vulkan0 with all layers offloaded.
- `fixtures/ticket-common.json` keeps the original state and criteria with the three question types both ONNX and llama.cpp support. The original four-question ONNX run has a multi-select question and is reported separately.
- llama.cpp Q4_K_M: 20 sequential HTTP requests per workload after 3 warmups, all 40 HTTP 200. Means 126.85 ms short and 860.54 ms long, with raw latencies in the oha SQLite files. Logs show 25/25 layers on Vulkan0 and `n_tokens=0` cached, so no prompt-cache effect. Long-input routing differs from packed ONNX and probability differences are material, so equal speed says nothing about equal answers.
- With `set_size=1`, upstream ONNX matches llama's token counts (161 / 1559), yet long routing and urgency still differ. Quantization and packing both move decisions, which is why the float32 reference was added.

## NPU conversion and environment audit

- Tomoali/d1-2B-ryzen-ai-npu at 04c583fcf56ecfaa6d387d03aa8330c896b27f7c with the author runtime tomhda/d1-2b-ryzen-ai-npu at e94cf1b50db06eac3aa388c7e55f7610bb27c446. Framework-free; text plus one 640x384 image; noul, choice and score; 4,096 tokens and 12 questions per packed forward. It uses RyzenAILight language projections, not the generic VitisAI graph that crashed.
- The installed AMD EP bundle already contains `onnxruntime_providers_ryzenai.dll`. ORT 1.31.1 registers it and enumerates both the NPU and the 890M. A one-line local runtime patch keeps only `OrtHardwareDeviceType.NPU` (`results/tomoali-runtime-npu-only.patch`). Acceptance required 143 MatMulNBitsBf events per language call.
- `.cache/tomoali-sdk/deployment` is a workspace junction to the installed, read-only AMD runtime folder. No DLLs were copied. The author used SDK 1.8.0; this run pairs WinML EP 1.8.80 with ORT 1.31.
- AppX events between 17:12:30 and 17:15 UTC record only `Microsoft.WinML.AMD.NPU.EP.2_1.8.80.0_x64__8wekyb3d8bbwe`. Other Windows App Runtime packages are inventoried and not attributed to this work.
- WinML CLI created `%USERPROFILE%/.winml/telemetry/` with a zero-byte `winmlcli.cache.lock` at 01:13:14 local time. No consent config was written; later commands set `CI=1` to disable telemetry. No global uv or Bun upgrade, driver install, PATH change, WSL distribution or global Python package was installed.
- Global Bun skill cache entries predate this work (Nov 4) and were left unchanged.
- Reporting rule: give exact context, token and question counts, and separate end-to-end time from input-forward time. d1 has no autoregressive decode (`output_tokens=0`), so decode, TTFT and generation throughput do not apply. End-to-end latency is not prefill latency.
- Xet transfers stalled with unchanged counters and zero I/O while a small HTTP range probe worked. Only the owned transfer processes were stopped, then resumed with `HF_HUB_DISABLE_XET=1`. The Tomoali bundle finished in 77 s and the 3.836 GB reference in 118 s.
- Tomoali integrity: all 21 manifest files passed. The NPU-only profile on both text workloads (123 / 589 packed tokens) shows exactly 286 MatMulNBitsBf events (143 per forward), all on RyzenAILightExecutionProvider with `OrtHardwareDeviceType.NPU`. CPU runs attention, convolution, embedding, layer norm and the other listed ops (`results/tomoali-npu-profile.json`, `results/tomoali-npu-profiles/`).
- Tomoali's score is the mean level; upstream d1 defaults to the median. Keep the raw score and compare a median derived from the probabilities. A fractional mean is not a wrong label.
- Source parity: tokenizer, config, calibration and attention files match across the reference, official ONNX and Tomoali. `prompt.py` differs only in validation and media handling, which this text fixture does not reach (`results/checkpoint-source-parity.json`).
