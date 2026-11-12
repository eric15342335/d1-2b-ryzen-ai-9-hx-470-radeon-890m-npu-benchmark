# d1-2B benchmark on Radeon 890M and AMD NPU

On the two text fixtures, llama.cpp Q8_0 on the iGPU had the lowest short-request latency (128 ms mean) and Tomoali's CPU+NPU configuration had the lowest long-request latency (699 ms mean). Both accelerators completed real decision inference. The configurations differ in precision, graph outputs and question packing, so these are whole-runtime comparisons.

Measured on 2026-11-13 (Asia/Singapore). The NPU result uses the Tomoali conversion with the RyzenAILight provider. The generic VitisAI graph failed earlier and is listed under other attempted paths.

Regenerated tables: [RESULTS.md](RESULTS.md). Commands: [README.md](README.md).

## Hardware and execution

| Item | Recorded configuration |
| --- | --- |
| Laptop | MECHREVO Wing Dragon 15 Air (翼龙15 Air, 2026; sold as Phenom 15 Air outside China); WMI model "YILONG Series", BIOS N.1.12MRO01 |
| CPU | AMD Ryzen AI 9 HX 470 |
| iGPU | Radeon 890M; graphics driver 32.0.31041.1004 |
| NPU | GorgonPoint1, PCI device 17F0; driver 32.0.20102.3930 |
| Discrete GPU | NVIDIA GeForce RTX 5060 Laptop GPU; present but not benchmarked |
| Windows | Build 26200.9974, 25H2 |
| Memory | 32 GB installed; 33,446,285,312 bytes usable |
| Power | AC connected; Windows Balanced; NPU Default |
| Threads | Four ORT intra-op threads; llama.cpp four threads; one request at a time |

Device selection:

- The discrete RTX 5060 was excluded.
- WebGPU selected AMD vendor 1002, DXGI adapter 0.
- llama.cpp selected Vulkan0, separately identified as the Radeon 890M.
- The NPU filter selected only `OrtHardwareDeviceType.NPU`, vendor 1022, device 17F0.

Where each configuration runs its work:

- ONNX WebGPU: a separate trace places matrix multiplications, convolution and attention on `WebGpuExecutionProvider`. Control and index operations stay on CPU.
- llama.cpp: 25/25 layers are offloaded to Vulkan0. Tokenization, the server and readout run on CPU.
- Tomoali: a separate trace has 286 RyzenAILight MatMulNBitsBf events for two forwards, exactly 143 per forward. Attention, convolution, embeddings, layer normalization and the other listed operations run on CPU, so this is a CPU+NPU hybrid.

## Latency and throughput

Each request asks three questions: refund yes/no, routing choice and ordinal urgency.

- Native runs: the first call is excluded, then three more warmups, then 20 sequential timed requests per input.
- llama.cpp: three excluded warmup requests, then oha sends 20 requests at concurrency 1.
- Profiling was off during timing and collected in separate processes.

Times are in milliseconds. P95 is nearest rank over the raw samples (`ceil(0.95 × 20)`, the 19th ordered sample). Req/s is `1000 / mean_ms`; multiply by three for questions per second.

| Configuration | Input | E2E mean | Median | P95 | Input-forward mean | Req/s |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| CPU ONNX q4 | Short | 614.48 | 603.43 | 668.26 | 612.66 | 1.627 |
| CPU ONNX q4 | Long | 3,208.81 | 2,789.96 | 4,991.24 | 3,206.24 | 0.312 |
| iGPU WebGPU ONNX q4 | Short | 199.23 | 199.01 | 209.97 | 197.28 | 5.019 |
| iGPU WebGPU ONNX q4 | Long | 872.88 | 873.93 | 887.29 | 870.27 | 1.146 |
| CPU+NPU Tomoali W4 | Short | 154.47 | 154.83 | 158.62 | 152.52 | 6.474 |
| CPU+NPU Tomoali W4 | Long | 699.49 | 699.85 | 708.54 | 697.10 | 1.430 |
| iGPU llama.cpp Q4_K_M | Short | 130.51 | 130.45 | 135.22 | 128.86 | 7.662 |
| iGPU llama.cpp Q4_K_M | Long | 916.60 | 911.36 | 930.19 | 914.31 | 1.091 |
| iGPU llama.cpp Q8_0 | Short | 128.45 | 128.99 | 130.89 | 126.70 | 7.785 |
| iGPU llama.cpp Q8_0 | Long | 923.72 | 919.56 | 933.64 | 921.56 | 1.083 |

What the columns include:

- E2E covers tokenization, packing, model execution and typed readout. For llama.cpp it also covers loopback HTTP and client overhead.
- Input-forward for ONNX and Tomoali is synchronous `ORT.Run` wall time, including transfers and CPU partitions.
- Input-forward for llama.cpp is the before/after `prompt_seconds_total` counter delta divided by 20, including the server's prompt processing and synchronization.
- Neither is a kernel-only accelerator timing.

Input-forward throughput in tokens/s, short / long:

- CPU: 201 / 184
- WebGPU: 623 / 677
- CPU+NPU: 806 / 845
- llama.cpp Q4_K_M: 1,249 / 1,705; Q8_0: 1,271 / 1,692

Packing gives these token rates different amounts of work per request, so request latency is the more direct comparison.

d1 generates no response tokens. Every captured output reports zero generated tokens, and the llama.cpp counter deltas show zero generated and zero cached prompt tokens. There is no decode phase, so generated tokens/s and TTFT do not apply. llama.cpp's `n_decode_total` counts calls to its input-processing function, not generated tokens.

## Context and input layout

The checkpoint configuration allows 128,000 positions with a 256-token sliding attention window. Runtime limits and the tested sequence lengths are separate from that.

| Path | Configured context | Actual short input | Actual long input |
| --- | --- | --- | --- |
| ONNX CPU / WebGPU | Dynamic input; no fixed context override; checkpoint setting 128,000 | One packed forward: 123 tokens | One packed forward: 589 tokens |
| Tomoali CPU+NPU | Hard runtime limit 4,096; up to 12 questions/forward | One packed forward: 123 tokens | One packed forward: 589 tokens |
| llama.cpp Vulkan | 2,048 tokens/slot; one slot; batch 2,048; microbatch 2,048 | Three forwards: 43, 64, 54 tokens; total 161 | Three forwards: 509, 530, 520 tokens; total 1,559 |

- The short state has 74 characters. The long state is `(short_state + " ") × 32`, 2,400 characters.
- Both use the same three question texts and criteria.
- No images, video, padding or prior conversation were used.
- These tests do not measure maximum-context capacity or performance. The publisher's WebGPU q4 guidance is also narrower than the checkpoint's context setting.

## Output correctness

The reference is the original `LiquidAI/d1-2B` weights loaded with Transformers on CPU in float32, run with default packed questions and with `set_size=1`. Tokenizer, configuration, calibration and attention-source hashes agree with the tested exports. The prompt-source differences concern input validation and media handling, which these text fixtures do not reach.

Each implementation is compared with the reference at matching packing. Packing alone changes the long input's reference routing and urgency, so a single packed reference would misjudge llama.cpp.

| Configuration | Label agreement | Largest absolute probability difference |
| --- | ---: | ---: |
| ONNX CPU q4 vs packed float32 | 6/6 | 9.290 percentage points |
| ONNX WebGPU q4 vs packed float32 | 6/6 | 9.249 percentage points |
| Tomoali W4 vs packed float32 | 6/6* | 16.550 percentage points |
| llama.cpp Q4_K_M vs separate float32 | 5/6 | 8.751 percentage points |
| llama.cpp Q8_0 vs separate float32 | 5/6 | 0.684 percentage points |

\* Tomoali returns the mean ordinal level, while the original API defaults to the median level. Its raw fractional scores are kept. The label comparison derives a median from Tomoali's returned distribution instead of comparing a mean with a median.

- Q8_0's one label difference is long-input urgency. The reference gives level 0 probability 0.502579 and Q8 gives 0.497980, which crosses the median boundary.
- Q8's probabilities are much closer to the reference than the 4-bit configurations, despite the same 5/6 count.
- CPU and WebGPU with the same ONNX weights agree on all six labels, with a maximum probability difference of 0.794 percentage points.

This is a six-decision numerical check, not a task-level accuracy or calibration benchmark. The probability drift matters for applications that use probability thresholds. All timed native Python outputs were identical across repeats. llama.cpp validation uses the captured typed responses and 20 successful HTTP samples per case.

## Artifacts and runtime versions

| Component | Revision / precision |
| --- | --- |
| Original reference | `LiquidAI/d1-2B` @ `652f920e544ea37f81e4b0ce89b81d4ed0da90b1`; loaded float32 |
| Official ONNX | `LiquidAI/d1-2B-ONNX` @ `2fd61486312020c30a1e3aa5aa677424672b2b37`; q4, FP32 activations, mixed INT4/INT8 weights |
| ONNX host runtime | LiquidONNX @ `26857c2e4d12381c6c630e7f7daf00bb3589f721` |
| Native WebGPU | ORT 1.31.1, `onnxruntime-ep-webgpu` 0.5.0 |
| NPU conversion | `Tomoali/d1-2B-ryzen-ai-npu` @ `04c583fcf56ecfaa6d387d03aa8330c896b27f7c`; W4 dense projections, W8 embeddings, BF16 projection execution, FP32 CPU work |
| NPU host runtime | `tomhda/d1-2b-ryzen-ai-npu` @ `e94cf1b50db06eac3aa388c7e55f7610bb27c446`; local NPU-only device-filter patch preserved |
| Tested NPU runtime pairing | ORT 1.31.1 + AMD WinML NPU EP 1.8.80.0; differs from author's documented SDK 1.8.0 setup |
| GGUF | `LiquidAI/d1-2B-GGUF` @ `4a181094d1b6addfaa3a0956f78e7ab2aa406a0c`; Q4_K_M and Q8_0 |
| llama.cpp | b12110, commit `cd959bae99ff4bed22bef2604f45241c09bbb8c8`; official Windows Vulkan x64 binary |
| HTTP benchmark | oha 1.16.0; 20 requests, concurrency 1; raw SQLite samples |
| Reference libraries | Torch 2.15.0+cpu, Transformers 5.22.2, Python 3.13.15 |

Publisher SHA256 checks passed for all 21 Tomoali manifest files. Reference and GGUF weight hashes match the upstream LFS hashes, and the selected official ONNX artifacts were also verified. The integrity JSON files are in `results/`.

## Other attempted paths

- DirectML: official q4 and GPU-oriented q4f16 both fail at `/model/layers.2/attn/MultiHeadAttention`, HRESULT 80070057, in stable ORT 1.24.4 and the available 1.25 nightly. The graph's attention bias has a single head, and that implementation requires an explicit head axis. No DirectML latency is reported.
- Generic VitisAI ONNX: the AMD EP installed and recognized the NPU. Initializing and compiling the unadapted official q4 graph emitted `Invalid NodeIndex` errors and crashed with 0xC0000005 in both tested core paths. The working Tomoali route uses a dedicated conversion and a different language provider.
- vLLM: native Windows binary resolution fails. The upstream model registry at `a7933519764efd3b51782ebd7df4362c527e4a4c` does not register this checkpoint's `D1Model` decision wrapper. Its documented System One support is for d1-3B, and generic LFM3 generation or pooling skips the custom d1-2B decision readout. No faithful vLLM d1-2B measurement exists. WSL2 Debian is present, but a Linux engine would not add the missing model integration.

## Reproducibility

- Commands are in [README.md](README.md). Setup steps, including the one Windows package this benchmark installs (the AMD WinML NPU execution provider), are in the "Environment setup" section of [README.md](README.md#environment-setup).
- `results/` holds the raw per-request samples, reference outputs, context layout, server properties, prompt counters and placement traces.
- `d1bench report` rebuilds the table data and the numerical comparison into [RESULTS.md](RESULTS.md). SQLite summaries select the latest oha run and require HTTP 200 for every request.
- Profiling artifacts are separate from latency samples. Earlier diagnostic and overlap results keep separate names and are not in the tables.

## Limitations

- Final timing passes ran one at a time, with no other agent-controlled inference, download or hashing running.
- Background OS and application activity and thermal state were not controlled.
- Each configuration has one final timed process with 20 samples per input, not repeated independent processes.
- The CPU long-input tail varies a lot, so read its median and P95 alongside the mean.
- No maximum-context, image, video, energy or sustained-load claim is made.

## Primary sources

- [Original model](https://huggingface.co/LiquidAI/d1-2B), [official ONNX bundle](https://huggingface.co/LiquidAI/d1-2B-ONNX), [LiquidONNX runtime](https://github.com/Liquid4All/onnx-export).
- [Tomoali NPU conversion](https://huggingface.co/Tomoali/d1-2B-ryzen-ai-npu), [its runtime](https://github.com/tomhda/d1-2b-ryzen-ai-npu).
- [Official GGUF bundle and System One scope](https://huggingface.co/LiquidAI/d1-2B-GGUF), [pinned llama.cpp server](https://github.com/ggml-org/llama.cpp/tree/cd959bae99ff4bed22bef2604f45241c09bbb8c8/tools/server).
- [Native WebGPU provider](https://onnxruntime.ai/docs/execution-providers/WebGPU-ExecutionProvider.html), [DirectML attention implementation](https://github.com/microsoft/onnxruntime/blob/rel-1.24.4/onnxruntime/core/providers/dml/DmlExecutionProvider/src/Operators/DmlOperatorMultiHeadAttention.cpp).
- [vLLM structured decisions](https://docs.vllm.ai/en/v0.32.0/serving/online_serving/structured_decisions/), [audited registry](https://github.com/vllm-project/vllm/blob/a7933519764efd3b51782ebd7df4362c527e4a4c/vllm/model_executor/models/registry.py).
