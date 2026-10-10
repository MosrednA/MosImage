# Official Qwen 2.1 Turbo quantization evaluation — 10 October 2026

Status: evaluation complete; **the user prefers FLUX overall and Qwen is not selected for integration**. This evaluates the official eight-step Turbo release, separate from the earlier six-step Viggle community model in the [9 October comparison](MODEL_COMPARISON_2026-10-09.md).

The target is a second main model beside FLUX.2 Klein 9B NVFP4 on the RTX 5070 Ti 16 GB / Ryzen 7 9700X / 32 GB Windows system. The registry and application runtime have no Qwen changes. Model downloads, optional GGUF dependencies and outputs stay under ignored `models/research-cache/` and `data/` directories.

## Sources and runtime

| Component | Pinned revision | File size |
|---|---|---|
| [Official Qwen 2.1 Turbo](https://huggingface.co/Qwen/Qwen-Image-2.1-Turbo) encoder, processor, VAE and configs | `d65dbc9a7e8f6b5479e33dee6030eaab2a906509` | BF16 encoder about 17.5 GB, VAE 0.676 GB |
| [BennyDaBall NVFP4](https://huggingface.co/BennyDaBall/Qwen-Image-2.1-Turbo-NVFP4) transformer | `cb5a4e68cb747ca8a753a9f15416c3aeea6c8b67` | 5.693 GB |
| Same release, mixed NVFP4 encoder | Same | 9.335 GB, including the unused language head |
| [Unsloth GGUF Q4_K_M](https://huggingface.co/unsloth/Qwen-Image-2.1-Turbo-GGUF) transformer | `5d1f9f759ea877ae556cda88dbec958b62ad3809` | 4.200 GB |

The Unsloth catalog also offers Q5_K_M (5.390 GB), Q6_K (6.272 GB) and Q8_0 (7.641 GB). Only Q4_K_M is GPU-tested here. File sizes do not describe complete GPU residency.

Packages: PyTorch 2.14.1+cu130, Diffusers 0.41.0, Transformers 5.19.0, Comfy Kitchen 0.2.37 and isolated `gguf==0.18.0`. No global/app dependency upgrades or remote kernel code are used. Model cards describe other hardware; their RTX 5090 timing is not used as this card's timing.

The official schedule is passed explicitly because the installed pipeline does not automatically apply the newer `sample_sigmas` checkpoint property: `1, .978453, .95418, .926626, .89508, .845148, .704534, .414568`, followed by the scheduler's terminal zero, Euler, CFG 1, eight steps. Processor and VAE files come from the official pinned release; the cached older model has different packaging/hashes and is not used for these Qwen components.

NVFP4 retains the publisher's sensitive BF16 matrices. Its 144 source quantized transformer matrices become 176 native linears after aligned gate/up splitting; the encoder has 188 native linears. This benchmark uses native CUDA NVFP4 GEMM for both components with dynamic activation scales; eager/Triton fallbacks are disabled. Three native/split probes match the dequantized-weight/quantized-activation reference at cosine >0.99999. This validates the adapter, not equality to original BF16 model outputs.

Diffusers' stock GGUF loader preserved four BF16 tensors as byte-shaped parameters, including `txt_in.text_norm.weight`, causing a 4096/8192 shape error before sampling. The benchmark reinterprets those original BF16 bytes at their declared shapes; quantized matrices still use Diffusers' standard dynamic GPU dequantization and BF16 matmul. The failed stock-loader attempts are retained separately. No installed Diffusers source or app code is patched.

## Protocol and results

Nine fixed cases, seeds **7 and 42**: two-person roller-coaster composition, DJ edit, phone-to-sword whole edit, mushroom replacement, brooch embroidery, framed art replacement, reference-cup insertion, exact Dutch poster text and native transparent cutout. All sources, masks, prompts, raw renders, composites and failed attempts are retained. Sources and masks come from the previous controlled cohort; the two generation prompts are fixed before rendering. The masked cases use its frozen crop/composite recipe, rather than claiming to reproduce all current editor defaults. Klein uses four distilled steps with the same 32-aligned output sizes.

`data/qwen-turbo-official-2026-10-10/` contains `cases.json`, asset/weight hashes, encoding reports, per-variant result JSON, complete PNGs, raw model outputs, comparison sheets and `index.html`. Gallery controls reveal memory/loader experiments and preserve browser-local review marks. All declared seeds remain visible.

Transformer comparisons use identical precomputed official BF16 embeddings and a tiled VAE, separating quantization from the encoder. A native NVFP4 encoder comparison and a separate full lifecycle test cover the proposed faster encoder recipe. Render-only time includes reference VAE conditioning, denoising, decoding, PNG saving and masked compositing. It excludes prompt encoding and model loading. First calls retain CUDA/kernel initialization effects; they are not silently discarded.

| Variant | Completed | Median render-only | Range, including first call | Peak allocated / reserved CUDA |
|---|---:|---:|---:|---:|
| Klein 9B NVFP4, retained BF16-encoder recipe | 18/18 | 3.128 s | 0.652–8.140 s | 8.015 / 9.402 GiB |
| Qwen NVFP4 + BF16 encoder, 512 conditioning + KV | 18/18 | 6.624 s | 0.957–23.886 s | 8.178 / 8.877 GiB |
| Qwen NVFP4 + native NVFP4 encoder, same conditioning | 18/18 | 6.685 s | 1.001–23.594 s | 8.178 / 8.877 GiB |
| Qwen GGUF Q4_K_M + same BF16 embeddings/conditioning | 18/18 | 11.373 s | 3.035–28.541 s | 6.522 / 7.381 GiB |

The nine-case medians mix small cropped retouches with larger generation/edits; they are not generic one-megapixel latency predictions. On the paired BF16-encoder cohort NVFP4's median is about **1.72× faster** than this Diffusers GGUF path, while GGUF saves about 1.66 GiB peak render allocation. Encoder memory is separate: the official BF16 stage peaked at 8.216 GiB CUDA and 9.555 GiB RSS. A smaller GGUF transformer therefore does not mean that the whole prompt-to-file flow peaks at only 6.522 GiB.

The renderer-only comparison excludes encoder/model loads for all arms. The Klein baseline specifically uses the retained comparison helper's BF16 encoder, not the production RAM-streamed FP8 encoder; its combined preparation/load took 39.372 seconds. Qwen renderer construction/placement took 5.470–6.595 seconds. BF16 encoder construction took 10.863 seconds; native NVFP4 construction in the original stage took 7.769 seconds. First encodes cost roughly 16 seconds of initialization; subsequent 512-conditioning native encodes took 0.072–0.158 seconds, versus 0.630–0.875 seconds with BF16. None of these isolated stage times substitutes for the lifecycle measurements below.

All **32 masked results in these four main cohorts** have exactly zero changed protected RGBA pixels and nonzero changed selected pixels. That is a compositor contract, not a seamless-edit quality score. Main cohorts total 72 completed images. Across all renderer experiments plus the eight lifecycle outputs, **113 images completed**, three render attempts exhausted VRAM and 18 stock-loader attempts failed before rendering. All evidence is retained; automatic completion is not visual acceptance.

The first 1024-conditioning KV-cache experiment failed three of 18 attempts: both two-reference composition seeds and one cup-reference seed. Its maximum completed allocation was 11.858 GiB and reservation 13.377 GiB, alongside the desktop's GPU allocations. Disabling KV caching completed all 18, at up to 8.378 GiB allocation, but two-reference composition took 26–44 seconds. Using **512 conditioning while retaining approximately one-megapixel output dimensions** allows KV caching with much lower memory. This trades some source/reference detail for speed and must be reviewed visually.

WDDM's CUDA free-memory figure overstated physical headroom in the first run. Subsequent workers use the lower of CUDA free memory and `nvidia-smi` physical free memory, minus 1.25 GiB desktop headroom; the lifecycle budget was 10.217 GiB. PyTorch allocated/reserved measurements are per-process and exclude desktop/driver allocations. Only one GPU benchmark worker samples at a time, with API models unloaded.

## Full prompt-to-file lifecycle

The native encoder and transformer keep CPU memory-mapped checkpoint views. Each new prompt uploads only the encoder, encodes, releases it, uploads the transformer, samples and releases it before full-frame VAE decode. No precomputed prompt result is reused in this test. All four cases use 512 conditioning, one-megapixel output, two seeds and KV caching.

| Case | Seed 7 total | Seed 42 total |
|---|---:|---:|
| Two-reference composition | 34.203 s, first call | 20.656 s |
| DJ whole edit | 8.953 s | 8.333 s |
| Dutch poster | 7.337 s | 7.487 s |
| Transparent product | 7.420 s | 7.428 s |

All eight completed. Peak CUDA allocation was **8.784 GiB**, reservation **10.217 GiB**. Lazy model construction took 1.319 seconds separately; checkpoint page reads and CUDA initialization appear in the first request. The first composition therefore took about 35.5 seconds after starting measured model construction, excluding Python imports. Six subsequent single-input/generation requests took 7.34–8.95 seconds including new encoding, both transfers and PNG save.

**RAM limits matter:** peak process RSS was 12.329 GiB and minimum available system RAM fell to **0.156 GiB**. The second composition spent 10.184 seconds encoding/reading its encoder state and 2.984 seconds uploading renderer state, compared with roughly 0.6–1.4 seconds encoding and 0.44–0.47 seconds renderer upload on later cases. GPU fit does not imply ample total memory or uniformly fast repeated requests on this busy 32 GB machine. Coexistence with loaded Klein is not tested; integration would need to preserve one-pipeline residency.

The tiled VAE outputs show vertical decode seams on the cup/poster. The lifecycle test releases transformer weights before full-frame decode; the visible seam is absent in inspected full-frame outputs. Native PNGs do contain alpha zero outside the product, but cutout-edge quality still needs review. Readable poster text is not proof of exact spelling: one inspected poster says **"Elle dag vers."** rather than the requested **"Elke dag vers"**.

## Visual findings and decision

In the inspected local edits, Klein's mushroom has more natural detail and lighting. Qwen's mushroom patch has a visibly painted, blocky boundary; the reference-cup insertion is oversized and its surrounding repaint is more apparent. The native encoder changes object scale and embroidery detail compared with the BF16 control. Zero changed protected pixels therefore does not establish a seamless or faithful retouch. Qwen's native transparency is useful, but exact text and cutout edges remain limitations.

NVFP4 is the faster Qwen quantization on this measured Blackwell path; the GGUF file is smaller and offers a useful memory/speed comparison. That runtime result does not establish better image quality. After reviewing the comparison, the user said **"Vind flux denk ik overall beter :D"**. Record this as an overall preference for FLUX, not individual approval or rejection of every retained image. Klein 9B remains the Quality model and Qwen integration is not pursued from this evaluation.

The complete evidence is retained for reference. A future decision to revisit Qwen would require new user approval and complete lifecycle/cancellation/offload validation, lossless protected compositing, correct RGBA handling, RAM pressure checks and real app acceptance. This evaluation adds no provider, UI toggle or deployment. It needs no backend restart or Tauri executable rebuild.

## Reproduction

Use `backend/.venv/Scripts/python.exe`; the tool selects isolated settings/data before importing historical fixture helpers. `gguf` is installed with `--no-deps --target models/research-cache/qwen-benchmark-deps`; use project-local `TEMP`/`TMP` for installation in this sandbox. All pinned checkpoints must already exist at the paths defined in the runner.

```powershell
backend/.venv/Scripts/python.exe backend/tools/benchmark_qwen_turbo.py --stage prepare
backend/.venv/Scripts/python.exe backend/tools/benchmark_qwen_turbo.py --stage encode --encoder bf16 --conditioning-resolution 512
backend/.venv/Scripts/python.exe backend/tools/benchmark_qwen_turbo.py --stage run --variant nvfp4 --encoder bf16 --conditioning-resolution 512 --kv-cache
backend/.venv/Scripts/python.exe backend/tools/benchmark_qwen_turbo.py --stage run --variant gguf --encoder bf16 --conditioning-resolution 512 --kv-cache
backend/.venv/Scripts/python.exe backend/tools/benchmark_qwen_turbo.py --stage encode --encoder nvfp4 --conditioning-resolution 512
backend/.venv/Scripts/python.exe backend/tools/benchmark_qwen_turbo.py --stage run --variant nvfp4 --encoder nvfp4 --conditioning-resolution 512 --kv-cache
backend/.venv/Scripts/python.exe backend/tools/benchmark_qwen_turbo.py --stage cycle
backend/.venv/Scripts/python.exe backend/tools/benchmark_qwen_turbo.py --stage klein
backend/.venv/Scripts/python.exe backend/tools/summarize_qwen_turbo.py
```

Run these GPU stages sequentially. A rerun of the same variant intentionally replaces that variant's result files; copy its evidence to a separately named snapshot before changing a recipe. Preparation verifies the retained asset hashes but does not recreate fixture sources. `summary.json` is derived from complete per-variant records, not a hidden shortlist of good seeds.

Validation: all measured main/lifecycle outputs completed; native/split reference probes passed; each main masked output changed selected pixels and preserved protected pixels exactly. Frontend typecheck/build passed (Node/fnm v26.8.1 as observed in a fresh shell), backend/tool compilation passed, and 81 local documentation links resolve. All 113 final/raw PNG pairs passed image validation, all 479 gallery asset links resolve, and 120 unique browser image requests returned HTTP 200. A web build here validates the checkout; it is not a Tauri executable rebuild. Detailed checks are retained with the local evidence.
