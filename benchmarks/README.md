# Inpainting benchmark fixtures

The [v1 overview](fixtures/v1/overview.jpg) contains four original AI-generated scenes at 1254 × 1254. All four sources, masks and edit prompts were fixed before inspecting benchmark outputs.

| Fixture | Edit | What it exercises |
| --- | --- | --- |
| Garden | Replace a terracotta sphere with a small glowing mushroom | Flowers, occlusion, grounding, local glow and depth of field |
| Kitchen | Remove a mug, handle, reflection and shadow | Marble continuity and reconstruction without color residue |
| Portrait | Replace a coral brooch with an embroidered daisy | Small detail, fabric folds and exact face protection |
| Architecture | Replace an orange poster illustration with blue botanical art | Paper perspective, lighting and exact frame/brick protection |

[manifest.json](fixtures/v1/manifest.json) records source and mask SHA-256 hashes, task-specific criteria, the exact prompts and native crop/model dimensions. [generation-prompts.json](fixtures/v1/generation-prompts.json) records the built-in imagegen source-generation prompts and mode. These are synthetic local evaluation assets, not a representative dataset or ground-truth edits.

## Mask files

- `NAME.png`: untouched source.
- `NAME.mask.png`: browser-compatible white RGBA; alpha is the editable footprint.
- `NAME.mask-binary.png`: equivalent black/white L mask for external tools.
- `NAME.preview.png`: blue selection overlay with gold context boundary.
- `prepared/NAME/input.png` and `mask.png`: the exact model-facing context crop and inward soft mask.

White or nonzero alpha means editable. Black/zero alpha means protected. The mask includes the full subject and relevant contact shadow, rather than following an arbitrary tight silhouette. The frozen preset uses 320 native pixels of surrounding context, 16 px inward sampling/composite feather, no hidden growth, and a 1024 px model-facing maximum. Original full-resolution images remain unchanged.

Rebuild masks, metadata and previews from the original sources:

```powershell
cd backend
.\.venv\Scripts\python.exe tools/create_benchmark_fixtures.py
```

## Paired precision experiment

The evaluation cohort is seeds **7, 42, 20261007** for each image and precision: 12 outputs per model, all retained. A separate seed 99999 warms the pipeline and is excluded from visual/performance results. Both precisions use the same native `Flux2KleinInpaintPipeline`, BF16 text encoder and VAE, four distilled steps, guidance 1.0, strength 1.0, prompt construction and exact compositor. The only changed component is the transformer and its GEMM precision.

NVFP4 uses the official distilled transformer checkpoint with its calibrated input/global/block scales. The experimental loader calls Comfy Kitchen's CUDA activation quantization and NVFP4 GEMM directly, disables eager/Triton fallbacks and records successful kernel calls. It does not dequantize weights to BF16 for inference. Remaining transformer projections/norms and pipeline components retain their original BF16 precision. QKV splits preserve complete 128-row scale tiles.

The kernel library is an isolated optional install, separate from the application requirements:

```powershell
cd backend
.\.venv\Scripts\python.exe -m pip install --only-binary=:all: --no-deps --target .nvfp4-deps -r requirements-nvfp4-benchmark.txt
```

Version 0.2.37 was checked against PyPI on 7 October 2026, supports the local Windows/Python 3.13/CUDA 13.0 runtime, and had no known OSV advisory at the time of the check. This is not a security audit. Model files use safetensors. The Apache-2.0 kernel package is used directly; no ComfyUI GPL source is copied into MosImage.

Run **sequentially**, with all API pipelines unloaded and no editor job running:

```powershell
cd backend
.\.venv\Scripts\python.exe tools/benchmark_klein_precision.py --precision bf16 --output ../data/nvfp4-benchmark/bf16
.\.venv\Scripts\python.exe tools/benchmark_klein_precision.py --precision nvfp4 --output ../data/nvfp4-benchmark/nvfp4
```

The script refuses to run if the API has a loaded model and verifies frozen asset hashes. Each precision runs in a separate process; both use component CPU offload. Result JSON stores exact model revisions, package versions, complete rendered prompts, warmed end-to-end latency, transformer storage, peak PyTorch CUDA allocated/reserved memory, sampled process RSS, successful NVFP4 GEMM calls and protected-pixel differences. End-to-end time includes prompt encoding, offload, VAE and PNG compositing; it is not pure transformer kernel timing. PyTorch allocator measurements exclude other desktop applications and some driver/library allocations.

Review **every seed**, scoring requested-content compliance, grounding/perspective and texture/lighting/seams separately. Protected pixels and output dimensions are automated checks, not evidence of a visually seamless edit. Matching seeds do not imply pixel-identical results after quantization. Never pick only the best seed when deciding the default runtime.

The UI's Quality provider remains BF16 until an application adapter and broader reliability validation are completed. This fixture suite can also compare future models and masking/context approaches without changing the source images or seed cohort.

## Completed run and review

See the [7 October 2026 results](NVFP4_RESULTS_2026-10-07.md) for all measurements and visual limitations. Rebuild the contact sheets and interactive review after both full cohorts finish:

```powershell
cd backend
.\.venv\Scripts\python.exe tools/summarize_precision_benchmark.py
```

Open `data/nvfp4-benchmark/index.html` in a browser, or run the restricted local review server from the repository root:

```powershell
.\backend\.venv\Scripts\python.exe -m http.server 5182 --bind 127.0.0.1 --directory data/nvfp4-benchmark
```

The review then appears at `http://127.0.0.1:5182`. It serves only benchmark results/assets, not the repository or environment files. The main editor remains on `http://localhost:5173`.

Optional adapter verification, requiring CUDA and the isolated kernel dependency:

```powershell
cd backend
.\.venv\Scripts\python.exe tools/check_nvfp4_kernels.py
```

For separate component-residency sampling, run each precision sequentially with `--fixture garden --seed-count 1 --profile-components` and a fresh output directory. These diagnostic runs are excluded from the latency/visual cohort. The returned samples include live tensors and are not component peak VRAM.
