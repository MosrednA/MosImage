# Inpainting benchmark fixtures

The [9 October model comparison](MODEL_COMPARISON_2026-10-09.md) adds the [v2 creative fixtures](fixtures/v2/manifest.json) (a man and a man holding a phone, generated locally) and compares Klein 4B/9B NVFP4, Qwen-Image 2.1 turbo and encoder variants across twelve cases.

The [8 October mask/reference investigation](INPAINT_QUALITY_2026-10-08.md) extends these fixtures with paired context/resolution tests, local identity and object references, and a separate measurement of the real app provider. Its recipes and all outcomes are retained under `data/inpaint-quality-2026-10-08/`.

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

NVFP4 uses the official distilled transformer checkpoint with its calibrated input/global/block scales. The native loader shared with the app calls Comfy Kitchen's CUDA activation quantization and NVFP4 GEMM directly, disables eager/Triton fallbacks and records successful kernel calls. It does not dequantize weights to BF16 for inference. Remaining transformer projections/norms and pipeline components retain their original BF16 precision. QKV splits preserve complete 128-row scale tiles.

The kernel library is pinned in the application requirements. The original experiment used an isolated install; the benchmark now imports the same app runtime. For a separately hash-verified Windows wheel install:

```powershell
cd backend
.\.venv\Scripts\python.exe -m pip install --only-binary=:all: --no-deps --require-hashes -r requirements-nvfp4-benchmark.txt
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

The UI's Quality provider now uses native NVFP4 on the target Blackwell GPU; BF16 remains selectable in Advanced controls. Integration does not establish broadly seamless editing quality. This fixture suite can also compare future models and masking/context approaches without changing the source images or seed cohort.

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

Optional adapter verification, requiring a Blackwell CUDA GPU and the app kernel dependency:

```powershell
cd backend
.\.venv\Scripts\python.exe tools/check_nvfp4_kernels.py
```

For separate component-residency sampling, run each precision sequentially with `--fixture garden --seed-count 1 --profile-components` and a fresh output directory. These diagnostic runs are excluded from the latency/visual cohort. The returned samples include live tensors and are not component peak VRAM.

## Anna/park reference insertion experiments

The [7 October Anna/park results](../INPAINTING_2026-10-07.md#controlled-annapark-experiments) use frozen local document assets, separate from the versioned v1 suite. They retain 72 completed outputs across 24 settings/model/prompt/guide/mask variants (three prompt-baseline outputs repeat previous PNGs exactly), plus failures and explicitly skipped seeds. The scripts require the original local source, mask and Anna reference paths recorded in the manifest; copying this checkout alone does not provide those ignored assets.

Run from `backend/`. The original API ablation tool requires explicit `--variants`; do not repeat the known failing long-Scene/BF16 context-320 profiles on this machine. The memory-bounded follow-up refuses a loaded API model, encodes text separately, releases it before loading a renderer, caps PyTorch allocation at 65%, and stops after a slow step or insufficient RAM. This is a diagnostic budget, not a total-system memory guarantee. Preserve every declared result and use a fresh output directory for another run.

```powershell
cd backend
.\.venv\Scripts\python.exe tools/benchmark_park_reference.py --output ../data/park-scale-new --variants context32 context192 context320 context512 strength075 steps8
# Unload all API models or stop the API before the isolated follow-up.
.\.venv\Scripts\python.exe tools/benchmark_park_base.py --input ../data/park-scale-new --output ../data/park-scale-new/base
.\.venv\Scripts\python.exe tools/benchmark_park_prompts.py --input ../data/park-scale-new --output ../data/park-scale-new/prompts
.\.venv\Scripts\python.exe tools/benchmark_park_prompts.py --input ../data/park-scale-new --output ../data/park-scale-new/prompts-contact --variants prompt-parallel prompt-short
.\.venv\Scripts\python.exe tools/summarize_park_reference.py --input ../data/park-scale-new
```

The Base tool needs the pinned, locally cached Base transformer described in the results and the existing distilled components. No editor provider is registered for Base. Manually review all outputs before writing `review.json`; the viewer compares settings by seed and preserves failures. Setup/encoding costs are separate from render times in the split-runtime phase. Serve only the review directory if using a local HTTP server.

The prompt tool uses the same bounded runner but needs only the cached distilled NVFP4 checkpoint and components; it does not require the Base transformer. Source, mask, reference resolution, seeds, context and eight-step settings are fixed. The first four prompts and the separately declared parallel/short follow-up are reviewed in separate result manifests. It uses a 15-second post-step render budget. All earlier 39 images were rejected by the user; relative research ratings and new prompt comparisons do not imply acceptance.

## Pose/layout reference follow-up

The [visual guide comparison](../INPAINTING_2026-10-07.md#visual-poselayout-guide-comparison) adds nine local renders on the same frozen scene. A generated transparent dummy/bench is a supporting fixture; final images still use local Klein NVFP4. The new identity and structural references share a 384 × 512 canvas, and auxiliary hashes are checked before loading. Both guided conditions share one role-aware prompt. The benchmark registers no new provider.

Use a new frozen fixture directory/output for another run; the preparation tool refuses to overwrite a completed guides directory. The source directory must already contain `results.json`, unchanged source/mask/reference files and `dummy-generation.json` recording the built-in generator's full prompt. The local cutout path comes from the built-in tool; copy it into the prepared fixture with this command:

```powershell
cd backend
.\.venv\Scripts\python.exe tools/prepare_park_guides.py --input ../data/park-scale-new --cutout C:/path/to/generated-dummy.png
.\.venv\Scripts\python.exe tools/benchmark_park_guides.py --input ../data/park-scale-new --output ../data/park-scale-new/guide-tests
.\.venv\Scripts\python.exe tools/summarize_park_reference.py --input ../data/park-scale-new
```

Placement centre, bottom and maximum size are explicit constants in the preparation script. They describe this garden fixture, not a generic scale estimator. Check that the complete dummy/bench fits the true mask before inference; the script rejects a meaningful-alpha overlap with protected pixels. The original document is preserved. Model loading, pose-asset generation and preparation are excluded from per-render timing and need separate product measurements.


## Paired mask-shape follow-up

The [paired mask comparison](../INPAINTING_2026-10-07.md#paired-mask-shape-comparison) adds six renders with exactly the guide cohort's prompts/references/seeds/crop, changing only its alpha mask. The box keeps the original bounding rectangle with a 16-px inward smooth edge. Its opaque interior restores baseline faces in all three seeds. Size, pose/contact and background seams remain imperfect.

```powershell
cd backend
.\.venv\Scripts\python.exe tools/benchmark_park_masks.py --input ../data/park-scale-new --output ../data/park-scale-new/mask-tests
.\.venv\Scripts\python.exe tools/summarize_park_reference.py --input ../data/park-scale-new
```

Run after preparing the same guide assets. The runner checks frozen hashes, refuses a loaded API pipeline and preserves the original selection. The box **expands test permissions inside the same rectangle**; its protection counters must not be represented as zero changes outside the original mask. The review displays both counters and the mask footprint. Do not apply this diagnostic mask silently to a user's document.
