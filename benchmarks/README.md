# Inpainting benchmark fixtures

The [15 October FLUX 3 Image Klein evaluation](FLUX3_KLEIN_2026-10-15.md) compares the official distilled NVFP4 model against the current Klein 9B provider: twelve frozen cases with two seeds, identical captions, a separate current-prompt control and six complete sequential encoder/renderer requests. All 78 images succeeded after repairing a reference-grid input mismatch. FLUX 3 fits the RTX 5070 Ti 16 GB, but cached renders take about twice as long and new prompts incur component reloads. The user rejected integration: selective reference/identity gains are insufficient. Keep Klein 9B as Quality; a future small model must be substantially faster at comparable quality or clearly better at difficult edits. The report preserves numerical kernel proof, failed attempts, memory/latency evidence and the complete local gallery.

The [11 October MLX comparison](MLX_KLEIN_2026-10-11.md) runs Klein 9B and 4B on an M5 Max in BF16 (PyTorch, MPS) and in MLX at 8 and 4 bits, through the app's own services: eleven cases with references, masked edits, Consistency, the detail enhancer and Specialized outpaint, two seeds each, plus an unquantized MLX control. All 88 renders succeeded. 8-bit lowers Quality's peak from 36–42 GiB to 24–30 GiB, 4-bit to 18–24 GiB, at 4–7% more time. 8-bit pictures differ from BF16 about as much as the unquantized MLX control does; 4-bit keeps each scene with more changed detail. It also measures releasing the GPU caches between jobs and the VAE's share of the peak.

The [10 October SAM 3 refinement follow-up](SAM3_REFINEMENT_2026-10-10.md) completes six real-photo calibration cases, twelve evaluation cases with official person masks, correction clicks/boxes, negative controls and the original seven-image prompt diagnostics. Refined point masks reach similar annotation agreement; SAM 3 improves GPU speed and rejects the absent astronaut concept. All 593 masks were verified and 276 evaluation repetitions were identical. Annotation limits and broader object/Klein acceptance remain explicit; app providers are unchanged.

The [10 October SAM 3 selection comparison](SAM3_SELECTION_2026-10-10.md) completes seven frozen point/text cases on CPU FP32 and GPU BF16 against the existing SAM 2.1 Tiny/Grounding DINO routes. GPU SAM 3 takes about 0.15 s for new-image points and 0.20 s for text, but default masks show no general quality improvement. CPU SAM 3 is much slower on new images; integration and loaded-Klein validation remain open.

The [10 October official Qwen Turbo quantization evaluation](QWEN_TURBO_QUANTS_2026-10-10.md) compares native NVFP4 and Unsloth Q4_K_M against Klein 9B, with frozen prompts/seeds, alpha/protected-pixel checks and a separate prompt-to-file lifecycle. It retains memory and loader failures. The user prefers FLUX overall; Qwen is not selected for integration.

The [10 October SeedVR2 3B test](SEEDVR2_3B_2026-10-10.md) adds 27 retained generative-restoration outputs on the same six sources, including two seeds/repeats and colour-correction ablations on three faces. Warm native 2x processing takes about 9 seconds on the one-megapixel man and 3.5–3.7 seconds on 512² portraits. It adds skin/beard texture but redraws faces, cloth and text. Standalone memory fits the 16GB card; Klein coexistence and user visual acceptance remain open.

The [10 October texture-preservation follow-up](UPSCALING_FIDELITY_2026-10-10.md) compares classical 2x SwinIR Light/M and Swin2SR on six native enlargements and reconstruction diagnostics, including one real camera photo. It records BF16 and larger-tile measurements, source/alpha/protected-pixel contracts and shared-runtime observations. Light is the next visual-review candidate; no AI upscale feature or quality default is approved.

The [10 October upscaling comparison](UPSCALING_2026-10-10.md) measures RealESRGAN_x2plus and compact x4v3, including PNG save time, four retained quality cases, tile/alpha/protected-pixel checks and coexistence with real Klein 9B edits. This is experimental tooling and evidence; AI upscaling is not yet integrated into Export or the editor.

The [9 October quality-candidate evaluation](QUALITY_CANDIDATES_2026-10-09.md) compares Consistency V2 on the native 9B provider, FAL green-border outpaint on the fast 4B NVFP4 runtime, and contextual detail crops. The published Base 4B recipe was stopped after six garden outputs because its 157–180-second edits exceed the user's latency requirement. The subsequent optional editor switches and six real GPU acceptance jobs are recorded separately in that report; all switches start off and the model registry retains its three distilled/BF16 Klein providers.

The [9 October edit-mask implementation comparison](EDIT_MASK_RESULTS_2026-10-09.md) retains 30 before/after renders on the real Klein 9B provider. It checks separate generation/final blend masks, expansion overlap at model resolution, protected pixels and visual limits. Its recipes use current production compositing; the v1 precision fixtures below keep their frozen historical settings.

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

The [7 October Anna/park results](../docs/notes/2026-10-07-inpainting.md#controlled-annapark-experiments) use frozen local document assets, separate from the versioned v1 suite. They retain 72 completed outputs across 24 settings/model/prompt/guide/mask variants (three prompt-baseline outputs repeat previous PNGs exactly), plus failures and explicitly skipped seeds. The scripts require the original local source, mask and Anna reference paths recorded in the manifest; copying this checkout alone does not provide those ignored assets.

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

The [visual guide comparison](../docs/notes/2026-10-07-inpainting.md#visual-poselayout-guide-comparison) adds nine local renders on the same frozen scene. A generated transparent dummy/bench is a supporting fixture; final images still use local Klein NVFP4. The new identity and structural references share a 384 × 512 canvas, and auxiliary hashes are checked before loading. Both guided conditions share one role-aware prompt. The benchmark registers no new provider.

Use a new frozen fixture directory/output for another run; the preparation tool refuses to overwrite a completed guides directory. The source directory must already contain `results.json`, unchanged source/mask/reference files and `dummy-generation.json` recording the built-in generator's full prompt. The local cutout path comes from the built-in tool; copy it into the prepared fixture with this command:

```powershell
cd backend
.\.venv\Scripts\python.exe tools/prepare_park_guides.py --input ../data/park-scale-new --cutout C:/path/to/generated-dummy.png
.\.venv\Scripts\python.exe tools/benchmark_park_guides.py --input ../data/park-scale-new --output ../data/park-scale-new/guide-tests
.\.venv\Scripts\python.exe tools/summarize_park_reference.py --input ../data/park-scale-new
```

Placement centre, bottom and maximum size are explicit constants in the preparation script. They describe this garden fixture, not a generic scale estimator. Check that the complete dummy/bench fits the true mask before inference; the script rejects a meaningful-alpha overlap with protected pixels. The original document is preserved. Model loading, pose-asset generation and preparation are excluded from per-render timing and need separate product measurements.


## Paired mask-shape follow-up

The [paired mask comparison](../docs/notes/2026-10-07-inpainting.md#paired-mask-shape-comparison) adds six renders with exactly the guide cohort's prompts/references/seeds/crop, changing only its alpha mask. The box keeps the original bounding rectangle with a 16-px inward smooth edge. Its opaque interior restores baseline faces in all three seeds. Size, pose/contact and background seams remain imperfect.

```powershell
cd backend
.\.venv\Scripts\python.exe tools/benchmark_park_masks.py --input ../data/park-scale-new --output ../data/park-scale-new/mask-tests
.\.venv\Scripts\python.exe tools/summarize_park_reference.py --input ../data/park-scale-new
```

Run after preparing the same guide assets. The runner checks frozen hashes, refuses a loaded API pipeline and preserves the original selection. The box **expands test permissions inside the same rectangle**; its protection counters must not be represented as zero changes outside the original mask. The review displays both counters and the mask footprint. Do not apply this diagnostic mask silently to a user's document.
