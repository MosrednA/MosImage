# Architecture

Status: Current reference
Last verified: 2026-10-11

MosImage is two local processes: a FastAPI engine that owns the models, the job queue and all files, and a React UI that talks to it over HTTP on `127.0.0.1:8000`. The optional Tauri shell starts the engine itself and shows the same UI in a desktop window.

```txt
┌──────────────────────────┐   HTTP + JSON / multipart   ┌───────────────────────────────┐
│ React UI (Vite / Tauri)  │ ──────────────────────────▶ │ FastAPI engine                │
│ canvas, layers, masks    │ ◀── job status polling ──── │ routers → services → providers│
└──────────────────────────┘                             │ job queue: GPU lane, CPU lane │
                                                         │ SQLite + data/images/         │
                                                         └───────────────────────────────┘
```

## Backend (`backend/app/`)

### Request flow

1. A router in `routes/` validates the upload and parameters.
2. It submits a job to `jobs.JobManager` and returns a job id at once.
3. The job runs in `services.py` (generation, edit, inpaint, outpaint) or a specialised module (`faces.py`, `enhance.py`, `selection.py`).
4. Services call an `ImageModelProvider` from `models/registry.py` and store results through `storage.py`.
5. The UI polls `GET /api/jobs/{id}` every 250 ms until the job completes, fails or is cancelled. A result shows at once; history and model status refresh in the background.

### Routers (`routes/`)

| Module | Prefix | Endpoints |
|---|---|---|
| `generation.py` | `/api` | generate, edit, inpaint, outpaint, faces and finding them, enhance, generation history |
| `selection.py` | `/api` | point selection, select by text, subject cut-out |
| `jobs.py` | `/api/jobs` | job status and cancellation |
| `documents.py` | `/api/v2` | documents, revisions, layer assets, layers, AI alternatives, undo/redo |
| `references.py` | `/api/v2` | loose and per-document references, subject isolation |
| `common.py` | | shared helpers: bounded upload reads, parameter parsing, capability and reference checks, `http_errors()` for mapping exceptions to status codes |

`main.py` keeps only the app setup: CORS, static file serving for `/files/images/`, health, system info, engine warm-up, model load/unload and shutdown.

### Jobs

`jobs.py` runs two single-worker lanes: one for GPU work (generation and editing) and one for CPU work (SAM selection, Grounding DINO, ViTMatte). Selecting therefore never waits behind a render. Each job gets a `JobControl` for progress and cooperative cancellation. Finished jobs are retained for `MOSIMAGE_JOB_RETENTION_SECONDS`. Under `start.cmd`, `start.sh` or the desktop app, a supervisor restarts the engine when a cancelled GPU call does not yield within eight seconds.

### Model providers (`models/`)

Every runtime sits behind the `ImageModelProvider` protocol in `base.py`. `registry.py` keeps one large pipeline resident at a time to fit 16 GB of VRAM. Every model is FLUX.2 Klein and supports every task: generation, whole-picture edits, masked edits and expansion.

| Provider | File | Role |
|---|---|---|
| FLUX.2 Klein 9B NVFP4 | `flux2_klein_nvfp4.py`, `klein_text_encoder.py` | Quality preset |
| FLUX.2 Klein 4B NVFP4 | `flux2_klein_nvfp4.py` | Fast preset; commercially licensed |
| FLUX.2 Klein 4B BF16 | `flux2_klein.py` | For GPUs without NVFP4 support |
| FLUX.2 Klein 9B (BF16, 8-bit or 4-bit) | `flux2_klein_apple.py`, `klein_text_encoder.py`, `klein_mlx.py` | Quality preset on Apple silicon |
| FLUX.2 Klein 4B (BF16, 8-bit or 4-bit) | `flux2_klein_apple.py`, `klein_mlx.py` | Fast preset on Apple silicon; Specialized outpaint |

The device profile decides which of these the registry offers. `config.py` defaults to the CUDA profile, or on macOS to the Apple silicon profile (`preferred_device=mps`), and names each profile's Quality and Fast model; request defaults, validators and the `preset` field of `/api/models` follow it, and the UI maps its presets to the models the engine reports. `runtime.py` selects CUDA, Metal (MPS) or the CPU, with bfloat16 on either GPU. On Apple silicon CPU and GPU share memory, so the providers keep every component resident, the 9B text encoder stays in BF16 on the GPU (trimmed to the layers Klein reads) and LoRAs attach to loaded weights.

`apple_weights` chooses how a Mac holds the two large components; by default it follows unified memory (BF16 from 64 GB, 8-bit from 48 GB, else 4-bit). With `bf16` the diffusers pipelines run on MPS as above. With `q8` or `q4`, `klein_mlx.py` loads the transformer and the trimmed Qwen3 encoder in MLX and quantizes their block linears at load (affine, groups of 64); embedders, modulation and the output projection stay BF16. Two torch-shaped wrappers take their place in the same diffusers pipelines, so references, masks, the scheduler and the VAE are untouched, and tensors cross between PyTorch and MLX without a copy (DLPack). The MLX modules carry diffusers' layer names, so `KleinLora` attaches the same adapters as a low-rank residual beside the quantized weights, at any scale and detachable. PyTorch and MLX each cache freed Metal buffers; on a Mac the job runner releases both after every GPU job (`runtime.release_cached_memory`). Where Quality's peak does not fit, the Mac also warms up Fast, and `apple_vae_tiling` makes `Flux2KleinAppleProvider._configure_memory` shrink the VAE's tiles from 1024 to 512 px. `system_info` reports `warm_up_model`, and the editor starts on that model's preset.

`nvfp4_runtime.py` loads the native CUDA NVFP4 transformer, shared with the benchmark tools.

Klein providers encode each prompt once and cache the embeddings. After a new encode they empty the CUDA cache. The 9B text encoder's weights (about 5 GiB) pass through the GPU per prompt, and Windows' system-memory fallback never makes an allocation fail. Without that step, the caching allocator kept those blocks while the denoiser's next allocations went to system RAM, about 20 times slower.

`klein_lora.py` provides reversible transformer-only low-rank residuals for the evaluated Consistency, FAL and detail-enhancer exports. It wraps BF16 or native NVFP4 linears, validates all targets before attachment and restores original modules on detach. Packed NVFP4 weights and scale buffers remain intact. App providers lazily resolve pinned, SHA-256-verified checkpoints under `models/adapters/` only for enabled options. Consistency V2 attaches at 0.6 on distilled 9B; FAL attaches at 1.1 before 4B placement/offload. The detail enhancer attaches on 9B for Enhance and Faithful Fix faces at the requested scale; it and Consistency patch the same 112 linears, so the provider detaches one before attaching the other. `EditRequest.strength` below 1 runs a whole-picture edit through the inpaint pipeline with a full mask, so only the later steps are redrawn. Ordinary jobs reset unused adapters to zero. Unload and failed initial placement release adapter references with the shared pipeline. The research-only undistilled Base 4B model is absent from the registry. The [candidate report](../benchmarks/QUALITY_CANDIDATES_2026-10-09.md) preserves compatibility evidence and visual limits.

### Image processing

| Module | Purpose |
|---|---|
| `compositing.py` | Context crops, model sizing, separate generation/blend masks, colour-drift correction, seam matching, grain matching and protected compositing |
| `selection.py` | SAM 2.1 Tiny point selection, text selection and subject cut-outs |
| `detection.py` | Grounding DINO tiny text-to-box detection |
| `matting.py` | ViTMatte edge refinement for hair and fur |
| `faces.py` | Face detection, crop, re-render and blend for Fix faces |
| `enhance.py` | Tile layout, tone keeping and overlap blending for Enhance |

Masked edits use opaque generation coverage inside the selected footprint and the original soft alpha for final compositing. Expansion grows its generation mask by 32 pixels after resizing to model resolution, while its final blend reaches only 16 native pixels into the original. Colour-drift reference pixels exclude the larger regenerated footprint. Inpaint grows its generation mask by one 16 px token after resizing; the tone offset in that hidden ring is carried into the selection with a tolerance, so only small offsets move. Grain lost to the model is restored as seeded monochrome noise measured outside the selection. After a cancel or a rejected request the model stays loaded; other failures unload it, since GPU memory may be in an unknown state. The service chooses the named inpaint or outpaint workspace; providers receive the prepared image and mask through the existing interface. [Paired validation](../benchmarks/EDIT_MASK_RESULTS_2026-10-09.md) records visual limits, protected pixels and runtime measurements.

Optional Fine detail sends the contextual selected crop through `provider.edit` at full strength, then uses the same protected compositor. Specialized outpaint sends the full green-border canvas through `provider.outpaint` on distilled Fast4, then maps the full result into the outpaint workspace for protected seam compositing. The API rejects incompatible models, sampling and operations before queueing. Generation migration `generations.0004_quality_options` stores `consistency_lora`, `detail_crop` and `specialized_outpaint` as booleans; older records default to false. `qualityOptions` in EditorSettings derives effective options from the current operation while retaining previous model, sampler and strength preferences.

### Storage

- `storage.py` runs SQLite with WAL and named migrations, and records generation history.
- `documents.py` stores documents, revisions, AI alternatives, undo and redo. A stale base revision raises `RevisionConflict`, which the API returns as HTTP 409.
- `layers.py` stores layer images (RGBA PNG) and masks (greyscale PNG). The opened picture is the bottom layer, `background`, which points at the document image itself; `revisions.composite_image_path` remains the canvas image that sets the size and the library thumbnail. An AI layer that replaced another keeps its `original_id`. A switched-off mask is kept with `mask_enabled: false`; stacks saved before that flag existed load with their masks on. A layer can hold up to eight same-sized `variant_paths`, the AI proposals it was applied with; `image_path` is the one shown and the others stay on disk until the document goes.
- `references.py` stores reference images and their optional isolated subjects.

All files live under `data/`. Only `data/images/` is served over HTTP.

## Frontend (`frontend/src/`)

`App.tsx` coordinates editor commands, keyboard handling, crop/application flow and document state through hooks and components. Model execution stays in the backend; canvas, mask and layer helpers live in their own modules.

| Folder | Contents |
|---|---|
| `api.ts` | Typed client for every endpoint |
| `hooks/` | State and side effects: engine connection and warm-up (`useEngine`), job submission and polling (`useJobRunner`), the visible picture and exports (`usePicture`), selection tools (`useSelectionTools`), layers, references, result preview and keyboard/paste handling |
| `generation/` | Pure logic: what a command submits (`submission.ts`), when it can run and what the hint says (`commandState.ts`), operations, presets, aspect ratios and progress text |
| `canvas/` | Editor canvas, viewport, mask and sketch layers, crop and expansion canvases, selection and mask-brush bars |
| `layers/` | Layer stack, transform geometry, bitmap helpers (mask display, selection to layer mask) and the soft mask brush (`brush.ts`) |
| `masks/` | Mask operations and refinement. The signed distance field for soft edges runs in a Web Worker (`maskWorker.ts` via `fieldWorker.ts`), with a synchronous fallback |
| `components/` | Panels, dialogs, top bar, tool rail, proposal tray and library |

Most of the logic in `generation/`, `masks/` and `layers/` is pure and covered by Vitest tests next to each module.

The UI runs in Chromium (WebView2 on Windows) and in WebKit (Safari, and the macOS app). Two WebKit differences matter for pixels: its `createImageBitmap` dithers, which turned black selection pixels into faint noise, so `decodeImage` in `layers/bitmaps.ts` decodes files through an `<img>`, which is exact in both engines; and its 2D canvas has no `filter`, so layer footprints blur in script (`layers/blur.ts`) there. A trackpad pinch arrives as Ctrl + wheel in Chromium and as gesture events in WebKit; `useViewport` zooms on both. `platform.ts` labels shortcuts with ⌘ and ⌥ on a Mac; the handlers accept Ctrl and Cmd everywhere.

## Desktop shell (`frontend/src-tauri/`)

The Tauri app starts the engine from `backend/.venv` without a console window and logs to `data/logs/engine.log`. If an engine is already running on port 8000, the app reuses it. It restarts the engine after a crash or hard cancel. On close it shuts the engine down through `/api/shutdown`, using a per-launch token. A Windows job object also ends the engine if the app itself crashes; on macOS and Linux the engine watches its parent process (`MOSIMAGE_EXIT_WITH_PARENT`) and exits once it is gone.
