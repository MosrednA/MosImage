# MosImage

MosImage is a local AI canvas editor with an English Focus interface. Generate an image, select an object, replace or remove it, or expand the canvas. FLUX.2 Klein 4B Distilled NVFP4 is the Quality default for generation and editing on the target RTX 5070 Ti; BF16 remains selectable in Advanced controls; Z-Image Turbo is the Fast option and the automatic Standard route for expansion. SAM 2.1 Tiny provides local point selection and refinement.

## Product direction and implementation status

- [Current v2 implementation and roadmap](V2_REFACTOR_PLAN.md)
- [Current Focus implementation and validation, 7 October 2026](FOCUS_IMPLEMENTATION_2026-10-07.md)
- [Inpainting, context crops and blending, 7 October 2026](INPAINTING_2026-10-07.md)
- [Four-image benchmark suite](benchmarks/README.md) and [Klein BF16/NVFP4 results, 7 October 2026](benchmarks/NVFP4_RESULTS_2026-10-07.md)
- [Model research and proposed tech/UX direction, 7 October 2026](MOSIMAGE_DIRECTION_2026-10-07.md) — research snapshot and longer-term proposals; Focus, point selection, and basic outpainting are now implemented.

## Layout

```txt
backend/   FastAPI API, SQLite metadata, model runtime providers
frontend/  React + Vite UI and optional Tauri desktop shell
data/      Local SQLite database and generated image files
models/    Optional local model/checkpoint storage
```

Local `.env` files, model/dependency caches, the SQLite database and all runtime/diagnostic files under `data/` are excluded from Git. The fixed inputs, masks and metadata under `benchmarks/fixtures/v1/` are versioned evaluation assets. Validation documents refer to local evidence in `data/`; those files are available in the original working copy or can be regenerated with the documented benchmark tools.

## Backend

Fast path from the repository root:

```powershell
.\start.cmd
```

This starts the FastAPI backend on `http://127.0.0.1:8000` and the Vite frontend on `http://localhost:5173`. It creates `backend/.env` from `.env.example` when missing and installs frontend dependencies when `node_modules/` is absent.

On the first run it also creates `backend/.venv` and installs `backend/requirements.txt`. To skip automatic installs and only launch with the current environment:

```powershell
.\start.cmd -NoInstall
```

Manual backend setup:

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
Copy-Item .env.example .env
pip install -r requirements-torch-cu130.txt
pip install -r requirements.txt
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

The default profile is tuned for Ryzen 7 9700X, 32 GB RAM, and RTX 5070 Ti 16 GB. `requirements.txt` includes Comfy Kitchen 0.2.37; an existing environment can install the validated Windows wheel separately with `python -m pip install --no-deps --only-binary=:all: --require-hashes -r requirements-nvfp4-benchmark.txt` from `backend/`. Restart the API after installing. Other GPUs can select FLUX.2 Klein 4B (BF16) under Generation settings → Advanced controls → Model.

Default runtime behavior:

- CUDA is preferred with `bfloat16`.
- TF32 is enabled for faster CUDA matrix operations.
- Only one large pipeline is kept resident by default to protect 16 GB VRAM.
- Model CPU offload is enabled by default because Z-Image Turbo can sit right on the edge of 16 GB VRAM on Windows.
- PyTorch uses `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` to reduce allocator fragmentation.

For Windows and RTX 5070 Ti, use Python 3.13 or 3.12 for the backend virtual environment. The startup script prefers `py -3.13`, then `py -3.12`. The current pinned runtime is PyTorch 2.14.1 with CUDA 13.0 from `backend/requirements-torch-cu130.txt`; the verified local NVIDIA driver is 617.42. The old CUDA 12.8 requirements are retained as a historical baseline and do not match the current dependency set.

If you created the venv with the wrong Python or a CPU-only PyTorch install, delete it and rerun startup:

```powershell
Remove-Item backend\.venv -Recurse -Force
.\start.cmd
```

If the UI reports CPU fallback, reinstall CUDA PyTorch:

```powershell
cd backend
.\.venv\Scripts\python.exe -m pip install -r requirements-torch-cu130.txt
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.version.cuda)"
```

The final command must print `True` for CUDA availability before generation will run. Install the CUDA requirements before the application requirements; the latter pin Torch to the CUDA build so a resolver cannot silently replace it with CPU-only Torch.

### Hugging Face login (optional)

Both generation defaults and SAM 2.1 Tiny are public and work without a Hugging Face token. Logging in is useful for higher authenticated download limits and is required for the optional gated FLUX.1 Fill specialist after accepting its license on Hugging Face:

```powershell
cd backend
.\.venv\Scripts\hf.exe auth login
```

Use a read-only token. Hugging Face stores it in the normal user cache and the model loaders automatically use that login. `HF_TOKEN` or `MOSIMAGE_HF_TOKEN` is also supported for temporary environments; never commit a real token.

## Frontend

```powershell
cd frontend
npm ci
npm run dev
```

Open `http://localhost:5173`. The frontend expects the API at `http://127.0.0.1:8000` unless `VITE_API_BASE_URL` is set.

The English Focus layout gives most of the viewport to the image in every mode. A compact tool rail and command dock stay visible; Generation settings and Canvas size open modal panels, selection refinement opens a contextual popover, and Library opens a separate drawer. Results replace the command dock with a small review strip; compare them with the original, then Apply or Discard. Fit, zoom, open and document undo/redo remain directly accessible. The engine indicator shows connection state; `/api/system` exposes the detailed hardware profile.

Object selection runs SAM 2.1 Tiny on CPU and caches one image embedding. Click to include a point, Alt+click to exclude a point, or use Brush/Eraser/Rectangle/Lasso to correct the mask. The first selection downloads the model. Ctrl+Z/Shift+Ctrl+Z undo/redo the mask first when its history is available, then document revisions. O/B/E/V choose Object/Brush/Expand/Pan; Ctrl+O opens an image and Ctrl+Enter submits an instruction.

References place a person, pet or object from a concept image into the open document. Drop or browse images into the References panel on the right; each document keeps up to 24. Pick up to three (a single upload is picked automatically; the In use badge shows the order), select where they go, then describe what they do, for example "Anna and Tom sitting on the bench". References are passed to FLUX.2 Klein as `image_reference` for Replace edits only, so it requires the Quality model; picking one switches to Quality and two proposals. The scissors button optionally isolates the subject with SAM 2.1 Tiny on a neutral background; the full image is used by default because it was more reliable in the local trial. Several references are padded to one shared size of about 0.36 MP each and named in order in the prompt ("Anna is reference image 1"), so each likeness stays separate. See [the reference trial](INPAINTING_2026-10-07.md#reference-images).

Layers stack loose images on the picture, separate from references. Add PNG, JPG or WebP files from the Layers tab, drop them on it, paste with Ctrl+V while a picture is open, or copy the current selection with Ctrl+J. With Transform (T), drag to move, corners scale (Shift stretches freely), edges stretch, and the knob rotates (Shift snaps to 15°); arrows nudge by 1 px, or 10 px with Shift. With a layer selected, the selection tools edit that layer's mask instead: Object cuts out what you click with SAM 2.1 Tiny on the layer's own image, Eraser hides, Brush restores, and Lasso/Box hide or restore an area. The panel sets visibility, opacity, order and name, and flips, rotates, resets, inverts the mask, duplicates or deletes. Every layer change is a document revision, so Ctrl+Z steps through it without touching the background. Layer images are stored as RGBA PNG under `data/images/layers/`, with masks as greyscale PNG; a document holds up to 32. Export flattens the visible layers onto the picture as PNG. AI edits still change the background only, with layers kept on top; Expand moves them along with the content.

Expand supports draggable edges, format presets, and exact pixel margins. It adds space without cropping or changing the original pixels. The native canvas is capped at 4096 × 4096; generation still uses a model-facing image capped at 1024 on its longest edge. Applying an expansion updates document dimensions; undo/redo and reopening restore the corresponding revision dimensions.

Expand automatically chooses a mask-conditioned provider: Z-Image Turbo at 8 steps and guidance 0, or the optional FLUX Fill specialist. A short description of the scene or desired new content is required: generic continuation text produced a decorative frame in the local fixture. Klein remains excluded from expansion after the earlier instruction-editing route left padding intact; its new native inpaint route has only been validated for regional edits. Z-Image receives descriptive content rather than instructions about gray margins.

Transparent images stay RGBA in documents and protected compositing. Model input is flattened onto neutral gray before resizing; this prevents fully transparent expansion margins from turning black during RGBA resampling.

## Runtime Modes

The UI exposes Quality/Fast presets and keeps technical controls under Settings:

- `Quality` uses FLUX.2 Klein 4B Distilled NVFP4 at four steps and guidance 1.0; BF16 remains selectable in Advanced controls. Generation uses `Flux2KleinPipeline`; editing uses `Flux2KleinInpaintPipeline`, which receives the actual mask and edit strength. Switching tasks or precision releases the previous pipeline. Balanced mode uses model CPU offload while inference runs on CUDA.
- `Fast` uses Z-Image Turbo at 8 steps for quick iteration.
- `FLUX.1 Fill Dev (specialist)` has an optional Advanced adapter, but its current BF16 component-offload route is not ready for the 16 GB profile: the 12B transformer alone exceeds that GPU's memory. A measured quantized or finer-grained offload route is required before use on this profile. The weights are gated and non-commercially licensed. Quality remains on FLUX.2 Klein by default; see the [next model comparisons](INPAINTING_2026-10-07.md#next-model-and-sampling-comparisons).

The three runtime modes are:

- `Balanced` uses model CPU offload for FLUX.2 Klein on the default 16 GB profile. Z-Image keeps text-to-image on GPU up to `896x896` and uses component-level offload for inpainting. Start here for these validated providers. Fill's current component-offload adapter needs a separate 16 GB memory implementation and validation.
- `Safe / lower VRAM` always uses model CPU offload. Use this for large text-to-image jobs or after an OOM.
- `Performance / GPU resident` keeps the model on GPU. Use this for fast iteration at `768x768` or `896x896`; it may OOM at `1024x1024`.

After a CUDA OOM, restart the backend before benchmarking another mode. PyTorch can hold fragmented allocations after an OOM.

Inpainting runs on a rectangle around the selection, with 320 native pixels of surrounding context by default, clamped to image edges. The Context toggle shows this protected boundary on the canvas. The selection bar on the canvas sets Expand, Soften and Surrounding context with a live overlay of the exact mask sent to the model; Scene notes live in generation settings. The model-facing crop is capped at 1024 pixels on its longest edge; the full image is not automatically added as another reference.

The model receives the selection exactly as shown: Expand defaults to +4 px and Soften to an automatic outward ramp of about the longest side / 40. Klein retains this soft mask during sampling, helping original details continue into the selection. The generated crop returns at native resolution with low-frequency colour drift matched to the untouched surroundings. Pixels outside the soft edge stay exact. Select enough space for the complete subject, its shadows and glow: the model cannot reliably fit every requested object inside an arbitrary silhouette. See the [inpainting reference](INPAINTING_2026-10-07.md) for the experiments and practical controls.

Regional visual quality remains experimental. A paired eight-seed check on the mushroom fixture produced four promising, two weak and two failed-subject results with the current defaults, despite exact protected pixels in every output. The [reliability protocol and full results](INPAINTING_2026-10-07.md#repeated-seed-reliability-check) distinguish subject compliance and scene integration from pixel protection; they are not a general success rate.

The editor stores opened images as persistent v2 documents. Accepted proposals become revisions, document Undo/Redo traverses those revisions, document names autosave, and documents can be reopened through Library. Existing generation history remains intact and any result can be imported as a document. Use one to four proposals; they run sequentially through the single worker. A proposal is bound to the source revision and cannot be accepted after that revision changes.

While a job is queued, loading or generating, **Cancel** replaces the primary Generate / Remove / Expand action in the instruction dock. The button changes to **Cancelling…** and the editor remains locked until cleanup finishes. Cancellation preserves the current image, prompt and selection; failed cancellation requests leave the action available to retry.

When launched with `start.cmd`, the backend runs under a local supervisor. Cancel first uses the model's cooperative diffusion callback; if native CUDA code does not yield within eight seconds, the worker is replaced automatically so GPU memory is released. The UI tolerates the brief reconnection and the cancelled job disappearing from the replacement worker, then restores the editing controls. A cancelled queued job can finish immediately; a running job reports `cancelling` until its callback and cleanup return.

The Quality app adapter loads the official `black-forest-labs/FLUX.2-klein-4b-nvfp4` distilled transformer with native Comfy Kitchen CUDA kernels, plus the original BF16 text encoder and VAE. The NVFP4 transformer and base components use the exact revisions in the benchmark protocol. The BF16 adapter remains available for comparison. Both support generation and native masked editing with one resident pipeline, cancellation, local context and inward feathering. NVFP4 requires a Blackwell GPU, CUDA 13 and the BF16 component profile; unsupported configurations report an explicit error before downloading weights. Across four fixed scenes and three seeds each, median warmed edit time fell from 9.82 to 6.26 s, with similar visual behavior and exact protected pixels. Transformer storage fell from 7.22 to 2.29 GiB; whole-pipeline peak PyTorch GPU allocation stayed near 9.1 GiB because text encoding remains BF16. The app and cached-only benchmark now share the same native loader. Quality uses four steps and guidance 1; inpainting defaults remain strength 1 and context 320 px, with the canvas edge controls described above. See the [full results and limitations](benchmarks/NVFP4_RESULTS_2026-10-07.md).

## Hot Loading

On start the backend warms up in the background: it loads SAM 2.1 Tiny, then the Quality model (`MOSIMAGE_WARM_UP_MODEL`, FLUX.2 Klein NVFP4 by default), and runs one 256 px, one-step pass for each task so the first edit skips the CUDA kernel set-up. This takes about 15–20 s; the top bar shows Warming up and then Ready, and `/api/engine` reports the job. Edits submitted meanwhile wait for it and say so. Set `MOSIMAGE_WARM_UP_ON_START=false` to load on first use instead. `/api/models/load` remains for scripted preloading.

Measured on the RTX 5070 Ti: loading Klein NVFP4 takes 5–9 s from the local cache (about 13 s cold), and the first run afterwards adds 3–6 s unless warmed up. Klein's text-to-image and inpaint pipelines share one set of weights, so switching between a new image and an edit is instant. Point selection runs in its own CPU queue and never waits behind a generation.

Prompt encodings are cached per loaded model, so a second proposal or "generate again" skips moving the 8 GB text encoder to the GPU. Two proposals with two references went from 22 s to about 18 s, and 15 s when repeated. Progress shows the phase, proposal, step and an estimate measured from the step time.

With the default 16 GB profile, MosImage keeps one large model hot at a time. Switching between Quality and Fast unloads the previous provider, and switching between GPU-resident and CPU-offload placement reloads the pipeline.

## Desktop Shell

The Tauri desktop app in `frontend/src-tauri` is the same UI in its own window. After installing Rust and dependencies:

```powershell
cd frontend
npm run tauri:dev    # development, with Vite
npm run tauri:build  # mosimage.exe plus MSI and NSIS installers under src-tauri/target/release
```

The app starts the engine itself from `backend/.venv` in the checkout it was built from (override with `MOSIMAGE_ROOT`), without a console window; output goes to `data/logs/engine.log`. If an engine is already listening on port 8000, for example from `start.cmd`, the app uses it and leaves it running. Like `backend/supervisor.ps1`, the app restarts its engine after a hard cancel or crash. On close it asks the engine to cancel jobs and unload models through `/api/shutdown`, guarded by a per-launch token, and ends it after 20 s if needed. A Windows job object ends the engine as well if the app itself crashes, so no Python process keeps RAM or GPU memory.

The engine is not bundled into the installer: Python with CUDA PyTorch, Diffusers and the NVFP4 kernels is several gigabytes, and the models download to the Hugging Face cache on first use.
