# MosImage guide

Setup, runtime behaviour and a detailed reference for every tool. For a quick overview see the [project README](../README.md); for design notes and research see the [documentation index](README.md).

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

The default profile is tuned for Ryzen 7 9700X, 32 GB RAM, and RTX 5070 Ti 16 GB. `requirements.txt` includes Comfy Kitchen 0.2.37; an existing environment can install the validated Windows wheel separately with `python -m pip install --no-deps --only-binary=:all: --require-hashes -r requirements-nvfp4-benchmark.txt` from `backend/`. Restart the API after installing. Other GPUs can select FLUX.2 Klein 4B (BF16) under Generation settings → Advanced controls → Model. The 9B default keeps about 6 GB of its text encoder in RAM, and its weights are licensed for non-commercial use only; choose Klein 4B NVFP4 for commercial work.

Default runtime behavior:

- CUDA is preferred with `bfloat16`.
- TF32 is enabled for faster CUDA matrix operations.
- Only one large pipeline is kept resident by default to protect 16 GB VRAM.
- Model CPU offload is enabled by default for the Klein 4B models in Balanced mode; Klein 9B stays resident on the GPU.
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

Every model MosImage uses is public and works without a Hugging Face token. Logging in only raises the download limits:

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

Library can delete saved documents and generation history entries after confirmation. Deleting a document removes its revisions, alternatives, references and unused layer images/masks, including assets kept only by older revisions. Assets shared with another document's revision history stay available for undo/redo. Deleting a generation retains input and mask files still used by another generation; imported documents and proposals keep their own copies.

Object selection runs SAM 2.1 Tiny on CPU and caches one image embedding. Click to include a point, Alt+click to exclude a point, or use Brush/Eraser/Rectangle/Lasso to correct the mask. The first selection downloads the model. Ctrl+Z/Shift+Ctrl+Z undo/redo the mask first when its history is available, then document revisions. O/B/E/V choose Object/Brush/Expand/Pan; Ctrl+O opens an image and Ctrl+Enter submits an instruction.

References place a person, pet or object from a concept image into the open document. Drop or browse images into the References panel on the right; each document keeps up to 24. Pick up to three (a single upload is picked automatically; the In use badge shows the order), select where they go, then describe what they do, for example "Anna and Tom sitting on the bench". A reference is named after its file, so "Anna.jpg" becomes Anna and can be named in the prompt. Code-like file names from cameras, chat apps and generators, such as `IMG_1234` or `T5S8DAP28-7b1f1113fc48`, become "Reference 1", "Reference 2" and so on, and the prompt then treats that reference as an unnamed subject. References need FLUX.2 Klein, so they require the Quality model; picking one switches to Quality and two proposals. Replace passes them to the inpaint pipeline as `image_reference`; Generate and Edit pass them as extra conditioning images.

References also work without a picture. On the start screen, the References panel keeps up to 24 loose references; pick up to three and describe the scene, for example "Anna and Tom riding a roller coaster together". Klein composes a new image from them at the chosen format. Apply opens it as a document and copies the references it used into that document, with their own files; the loose originals stay on the start screen for the next image. The scissors button optionally isolates the subject with SAM 2.1 Tiny on a neutral background; the full image is used by default because it was more reliable in the local trial. Several references are padded to one shared size of about 0.36 MP each and named in order in the prompt ("Anna is reference image 1"), so each likeness stays separate. See [the reference trial](notes/2026-10-07-inpainting.md#reference-images).

Layers stack loose images on the picture, separate from references. Add PNG, JPG or WebP files from the Layers tab, drop them on it, paste with Ctrl+V while a picture is open, or copy the current selection with Ctrl+J. With Transform (T), drag to move, corners scale (Shift stretches freely), edges stretch, and the knob rotates (Shift snaps to 15°); arrows nudge by 1 px, or 10 px with Shift. Object, Brush, Eraser, Lasso and Box always select on the visible picture, as in Photoshop, so picking one with a layer selected deselects the layer. To cut a layer down, press **Mask layer** in the panel: the same tools then edit that layer's mask. Object cuts out what you click with SAM 2.1 Tiny on the layer's own image, Eraser hides, Brush restores, and Lasso/Box hide or restore an area. Esc or Done masking ends it. The panel sets visibility, opacity, order and name, and flips, rotates, resets, inverts the mask, duplicates or deletes. Every layer change is a document revision, so Ctrl+Z steps through it without touching the background. Layer images are stored as RGBA PNG under `data/images/layers/`, with masks as greyscale PNG; a document holds up to 32. Export flattens the visible layers onto the picture as PNG.

Edit changes the whole picture with an instruction, such as "make it a snowy night" or "turn it into a watercolour", without a selection. The visible picture, including every visible layer, is scaled to about one megapixel in multiples of 16 and passed to Klein as image 1; picked references follow as images 2 and up and are named in the prompt. The prompt asks Klein to keep faces and everything the instruction does not mention. Proposals are scaled back to the canvas size, so Apply adds a layer that covers the canvas exactly. Once a document has layers, Edit offers **All visible** or **Current layer**. Current layer edits only the selected layer, through its mask, with transparent parts on neutral grey. The result keeps the layer's transparency, mask, position, scale, rotation and opacity. Apply puts it above the original and hides the original, like Blend in. Choosing Edit with a layer selected starts on that layer, and picking another layer in the panel switches the target. Detail above one megapixel is redrawn at that resolution, so very large pictures come back softer than the original; use Replace for a local change that keeps every original pixel. Edit needs the Quality model.

Sketch (P, or Sketch in the command dock) paints rough coloured strokes on their own drawing over the picture, with ten colours plus any colour, size, Draw/Erase (or Alt), Undo stroke and Clear. Describe what you drew and press Render. **Only the sketch** redraws the strokes plus a margin of about 2.5% of the longest side, fading out over another 2.5%, with Klein's inpaint pipeline at strength 1 and the sketched crop as its reference image. In the local trial this followed a crude deer and top hat closely without leaving stroke remnants; strength 1 or 0.85 without the reference left a sketch stub. The rest of the picture keeps every pixel, and Apply adds the patch as a layer. **Whole picture** sends the sketched picture through Edit, so light and style match around it, and Apply adds a full-canvas layer. **Blank canvas** on the start screen opens a white document at the format in generation settings, ready to sketch with Whole picture. Sketch needs the Quality model and ignores references; Ctrl+Z undoes strokes while the pencil is active.

Faces finds human faces with Grounding DINO tiny (the query is `human face.`, so pets are skipped) and redraws each one sharper with Klein. A square crop of about 2.2 times the face renders at 768 × 768 and scales back; its average colour is matched to the original, and a feathered ellipse pastes only the face, so hair and background stay. Up to eight faces are fixed, largest first; a selection limits the fix to faces centred inside it. The prompt is optional and adds to the built-in restoration prompt, and a picked reference makes the face match that person. Apply adds only the faces as a layer, cut to their shared mask. In the local trial, five faces in a 512 px pub photo came back visibly sharper and the dog was skipped, in 38 s including model loading. Faces needs the Quality model.

With Object, **Find…** selects everything matching a short English description, such as "the hat" or "dog, beer glass". Grounding DINO tiny finds the boxes and SAM 2.1 Tiny turns each into a mask, joined into one selection. Matching is loose: in the trial, "hat" selected bunny ears and "beer glasses" also caught eyeglasses. **Subject** selects the main people and animals (the strongest large detection, plus others at least a quarter of its size) and refines the edge with ViTMatte small, so hair and fur keep soft edges. **Remove background** in the layer panel does the same on the layer's own image and sets the result as its layer mask; while masking a layer, Find and Subject keep only the match on that layer. These run on CPU; Grounding DINO (about 0.7 GB) and ViTMatte (about 0.1 GB) download on first use and stay loaded.

In an open document, Generate creates a full image without requiring a selection. Apply adds it as a new layer, centred and fitted to the canvas without changing its aspect ratio; the preview uses the same placement. New still starts a separate document with its first image as the background. Replace and Remove always add the selected proposal as a new layer, including when the layer stack is empty or all layers are hidden. They work on the picture as you see it, including every visible layer. The selected patch is cropped to the mask's footprint; the result already contains the feathered blend, so its layer mask retains the footprint without blending the edge twice. Apply selects the result with Transform and opens Layers. Hide, delete, opacity, undo/redo and reopening keep the background available. A failed layer save leaves the proposal available for retry. Blend in keeps its lower-strength edit and places the blended copy above the original, hiding that original. Expand extends the background and moves the layers with the content. See [AI layer validation, 9 October](notes/2026-10-07-focus-implementation.md#ai-results-as-layers-9-october-2026).

Expand supports draggable edges, format presets, and exact pixel margins. It adds space without cropping, and only a 16 px band along the original edge blends into the new area. The native canvas is capped at 4096 × 4096; generation still uses a model-facing image capped at 1024 on its longest edge. Applying an expansion updates document dimensions; undo/redo and reopening restore the corresponding revision dimensions.

Expand runs on the chosen Klein model, like every other tool: 9B for Quality, 4B for Fast. Describe what the new area should show, such as "more beach and sea"; Klein is asked to extend the scene seamlessly, without a frame or border. The model also redraws a 16 px band inside the original. MosImage carries that band's tone offset into the new area and crossfades across it, so the new area meets the original without a visible line; every original pixel beyond the band stays exact. In the 9 October trial with 9B, widening a 1254 px garden and kitchen photo to 1894 px and a 832 × 1216 portrait to 1664 px each took 7–10 s once loaded, with seamless edges.

Transparent images stay RGBA in documents and protected compositing. Model input is flattened onto neutral gray before resizing; this prevents fully transparent expansion margins from turning black during RGBA resampling.

## Runtime Modes

Every AI tool runs on FLUX.2 Klein. The UI exposes Quality/Fast presets and keeps technical controls under Settings:

- `Quality` uses FLUX.2 Klein 9B Distilled NVFP4 at four steps and guidance 1.0. Generation uses `Flux2KleinPipeline`; editing uses `Flux2KleinInpaintPipeline`, which receives the actual mask and edit strength. Generation and editing share one set of weights; switching models releases the previous pipeline. 9B keeps its transformer and VAE on the GPU in every mode and streams its text encoder from RAM; the 4B adapters use model CPU offload in Balanced mode.
- `Fast` uses FLUX.2 Klein 4B Distilled NVFP4 with the same settings, at about 60% of 9B's time and under a commercial-friendly licence. It does everything Quality does, including references, sketches and faces, with less faithful identities and scenes.
- Advanced controls → Model also offers Klein 4B BF16, for GPUs without NVFP4 support.

Prompt embeddings are cached without autograd graphs. The [8 October app-provider check](../benchmarks/INPAINT_QUALITY_2026-10-08.md#actual-app-performance-and-the-verified-fix) fixed retained encoder activations: repeated reference edits used about 3.0 GiB of PyTorch allocation, and a changed prompt completed in 4.2 seconds on the target GPU. Cold loading remained about 33 seconds; these are measured small-crop results, not promises for every resolution. Restart the backend after updating to pick up this fix.

The three runtime modes are:

- `Balanced` uses model CPU offload for the Klein 4B models on the default 16 GB profile. Start here.
- `Safe / lower VRAM` always uses model CPU offload for 4B. Use this after an OOM.
- `Performance / GPU resident` keeps 4B entirely on the GPU, for the quickest iteration.

Klein 9B ignores these modes: its transformer and VAE always stay on the GPU and its text encoder streams from RAM.

After a CUDA OOM, restart the backend before benchmarking another mode. PyTorch can hold fragmented allocations after an OOM.

Inpainting runs on a rectangle around the selection, with 320 native pixels of surrounding context by default. The Context toggle shows the part inside the canvas. The model receives a centred crop with the missing picture context mirrored past the canvas edges. Mask coverage continues from the edge into this padding, so a selection touching the canvas does not acquire an artificial soft boundary there. Unselected edges remain protected. The selection bar on the canvas sets Expand, Soften and Surrounding context with a live overlay of the exact mask sent to the model; Scene notes live in generation settings. The model-facing crop is capped at 1024 pixels on its longest edge; the full image is not automatically added as another reference.

The model receives the selection exactly as shown: Expand defaults to +4 px and Soften to an automatic outward ramp of about the longest side / 40. Klein retains this soft mask during sampling, helping original details continue into the selection. The generated crop returns at native resolution with low-frequency colour drift matched to the untouched surroundings. Pixels outside the soft edge stay exact. Select enough space for the complete subject, its shadows and glow: the model cannot reliably fit every requested object inside an arbitrary silhouette. See the [inpainting reference](notes/2026-10-07-inpainting.md) for the experiments and practical controls.

Regional visual quality remains experimental. A paired eight-seed check on the mushroom fixture produced four promising, two weak and two failed-subject results with the current defaults, despite exact protected pixels in every output. The [reliability protocol and full results](notes/2026-10-07-inpainting.md#repeated-seed-reliability-check) distinguish subject compliance and scene integration from pixel protection; they are not a general success rate.

The editor stores opened images as persistent v2 documents. Accepted proposals become revisions, document Undo/Redo traverses those revisions, document names autosave, and documents can be reopened through Library. Existing generation history remains intact and any result can be imported as a document. Use one to four proposals; they run sequentially through the single worker. A proposal is bound to the source revision and cannot be accepted after that revision changes.

While a job is queued, loading or generating, **Cancel** replaces the primary Generate / Remove / Expand action in the instruction dock. The button changes to **Cancelling…** and the editor remains locked until cleanup finishes. Cancellation preserves the current image, prompt and selection; failed cancellation requests leave the action available to retry.

When launched with `start.cmd`, the backend runs under a local supervisor. Cancel first uses the model's cooperative diffusion callback; if native CUDA code does not yield within eight seconds, the worker is replaced automatically so GPU memory is released. The UI tolerates the brief reconnection and the cancelled job disappearing from the replacement worker, then restores the editing controls. A cancelled queued job can finish immediately; a running job reports `cancelling` until its callback and cleanup return.

The Quality app adapter loads the official `black-forest-labs/FLUX.2-klein-9b-nvfp4` distilled transformer with native Comfy Kitchen CUDA kernels and the BF16 VAE. Its 16 GB Qwen3-8B text encoder is trimmed to the 28 layers Klein reads, stored as per-row FP8 in pinned RAM (about 6 GB) and copied to the GPU for each new prompt, which takes about 0.4 s; the transformer and VAE stay resident. Measured through the provider, a 1024 × 1024 generation took 3.2 s and a full-frame 832 × 1216 edit 7.3–7.8 s, at a peak of 11 GiB PyTorch allocation. In the side-by-side comparison, 9B gave cleaner reference inserts, held-object swaps and two-person scenes than 4B at about 1.6× its time; see the [model comparison](../benchmarks/MODEL_COMPARISON_2026-10-09.md).

The Klein 4B NVFP4 adapter, the default until 9 October, loads the official `black-forest-labs/FLUX.2-klein-4b-nvfp4` distilled transformer with native Comfy Kitchen CUDA kernels, plus the original BF16 text encoder and VAE. The NVFP4 transformer and base components use the exact revisions in the benchmark protocol. The BF16 adapter remains available for comparison. Both support generation and native masked editing with one resident pipeline, cancellation, local context and inward feathering. NVFP4 requires a Blackwell GPU, CUDA 13 and the BF16 component profile; unsupported configurations report an explicit error before downloading weights. Across four fixed scenes and three seeds each, median warmed edit time fell from 9.82 to 6.26 s, with similar visual behavior and exact protected pixels. Transformer storage fell from 7.22 to 2.29 GiB; whole-pipeline peak PyTorch GPU allocation stayed near 9.1 GiB because text encoding remains BF16. The app and both benchmarks share the same native loader. Quality uses four steps and guidance 1; inpainting defaults remain strength 1 and context 320 px, with the canvas edge controls described above. See the [full results and limitations](../benchmarks/NVFP4_RESULTS_2026-10-07.md).

## Hot Loading

On start the backend warms up in the background: it loads SAM 2.1 Tiny, then the Quality model (`MOSIMAGE_WARM_UP_MODEL`, FLUX.2 Klein 9B NVFP4 by default), and runs one 256 px, one-step pass for each task so the first edit skips the CUDA kernel set-up. This takes about 25 s for 9B from the local cache; the top bar shows Warming up and then Ready, and `/api/engine` reports the job. Edits submitted meanwhile wait for it and say so. Set `MOSIMAGE_WARM_UP_ON_START=false` to load on first use instead. `/api/models/load` remains for scripted preloading.

Measured on the RTX 5070 Ti: loading Klein 4B NVFP4 takes 5–9 s from the local cache (about 13 s cold); 9B also builds its FP8 encoder, and the first run afterwards adds 3–6 s unless warmed up. Klein's text-to-image and inpaint pipelines share one set of weights, so switching between a new image and an edit is instant. Point selection runs in its own CPU queue and never waits behind a generation.

Prompt encodings are cached per loaded model, so a second proposal or "generate again" skips the text encoder. For 9B that saves its 0.4 s stream; under 4B offload it skips moving the 8 GB encoder to the GPU, which took two proposals with two references from 22 s to about 18 s, and 15 s when repeated. Progress shows the phase, proposal, step and an estimate measured from the step time.

With the default 16 GB profile, MosImage keeps one large model hot at a time. Switching between Quality and Fast unloads the previous provider, and switching between GPU-resident and CPU-offload placement reloads the pipeline.

## Local storage and request limits

SQLite uses WAL and records named, idempotent schema migrations. HTTP file access is restricted to `/files/images/`; the database and other files under `data/` are not served.

Each uploaded image or mask is limited to 64 MiB by `MOSIMAGE_MAX_UPLOAD_BYTES`. The API reads at most the limit plus one byte before returning HTTP 413 for an oversized file. `MOSIMAGE_MAX_IMAGE_PIXELS=64000000` sets Pillow's pixel warning threshold; decoding is refused above twice that value. Completed, failed and cancelled jobs remain queryable for `MOSIMAGE_JOB_RETENTION_SECONDS=600` seconds and are removed on the next status check or submission. Subject isolation shares the CPU worker with object selection.

See [runtime refactor validation, 8 October 2026](notes/2026-10-07-focus-implementation.md#runtime-refactor-validation-8-october-2026) for regression coverage and its limits.

## Desktop Shell

The Tauri desktop app in `frontend/src-tauri` is the same UI in its own window. After installing Rust and dependencies:

```powershell
cd frontend
npm run tauri:dev    # development, with Vite
npm run tauri:build  # mosimage.exe plus MSI and NSIS installers under src-tauri/target/release
```

The app starts the engine itself from `backend/.venv` in the checkout it was built from (override with `MOSIMAGE_ROOT`), without a console window; output goes to `data/logs/engine.log`. If an engine is already listening on port 8000, for example from `start.cmd`, the app uses it and leaves it running. Like `backend/supervisor.ps1`, the app restarts its engine after a hard cancel or crash. On close it asks the engine to cancel jobs and unload models through `/api/shutdown`, guarded by a per-launch token, and ends it after 20 s if needed. A Windows job object ends the engine as well if the app itself crashes, so no Python process keeps RAM or GPU memory.

For everyday use, start the app from a desktop shortcut to `frontend\src-tauri\target\release\mosimage.exe` instead of installing the MSI. `npm run tauri:build` replaces that file in place, so the shortcut always opens the newest build:

- Backend changes apply the next time the app starts, without a rebuild, because the engine runs from the checkout.
- Frontend changes are embedded in the executable, so they need `npm run tauri:build`.
- Close MosImage before building. Windows locks a running `mosimage.exe`, and the build cannot replace it.

To create the shortcut:

```powershell
$exe = "$PWD\frontend\src-tauri\target\release\mosimage.exe"
$shortcut = (New-Object -ComObject WScript.Shell).CreateShortcut("$([Environment]::GetFolderPath('Desktop'))\MosImage.lnk")
$shortcut.TargetPath = $exe; $shortcut.WorkingDirectory = Split-Path $exe; $shortcut.IconLocation = "$exe,0"; $shortcut.Save()
```

Run it from the repository root.

The engine is not bundled into the installer: Python with CUDA PyTorch, Diffusers and the NVFP4 kernels is several gigabytes, and the models download to the Hugging Face cache on first use.
