# MosImage guide

Status: Current reference
Last verified: 2026-10-11

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

When upgrading an existing `backend/.env`, remove `MOSIMAGE_INPAINT_MASK_GROWTH` and `MOSIMAGE_INPAINT_COMPOSITE_FEATHER` if present. These retired settings are rejected by the current configuration parser; selection growth and softness now come from the canvas controls. `start.cmd` creates `.env` only when it is missing and does not rewrite an existing file.

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

### macOS (Apple silicon)

Fast path from the repository root:

```bash
./start.sh
```

It does what `start.cmd` does: it creates `backend/.venv` with Python 3.13 or 3.12 (`python3.13` from Homebrew, or uv when that is installed), installs `requirements-torch-macos.txt` and `requirements.txt`, checks that PyTorch can use Metal, installs the frontend dependencies and starts the supervised engine and Vite. `./start.sh --no-install` skips the checks. Ctrl+C, closing the terminal or a plain `kill` stops both and waits for the engine to unload; the engine also exits by itself if the script is killed outright (`MOSIMAGE_WATCH_PID`). A second `./start.sh` stops again when the ports are taken, and leaves the first one running.

Manual setup:

```bash
cd backend
python3.13 -m venv .venv
.venv/bin/python -m pip install -r requirements-torch-macos.txt
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

On macOS the defaults switch to the Apple silicon profile (`MOSIMAGE_PREFERRED_DEVICE=mps`, `MOSIMAGE_HARDWARE_PROFILE=apple-silicon`); no `.env` setting is needed. NVFP4 needs Blackwell tensor cores, so a Mac runs the original BF16 weights the NVFP4 builds were quantized from, on the GPU through Metal (MPS), or quantizes them in MLX:

- `Quality` is FLUX.2 Klein 9B (`flux2-klein-9b`, `black-forest-labs/FLUX.2-klein-9B`). Its Qwen3-8B text encoder loads without the layers Klein never reads.
- `Fast` is FLUX.2 Klein 4B (`flux2-klein-4b`), which also runs Specialized outpaint.
- Consistency and the detail enhancer work with Quality, as on Windows.
- CPU and GPU share one memory, so every component stays resident and the memory usage setting is hidden: offloading would only copy weights around. Switching between Quality and Fast releases the other.
- `MOSIMAGE_APPLE_WEIGHTS` sets how the transformer and text encoder hold their weights. `bf16` runs them in PyTorch on MPS. `q8` and `q4` run them in MLX, quantized to 8 or 4 bits at load. Masked edits, references and every LoRA work the same in all three, and the model menu shows the choice ("FLUX.2 Klein 9B (8-bit)"). Left empty, it follows unified memory: BF16 from 64 GB, 8-bit from 48 GB, 4-bit below that.
- Quality warms up at start when its peak fits with about 12 GB to spare (BF16 from 64 GB, 8-bit from 48 GB, 4-bit from 36 GB); otherwise Fast warms up, and you should choose Fast in the editor as well. `MOSIMAGE_WARM_UP_MODEL` overrides the choice.
- After every edit the engine hands PyTorch's and MLX's GPU caches back to macOS, so memory use falls back between jobs.
- `PYTORCH_ENABLE_MPS_FALLBACK=1` is set before PyTorch loads, so an operator Metal lacks runs on the CPU instead of failing.

Memory on an M5 Max, as Activity Monitor counts it (the [MLX comparison](../benchmarks/MLX_KLEIN_2026-10-11.md)):

| Weights | Quality: after load / peak | Fast: after load / peak | Time against BF16 | Pictures against BF16 |
|---|---|---|---|---|
| BF16 | 30 / 36–42 GiB | 17 / 23–27 GiB | | |
| 8-bit (q8) | 19 / 24–30 GiB | 11 / 19 GiB | 4–13% longer | the same; fine details differ about as much as between two BF16 runtimes |
| 4-bit (q4) | 13 / 18–24 GiB | 8 / 16 GiB | 2–6% longer | the same scene, more changed detail |

Measured warm through the API on an M5 Max (40-core GPU, 128 GB), macOS 27, PyTorch 2.14.1, with BF16 weights:

| | Quality 9B | Fast 4B |
|---|---|---|
| Generate 1024 × 1024 | 7.0 s | 3.7 s |
| Edit the whole 832 × 1216 picture | 14.5 s | 7.6 s |
| Replace a selection | 11.0 s | 5.8 s |
| Load and warm up from the local cache | 10–25 s | 5 s |

Float16 ran no faster than bfloat16 on the M5 Max, so the profile keeps bfloat16.

### Hugging Face login

FLUX.2 Klein 9B, the Quality model, is gated on Hugging Face: open [its model page](https://huggingface.co/black-forest-labs/FLUX.2-klein-9B) while logged in and accept the licence (approval is automatic), then log in once on this machine:

```powershell
cd backend
.\.venv\Scripts\hf.exe auth login   # macOS: .venv/bin/hf auth login
```

The other models are public; for them a login only raises the download limits. Use a read-only token. Hugging Face stores it in the normal user cache and the model loaders automatically use that login. `HF_TOKEN` or `MOSIMAGE_HF_TOKEN` (in `backend/.env`) is also supported; never commit a real token.

## Frontend

```powershell
cd frontend
npm ci
npm run dev
```

Open `http://localhost:5173`. The frontend expects the API at `http://127.0.0.1:8000` unless `VITE_API_BASE_URL` is set.

The English Focus layout gives most of the viewport to the image in every mode. A compact tool rail and command dock stay visible; Generation settings and Canvas size open modal panels, selection refinement opens a contextual popover, and Library opens a separate drawer. Results replace the command dock with a small review strip; compare them with the original, then Apply or Discard. Fit, zoom, open and document undo/redo remain directly accessible. The engine indicator shows connection state; `/api/system` exposes the detailed hardware profile.

Library can delete saved documents and generation history entries after confirmation. Deleting a document removes its revisions, alternatives, references and unused layer images/masks, including assets kept only by older revisions. Assets shared with another document's revision history stay available for undo/redo. Deleting a generation retains input and mask files still used by another generation; imported documents and proposals keep their own copies.

Object selection runs SAM 2.1 Tiny on CPU and caches one image embedding. Click to include a point, Alt+click to exclude a point, or use Brush/Eraser/Rectangle/Lasso to correct the mask. The first selection downloads the model. Ctrl+Z/Shift+Ctrl+Z undo/redo the mask first when its history is available, then document revisions. The tool keys follow Photoshop: W Object (quick selection), B Brush, E Eraser, L Lasso, M Box, V or Ctrl+T Transform, H Pan, C Crop, Shift+C Expand. Ctrl+D deselects, Ctrl+Shift+I inverts and Shift+Backspace puts the cursor in the prompt to fill the selection. Ctrl+O opens an image and Ctrl+Enter submits an instruction. A focused slider keeps only its arrow, Home/End and Page keys, so the other shortcuts still work after dragging one.

A selection gets a floating bar beside it, like Photoshop's contextual task bar: **Generate** puts the cursor in the prompt, **Remove**, **Faces** and **Enhance** start at once, **To layer** copies the selection to a layer, **Mask “layer”** turns it into a mask for the layer you last worked on, and the last two buttons invert or drop the selection. The mode follows the selection: making one switches Generate or Edit to the last regional mode (Replace or Remove), and dropping it switches Replace or Remove to Edit on the whole picture.

References place a person, pet or object from a concept image into the open document. Drop or browse images into the References panel on the right; each document keeps up to 24. Pick up to three (a single upload is picked automatically; the In use badge shows the order), select where they go, then describe what they do, for example "Anna and Tom sitting on the bench". A reference is named after its file, so "Anna.jpg" becomes Anna and can be named in the prompt. Code-like file names from cameras, chat apps and generators, such as `IMG_1234` or `T5S8DAP28-7b1f1113fc48`, become "Reference 1", "Reference 2" and so on, and the prompt then treats that reference as an unnamed subject. References work with every registered Klein model. Picking one keeps your chosen model and proposal count; Quality gives more faithful identities in the local comparison. Replace passes them to the inpaint pipeline as `image_reference`; Generate and Edit pass them as extra conditioning images.

References also work without a picture. On the start screen, the References panel keeps up to 24 loose references; pick up to three and describe the scene, for example "Anna and Tom riding a roller coaster together". Klein composes a new image from them at the chosen format. References belong to one project: whatever starts the next document (Apply, Open or Blank canvas) moves the start screen's references into it, so the next new image starts with an empty panel. Removing and adding references only waits while the engine works; picking one waits until a proposal is applied or discarded. Phone photos are turned upright once on opening, following their EXIF orientation, so every tool sees the picture the way it is shown. The scissors button optionally isolates the subject with SAM 2.1 Tiny on a neutral background; the full image is used by default because it was more reliable in the local trial. Several references are padded to one shared size of about 0.36 MP each and named in order in the prompt ("Anna is reference image 1"), so each likeness stays separate. See [the reference trial](notes/2026-10-07-inpainting.md#reference-images).

Layers make up the picture, separate from references. The picture you open is the bottom layer, the Background, and an ordinary layer like the rest, as when you turn Photoshop's background into a layer: hide, move, mask, merge or delete it. Add PNG, JPG or WebP files from the Layers tab, drop them on it, paste with Ctrl+V while a picture is open, or copy the current selection with Ctrl+J. With Transform (V or Ctrl+T), drag to move, corners scale (Shift stretches freely), edges stretch, and the knob rotates (Shift snaps to 15°); arrows nudge by 1 px, or 10 px with Shift. Object, Lasso and Box select on the visible picture, so picking one with a layer selected deselects the layer. Brush and Eraser (B and E) paint the selected layer's mask instead, as in Photoshop; press Esc first to select on the picture. Layer masks work as in Photoshop. **Add mask** in the panel adds a mask that shows the whole layer, or one that hides it all with Alt+click; with a selection on the picture it becomes **Mask from selection**, which follows the layer's position, scale and rotation. A masked layer shows the mask thumbnail beside its own: click it to paint the mask, Shift+click to switch the mask off (crossed out in red) and Alt+click to view the mask alone in grey. While masking, the same tools edit the mask: Object cuts out what you click with SAM 2.1 Tiny on the layer's own image, Eraser hides, Brush paints its tone from white (shows) to black (hides) with any grey between (Alt paints black and makes the Eraser restore; X swaps white and black), and Lasso/Box hide or restore an area. The mask brush has a Layer mask bar of its own under the toolbar, with Tone, Softness and Opacity. One stroke never goes past its opacity, however often it crosses a pixel, and a grey stroke at full opacity sets the mask to that grey, whatever was there. `\` tints the hidden parts red. Clicking the layer thumbnail, Esc or Done painting ends masking. The panel below the list has two groups: Layer sets opacity, flips, rotates, resets, duplicates or deletes the layer and blends it in; Mask paints, switches off, views, inverts or deletes the mask, and Remove background makes one. The list sets visibility, order and name. Every layer change is a document revision, so Ctrl+Z steps through it. Layer images are stored as RGBA PNG under `data/images/layers/`, with masks as greyscale PNG; a document holds up to 40. Export flattens the visible layers as PNG; a Background on its own goes out as it came in.

Click a layer to select it, Ctrl+click to add or remove one, and Shift+click to select the layers in between; selected layers hide, show, delete and merge together. Right-click a layer, or use the ⋯ button above the list, for the layer menu:

- **Merge down** (Ctrl+E) merges the layer onto the next visible layer below, which keeps its place and name. With several layers selected, Ctrl+E merges the visible ones at the top one's place.
- **Merge visible** (Ctrl+Shift+E) merges every visible layer at the bottom one's place and leaves hidden layers alone.
- **Flatten image** turns everything into one Background the size of the canvas and discards hidden layers; Ctrl+Z brings them back.
- **Select pixels**, or Ctrl+click on a layer thumbnail, selects where the layer shows, soft edges included, to fill, mask or copy exactly that area. It is also in the Layer group below the list, beside Merge down.

A merged layer shows exactly what its parts showed, without a mask and at full opacity. It keeps pixels up to half a canvas past each edge and is stored at most 8192 px a side. Blend in and Edit on a layer keep the layer they replace hidden just below the result, marked **Original of …** in the list. Deleting the result shows the original again; merging or flattening the result lets the original go.

Edit changes the whole picture with an instruction, such as "make it a snowy night" or "turn it into a watercolour", without a selection. A newly opened picture starts in Edit with the Object tool ready: type an instruction for the whole picture, or click an object and the dock switches to Replace. The visible picture, including every visible layer, is scaled to about one megapixel in multiples of 16 and passed to Klein as image 1; picked references follow as images 2 and up and are named in the prompt. The prompt asks Klein to keep faces and everything the instruction does not mention. Proposals are scaled back to the canvas size, so Apply adds a layer that covers the canvas exactly. Once a document has layers, Edit offers **All visible** or **Current layer**. Current layer edits only the selected layer, through its mask, with transparent parts on neutral grey. The result keeps the layer's transparency, mask, position, scale, rotation and opacity. Apply puts it above the original and hides the original, like Blend in. Choosing Edit with a layer selected starts on that layer, and picking another layer in the panel switches the target. Detail above one megapixel is redrawn at that resolution, so very large pictures come back softer than the original; use Replace for a local change that keeps every original pixel. Edit uses the chosen Quality or Fast Klein model.

Sketch (P, or Sketch in the command dock) paints rough coloured strokes on their own drawing over the picture, with ten colours plus any colour, size, Draw/Erase (or Alt), Undo stroke and Clear. Describe what you drew and press Render. **Only the sketch** redraws the strokes plus a margin of about 2.5% of the longest side, fading out over another 2.5%, with Klein's inpaint pipeline at strength 1 and the sketched crop as its reference image. In the local trial this followed a crude deer and top hat closely without leaving stroke remnants; strength 1 or 0.85 without the reference left a sketch stub. The rest of the picture keeps every pixel, and Apply adds the patch as a layer. **Whole picture** sends the sketched picture through Edit, so light and style match around it, and Apply adds a full-canvas layer. **Blank canvas** on the start screen opens a white document at the format in generation settings, ready to sketch with Whole picture. Sketch uses the chosen Klein model and ignores references; Ctrl+Z undoes strokes while the pencil is active.

Faces finds human faces with Grounding DINO tiny (the query is `human face.`, so pets are skipped) and redraws each one sharper with Klein. The detector also calls cakes, signs and other objects a face, with scores as low as small, blurry real faces, so a face only counts when a detected `person.` holds most of it and is at least 1.5 times its size. A square crop of about 2.2 times the face renders at 768 × 768 and scales back; its average colour is matched to the original, and a feathered ellipse pastes only the face, so hair and background stay. Choosing Faces shows every face found on the canvas: the real ones picked in mint, the rest dashed. Click a face to leave it out or add it; up to eight faces are fixed at a time, largest first, and with a selection only faces centred inside it start out picked. Faces from the selection bar skip the canvas and fix the real faces in the selection. The prompt is optional and adds to the built-in restoration prompt, and a picked reference makes the face match that person. Apply adds only the faces as a layer, cut to their shared mask. In the local trial, five faces in a 512 px pub photo came back visibly sharper and the dog was skipped, in 38 s including model loading.

Faces offers two ways to redraw. **Faithful**, the default, adds noise to the face crop and runs only the later steps (twice the steps at strength 0.625, so five model passes from noise level 0.93), with the detail-enhancer LoRA at 0.4 on Quality. Expression, lighting and identity stay; small, blurry faces gain less sharpness. **Redraw** draws the face anew from its crop, as before: sharper on faces of a few dozen pixels, but the model then has to invent the expression and the light. In the trial on four faces, Redraw turned a neutral 95 px face under blue light into a smiling, evenly lit one, while Faithful kept both; on large faces the two looked alike. The built-in prompt no longer names teeth, which made Klein open closed mouths into a grin in either mode. Faithful takes about 4.5 s per face on Quality, Redraw about 3 s.

Enhance adds fine detail, such as pores, knit and foliage, to the whole picture or to a selection. It uses [Elusarca's Detail Enhancer](https://huggingface.co/reverentelusarca/detail-enhancer-flux-klein-9b) LoRA for Klein 9B with its own prompt; the optional prompt adds to it. **Subtle** sets the LoRA to 0.4, which kept colours closest to the original in the A/B test; **Strong** sets 0.7, which adds more texture but can reinvent small details such as a face's shape or a marble pattern. Each part renders at about one megapixel as a partial redraw (strength 0.625), and the broad tones of the original are restored under the new detail, so colour and light do not drift. A picture of up to about 1.1 megapixels renders in one pass; a larger picture or selection is split into up to nine overlapping tiles of at most 1024 × 1024 that blend across 128 px, so every pixel keeps at least its own resolution. A region larger than about 2800 × 2800 pixels asks for a smaller selection. Each part takes about ten seconds: in the local trial, a 1254 × 1254 generated picture took 20 s in two parts and a 2000 × 2400 selection on a 12-megapixel photo took 82 s in eight. A full redraw instead of the partial one doubled the colour drift and left double edges where tiles overlap. Apply adds the result as a layer: covering the canvas without a selection, or cut to the selection. Enhance always uses the Quality model and ignores references. The LoRA is under the FLUX.1 [dev] non-commercial licence, like the 9B weights, and downloads (632 MB, checksum-verified) on first use.

With Object, **Find…** selects everything matching a short English description, such as "the hat" or "dog, beer glass". Grounding DINO tiny finds the boxes and SAM 2.1 Tiny turns each into a mask, joined into one selection. Matching is loose: in the trial, "hat" selected bunny ears and "beer glasses" also caught eyeglasses. **Subject** selects the main people and animals (the strongest large detection, plus others at least a quarter of its size) and refines the edge with ViTMatte small, so hair and fur keep soft edges. **Remove background** in the layer panel does the same on the layer's own image and sets the result as its layer mask; while masking a layer, Find and Subject keep only the match on that layer. These run on CPU; Grounding DINO (about 0.7 GB) and ViTMatte (about 0.1 GB) download on first use and stay loaded.

In an open document, Generate creates a full image without requiring a selection. Apply adds it as a new layer, centred and fitted to the canvas without changing its aspect ratio; the preview uses the same placement. New still starts a separate document with its first image as the Background layer. Replace and Remove always add the selected proposal as a new layer, including when the layer stack is empty or all layers are hidden. They work on the picture as you see it, including every visible layer. The selected patch is cropped to the mask's footprint; the result already contains the feathered blend, so its layer mask retains the footprint without blending the edge twice. Apply selects the result with Transform and opens Layers. With several proposals, Apply keeps them all in that one layer as variants, like Photoshop's generative layers: the thumbnail shows which one is visible (2/3), `,` and `.` switch the selected layer's variant, and the layer panel shows each variant to click, with **To layers** to give every variant a layer of its own. **All as layers** in the proposal tray does that at once, keeping the other proposals as hidden layers. Each switch is an undo step. Hide, delete, opacity, undo/redo and reopening leave the other layers as they were. A failed layer save leaves the proposal available for retry. Blend in keeps its lower-strength edit and places the blended copy above the original, hiding that original. Expand continues the picture as you see it, so the new edges match the visible layers. The layers stay separate and move with the content: a Background that covers the canvas grows into the new edges and keeps its own pixels under the other layers; otherwise the new picture goes under them as a layer named Expanded. See [AI layer validation, 9 October](notes/2026-10-07-focus-implementation.md#ai-results-as-layers-9-october-2026).

Expand supports draggable edges, format presets, and exact pixel margins. It adds space without cropping, and only a 16 px band along the original edge blends into the new area. The native canvas is capped at 4096 × 4096; the model-facing crop has an area budget of about one megapixel (1024²), with its longest side capped at 2048 px. Crops below 768² pixels are enlarged for inference and scaled back. Applying an expansion updates document dimensions; undo/redo and reopening restore the corresponding revision dimensions.

Crop (C, as in Photoshop) resizes the canvas without AI. Shift+C now opens Expand. Crop shows the picture as you see it, with its visible layers, and a box with rule-of-thirds lines.

- **Edges and corners:** drag them in to cut. Drag them past the picture to add space, which stays transparent; dragging inside moves the box.
- **Ratios:** Free, Original, 1:1, 4:5, 3:2, 16:9 and 9:16, applied from the largest centred crop.
- **Size:** exact width and height fields resize around the centre.

Enter or **Crop** applies and Esc cancels. The canvas becomes a new revision of 16 to 4096 px per side, which must keep part of the picture. Space past the picture stays transparent. Layers, the Background among them, keep their pixels past the new edges and move with the content, and Ctrl+Z restores the old canvas.

Expand normally runs on the chosen Klein model: 9B for Quality, 4B for Fast. Describe what the new area should show, such as "more beach and sea"; Klein is asked to extend the scene seamlessly, without a frame or border. Its generation mask reaches 32 pixels into the original at model resolution, keeping useful overlap even when a large canvas is downscaled. The final blend still changes only a 16 px band at native resolution. MosImage matches low-frequency colour using untouched context outside the generation footprint, carries the edge band's tone offset into the new area and crossfades across it. Every original pixel beyond the final band stays exact. Structural joins and texture can still vary by seed; see the [paired mask comparison](../benchmarks/EDIT_MASK_RESULTS_2026-10-09.md) for outputs and timings. Settings also offers the optional Specialized outpaint route described below.

Transparent images stay RGBA in documents and protected compositing. Model input is flattened onto neutral gray before resizing; this prevents fully transparent expansion margins from turning black during RGBA resampling.

## Runtime Modes

Every AI tool runs on FLUX.2 Klein. The UI exposes Quality/Fast presets and keeps technical controls under Settings:

- `Quality` uses FLUX.2 Klein 9B Distilled NVFP4 at four steps and guidance 1.0. Generation uses `Flux2KleinPipeline`; editing uses `Flux2KleinInpaintPipeline`, which receives the actual mask and edit strength. Generation and editing share one set of weights; switching models releases the previous pipeline. 9B keeps its transformer and VAE on the GPU in every mode and streams its text encoder from RAM; the 4B adapters use model CPU offload in Balanced mode.
- `Fast` uses FLUX.2 Klein 4B Distilled NVFP4 with the same settings, at about 60% of 9B's time and under a commercial-friendly licence. It supports generation, editing, references, sketches, faces and expansion, with less faithful identities and scenes. Enhance always uses Quality; the detail-enhancer and Consistency adapters are also 9B-only.
- Advanced controls → Model also offers Klein 4B BF16, for GPUs without NVFP4 support.

Prompt embeddings are cached without autograd graphs. The [8 October app-provider check](../benchmarks/INPAINT_QUALITY_2026-10-08.md#actual-app-performance-and-the-verified-fix) fixed retained encoder activations: repeated reference edits used about 3.0 GiB of PyTorch allocation, and a changed prompt completed in 4.2 seconds on the target GPU. Cold loading remained about 33 seconds; these are measured small-crop results, not promises for every resolution. Restart the backend after updating to pick up this fix.

The three runtime modes are:

- `Balanced` uses model CPU offload for the Klein 4B models on the default 16 GB profile. Start here.
- `Safe / lower VRAM` always uses model CPU offload for 4B. Use this after an OOM.
- `Performance / GPU resident` keeps 4B entirely on the GPU, for the quickest iteration.

Klein 9B ignores these modes: its transformer and VAE always stay on the GPU and its text encoder streams from RAM.

After a CUDA OOM, restart the backend before benchmarking another mode. PyTorch can hold fragmented allocations after an OOM.

Inpainting runs on a rectangle around the selection, with 320 native pixels of surrounding context by default. The Context toggle shows the part inside the canvas. The model receives a centred crop with the missing picture context mirrored past the canvas edges. Mask coverage continues from the edge into this padding, so a selection touching the canvas does not acquire an artificial boundary there. Unselected edges remain protected. The selection bar on the canvas sets Expand, Soften and Surrounding context with a live overlay of the final blend selection; Scene notes live in generation settings. The model-facing crop gets an area budget of about one megapixel (1024², with the long side at most 2048), so a wide crop keeps as much detail as a square one. Crops smaller than 768 × 768 are rendered at that size and scaled back, because Klein draws flat, plastic detail at low resolution; this costs about a second on small selections. The source crop is always the model's reference image, the same as ComfyUI's ReferenceLatent plus latent noise mask.

Expand and Soften scale with the picture: Expand about the longest side / 200 (4–24 px; double-click the slider to return to automatic) and Soften an outward ramp of about the longest side / 150 (6–32 px). A wide soft ramp smeared the old content's colour across the new edge; a little growth with a short ramp keeps edges crisp without leaving a rim. Generation and final blending use separate masks: Klein gets full coverage wherever the selection has nonzero alpha, with resampling coverage retained when resizing; the final composite uses the displayed soft alpha. This lets the model replace the source throughout the editable area while keeping the chosen transition in the finished image. The generated crop returns at native resolution with low-frequency colour drift matched to the untouched surroundings. Klein regenerates one extra 16 px token around the selection, hidden by the final blend, so tokens the edge cuts through do not keep old colour. That ring also measures how the new content meets the old: small tone offsets, such as a removed object's shadow, fade into the fill, while different content is ignored. Finally the photo's fine grain, which the model and resizing smooth away, is measured outside the selection and added back as up to 5 levels of monochrome noise. Pixels outside the soft edge stay exact. A soft final blend can still retain old colour at an object's outline. Select enough space for the complete subject, its shadows and glow: the model cannot reliably fit every requested object inside an arbitrary silhouette. See the [paired mask comparison](../benchmarks/EDIT_MASK_RESULTS_2026-10-09.md) and [inpainting reference](notes/2026-10-07-inpainting.md) for results and practical controls.

Regional visual quality remains experimental. A paired eight-seed check on the mushroom fixture produced four promising, two weak and two failed-subject results with the current defaults, despite exact protected pixels in every output. The [reliability protocol and full results](notes/2026-10-07-inpainting.md#repeated-seed-reliability-check) distinguish subject compliance and scene integration from pixel protection; they are not a general success rate.

Generation settings → Optional refinements offers three switches, all off initially:

- **Consistency** uses exactly `Flux2-Klein-9B-consistency-V2.safetensors` at weight 0.6, without a trigger word. It is available for Quality Edit, Replace and ordinary Expand. It can hold colours and framing closer, but can soften larger requested changes. Fast disables this switch without losing your preference.
- **Specialized outpaint** appears in Expand. It temporarily uses distilled Fast 4B NVFP4, four steps and guidance 1, with the FAL adapter at weight 1.1 and its green-border instruction. Model and sampling controls are locked while enabled; disabling restores your earlier settings. It renders the full expanded canvas at about one megapixel, then applies the same protected core and 16 native-pixel seam blend. The published 50-step Base route took 157–180 seconds and is absent from the editor. The measured Fast experiment took about 6–7 seconds warm; first model/adapter loading adds time.
- **Fine detail** appears in Replace with a selection, except Blend in. It sends the selected crop and surrounding context through image-conditioned Edit at about one megapixel, then composites with your displayed selection alpha. It uses full strength while enabled and restores your previous strength when disabled. Scene notes, references, colour matching and exact protection outside the selection remain active. Smaller Surrounding context can spend more generated detail on the selected material; inspect the Context preview. It can invent texture or change local colour. The bounded study used 96 px context and took about 8 seconds warm.

Whole-picture Edit also renders around one megapixel and resizes to the original dimensions. A larger export therefore adds no generated detail; Enhance renders large pictures in tiles instead. The [LoRA/outpaint/detail evaluation](../benchmarks/QUALITY_CANDIDATES_2026-10-09.md) retains the visual tradeoffs; these switches are optional comparisons. Their preferences last for the current app session. The first enabled LoRA job downloads a pinned, checksum-verified public checkpoint into `models/adapters/` if missing. Switching a LoRA off sets its residual to zero; switching models or unloading releases it. Consistency and the detail enhancer patch the same layers, so using one detaches the other, which costs about a second when switching. The first specialized job may reload an already loaded Fast model once to attach the adapter before CPU-offload hooks.

The editor stores opened images as persistent v2 documents. Accepted proposals become revisions, document Undo/Redo traverses those revisions, document names autosave, and documents can be reopened through Library. Existing generation history remains intact and any result can be imported as a document. Use one to four proposals; they run sequentially through the single worker. A proposal is bound to the source revision and cannot be accepted after that revision changes.

While a job is queued, loading or generating, **Cancel** replaces the primary Generate / Remove / Expand action in the instruction dock. The button changes to **Cancelling…** and the editor remains locked until cleanup finishes. Cancellation preserves the current image, prompt and selection; failed cancellation requests leave the action available to retry.

Completed, failed and cancelled jobs freeze their elapsed time at completion, so later status checks do not inflate generation time.

When launched with `start.cmd`, `start.sh` or the desktop app, the backend runs under a local supervisor. Cancel first uses the model's cooperative diffusion callback; if native CUDA code does not yield within eight seconds, the worker is replaced automatically so GPU memory is released. The UI tolerates the brief reconnection and the cancelled job disappearing from the replacement worker, then restores the editing controls. A cancelled queued job can finish immediately; a running job reports `cancelling` until its callback and cleanup return.

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

On a Mac, `npm run tauri:build:mac` builds `MosImage.app` and installs it in `/Applications`; open it from Launchpad, Spotlight or the Dock. Run `./start.sh` once first, because the app uses `backend/.venv`. The same rules apply as on Windows: backend changes need only a restart of the app, frontend changes a rebuild. Quit with Cmd+Q, which unloads the models; if the app crashes or is force-quit, the engine notices that its parent is gone and exits by itself (`MOSIMAGE_EXIT_WITH_PARENT`). The app finds the engine through the checkout it was built from, so rebuild it after moving the repository. The window lets the page handle dropped files, so images can be dragged onto the canvas, and the standard Edit menu provides Cut, Copy, Paste and Select All in text fields. Shortcuts use Cmd where Windows uses Ctrl, and the labels show ⌘ and ⌥.

For everyday use on Windows, start the app from a desktop shortcut to `frontend\src-tauri\target\release\mosimage.exe` instead of installing the MSI. `npm run tauri:build` replaces that file in place, so the shortcut always opens the newest build:

- Backend changes apply the next time the app starts, without a rebuild, because the engine runs from the checkout.
- Frontend changes are embedded in the executable, so they need `npm run tauri:build`.
- Close MosImage before building. Windows locks a running `mosimage.exe`, and the build cannot replace it.

In the app, Export (Ctrl+S) opens a Save As dialog through `tauri-plugin-dialog`. The image goes to the `save_image` command as raw bytes, and a notice shows where the file was saved. In a browser, Export downloads the file instead.

To create the shortcut:

```powershell
$exe = "$PWD\frontend\src-tauri\target\release\mosimage.exe"
$shortcut = (New-Object -ComObject WScript.Shell).CreateShortcut("$([Environment]::GetFolderPath('Desktop'))\MosImage.lnk")
$shortcut.TargetPath = $exe; $shortcut.WorkingDirectory = Split-Path $exe; $shortcut.IconLocation = "$exe,0"; $shortcut.Save()
```

Run it from the repository root.

The engine is not bundled into the installer: Python with CUDA PyTorch, Diffusers and the NVFP4 kernels is several gigabytes, and the models download to the Hugging Face cache on first use.
