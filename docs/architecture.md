# Architecture

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
3. The job runs in `services.py` (generation, edit, inpaint, outpaint) or a specialised module (`faces.py`, `selection.py`).
4. Services call an `ImageModelProvider` from `models/registry.py` and store results through `storage.py`.
5. The UI polls `GET /api/jobs/{id}` until the job completes, fails or is cancelled.

### Routers (`routes/`)

| Module | Prefix | Endpoints |
|---|---|---|
| `generation.py` | `/api` | generate, edit, inpaint, outpaint, faces, generation history |
| `selection.py` | `/api` | point selection, select by text, subject cut-out |
| `jobs.py` | `/api/jobs` | job status and cancellation |
| `documents.py` | `/api/v2` | documents, revisions, layer assets, layers, AI alternatives, undo/redo |
| `references.py` | `/api/v2` | loose and per-document references, subject isolation |
| `common.py` | | shared helpers: bounded upload reads, parameter parsing, capability and reference checks, `http_errors()` for mapping exceptions to status codes |

`main.py` keeps only the app setup: CORS, static file serving for `/files/images/`, health, system info, engine warm-up, model load/unload and shutdown.

### Jobs

`jobs.py` runs two single-worker lanes: one for GPU work (generation and editing) and one for CPU work (SAM selection, Grounding DINO, ViTMatte). Selecting therefore never waits behind a render. Each job gets a `JobControl` for progress and cooperative cancellation. Finished jobs are retained for `MOSIMAGE_JOB_RETENTION_SECONDS`. Under `start.cmd` or the desktop app, a supervisor restarts the engine when a cancelled CUDA call does not yield within eight seconds.

### Model providers (`models/`)

Every runtime sits behind the `ImageModelProvider` protocol in `base.py`. `registry.py` keeps one large pipeline resident at a time to fit 16 GB of VRAM.

| Provider | File | Role |
|---|---|---|
| FLUX.2 Klein 9B NVFP4 | `flux2_klein_nvfp4.py`, `klein_text_encoder.py` | Quality default: generation, editing, references, sketch, faces |
| FLUX.2 Klein 4B NVFP4 / BF16 | `flux2_klein.py` | Lighter and commercially licensed alternatives |
| Z-Image Turbo | `z_image.py` | Fast mode and canvas expansion |
| FLUX.1 Fill Dev | `flux_fill.py` | Optional specialist; not ready for the 16 GB profile |

`nvfp4_runtime.py` loads the native CUDA NVFP4 transformer, shared with the benchmark tools.

### Image processing

| Module | Purpose |
|---|---|
| `compositing.py` | Context crops, soft masks, colour-drift correction and protected compositing for inpainting |
| `selection.py` | SAM 2.1 Tiny point selection, text selection and subject cut-outs |
| `detection.py` | Grounding DINO tiny text-to-box detection |
| `matting.py` | ViTMatte edge refinement for hair and fur |
| `faces.py` | Face detection, crop, re-render and blend for Fix faces |

### Storage

- `storage.py` runs SQLite with WAL and named migrations, and records generation history.
- `documents.py` stores documents, revisions, AI alternatives, undo and redo. A stale base revision raises `RevisionConflict`, which the API returns as HTTP 409.
- `layers.py` stores layer images (RGBA PNG) and masks (greyscale PNG).
- `references.py` stores reference images and their optional isolated subjects.

All files live under `data/`. Only `data/images/` is served over HTTP.

## Frontend (`frontend/src/`)

`App.tsx` composes the editor out of hooks and components; it holds no model or canvas logic itself.

| Folder | Contents |
|---|---|
| `api.ts` | Typed client for every endpoint |
| `hooks/` | State and side effects: engine connection and warm-up (`useEngine`), job submission and polling (`useJobRunner`), the visible picture and exports (`usePicture`), selection tools (`useSelectionTools`), layers, references, result preview and keyboard/paste handling |
| `generation/` | Pure logic: what a command submits (`submission.ts`), when it can run and what the hint says (`commandState.ts`), operations, presets, aspect ratios and progress text |
| `canvas/` | Editor canvas, viewport, mask and sketch layers, expansion canvas and the selection bar |
| `layers/` | Layer stack, transform geometry and bitmap helpers |
| `masks/` | Mask operations and refinement. The signed distance field for soft edges runs in a Web Worker (`maskWorker.ts` via `fieldWorker.ts`), with a synchronous fallback |
| `components/` | Panels, dialogs, top bar, tool rail, proposal tray and library |

Most of the logic in `generation/`, `masks/` and `layers/` is pure and covered by Vitest tests next to each module.

## Desktop shell (`frontend/src-tauri/`)

The Tauri app starts the engine from `backend/.venv` without a console window and logs to `data/logs/engine.log`. If an engine is already running on port 8000, the app reuses it. It restarts the engine after a crash or hard cancel. On close it shuts the engine down through `/api/shutdown`, using a per-launch token. A Windows job object also ends the engine if the app itself crashes.
