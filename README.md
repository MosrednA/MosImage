# MosImage

A local AI image editor. Generate pictures, edit them with plain-language instructions, select objects with a click, and keep every result as a layer you can move, mask or undo. Everything runs on your own GPU: no cloud and no account.

Built for one NVIDIA RTX 5070 Ti (16 GB), with FLUX.2 Klein 9B in NVFP4 as the default model.

## Features

**Create**
- **Generate** an image from a prompt. In an open document the result is added as a new layer.
- **References**: put people, pets or objects from your own photos into a new scene ("Anna and Tom riding a roller coaster"), up to three at once.
- **Sketch** rough coloured strokes and describe them. MosImage renders just the sketch, or the whole picture around it.

**Edit**
- **Replace** or **Remove** a selected area. Pixels outside the selection stay exact.
- **Edit** the whole picture or a single layer with an instruction, such as "make it a snowy night".
- **Fix faces**: finds faces and redraws them sharper, optionally matching a reference photo.
- **Expand** the canvas in any direction, with format presets.

**Select**
- **Object**: click to select with SAM 2.1 Tiny, Alt+click to exclude.
- **Find**: select by description, such as "the dog" or "hat, beer glass".
- **Subject** and **Remove background**, with soft edges for hair and fur.
- Brush, eraser, lasso and box tools, plus grow, shrink, feather and invert.

**Work with layers**
- Every AI result can land on its own layer, with transform, mask, opacity and order.
- Documents keep a full revision history: undo and redo cover the background and every layer.
- Results come as one to four proposals. Compare them with the original, then apply or discard.

## Requirements

| | Tested profile |
|---|---|
| OS | Windows 11 |
| GPU | NVIDIA RTX 5070 Ti 16 GB; NVFP4 needs a Blackwell GPU and CUDA 13 |
| RAM | 32 GB (the 9B text encoder keeps about 6 GB in RAM) |
| Python | 3.13 or 3.12 |
| Node.js | A current LTS release |
| Optional | Rust and the Tauri prerequisites, for the desktop app |

Other NVIDIA GPUs can use FLUX.2 Klein 4B (BF16) under Generation settings → Advanced controls.

## Quick start

From the repository root:

```powershell
.\start.cmd
```

The first run creates `backend/.venv`, installs the CUDA PyTorch build and the backend requirements, creates `backend/.env` from `.env.example` and installs the frontend dependencies. It then starts:

- the engine on `http://127.0.0.1:8000`
- the editor on `http://localhost:5173`

Models download from Hugging Face on first use; no token is needed. On later runs, `.\start.cmd -NoInstall` skips the install checks.

For manual setup, troubleshooting a CPU-only PyTorch install and every runtime option, see the [guide](docs/guide.md).

## Models

| Model | Used for | Notes |
|---|---|---|
| FLUX.2 Klein 9B NVFP4 | Quality preset: every AI tool | Non-commercial licence |
| FLUX.2 Klein 4B NVFP4 | Fast preset: every AI tool, quicker and lighter | Use this for commercial work |
| FLUX.2 Klein 4B BF16 | For GPUs without NVFP4 support | Advanced controls |
| SAM 2.1 Tiny | Click selection and layer cut-outs | Runs on the CPU |
| Grounding DINO tiny | Find, Subject and face detection | Runs on the CPU |
| ViTMatte small | Soft edges for Subject and Remove background | Runs on the CPU |

One FLUX.2 Klein model does all the AI work: generate, edit, replace, remove, sketch, fix faces and expand. The CPU models handle selection. On the target GPU a 1024 × 1024 generation takes about 3 s, a full-frame edit about 8 s, and widening a 1254 px photo by half about 10 s. The engine warms up the Quality model in the background at start-up. See the [model comparison](benchmarks/MODEL_COMPARISON_2026-10-09.md) for measurements.

## Desktop app

```powershell
cd frontend
npm run tauri:build
```

This builds `frontend/src-tauri/target/release/mosimage.exe` plus MSI and NSIS installers. The app starts and stops the engine itself, or reuses one that `start.cmd` already started. Close a running `mosimage.exe` before rebuilding, because Windows locks the file.

## Project layout

```txt
backend/
  app/
    routes/     API routers: generation, selection, jobs, documents, references
    models/     Model providers behind one ImageModelProvider interface
    ...         Job queue, compositing, selection, documents, storage
  tests/        unittest suite
  tools/        Benchmark and fixture scripts
frontend/
  src/
    hooks/      Engine, jobs, picture, selection, layers, references
    generation/ Submission and command logic
    canvas/     Editor canvas, masks, sketch, expansion
    components/ Panels, dialogs, tool rail, library
  src-tauri/    Desktop shell
benchmarks/     Versioned fixtures and benchmark reports
docs/           Guide, architecture, roadmap and research notes
data/           Local database and images (not in Git)
models/         Optional local checkpoints (not in Git)
```

## Development

```powershell
# Backend tests and syntax check
cd backend
.\.venv\Scripts\python.exe -m unittest discover -s tests
.\.venv\Scripts\python.exe -m compileall app

# Frontend tests, type check and build
cd frontend
npm test
npm run build
```

Contributor conventions live in [AGENTS.md](AGENTS.md). For how the pieces fit together, read the [architecture overview](docs/architecture.md).

## Documentation

| | |
|---|---|
| [Guide](docs/guide.md) | Setup, runtime modes and a reference for every tool |
| [Architecture](docs/architecture.md) | Engine, job queue, providers and UI structure |
| [Roadmap](docs/roadmap.md) | Implementation status and next steps |
| [Benchmarks](benchmarks/README.md) | Fixtures, protocol and results |
| [All docs](docs/README.md) | Index, including the dated research notes |
