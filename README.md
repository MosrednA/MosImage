<div align="center">

<img src="frontend/src-tauri/icons/128x128@2x.png" width="96" alt="MosImage logo">

# MosImage

**A local AI image editor. Your GPU, your pictures, no cloud.**

Generate pictures, edit them with plain-language instructions, select objects with a click,<br>
and keep every result as a layer you can move, mask or undo.

[![License: MIT](https://img.shields.io/badge/license-MIT-3b82f6)](LICENSE)
![Windows 11](https://img.shields.io/badge/Windows-11-0078D4?logo=windows11&logoColor=white)
![NVIDIA RTX](https://img.shields.io/badge/NVIDIA-RTX%2050%20series-76B900?logo=nvidia&logoColor=white)
![macOS](https://img.shields.io/badge/macOS-Apple%20silicon-000000?logo=apple&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.12%20%7C%203.13-3776AB?logo=python&logoColor=white)
![React](https://img.shields.io/badge/React-Vite-61DAFB?logo=react&logoColor=black)
![FLUX.2 Klein](https://img.shields.io/badge/model-FLUX.2%20Klein-111827)

<img src="docs/images/editor.jpg" alt="The MosImage editor with the cat selected by a single click" width="100%">

<sub>One click on the cat selects it with SAM 2.1. Type what should appear instead and press Generate.</sub>

</div>

## See it work

Three steps on one picture, each about 3–10 seconds on an RTX 5070 Ti (7–15 seconds on an M5 Max), straight from the app:

| 1 · Generate | 2 · Replace a selection | 3 · Edit with an instruction |
|:---:|:---:|:---:|
| <img src="docs/images/generate.jpg" alt="Generated greenhouse reading nook with a sleeping cat" width="260"> | <img src="docs/images/replace.jpg" alt="The cat replaced by a sleeping puppy" width="260"> | <img src="docs/images/edit.jpg" alt="The same scene turned into a snowy winter night" width="260"> |
| *"A cosy reading nook in a glass greenhouse at golden hour, an orange cat asleep on a velvet armchair"* | Click the cat, then *"A fluffy golden retriever puppy curled up asleep"*. Everything outside the selection stays pixel-exact. | *"Make it a snowy winter night … a warm glowing floor lamp beside the armchair"* |

## Why MosImage

- 🔒 **Private by design.** Images, prompts and history stay on your disk. No account and no image uploads; models and optional adapters download from Hugging Face on first use.
- ⚡ **Fast.** FLUX.2 Klein in NVFP4 runs a 1024 × 1024 image in about 3 seconds on a 16 GB card, and in BF16 in about 7 seconds on an M5 Max.
- 🎯 **Precise.** Edits stay inside your selection, and every result lands as a layer you can mask, move or undo.
- 🧩 **A real editor.** Layers, revision history, proposals to compare, and brush, lasso, box and click selection.

## Features

**Create**
- **Generate** an image from a prompt. In an open document the result is added as a new layer.
- **References**: put people, pets or objects from your own photos into a new scene ("Anna and Tom riding a roller coaster"), up to three at once.
- **Sketch** rough coloured strokes and describe them. MosImage renders just the sketch, or the whole picture around it. With references picked, the sketch places and poses them.

**Edit**
- **Replace** or **Remove** a selected area. Pixels outside the selection stay exact.
- **Edit** the whole picture or a single layer with an instruction, such as "make it a snowy night".
- **Fix faces**: pick the detected faces to sharpen. **Faithful** keeps expression and lighting closer; **Redraw** restores very blurry faces and can change their expression. A reference can guide likeness.
- **Enhance** adds fine detail to the whole picture or a selection, with **Subtle** and **Strong** settings. Large areas render in tiles at their original resolution; small details can change.
- **Expand** the canvas in any direction, with format presets.
- **Crop** resizes the canvas without AI, with ratios and exact dimensions. Use C for Crop and Shift+C for Expand.

**Select**
- **Object**: click to select with SAM 2.1 Tiny, Alt+click (⌥-click on a Mac) to exclude.
- **Find**: select by description, such as "the dog" or "hat, beer glass".
- **Subject** and **Remove background**, with soft edges for hair and fur.
- Brush, eraser, lasso and box tools, plus grow, shrink, feather and invert.

**Work with layers**
- Every AI result can land on its own layer, with transform, mask, opacity and order.
- Select several layers, merge them down or flatten, as in Photoshop; the picture itself is a layer too.
- Documents keep a full revision history: undo and redo cover every layer.
- Results come as one to four proposals. Compare them with the original, then apply or discard.

## Requirements

| | Windows (tested profile) | macOS (tested profile) |
|---|---|---|
| OS | Windows 11 | macOS 27; macOS 14 or newer on Apple silicon |
| GPU | NVIDIA RTX 5070 Ti 16 GB; NVFP4 needs a Blackwell GPU and CUDA 13 | Apple M5 Max, 40-core GPU, through Metal (MPS) |
| Memory | 32 GB RAM (the 9B text encoder keeps about 6 GB in RAM) | 128 GB unified memory; Quality uses about 30 GB in BF16, 19 GB in 8-bit and 13 GB in 4-bit |
| Python | 3.13 or 3.12 | 3.13 or 3.12, from Homebrew or uv |
| Node.js | A current LTS release | A current LTS release |
| Optional | Rust and the Tauri prerequisites, for the desktop app | Rust (`brew install rust`) and the Xcode command-line tools, for the desktop app |

Other NVIDIA GPUs can use FLUX.2 Klein 4B (BF16) under Generation settings → Advanced controls. A Mac runs the BF16 weights the NVFP4 builds come from, or quantizes them to 8 or 4 bits in MLX: from 64 GB of memory it uses BF16, from 48 GB 8-bit, below that 4-bit (`MOSIMAGE_APPLE_WEIGHTS` overrides this). Below 36 GB the engine warms up Fast, the editor starts on Fast, and the VAE works in 512 px tiles to save memory.

## Quick start

From the repository root, on Windows:

```powershell
.\start.cmd
```

or on a Mac:

```bash
./start.sh
```

The first run creates `backend/.venv`, installs PyTorch (the CUDA build on Windows, the Metal build on a Mac) and the backend requirements, creates `backend/.env` and installs the frontend dependencies. It then starts:

- the engine on `http://127.0.0.1:8000`
- the editor on `http://localhost:5173`

Models download from Hugging Face on first use. FLUX.2 Klein 9B, the Quality model, is gated: accept its licence on [its model page](https://huggingface.co/black-forest-labs/FLUX.2-klein-9B) and log in once with `hf auth login`, or set `MOSIMAGE_HF_TOKEN` in `backend/.env`. The other models need no token. On later runs, `.\start.cmd -NoInstall` or `./start.sh --no-install` skips the install checks.

For manual setup, troubleshooting a CPU-only PyTorch install and every runtime option, see the [guide](docs/guide.md).

## Models

| Model | Used for | Notes |
|---|---|---|
| FLUX.2 Klein 9B NVFP4 | Quality preset on Windows: every AI tool | Non-commercial licence |
| FLUX.2 Klein 4B NVFP4 | Fast preset on Windows: generation and editing, quicker and lighter | Commercially licensed; Enhance runs without the detail enhancer |
| FLUX.2 Klein 9B (BF16, 8-bit or 4-bit) | Quality preset on a Mac | Non-commercial licence |
| FLUX.2 Klein 4B (BF16, 8-bit or 4-bit) | Fast preset on a Mac | Commercially licensed |
| FLUX.2 Klein 4B BF16 | On Windows for GPUs without NVFP4 support | Advanced controls |
| SAM 2.1 Tiny | Click selection and layer cut-outs | Runs on the CPU |
| Grounding DINO tiny | Find, Subject and face detection | Runs on the CPU |
| ViTMatte small | Soft edges for Subject and Remove background | Runs on the CPU |

FLUX.2 Klein does all the AI work: generate, edit, replace, remove, sketch, fix faces, enhance and expand. Enhance on Quality 9B adds its non-commercial detail-enhancer adapter; on Fast it redraws without it. The CPU models handle selection. With the model loaded on the target GPU, a 1024 × 1024 generation takes about 3 s, a full-frame edit about 8 s, and widening a 1254 px photo by half about 10 s. Enhance takes about ten seconds per tile. On an M5 Max, Quality takes about 7 s to generate, 11 s to replace a selection and 15 s for a full-frame edit; Fast about half that. The engine warms up the Quality model in the background at start-up; downloads and loading add time. See the [model comparison](benchmarks/MODEL_COMPARISON_2026-10-09.md) and [Enhance evaluation](docs/notes/2026-10-10-quality-speed-sweep.md#enhance) for measurements.

## Desktop app

```powershell
cd frontend
npm run tauri:build
```

This builds `frontend/src-tauri/target/release/mosimage.exe` in a few seconds after the first build; `npm run tauri:installers` also makes MSI and NSIS installers. The app starts and stops the engine itself, or reuses one that `start.cmd` already started. Close a running `mosimage.exe` before rebuilding, because Windows locks the file.

On a Mac, after `./start.sh` has set up the engine once:

```bash
cd frontend
npm run tauri:build:mac
```

This builds `MosImage.app` and installs it in `/Applications`, so it opens from Launchpad, Spotlight or the Dock like any other app.

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
# Backend tests and syntax check (on a Mac: .venv/bin/python)
cd backend
.\.venv\Scripts\python.exe -m unittest discover -s tests -t .
.\.venv\Scripts\python.exe -m compileall app

# Frontend tests, type check and build
cd ..\frontend
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

## License

The MosImage source code is released under the [MIT License](LICENSE).

Model weights are not part of this repository. They download from Hugging Face under their own licences: FLUX.2 Klein 9B is non-commercial, while FLUX.2 Klein 4B, SAM 2.1, Grounding DINO and ViTMatte permit commercial use. Check each model card before you use results commercially.

The screenshots and example images in [docs/images](docs/images) were generated with MosImage itself.
