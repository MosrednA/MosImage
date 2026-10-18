# Repository Guidelines

## Project Structure & Module Organization

MosImage uses a split local app architecture:

- `backend/` contains the FastAPI API, SQLite metadata layer, job queue, and model provider registry.
- `backend/app/models/` contains runtime adapters. The Quality default is FLUX.2 Klein 9B NVFP4 (`flux2_klein_nvfp4.py`, with its RAM-streamed FP8 text encoder in `klein_text_encoder.py`); Klein 4B NVFP4 (the Fast preset) and 4B BF16 (`flux2_klein.py`) are the other models in `registry.py`. On Apple silicon the registry offers Klein 9B (Quality) and 4B (Fast) from `flux2_klein_apple.py` instead: BF16 on Metal (MPS), or 8-bit/4-bit in MLX (`klein_mlx.py`, `MOSIMAGE_APPLE_WEIGHTS`) inside the same diffusers pipelines. Every AI tool, including expansion, runs on FLUX.2 Klein; selection runs on CPU models (SAM 2.1 Tiny, Grounding DINO tiny, ViTMatte).
- `frontend/` contains the React + Vite UI and optional Tauri shell in `frontend/src-tauri/`.
- `data/images/{inputs,masks,outputs}/` stores local image files. `data/app.db` stores metadata.
- `models/` is reserved for local checkpoints or manually downloaded model files.
- `docs/` holds the guide, architecture overview, roadmap and dated research notes (`docs/notes/`). `benchmarks/` holds versioned fixtures and benchmark reports. Keep the root `README.md` a short overview and put detail in `docs/`.
- `backend/.env.example` contains the default RTX 5070 Ti 16 GB runtime profile. On macOS the defaults in `config.py` switch to the Apple silicon profile by themselves.

## Build, Test, and Development Commands

- `cd backend; uvicorn app.main:app --reload --host 127.0.0.1 --port 8000` starts the API.
- `.\start.cmd` starts backend and frontend together from the repository root; `./start.sh` does the same on a Mac.
- `npm install` from `frontend/` installs UI dependencies.
- `npm run dev` from `frontend/` starts Vite on port `5173`.
- `npm run build` from `frontend/` type-checks and builds the UI.
- `npm run tauri:dev` from `frontend/` starts the optional desktop shell after Rust/Tauri prerequisites are installed.
- `npm run tauri:build` from `frontend/` rebuilds `src-tauri/target/release/mosimage.exe` (incremental, without installers). The user's desktop shortcut opens that file, so a build updates the app they launch. Close a running `mosimage.exe` first, because Windows locks it. `npm run tauri:installers` also builds the MSI and NSIS installers.
- `npm run tauri:build:mac` from `frontend/` builds `MosImage.app` and installs it in `/Applications`, which the user opens on their Mac.

## Coding Style & Naming Conventions

Python uses typed modules, Pydantic schemas, and small service/provider boundaries. Keep runtime-specific code behind `ImageModelProvider` implementations. TypeScript uses strict mode, React function components, and explicit API types in `frontend/src/api.ts`.

Use descriptive snake_case for Python files and camelCase for TypeScript variables/functions. Keep generated files, model caches, and local image outputs out of commits.

Default runtime assumptions are Ryzen 7 9700X, 32 GB RAM, and RTX 5070 Ti 16 GB. Preserve CUDA `bfloat16`, TF32, and one-pipeline residency unless a change explicitly targets another profile. The Apple silicon profile (M5 Max, 128 GB) runs MPS `bfloat16` with every component resident, or MLX-quantized weights on smaller Macs; refer to models by role (`settings.quality_model`, `settings.fast_model`) rather than by NVFP4 name. The UI also runs in WebKit there: avoid `createImageBitmap` for pixels that matter (use `decodeImage`) and canvas `filter`, and label shortcuts through `platform.ts`.

## Testing Guidelines

Backend tests live under `backend/tests/`. Frontend Vitest tests live beside their modules; run them with `npm test` from `frontend/`. Extend these suites for behavior changes and add regression tests for bug fixes where practical.

Before submitting changes, run `npm run build` in `frontend/` and at least `python -m compileall backend\app` for backend syntax checks. Run the backend suite from `backend/` with `python -m unittest discover -s tests -t .`; `tests/__init__.py` pins the CUDA profile, so the suite means the same on every machine.

## Commit & Pull Request Guidelines

Use concise, imperative commit messages such as `Add Klein expansion` or `Fix inpaint mask upload`.

Pull requests should include a summary, test results, linked issues when applicable, and screenshots for UI changes. Keep PRs scoped to one logical change.
