# Repository Guidelines

## Project Structure & Module Organization

MosImage uses a split local app architecture:

- `backend/` contains the FastAPI API, SQLite metadata layer, job queue, and model provider registry.
- `backend/app/models/` contains runtime adapters. `z_image.py` is the first provider for `Tongyi-MAI/Z-Image-Turbo`.
- `frontend/` contains the React + Vite UI and optional Tauri shell in `frontend/src-tauri/`.
- `data/images/{inputs,masks,outputs}/` stores local image files. `data/app.db` stores metadata.
- `models/` is reserved for local checkpoints or manually downloaded model files.
- `backend/.env.example` contains the default RTX 5070 Ti 16 GB runtime profile.

## Build, Test, and Development Commands

- `cd backend; uvicorn app.main:app --reload --host 127.0.0.1 --port 8000` starts the API.
- `.\start.cmd` starts backend and frontend together from the repository root.
- `npm install` from `frontend/` installs UI dependencies.
- `npm run dev` from `frontend/` starts Vite on port `5173`.
- `npm run build` from `frontend/` type-checks and builds the UI.
- `npm run tauri:dev` from `frontend/` starts the optional desktop shell after Rust/Tauri prerequisites are installed.
- `npm run tauri:build` from `frontend/` rebuilds `src-tauri/target/release/mosimage.exe`. The user's desktop shortcut opens that file, so a build updates the app they launch. Close a running `mosimage.exe` first, because Windows locks it.

## Coding Style & Naming Conventions

Python uses typed modules, Pydantic schemas, and small service/provider boundaries. Keep runtime-specific code behind `ImageModelProvider` implementations. TypeScript uses strict mode, React function components, and explicit API types in `frontend/src/api.ts`.

Use descriptive snake_case for Python files and camelCase for TypeScript variables/functions. Keep generated files, model caches, and local image outputs out of commits.

Default runtime assumptions are Ryzen 7 9700X, 32 GB RAM, and RTX 5070 Ti 16 GB. Preserve CUDA `bfloat16`, TF32, and one-pipeline residency unless a change explicitly targets another profile.

## Testing Guidelines

Backend tests live under `backend/tests/`. Frontend Vitest tests live beside their modules; run them with `npm test` from `frontend/`. Extend these suites for behavior changes and add regression tests for bug fixes where practical.

Before submitting changes, run `npm run build` in `frontend/` and at least `python -m compileall backend\app` for backend syntax checks.

## Commit & Pull Request Guidelines

Use concise, imperative commit messages such as `Add Z-Image provider` or `Fix inpaint mask upload`.

Pull requests should include a summary, test results, linked issues when applicable, and screenshots for UI changes. Keep PRs scoped to one logical change.
