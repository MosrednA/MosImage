from __future__ import annotations

import asyncio
import os
import secrets
import signal
import threading
import time
from contextlib import asynccontextmanager
from typing import Awaitable, Callable

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from starlette.responses import Response
from PIL import Image

from . import documents, references, storage
from .config import settings
from .jobs import JobCancelled, JobControl, jobs
from .models.registry import registry
from .routes import documents as document_routes
from .routes import generation
from .routes import jobs as job_routes
from .routes import references as reference_routes
from .routes import selection as selection_routes
from .routes.common import job_response, read_uploaded_image  # noqa: F401  (tests import it from here)
from .runtime import system_info
from .schemas import EngineStatus, JobResponse, LoadModelRequest, ModelInfo
from .selection import selector

Image.MAX_IMAGE_PIXELS = settings.max_image_pixels

storage.ensure_storage()
documents.ensure_document_storage()
references.ensure_reference_storage()


warm_up_job_id: str | None = None


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global warm_up_job_id
    if settings.exit_with_parent and os.name == "posix":
        threading.Thread(target=_exit_with_parent, args=(os.getppid(), settings.watch_pid), name="parent-watch", daemon=True).start()
    if settings.warm_up_on_start:
        warm_up_job_id = jobs.submit("load-model", _warm_up_engine).id
    try:
        yield
    finally:
        jobs.shutdown(wait=True)
        registry.unload_all()


app = FastAPI(title=settings.app_name, lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Mount images only so app.db and other server data are never exposed via HTTP.
app.mount("/files/images", StaticFiles(directory=settings.image_dir), name="images")


@app.middleware("http")
async def vary_image_origin(request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
    response = await call_next(request)
    # An <img> may cache a response without Origin before fetch reopens it.
    # That cached response must not be reused for a CORS-enabled request.
    if request.url.path.startswith("/files/"):
        vary = response.headers.get("vary", "")
        if "origin" not in {value.strip().lower() for value in vary.split(",")}:
            response.headers.add_vary_header("Origin")
    return response


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/system")
def system() -> dict:
    return system_info()


@app.post("/api/shutdown", status_code=202)
async def shutdown_engine(x_mosimage_token: str | None = Header(None)) -> dict[str, str]:
    """Stops the engine the way Ctrl+C does: cancel jobs, unload models, exit."""
    if not settings.shutdown_token:
        raise HTTPException(status_code=404, detail="Not Found")
    if not x_mosimage_token or not secrets.compare_digest(x_mosimage_token, settings.shutdown_token):
        raise HTTPException(status_code=403, detail="Invalid shutdown token.")
    _schedule_exit()
    return {"status": "stopping"}


def _schedule_exit() -> None:
    # Just after the response is sent; uvicorn then runs the lifespan shutdown.
    asyncio.get_running_loop().call_later(0.1, signal.raise_signal, signal.SIGINT)


def _exit_with_parent(parent: int, watched: int | None = None) -> None:
    """Stops the engine once its parent, or the ``watched`` process, is gone.

    POSIX re-parents an orphan, so its parent ID changes. Windows has no such
    signal; the desktop app's job object ends the engine there.
    """
    while os.getppid() == parent and (watched is None or _running(watched)):
        time.sleep(1)
    os.kill(os.getpid(), signal.SIGINT)
    # A running edit can hold the graceful shutdown; the memory must be released regardless.
    time.sleep(settings.cancel_grace_seconds + 5)
    os._exit(0)


def _running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


@app.get("/api/engine", response_model=EngineStatus)
def engine() -> EngineStatus:
    return EngineStatus(warm_up=jobs.status(warm_up_job_id) if warm_up_job_id else None, models=registry.list())


@app.get("/api/models", response_model=list[ModelInfo])
def list_models() -> list[ModelInfo]:
    return registry.list()


@app.post("/api/models/{model_name}/unload")
def unload_model(model_name: str) -> dict[str, str]:
    try:
        registry.unload(model_name)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {"status": "unloaded"}


@app.post("/api/models/load", response_model=JobResponse)
def load_model(request: LoadModelRequest) -> JobResponse:
    return job_response(jobs.submit("load-model", lambda control: _preload_model(request, control)))


for router in (generation.router, selection_routes.router, job_routes.router, document_routes.router, reference_routes.router):
    app.include_router(router)


def _warm_up_engine(control: JobControl) -> None:
    """Load SAM and the Quality model before the first click or edit."""
    control.set_phase("loading", "SAM 2.1 Tiny")
    selector.warm_up()
    # SAM takes under a second; the Quality model the other 25 or so.
    control.set_progress(0.03)
    request = LoadModelRequest(model_name=settings.warm_up_model, task="inpaint", warm_up=True)
    _preload_model(request, control.subrange(0.03, 1.0))


def _preload_model(request: LoadModelRequest, control: JobControl) -> None:
    try:
        registry.preload(request, control)
    except JobCancelled:
        registry.unload(request.model_name)
        raise
    return None
