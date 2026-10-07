from __future__ import annotations

import asyncio
import os
import secrets
import signal
from contextlib import asynccontextmanager
from threading import Thread
from typing import Awaitable, Callable

from fastapi import FastAPI, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from starlette.responses import Response
from pydantic import ValidationError
from PIL import Image

from . import documents, layers, references, storage
from .config import settings
from .jobs import JobCancelled, JobControl, jobs
from .models.registry import registry
from .runtime import system_info
from .schemas import (
    GenerationRecord,
    AlternativeProposalRequest,
    DocumentRecord,
    DocumentUpdate,
    EngineStatus,
    InpaintForm,
    LayerAsset,
    LayersUpdate,
    OutpaintForm,
    ReferenceRecord,
    SelectionRequest,
    JobResponse,
    JobStatus,
    LoadModelRequest,
    ModelInfo,
    RevisionRecord,
    TextToImageRequest,
)
from .models.flux2_klein_nvfp4 import KLEIN_MODELS
from .services import run_inpaint, run_outpaint, run_text_to_image
from .selection import selector

storage.ensure_storage()
documents.ensure_document_storage()
references.ensure_reference_storage()


warm_up_job_id: str | None = None


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global warm_up_job_id
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
app.mount("/files", StaticFiles(directory=settings.data_dir), name="files")


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
    job = jobs.submit("load-model", lambda control: _preload_model(request, control))
    return JobResponse(id=job.id, kind=job.kind, status=job.status)


@app.post("/api/generate", response_model=JobResponse)
def generate(request: TextToImageRequest) -> JobResponse:
    job = jobs.submit("text-to-image", lambda control: run_text_to_image(request, control))
    return JobResponse(id=job.id, kind=job.kind, status=job.status)


@app.post("/api/inpaint", response_model=JobResponse)
async def inpaint(
    prompt: str = Form(...),
    negative_prompt: str | None = Form(None),
    model_name: str = Form("flux2-klein-4b-nvfp4"),
    seed: int | None = Form(None),
    runtime_mode: str = Form("balanced"),
    steps: int = Form(4),
    guidance: float = Form(1.0),
    strength: float = Form(1.0),
    count: int = Form(1),
    edit_operation: str = Form("replace"),
    blend_feather: float = Form(16),
    context_padding: int = Form(320),
    scene_context: str = Form(""),
    reference_ids: list[int] = Form([]),
    image: UploadFile = File(...),
    mask: UploadFile = File(...),
) -> JobResponse:
    try:
        request = InpaintForm(
            prompt=prompt,
            negative_prompt=negative_prompt,
            model_name=model_name,
            seed=seed,
            runtime_mode=runtime_mode,
            steps=steps,
            guidance=guidance,
            strength=strength,
            count=count,
            edit_operation=edit_operation,
            blend_feather=blend_feather,
            context_padding=context_padding,
            scene_context=scene_context,
            reference_ids=reference_ids,
        )
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    if request.reference_ids:
        if any(references.get_reference(reference_id) is None for reference_id in request.reference_ids):
            raise HTTPException(status_code=422, detail="A selected reference no longer exists.")
        if request.model_name not in KLEIN_MODELS:
            raise HTTPException(status_code=422, detail="References need the Quality model (FLUX.2 Klein).")
    image_bytes = await image.read()
    mask_bytes = await mask.read()
    job = jobs.submit(
        "inpaint",
        lambda control: run_inpaint(request, image_bytes, image.filename, mask_bytes, mask.filename, control),
    )
    return JobResponse(id=job.id, kind=job.kind, status=job.status)


@app.get("/api/jobs/{job_id}", response_model=JobStatus)
def job_status(job_id: str) -> JobStatus:
    status = jobs.status(job_id)
    if status is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return status


@app.post("/api/outpaint", response_model=JobResponse)
async def outpaint(
    image: UploadFile = File(...),
    parameters: str = Form(...),
) -> JobResponse:
    try:
        request = OutpaintForm.model_validate_json(parameters)
        if "outpaint" not in registry.get(request.model_name).info().capabilities:
            raise ValueError("Choose a mask-conditioned model for expansion: Z-Image Turbo or FLUX Fill.")
    except (ValidationError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    content = await image.read()
    job = jobs.submit("outpaint", lambda control: run_outpaint(request, content, control))
    return JobResponse(id=job.id, kind=job.kind, status=job.status)


@app.post("/api/select", response_model=JobResponse)
async def select_object(image: UploadFile = File(...), parameters: str = Form(...)) -> JobResponse:
    try:
        request = SelectionRequest.model_validate_json(parameters)
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    content = await image.read()
    job = jobs.submit("select", lambda control: selector.select(content, request, control))
    return JobResponse(id=job.id, kind=job.kind, status=job.status)


@app.post("/api/jobs/{job_id}/cancel", response_model=JobStatus)
def cancel_job(job_id: str) -> JobStatus:
    status = jobs.cancel(job_id)
    if status is None:
        raise HTTPException(status_code=404, detail="Job not found")
    if settings.supervised and jobs.claim_hard_cancel(job_id):
        Thread(target=_enforce_cancel, args=(job_id,), daemon=True).start()
    return status


@app.get("/api/history", response_model=list[GenerationRecord])
def history(limit: int = 50) -> list[dict]:
    return storage.list_generations(limit=limit)


@app.get("/api/history/{generation_id}", response_model=GenerationRecord)
def history_item(generation_id: int) -> dict:
    record = storage.get_generation(generation_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Generation not found")
    return record


@app.get("/api/v2/health")
def v2_health() -> dict[str, str]:
    return {"status": "ok", "api": "v2"}


@app.get("/api/v2/documents", response_model=list[DocumentRecord])
def list_documents(limit: int = 50) -> list[dict]:
    return documents.list_documents(limit=max(1, min(limit, 100)))


@app.post("/api/v2/documents", response_model=DocumentRecord)
async def create_document(
    image: UploadFile = File(...),
    name: str = Form("Untitled"),
) -> dict:
    content = await image.read()
    try:
        return documents.create_from_upload(name, content, image.filename)
    except (OSError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/api/v2/documents/import/{generation_id}", response_model=DocumentRecord)
def import_generation_document(generation_id: int) -> dict:
    try:
        return documents.import_generation(generation_id)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except (OSError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.get("/api/v2/documents/{document_id}", response_model=DocumentRecord)
def get_document(document_id: int) -> dict:
    record = documents.get_document(document_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Document not found")
    return record


@app.patch("/api/v2/documents/{document_id}", response_model=DocumentRecord)
def update_document(document_id: int, request: DocumentUpdate) -> dict:
    try:
        return documents.update_document(document_id, request.name)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/api/v2/documents/{document_id}/revisions", response_model=RevisionRecord)
async def create_document_revision(
    document_id: int,
    image: UploadFile = File(...),
    operation_type: str = Form("manual"),
) -> dict:
    if operation_type not in {"manual", "autosave"}:
        raise HTTPException(status_code=422, detail="Unsupported revision operation")
    try:
        return documents.create_revision_from_upload(
            document_id,
            await image.read(),
            image.filename,
            operation_type,
        )
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except (OSError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/api/v2/documents/{document_id}/layer-assets", response_model=LayerAsset)
async def upload_layer_asset(document_id: int, image: UploadFile = File(...), kind: str = Form("image")) -> dict:
    if kind not in {"image", "mask"}:
        raise HTTPException(status_code=422, detail="Unsupported layer asset")
    if documents.get_document(document_id) is None:
        raise HTTPException(status_code=404, detail="Document not found")
    try:
        return layers.save_asset(await image.read(), kind)  # type: ignore[arg-type]
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.put("/api/v2/documents/{document_id}/layers", response_model=DocumentRecord)
def save_document_layers(document_id: int, request: LayersUpdate) -> dict:
    try:
        return documents.save_layers(
            document_id, request.base_revision_id, [layer.model_dump() for layer in request.layers], request.label
        )
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=409 if "changed" in str(exc) else 422, detail=str(exc)) from exc


@app.post("/api/v2/documents/{document_id}/alternatives", response_model=RevisionRecord)
def create_alternative_proposal(document_id: int, request: AlternativeProposalRequest) -> dict:
    try:
        return documents.create_alternative_proposal(
            document_id,
            request.generation_ids,
            request.operation_type,
            request.base_revision_id,
            (request.layer_offset_x, request.layer_offset_y),
        )
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except (OSError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post(
    "/api/v2/documents/{document_id}/revisions/{revision_id}/alternatives/{alternative_id}/accept",
    response_model=DocumentRecord,
)
def accept_document_alternative(document_id: int, revision_id: int, alternative_id: int) -> dict:
    try:
        return documents.accept_alternative(document_id, revision_id, alternative_id)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.post("/api/v2/documents/{document_id}/undo", response_model=DocumentRecord)
def undo_document(document_id: int) -> dict:
    try:
        return documents.undo(document_id)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.post("/api/v2/documents/{document_id}/redo", response_model=DocumentRecord)
def redo_document(document_id: int) -> dict:
    try:
        return documents.redo(document_id)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.get("/api/v2/documents/{document_id}/references", response_model=list[ReferenceRecord])
def list_document_references(document_id: int) -> list[dict]:
    return references.list_references(document_id)


@app.post("/api/v2/documents/{document_id}/references", response_model=ReferenceRecord)
async def add_document_reference(document_id: int, image: UploadFile = File(...)) -> dict:
    try:
        return references.create_reference(document_id, await image.read(), image.filename)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except (OSError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.delete("/api/v2/references/{reference_id}")
def delete_document_reference(reference_id: int) -> dict[str, str]:
    try:
        references.delete_reference(reference_id)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {"status": "deleted"}


@app.post("/api/v2/references/{reference_id}/subject", response_model=ReferenceRecord)
def isolate_reference_subject(reference_id: int) -> dict:
    try:
        source_path = references.original_image_path(reference_id)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    with Image.open(source_path) as source:
        image = source.convert("RGB")
    mask = selector.subject_mask(image)
    if mask is None:
        raise HTTPException(status_code=422, detail="No clear subject found. Crop the image around the subject and add it again.")
    return references.set_subject(reference_id, references.extract_subject(image, mask))


@app.delete("/api/v2/references/{reference_id}/subject", response_model=ReferenceRecord)
def restore_reference_background(reference_id: int) -> dict:
    try:
        return references.clear_subject(reference_id)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


def _warm_up_engine(control: JobControl) -> None:
    """Load SAM and the Quality model before the first click or edit."""
    control.set_phase("loading", "SAM 2.1 Tiny")
    selector.warm_up()
    control.set_progress(0.2)
    request = LoadModelRequest(model_name=settings.warm_up_model, task="inpaint", warm_up=True)
    _preload_model(request, control.subrange(0.2, 1.0))


def _preload_model(request: LoadModelRequest, control: JobControl) -> None:
    try:
        registry.preload(request, control)
    except JobCancelled:
        registry.unload(request.model_name)
        raise
    return None


def _enforce_cancel(job_id: str) -> None:
    if jobs.wait_for_completion(job_id, settings.cancel_grace_seconds):
        return
    # CUDA kernels and native model code cannot be interrupted safely from
    # Python. The local supervisor immediately replaces this worker.
    os._exit(75)
