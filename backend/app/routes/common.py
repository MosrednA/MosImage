"""Helpers shared by the API routers: uploads, form parsing and error mapping."""
from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator, Mapping, TypeVar

from fastapi import HTTPException, UploadFile
from pydantic import BaseModel, ValidationError

from .. import references
from ..config import settings
from ..jobs import Job
from ..loras import ARCHITECTURE_NAMES, inspect, lora_file
from ..models.registry import registry
from ..schemas import JobResponse, LoraChoice

Model = TypeVar("Model", bound=BaseModel)


async def read_uploaded_image(file: UploadFile, max_bytes: int | None = None) -> bytes:
    limit = max_bytes if max_bytes is not None else settings.max_upload_bytes
    content = await file.read(limit + 1)
    if len(content) > limit:
        max_mb = max(1, limit // (1024 * 1024))
        raise HTTPException(
            status_code=413,
            detail=f"File {file.filename or 'upload'} exceeds maximum size of {max_mb} MB.",
        )
    return content


def parse_parameters(model: type[Model], parameters: str) -> Model:
    """The JSON `parameters` form field of a multipart request."""
    try:
        return model.model_validate_json(parameters)
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


def require_capability(model_name: str, capability: str, message: str) -> None:
    try:
        supported = capability in registry.get(model_name).info().capabilities
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    if not supported:
        raise HTTPException(status_code=422, detail=message)


def check_references(reference_ids: list[int]) -> None:
    if not reference_ids:
        return
    if any(references.get_reference(reference_id) is None for reference_id in reference_ids):
        raise HTTPException(status_code=422, detail="A selected reference no longer exists.")


def check_loras(model_name: str, loras: list[LoraChoice]) -> None:
    """Refuses a missing, unreadable or mismatched LoRA before its job waits in the queue."""
    if not loras:
        return
    architecture = registry.get(model_name).info().architecture
    for choice in loras:
        with http_errors({ValueError: 422, OSError: 503}):
            inspection = inspect(lora_file(choice.name))
        if inspection.problem:
            raise HTTPException(status_code=422, detail=f"{choice.name}: {inspection.problem}")
        if architecture not in inspection.fits:
            made_for = " or ".join(ARCHITECTURE_NAMES[fit] for fit in inspection.fits)
            raise HTTPException(status_code=422, detail=f"{choice.name} is made for {made_for}. Turn it off or switch models.")


def job_response(job: Job) -> JobResponse:
    return JobResponse(id=job.id, kind=job.kind, status=job.status)


@contextmanager
def http_errors(statuses: Mapping[type[Exception], int]) -> Iterator[None]:
    """Turns domain errors into HTTP errors; the first matching type wins, so list subclasses first."""
    try:
        yield
    except tuple(statuses) as exc:
        status = next(code for kind, code in statuses.items() if isinstance(exc, kind))
        raise HTTPException(status_code=status, detail=str(exc)) from exc
