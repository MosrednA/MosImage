"""References: people and objects to place, loose on the start screen or kept with a document."""
from __future__ import annotations

from fastapi import APIRouter, File, HTTPException, UploadFile
from PIL import Image

from .. import references
from ..jobs import jobs
from ..schemas import ReferenceRecord, ReferenceUpdate
from ..selection import selector
from .common import http_errors, read_uploaded_image

router = APIRouter(prefix="/api/v2")

NOT_FOUND = {LookupError: 404}
UNREADABLE = {OSError: 422, ValueError: 422}


@router.get("/references", response_model=list[ReferenceRecord])
def list_loose_references() -> list[dict]:
    return references.list_references(None)


@router.post("/references", response_model=ReferenceRecord)
async def add_loose_reference(image: UploadFile = File(...)) -> dict:
    content = await read_uploaded_image(image)
    with http_errors(UNREADABLE):
        return references.create_reference(None, content, image.filename)


@router.post("/documents/{document_id}/references/adopt", response_model=list[ReferenceRecord])
def adopt_start_references(document_id: int) -> list[dict]:
    """Moves the start screen's references into a new document."""
    with http_errors(NOT_FOUND):
        return references.adopt_loose_references(document_id)


@router.get("/documents/{document_id}/references", response_model=list[ReferenceRecord])
def list_document_references(document_id: int) -> list[dict]:
    return references.list_references(document_id)


@router.post("/documents/{document_id}/references", response_model=ReferenceRecord)
async def add_document_reference(document_id: int, image: UploadFile = File(...)) -> dict:
    content = await read_uploaded_image(image)
    with http_errors({**NOT_FOUND, **UNREADABLE}):
        return references.create_reference(document_id, content, image.filename)


@router.patch("/references/{reference_id}", response_model=ReferenceRecord)
def update_document_reference(reference_id: int, body: ReferenceUpdate) -> dict:
    with http_errors({**NOT_FOUND, **UNREADABLE}):
        return references.update_reference(reference_id, name=body.name, role=body.role)


@router.delete("/references/{reference_id}")
def delete_document_reference(reference_id: int) -> dict[str, str]:
    with http_errors(NOT_FOUND):
        references.delete_reference(reference_id)
    return {"status": "deleted"}


@router.post("/references/{reference_id}/subject", response_model=ReferenceRecord)
def isolate_reference_subject(reference_id: int) -> dict:
    with http_errors(NOT_FOUND):
        source_path = references.original_image_path(reference_id)
    with Image.open(source_path) as source:
        image = source.convert("RGB")
    # Run subject segmentation on the CPU executor lane to serialize with select jobs
    mask = jobs.run_cpu(lambda: selector.subject_mask(image))
    if mask is None:
        raise HTTPException(status_code=422, detail="No clear subject found. Crop the image around the subject and add it again.")
    return references.set_subject(reference_id, references.extract_subject(image, mask))


@router.delete("/references/{reference_id}/subject", response_model=ReferenceRecord)
def restore_reference_background(reference_id: int) -> dict:
    with http_errors(NOT_FOUND):
        return references.clear_subject(reference_id)
