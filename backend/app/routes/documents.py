"""Documents: revisions, layers, AI alternatives and undo/redo."""
from __future__ import annotations

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from .. import documents, layers
from ..documents import RevisionConflict
from ..schemas import AlternativeProposalRequest, DocumentRecord, DocumentUpdate, LayerAsset, LayersUpdate, RevisionRecord
from .common import http_errors, read_uploaded_image

router = APIRouter(prefix="/api/v2")

NOT_FOUND = {LookupError: 404}
UNREADABLE = {OSError: 422, ValueError: 422}
CHANGED = {LookupError: 404, RevisionConflict: 409, ValueError: 422}


@router.get("/health")
def v2_health() -> dict[str, str]:
    return {"status": "ok", "api": "v2"}


@router.get("/documents", response_model=list[DocumentRecord])
def list_documents(limit: int = 50) -> list[dict]:
    return documents.list_documents(limit=max(1, min(limit, 100)))


@router.post("/documents", response_model=DocumentRecord)
async def create_document(image: UploadFile = File(...), name: str = Form("Untitled")) -> dict:
    content = await read_uploaded_image(image)
    with http_errors(UNREADABLE):
        return documents.create_from_upload(name, content, image.filename)


@router.delete("/documents/{document_id}")
def delete_document(document_id: int) -> dict[str, str]:
    with http_errors(NOT_FOUND):
        documents.delete_document(document_id)
    return {"status": "deleted"}


@router.post("/documents/import/{generation_id}", response_model=DocumentRecord)
def import_generation_document(generation_id: int) -> dict:
    with http_errors({**NOT_FOUND, **UNREADABLE}):
        return documents.import_generation(generation_id)


@router.get("/documents/{document_id}", response_model=DocumentRecord)
def get_document(document_id: int) -> dict:
    record = documents.get_document(document_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Document not found")
    return record


@router.patch("/documents/{document_id}", response_model=DocumentRecord)
def update_document(document_id: int, request: DocumentUpdate) -> dict:
    with http_errors(NOT_FOUND):
        return documents.update_document(document_id, request.name)


@router.post("/documents/{document_id}/revisions", response_model=RevisionRecord)
async def create_document_revision(document_id: int, image: UploadFile = File(...), operation_type: str = Form("manual")) -> dict:
    if operation_type not in {"manual", "autosave"}:
        raise HTTPException(status_code=422, detail="Unsupported revision operation")
    content = await read_uploaded_image(image)
    with http_errors({**NOT_FOUND, **UNREADABLE}):
        return documents.create_revision_from_upload(document_id, content, image.filename, operation_type)


@router.post("/documents/{document_id}/layer-assets", response_model=LayerAsset)
async def upload_layer_asset(document_id: int, image: UploadFile = File(...), kind: str = Form("image")) -> dict:
    if kind not in {"image", "mask"}:
        raise HTTPException(status_code=422, detail="Unsupported layer asset")
    if documents.get_document(document_id) is None:
        raise HTTPException(status_code=404, detail="Document not found")
    content = await read_uploaded_image(image)
    with http_errors({ValueError: 422}):
        return layers.save_asset(content, kind)  # type: ignore[arg-type]


@router.put("/documents/{document_id}/layers", response_model=DocumentRecord)
def save_document_layers(document_id: int, request: LayersUpdate) -> dict:
    with http_errors(CHANGED):
        return documents.save_layers(document_id, request.base_revision_id, [layer.model_dump() for layer in request.layers], request.label)


@router.post("/documents/{document_id}/alternatives", response_model=RevisionRecord)
def create_alternative_proposal(document_id: int, request: AlternativeProposalRequest) -> dict:
    with http_errors({**CHANGED, OSError: 422}):
        return documents.create_alternative_proposal(
            document_id,
            request.generation_ids,
            request.operation_type,
            request.base_revision_id,
            (request.layer_offset_x, request.layer_offset_y),
        )


@router.post("/documents/{document_id}/revisions/{revision_id}/alternatives/{alternative_id}/accept", response_model=DocumentRecord)
def accept_document_alternative(document_id: int, revision_id: int, alternative_id: int) -> dict:
    with http_errors(CHANGED):
        return documents.accept_alternative(document_id, revision_id, alternative_id)


@router.post("/documents/{document_id}/undo", response_model=DocumentRecord)
def undo_document(document_id: int) -> dict:
    # Nothing left to undo is a conflict with the document's state.
    with http_errors({LookupError: 404, ValueError: 409}):
        return documents.undo(document_id)


@router.post("/documents/{document_id}/redo", response_model=DocumentRecord)
def redo_document(document_id: int) -> dict:
    with http_errors({LookupError: 404, ValueError: 409}):
        return documents.redo(document_id)
