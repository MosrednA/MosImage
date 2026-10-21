"""AI image jobs: generate, edit, inpaint, expand, fix faces and enhance, plus the generation history."""
from __future__ import annotations

from io import BytesIO

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from PIL import Image, UnidentifiedImageError
from pydantic import TypeAdapter, ValidationError

from .. import storage
from ..enhance import run_enhance
from ..faces import face_candidates, run_face_fix
from ..jobs import jobs
from ..schemas import (EditRequest, EnhanceRequest, FaceCandidate, FaceDetectionResponse, FaceFixRequest, GenerationRecord, InpaintForm, JobResponse,
                       LoraChoice, OutpaintForm, TextToImageRequest)
from ..services import run_image_edit, run_inpaint, run_outpaint, run_text_to_image
from .common import check_loras, check_references, http_errors, job_response, parse_parameters, read_uploaded_image, require_capability

router = APIRouter(prefix="/api")
LORA_LIST = TypeAdapter(list[LoraChoice])


@router.post("/generate", response_model=JobResponse)
def generate(request: TextToImageRequest) -> JobResponse:
    require_capability(request.model_name, "text-to-image", "This model cannot generate images.")
    check_references(request.reference_ids)
    check_loras(request.model_name, request.loras)
    return job_response(jobs.submit("text-to-image", lambda control: run_text_to_image(request, control)))


@router.post("/edit", response_model=JobResponse)
async def edit_image(image: UploadFile = File(...), parameters: str = Form(...)) -> JobResponse:
    request = parse_parameters(EditRequest, parameters)
    require_capability(request.model_name, "edit", "This model cannot edit the whole image.")
    check_references(request.reference_ids)
    await run_in_threadpool(check_loras, request.model_name, request.loras)
    content = await read_uploaded_image(image)
    return job_response(jobs.submit("edit", lambda control: run_image_edit(request, content, image.filename, control)))


@router.post("/inpaint", response_model=JobResponse)
async def inpaint(
    prompt: str = Form(...),
    negative_prompt: str | None = Form(None),
    # Empty means the Quality model of this device.
    model_name: str | None = Form(None),
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
    consistency_lora: bool = Form(False),
    detail_crop: bool = Form(False),
    reference_ids: list[int] = Form([]),
    # JSON, as in the other routes' parameters: [{"name": ..., "scale": ...}].
    loras: str = Form("[]"),
    image: UploadFile = File(...),
    mask: UploadFile = File(...),
) -> JobResponse:
    try:
        request = InpaintForm(
            prompt=prompt,
            negative_prompt=negative_prompt,
            **({"model_name": model_name} if model_name else {}),
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
            consistency_lora=consistency_lora,
            detail_crop=detail_crop,
            reference_ids=reference_ids,
            loras=LORA_LIST.validate_json(loras),
        )
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    require_capability(request.model_name, "inpaint", "This model cannot edit a selection.")
    check_references(request.reference_ids)
    await run_in_threadpool(check_loras, request.model_name, request.loras)
    image_bytes = await read_uploaded_image(image)
    mask_bytes = await read_uploaded_image(mask)
    return job_response(jobs.submit(
        "inpaint",
        lambda control: run_inpaint(request, image_bytes, image.filename, mask_bytes, mask.filename, control),
    ))


@router.post("/outpaint", response_model=JobResponse)
async def outpaint(image: UploadFile = File(...), parameters: str = Form(...)) -> JobResponse:
    request = parse_parameters(OutpaintForm, parameters)
    require_capability(request.model_name, "outpaint", "This model cannot expand the canvas.")
    await run_in_threadpool(check_loras, request.model_name, request.loras)
    content = await read_uploaded_image(image)
    return job_response(jobs.submit("outpaint", lambda control: run_outpaint(request, content, control)))


@router.post("/faces", response_model=JobResponse)
async def fix_faces(image: UploadFile = File(...), parameters: str = Form(...), mask: UploadFile | None = File(None)) -> JobResponse:
    request = parse_parameters(FaceFixRequest, parameters)
    require_capability(request.model_name, "edit", "This model cannot fix faces.")
    check_references(request.reference_ids)
    await run_in_threadpool(check_loras, request.model_name, request.loras)
    content = await read_uploaded_image(image)
    mask_bytes = await read_uploaded_image(mask) if mask is not None else None
    return job_response(jobs.submit("edit", lambda control: run_face_fix(request, content, image.filename, mask_bytes, control)))


@router.post("/faces/detect", response_model=FaceDetectionResponse)
async def detect_faces(image: UploadFile = File(...)) -> FaceDetectionResponse:
    content = await read_uploaded_image(image)

    def candidates() -> list[FaceCandidate]:
        with Image.open(BytesIO(content)) as uploaded:
            return face_candidates(uploaded.convert("RGB"))

    # Grounding DINO runs on the CPU lane, after any selection in progress.
    with http_errors({UnidentifiedImageError: 422}):
        return FaceDetectionResponse(faces=await run_in_threadpool(jobs.run_cpu, candidates))


@router.post("/enhance", response_model=JobResponse)
async def enhance(image: UploadFile = File(...), parameters: str = Form(...), mask: UploadFile | None = File(None)) -> JobResponse:
    request = parse_parameters(EnhanceRequest, parameters)
    require_capability(request.model_name, "edit", "This model cannot enhance pictures.")
    await run_in_threadpool(check_loras, request.model_name, request.loras)
    content = await read_uploaded_image(image)
    mask_bytes = await read_uploaded_image(mask) if mask is not None else None
    return job_response(jobs.submit("edit", lambda control: run_enhance(request, content, image.filename, mask_bytes, control)))


@router.get("/history", response_model=list[GenerationRecord])
def history(limit: int = 50) -> list[dict]:
    return storage.list_generations(limit=max(1, min(limit, 100)))


@router.get("/history/{generation_id}", response_model=GenerationRecord)
def history_item(generation_id: int) -> dict:
    record = storage.get_generation(generation_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Generation not found")
    return record


@router.delete("/history/{generation_id}")
def delete_generation(generation_id: int) -> dict[str, str]:
    with http_errors({LookupError: 404}):
        storage.delete_generation(generation_id)
    return {"status": "deleted"}
