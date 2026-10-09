"""Selection jobs: click-to-select, select by text and subject cut-out."""
from __future__ import annotations

from fastapi import APIRouter, File, Form, UploadFile

from ..jobs import jobs
from ..schemas import CutoutRequest, JobResponse, SelectionRequest, TextSelectionRequest
from ..selection import selector
from .common import job_response, parse_parameters, read_uploaded_image

router = APIRouter(prefix="/api")


@router.post("/select", response_model=JobResponse)
async def select_object(image: UploadFile = File(...), parameters: str = Form(...)) -> JobResponse:
    request = parse_parameters(SelectionRequest, parameters)
    content = await read_uploaded_image(image)
    return job_response(jobs.submit("select", lambda control: selector.select(content, request, control)))


@router.post("/select-text", response_model=JobResponse)
async def select_by_text(image: UploadFile = File(...), parameters: str = Form(...)) -> JobResponse:
    request = parse_parameters(TextSelectionRequest, parameters)
    content = await read_uploaded_image(image)
    return job_response(jobs.submit("select", lambda control: selector.select_text(content, request, control)))


@router.post("/cutout", response_model=JobResponse)
async def cut_out_subject(image: UploadFile = File(...), parameters: str = Form("{}")) -> JobResponse:
    request = parse_parameters(CutoutRequest, parameters)
    content = await read_uploaded_image(image)
    return job_response(jobs.submit("select", lambda control: selector.cutout(content, request, control)))
