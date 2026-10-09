"""Job status and cancellation."""
from __future__ import annotations

import os
from threading import Thread

from fastapi import APIRouter, HTTPException

from ..config import settings
from ..jobs import jobs
from ..schemas import JobStatus

router = APIRouter(prefix="/api/jobs")


@router.get("/{job_id}", response_model=JobStatus)
def job_status(job_id: str) -> JobStatus:
    status = jobs.status(job_id)
    if status is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return status


@router.post("/{job_id}/cancel", response_model=JobStatus)
def cancel_job(job_id: str) -> JobStatus:
    status = jobs.cancel(job_id)
    if status is None:
        raise HTTPException(status_code=404, detail="Job not found")
    if settings.supervised and jobs.claim_hard_cancel(job_id):
        Thread(target=_enforce_cancel, args=(job_id,), daemon=True).start()
    return status


def _enforce_cancel(job_id: str) -> None:
    if jobs.wait_for_completion(job_id, settings.cancel_grace_seconds):
        return
    # CUDA kernels and native model code cannot be interrupted safely from
    # Python. The local supervisor immediately replaces this worker.
    os._exit(75)
