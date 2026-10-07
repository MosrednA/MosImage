from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor, TimeoutError
from dataclasses import dataclass, field
from threading import Event, Lock
from typing import Any, Callable
from uuid import uuid4

from .schemas import JobState, JobStatus, TaskKind


class JobCancelled(Exception):
    """Raised cooperatively when an inference job has been cancelled."""


@dataclass(frozen=True)
class SelectionResult:
    mask_image_url: str


JobResult = int | list[int] | SelectionResult | None


class JobControl:
    def __init__(self, manager: JobManager, job_id: str, progress_start: float = 0.0, progress_end: float = 1.0) -> None:
        self._manager = manager
        self.job_id = job_id
        self._progress_start = progress_start
        self._progress_end = progress_end

    def raise_if_cancelled(self) -> None:
        if self._manager.is_cancel_requested(self.job_id):
            raise JobCancelled("Job cancelled")

    def set_progress(self, progress: float) -> None:
        self.raise_if_cancelled()
        scaled = self._progress_start + ((self._progress_end - self._progress_start) * progress)
        self._manager.update_progress(self.job_id, scaled)

    def slice(self, index: int, count: int) -> JobControl:
        if count < 1 or index < 0 or index >= count:
            raise ValueError("Invalid job progress slice")
        span = self._progress_end - self._progress_start
        start = self._progress_start + (span * index / count)
        end = self._progress_start + (span * (index + 1) / count)
        return JobControl(self._manager, self.job_id, start, end)

    def diffusion_callback(self, total_steps: int) -> Callable[[Any, int, Any, dict[str, Any]], dict[str, Any]]:
        step_count = max(total_steps, 1)

        def on_step_end(_pipeline: Any, step_index: int, _timestep: Any, callback_kwargs: dict[str, Any]) -> dict[str, Any]:
            self.set_progress(0.15 + (0.75 * ((step_index + 1) / step_count)))
            return callback_kwargs

        return on_step_end


@dataclass
class Job:
    id: str
    kind: TaskKind
    status: JobState = "queued"
    progress: float = 0.0
    error: str | None = None
    generation_id: int | None = None
    generation_ids: list[int] = field(default_factory=list)
    mask_image_url: str | None = None
    future: Future | None = None
    cancel_event: Event = field(default_factory=Event)
    hard_cancel_claimed: bool = False


class JobManager:
    def __init__(self) -> None:
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._jobs: dict[str, Job] = {}
        self._lock = Lock()

    def submit(self, kind: TaskKind, fn: Callable[[JobControl], JobResult]) -> Job:
        job = Job(id=str(uuid4()), kind=kind)
        with self._lock:
            self._jobs[job.id] = job
        job.future = self._executor.submit(self._run, job.id, fn)
        return job

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def status(self, job_id: str) -> JobStatus | None:
        job = self.get(job_id)
        if job is None:
            return None
        return JobStatus(
            id=job.id,
            kind=job.kind,
            status=job.status,
            progress=job.progress,
            error=job.error,
            generation_id=job.generation_id,
            generation_ids=job.generation_ids,
            mask_image_url=job.mask_image_url,
        )

    def cancel(self, job_id: str) -> JobStatus | None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return None
            if job.status in {"cancelled", "completed", "failed"}:
                return self._status_for(job)
            job.cancel_event.set()
            if job.future is not None and job.future.cancel():
                job.status = "cancelled"
            else:
                # Keep the client busy until the callback finishes cleanup or
                # the supervisor replaces a worker blocked in native code.
                job.status = "cancelling"
            return self._status_for(job)

    def cancel_all(self) -> None:
        with self._lock:
            job_ids = [
                job_id
                for job_id, job in self._jobs.items()
                if job.status not in {"cancelled", "completed", "failed"}
            ]
        for job_id in job_ids:
            self.cancel(job_id)

    def is_cancel_requested(self, job_id: str) -> bool:
        with self._lock:
            job = self._jobs[job_id]
            return job.cancel_event.is_set()

    def update_progress(self, job_id: str, progress: float) -> None:
        self._update(job_id, progress=min(max(progress, 0.0), 1.0))

    def claim_hard_cancel(self, job_id: str) -> bool:
        with self._lock:
            job = self._jobs[job_id]
            if not job.cancel_event.is_set() or job.hard_cancel_claimed or job.future is None or job.future.done():
                return False
            job.hard_cancel_claimed = True
            return True

    def wait_for_completion(self, job_id: str, timeout: float) -> bool:
        with self._lock:
            future = self._jobs[job_id].future
        if future is None:
            return True
        try:
            future.result(timeout=timeout)
        except TimeoutError:
            return False
        except BaseException:
            return True
        return True

    def shutdown(self, wait: bool = True) -> None:
        self.cancel_all()
        self._executor.shutdown(wait=wait, cancel_futures=True)

    def _run(self, job_id: str, fn: Callable[[JobControl], JobResult]) -> None:
        control = JobControl(self, job_id)
        with self._lock:
            job = self._jobs[job_id]
            if job.cancel_event.is_set():
                job.status = "cancelled"
                return
            job.status = "running"
            job.progress = 0.05
        try:
            result = fn(control)
        except JobCancelled:
            self._update(job_id, status="cancelled")
            return
        except Exception as exc:  # The API should expose failures without killing the worker.
            if self.is_cancel_requested(job_id):
                self._update(job_id, status="cancelled")
                return
            self._update(job_id, status="failed", progress=1.0, error=str(exc))
            return
        selection = result if isinstance(result, SelectionResult) else None
        generation_ids = result if isinstance(result, list) else ([result] if isinstance(result, int) else [])
        with self._lock:
            job = self._jobs[job_id]
            if job.cancel_event.is_set():
                job.status = "cancelled"
                return
            job.status = "completed"
            job.progress = 1.0
            job.generation_id = generation_ids[0] if generation_ids else None
            job.generation_ids = generation_ids
            job.mask_image_url = selection.mask_image_url if selection else None

    @staticmethod
    def _status_for(job: Job) -> JobStatus:
        return JobStatus(
            id=job.id,
            kind=job.kind,
            status=job.status,
            progress=job.progress,
            error=job.error,
            generation_id=job.generation_id,
            generation_ids=job.generation_ids,
            mask_image_url=job.mask_image_url,
        )

    def _update(self, job_id: str, **changes: object) -> None:
        with self._lock:
            job = self._jobs[job_id]
            for key, value in changes.items():
                setattr(job, key, value)


jobs = JobManager()
