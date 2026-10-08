from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor, TimeoutError
from dataclasses import dataclass, field
from threading import Event, Lock
from time import monotonic
from typing import Any, Callable
from uuid import uuid4

from .config import settings
from .schemas import JobPhase, JobState, JobStatus, TaskKind


class JobCancelled(Exception):
    """Raised cooperatively when an inference job has been cancelled."""


@dataclass(frozen=True)
class SelectionResult:
    mask_image_url: str


JobResult = int | list[int] | SelectionResult | None


class JobControl:
    def __init__(
        self,
        manager: JobManager,
        job_id: str,
        progress_start: float = 0.0,
        progress_end: float = 1.0,
        item: tuple[int, int] | None = None,
    ) -> None:
        self._manager = manager
        self.job_id = job_id
        self._progress_start = progress_start
        self._progress_end = progress_end
        self._item = item

    def raise_if_cancelled(self) -> None:
        if self._manager.is_cancel_requested(self.job_id):
            raise JobCancelled("Job cancelled")

    def set_progress(self, progress: float) -> None:
        self.raise_if_cancelled()
        scaled = self._progress_start + ((self._progress_end - self._progress_start) * progress)
        self._manager.update_progress(self.job_id, scaled)

    def set_phase(self, phase: JobPhase, detail: str | None = None) -> None:
        """Tell the user what the job is doing; the step count resets with every phase."""
        self.raise_if_cancelled()
        self._manager.update_phase(self.job_id, phase, detail, self._item)

    def subrange(self, start: float, end: float) -> JobControl:
        """A share of this progress range, for one stage of a longer job."""
        span = self._progress_end - self._progress_start
        return JobControl(self._manager, self.job_id, self._progress_start + span * start, self._progress_start + span * end, self._item)

    def slice(self, index: int, count: int) -> JobControl:
        if count < 1 or index < 0 or index >= count:
            raise ValueError("Invalid job progress slice")
        span = self._progress_end - self._progress_start
        start = self._progress_start + (span * index / count)
        end = self._progress_start + (span * (index + 1) / count)
        return JobControl(self._manager, self.job_id, start, end, (index + 1, count))

    def diffusion_callback(self, total_steps: int) -> Callable[[Any, int, Any, dict[str, Any]], dict[str, Any]]:
        step_count = max(total_steps, 1)

        def on_step_end(pipeline: Any, step_index: int, _timestep: Any, callback_kwargs: dict[str, Any]) -> dict[str, Any]:
            # Strength below 1 skips early timesteps; the pipeline knows the real count.
            steps = getattr(pipeline, "num_timesteps", None)
            steps = steps if isinstance(steps, int) and steps > 0 else step_count
            self.set_progress(0.15 + (0.75 * min((step_index + 1) / steps, 1.0)))
            self._manager.update_step(self.job_id, step_index + 1, steps)
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
    phase: JobPhase | None = None
    detail: str | None = None
    step: int | None = None
    steps: int | None = None
    item: int | None = None
    items: int | None = None
    started_at: float | None = None
    phase_started_at: float | None = None
    completed_at: float | None = None
    # Timing of the current proposal and the finished ones, for the estimate.
    timed_item: int | None = None
    item_started_at: float | None = None
    first_step_at: float | None = None
    last_step_at: float | None = None
    prep_seconds: float | None = None
    step_seconds: float | None = None
    item_steps: int | None = None
    item_durations: list[float] = field(default_factory=list)


def _lane(kind: TaskKind) -> str:
    # Object selection runs on the CPU, so it never waits behind GPU work.
    return "cpu" if kind == "select" else "gpu"


class JobManager:
    def __init__(self) -> None:
        self._executors = {"gpu": ThreadPoolExecutor(max_workers=1), "cpu": ThreadPoolExecutor(max_workers=1)}
        self._jobs: dict[str, Job] = {}
        self._lock = Lock()

    def submit(self, kind: TaskKind, fn: Callable[[JobControl], JobResult]) -> Job:
        job = Job(id=str(uuid4()), kind=kind)
        with self._lock:
            self._cleanup_retained_jobs()
            self._jobs[job.id] = job
        job.future = self._executors[_lane(kind)].submit(self._run, job.id, fn)
        return job

    def run_cpu(self, fn: Callable[[], Any]) -> Any:
        """Runs a synchronous operation on the CPU worker lane, sharing it with selection jobs."""
        future = self._executors["cpu"].submit(fn)
        return future.result()

    def _cleanup_retained_jobs(self) -> None:
        now = monotonic()
        retention = getattr(settings, "job_retention_seconds", 600.0)
        expired = [
            job_id
            for job_id, job in self._jobs.items()
            if job.status in {"completed", "failed", "cancelled"}
            and job.completed_at is not None
            and (now - job.completed_at) > retention
        ]
        for job_id in expired:
            del self._jobs[job_id]

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def status(self, job_id: str) -> JobStatus | None:
        with self._lock:
            self._cleanup_retained_jobs()
            job = self._jobs.get(job_id)
            if job is None:
                return None
            status = self._status_for(job)
            if job.status == "queued":
                running = next((other for other in self._jobs.values()
                                if other.status in {"running", "cancelling"} and _lane(other.kind) == _lane(job.kind)), None)
                status.waiting_for = running.kind if running else None
            return status

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
                job.completed_at = monotonic()
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

    def update_phase(self, job_id: str, phase: JobPhase, detail: str | None, item: tuple[int, int] | None) -> None:
        with self._lock:
            job = self._jobs[job_id]
            now = monotonic()
            if job.phase != phase or (item is not None and item[0] != job.item):
                job.phase_started_at = now
            job.phase, job.detail, job.step, job.steps = phase, detail, None, None
            if item is not None:
                job.item, job.items = item
            # A proposal is timed from sampling set-up, after prompt encoding, which
            # later proposals reuse; the previous proposal ends where the next begins.
            if phase == "generating" and (job.item_started_at is None or job.timed_item != job.item):
                if job.item_started_at is not None:
                    job.item_durations.append(now - job.item_started_at)
                job.timed_item, job.item_started_at, job.first_step_at, job.last_step_at = job.item, now, None, None

    def update_step(self, job_id: str, step: int, steps: int) -> None:
        with self._lock:
            job = self._jobs[job_id]
            now = monotonic()
            job.step, job.steps, job.item_steps, job.last_step_at = step, steps, steps, now
            if job.first_step_at is None:
                job.first_step_at = now
                if job.item_started_at is not None:
                    job.prep_seconds = now - job.item_started_at
            elif step >= 2:
                job.step_seconds = (now - job.first_step_at) / (step - 1)

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
        for executor in self._executors.values():
            executor.shutdown(wait=wait, cancel_futures=True)

    def _run(self, job_id: str, fn: Callable[[JobControl], JobResult]) -> None:
        control = JobControl(self, job_id)
        with self._lock:
            job = self._jobs[job_id]
            if job.cancel_event.is_set():
                job.status = "cancelled"
                job.completed_at = monotonic()
                return
            job.status = "running"
            job.progress = 0.05
            job.started_at = job.phase_started_at = monotonic()
        try:
            result = fn(control)
        except JobCancelled:
            self._update(job_id, status="cancelled", completed_at=monotonic())
            return
        except Exception as exc:  # The API should expose failures without killing the worker.
            now = monotonic()
            if self.is_cancel_requested(job_id):
                self._update(job_id, status="cancelled", completed_at=now)
                return
            self._update(job_id, status="failed", progress=1.0, error=str(exc), completed_at=now)
            return
        selection = result if isinstance(result, SelectionResult) else None
        generation_ids = result if isinstance(result, list) else ([result] if isinstance(result, int) else [])
        with self._lock:
            job = self._jobs[job_id]
            now = monotonic()
            if job.cancel_event.is_set():
                job.status = "cancelled"
                job.completed_at = now
                return
            job.status = "completed"
            job.completed_at = now
            job.progress = 1.0
            job.generation_id = generation_ids[0] if generation_ids else None
            job.generation_ids = generation_ids
            job.mask_image_url = selection.mask_image_url if selection else None

    @staticmethod
    def _status_for(job: Job) -> JobStatus:
        now = monotonic()
        eta = _estimate_remaining(job, now) if job.status == "running" else None
        return JobStatus(
            id=job.id,
            kind=job.kind,
            status=job.status,
            progress=job.progress,
            error=job.error,
            generation_id=job.generation_id,
            generation_ids=job.generation_ids,
            mask_image_url=job.mask_image_url,
            phase=job.phase,
            detail=job.detail,
            step=job.step,
            steps=job.steps,
            item=job.item,
            items=job.items,
            elapsed_seconds=round(now - job.started_at, 1) if job.started_at is not None else None,
            phase_elapsed_seconds=round(now - job.phase_started_at, 1) if job.phase_started_at is not None else None,
            eta_seconds=round(eta, 1) if eta is not None else None,
        )

    def _update(self, job_id: str, **changes: object) -> None:
        with self._lock:
            job = self._jobs[job_id]
            for key, value in changes.items():
                setattr(job, key, value)


# Decoding and blending after the last step; replaced by measured proposals.
FINISH_SECONDS = 1.5


def _estimate_remaining(job: Job, now: float) -> float | None:
    """Seconds left: the rest of the current proposal plus the whole ones after it.

    Step time is measured between steps, so set-up before the first step does
    not inflate it; set-up counts once per proposal instead.
    """
    if job.item_started_at is None:
        return None
    if job.item_durations:
        per_item: float | None = sum(job.item_durations) / len(job.item_durations)
    elif job.step_seconds is not None and job.prep_seconds is not None and job.item_steps:
        # Set-up is measured to the end of step 1, so that step is already in it.
        per_item = job.prep_seconds + (job.item_steps - 1) * job.step_seconds + FINISH_SECONDS
    else:
        per_item = None
    if job.phase == "finishing":
        current: float | None = max(FINISH_SECONDS - (now - (job.phase_started_at or now)), 0.0)
    elif job.step and job.steps and job.step_seconds is not None and job.last_step_at is not None:
        current = max((job.steps - job.step) * job.step_seconds + FINISH_SECONDS - (now - job.last_step_at), 0.0)
    elif per_item is not None:
        current = max(per_item - (now - job.item_started_at), 0.0)
    else:
        current = None
    later = max((job.items or 1) - (job.item or 1), 0)
    if current is None or (later and per_item is None):
        return None
    return current + later * (per_item or 0.0)

jobs = JobManager()
