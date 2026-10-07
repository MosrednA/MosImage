from __future__ import annotations

import time
import unittest
from threading import Event

from app.jobs import JobControl, JobManager


def wait_for_status(manager: JobManager, job_id: str, expected: str, timeout: float = 2.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        status = manager.status(job_id)
        if status is not None and status.status == expected:
            return
        time.sleep(0.01)
    current = manager.status(job_id)
    raise AssertionError(f"Job {job_id} did not reach {expected!r}; current status is {current}")


class JobManagerCancellationTests(unittest.TestCase):
    def test_reports_all_generation_ids_from_an_alternatives_job(self) -> None:
        manager = JobManager()
        try:
            job = manager.submit("text-to-image", lambda control: [11, 12, 13])
            wait_for_status(manager, job.id, "completed")
            status = manager.status(job.id)
            self.assertIsNotNone(status)
            self.assertEqual(status.generation_id, 11)
            self.assertEqual(status.generation_ids, [11, 12, 13])
        finally:
            manager.shutdown()

    def test_progress_slice_maps_an_alternative_into_the_parent_job(self) -> None:
        manager = JobManager()
        try:
            job = manager.submit("text-to-image", lambda control: _exercise_second_slice(control))
            wait_for_status(manager, job.id, "completed")
            status = manager.status(job.id)
            self.assertIsNotNone(status)
            self.assertEqual(status.generation_ids, [7])
        finally:
            manager.shutdown()

    def test_cancels_a_queued_job_without_running_it(self) -> None:
        manager = JobManager()
        blocker_started = Event()
        release_blocker = Event()
        queued_job_ran = Event()

        def block_worker(_control: JobControl) -> None:
            blocker_started.set()
            release_blocker.wait(timeout=2.0)

        def queued_work(_control: JobControl) -> None:
            queued_job_ran.set()

        try:
            blocker = manager.submit("load-model", block_worker)
            self.assertTrue(blocker_started.wait(timeout=1.0))
            queued = manager.submit("text-to-image", queued_work)

            cancelled = manager.cancel(queued.id)

            self.assertIsNotNone(cancelled)
            self.assertEqual(cancelled.status, "cancelled")
            release_blocker.set()
            wait_for_status(manager, blocker.id, "completed")
            self.assertFalse(queued_job_ran.is_set())
        finally:
            release_blocker.set()
            manager.shutdown()

    def test_running_job_cooperatively_cancels_on_progress_callback(self) -> None:
        manager = JobManager()
        started = Event()

        def inference(control: JobControl) -> int:
            callback = control.diffusion_callback(total_steps=100)
            started.set()
            for step in range(100):
                callback(None, step, None, {})
                time.sleep(0.005)
            return 42

        try:
            job = manager.submit("text-to-image", inference)
            self.assertTrue(started.wait(timeout=1.0))

            cancelling = manager.cancel(job.id)

            self.assertIsNotNone(cancelling)
            self.assertIn(cancelling.status, {"cancelling", "cancelled"})
            wait_for_status(manager, job.id, "cancelled")
            final = manager.status(job.id)
            self.assertIsNotNone(final)
            self.assertIsNone(final.generation_id)
        finally:
            manager.shutdown()

    def test_running_job_claims_only_one_hard_cancel_watchdog(self) -> None:
        manager = JobManager()
        started = Event()
        release = Event()

        def native_work(_control: JobControl) -> None:
            started.set()
            release.wait(timeout=2.0)

        try:
            job = manager.submit("inpaint", native_work)
            self.assertTrue(started.wait(timeout=1.0))
            cancelled = manager.cancel(job.id)

            self.assertIsNotNone(cancelled)
            self.assertEqual(cancelled.status, "cancelling")
            self.assertTrue(manager.claim_hard_cancel(job.id))
            self.assertFalse(manager.claim_hard_cancel(job.id))
            self.assertFalse(manager.wait_for_completion(job.id, timeout=0.01))
            release.set()
            wait_for_status(manager, job.id, "cancelled")
        finally:
            release.set()
            manager.shutdown()

    def test_keeps_running_cancellation_pending_until_cleanup_finishes(self) -> None:
        manager = JobManager()
        started = Event()
        cleanup_started = Event()
        release_cleanup = Event()

        def inference(control: JobControl) -> int:
            started.set()
            try:
                while True:
                    control.raise_if_cancelled()
                    time.sleep(0.005)
            finally:
                cleanup_started.set()
                release_cleanup.wait(timeout=2.0)

        try:
            job = manager.submit("inpaint", inference)
            self.assertTrue(started.wait(timeout=1.0))
            self.assertFalse(manager.claim_hard_cancel(job.id))
            self.assertEqual(manager.cancel(job.id).status, "cancelling")
            self.assertTrue(cleanup_started.wait(timeout=1.0))
            self.assertEqual(manager.status(job.id).status, "cancelling")
            self.assertEqual(manager.cancel(job.id).status, "cancelling")
            release_cleanup.set()
            wait_for_status(manager, job.id, "cancelled")
            self.assertIsNone(manager.status(job.id).generation_id)
        finally:
            release_cleanup.set()
            manager.shutdown()

    def test_cannot_cancel_or_claim_watchdog_for_a_completed_job(self) -> None:
        manager = JobManager()
        try:
            job = manager.submit("text-to-image", lambda control: 42)
            wait_for_status(manager, job.id, "completed")
            self.assertEqual(manager.cancel(job.id).status, "completed")
            self.assertFalse(manager.claim_hard_cancel(job.id))
            self.assertEqual(manager.status(job.id).generation_id, 42)
        finally:
            manager.shutdown()


def _exercise_second_slice(control: JobControl) -> list[int]:
    second = control.slice(1, 2)
    second.set_progress(0.5)
    return [7]


if __name__ == "__main__":
    unittest.main()
