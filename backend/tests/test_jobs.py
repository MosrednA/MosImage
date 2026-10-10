from __future__ import annotations

import time
import unittest
from threading import Event
from types import SimpleNamespace
from unittest.mock import patch

from app.jobs import FINISH_SECONDS, Job, JobControl, JobManager, _estimate_remaining


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


class JobProgressDetailTests(unittest.TestCase):
    def test_reports_phase_proposal_step_and_an_estimate_once_sampling_runs(self) -> None:
        manager = JobManager()
        at_step, release = Event(), Event()

        def work(control: JobControl) -> int:
            control.set_phase("loading", "FLUX.2 Klein")
            second = control.slice(1, 2)
            second.set_progress(0.15)
            second.set_phase("generating")
            callback = second.diffusion_callback(total_steps=4)
            callback(None, 0, None, {})
            time.sleep(0.3)
            callback(SimpleNamespace(num_timesteps=3), 1, None, {})
            at_step.set()
            release.wait(2)
            return 1

        try:
            job = manager.submit("inpaint", work)
            self.assertTrue(at_step.wait(2))
            status = manager.status(job.id)
            self.assertEqual((status.phase, status.item, status.items), ("generating", 2, 2))
            # Strength below 1 runs fewer timesteps than requested; the pipeline count wins.
            self.assertEqual((status.step, status.steps), (2, 3))
            self.assertIsNotNone(status.eta_seconds)
            self.assertGreater(status.eta_seconds, 0)
            self.assertGreaterEqual(status.elapsed_seconds, status.phase_elapsed_seconds)
            release.set()
            wait_for_status(manager, job.id, "completed")
        finally:
            release.set()
            manager.shutdown()

    def test_estimate_ignores_set_up_in_the_step_time_and_counts_remaining_proposals(self) -> None:
        # Proposal 1 of 2: step 1 ends 2 s after set-up starts, step 2 one second later, out of 4.
        job = Job(id="a", kind="inpaint", status="running", phase="generating", item=1, items=2, step=2, steps=4,
                  item_started_at=0.0, first_step_at=2.0, last_step_at=3.0, prep_seconds=2.0, step_seconds=1.0, item_steps=4)
        current = 2 * 1.0 + FINISH_SECONDS
        whole_proposal = 2.0 + 3 * 1.0 + FINISH_SECONDS
        self.assertAlmostEqual(_estimate_remaining(job, now=3.0), current + whole_proposal)

    def test_estimate_waits_for_a_step_time_unless_a_proposal_finished(self) -> None:
        first = Job(id="a", kind="inpaint", status="running", phase="generating", item=1, items=2, step=1, steps=4,
                    item_started_at=0.0, first_step_at=2.0, last_step_at=2.0, prep_seconds=2.0)
        self.assertIsNone(_estimate_remaining(first, now=2.0))
        second = Job(id="b", kind="inpaint", status="running", phase="generating", item=2, items=2,
                     item_started_at=10.0, item_durations=[10.0])
        self.assertAlmostEqual(_estimate_remaining(second, now=12.0), 8.0)

    def test_prompt_encoding_before_sampling_is_not_timed_as_part_of_a_proposal(self) -> None:
        manager = JobManager()
        try:
            def work(control: JobControl) -> int:
                for index in range(2):
                    proposal = control.slice(index, 2)
                    proposal.set_phase("preparing")
                    if index == 0:
                        time.sleep(0.4)  # encoding happens once; the second proposal reuses it
                    proposal.set_phase("generating")
                    proposal.set_phase("finishing")
                return 1

            job = manager.submit("inpaint", work)
            wait_for_status(manager, job.id, "completed")
            durations = manager._jobs[job.id].item_durations
            self.assertEqual(len(durations), 1)
            self.assertLess(durations[0], 0.2)
        finally:
            manager.shutdown()

    def test_queued_job_names_what_it_waits_for_and_selection_skips_the_gpu_queue(self) -> None:
        manager = JobManager()
        release = Event()
        try:
            warm_up = manager.submit("load-model", lambda control: release.wait(2) and None)
            wait_for_status(manager, warm_up.id, "running")
            edit = manager.submit("inpaint", lambda control: 1)
            self.assertEqual(manager.status(edit.id).waiting_for, "load-model")
            selection = manager.submit("select", lambda control: None)
            wait_for_status(manager, selection.id, "completed")
            self.assertEqual(manager.status(edit.id).status, "queued")
            release.set()
            wait_for_status(manager, edit.id, "completed")
            self.assertIsNone(manager.status(edit.id).waiting_for)
        finally:
            release.set()
            manager.shutdown()

    def test_run_cpu_executes_on_cpu_worker_lane(self) -> None:
        manager = JobManager()
        try:
            result = manager.run_cpu(lambda: 42 * 2)
            self.assertEqual(result, 84)
        finally:
            manager.shutdown()

    def test_expired_retained_jobs_are_cleaned_up(self) -> None:
        manager = JobManager()
        try:
            job = manager.submit("select", lambda control: None)
            wait_for_status(manager, job.id, "completed")
            self.assertIsNotNone(manager.status(job.id))
            # Simulate passage of time past retention window
            with patch("app.jobs.settings.job_retention_seconds", 0.05):
                time.sleep(0.06)
                # Next status check or submit triggers cleanup
                self.assertIsNone(manager.status(job.id))
        finally:
            manager.shutdown()

    def test_finished_job_timers_stop_at_completion(self) -> None:
        for state in ("completed", "failed", "cancelled"):
            job = Job(id="finished", kind="inpaint", status=state, started_at=100.0,
                      phase_started_at=106.0, completed_at=108.0)
            with self.subTest(state=state), patch("app.jobs.monotonic", return_value=150.0):
                status = JobManager._status_for(job)
                self.assertEqual(status.elapsed_seconds, 8.0)
                self.assertEqual(status.phase_elapsed_seconds, 2.0)
                self.assertIsNone(status.eta_seconds)


def _exercise_second_slice(control: JobControl) -> list[int]:
    second = control.slice(1, 2)
    second.set_progress(0.5)
    return [7]


if __name__ == "__main__":
    unittest.main()
