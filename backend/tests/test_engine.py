from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from app import main
from app.config import settings


class EngineWarmUpTests(unittest.TestCase):
    def test_warm_up_loads_selection_then_warms_the_quality_inpaint_pipeline(self) -> None:
        control = MagicMock()
        with patch("app.main.selector.warm_up") as selection, patch("app.main.registry.preload") as preload:
            main._warm_up_engine(control)
        selection.assert_called_once()
        request = preload.call_args.args[0]
        self.assertEqual((request.model_name, request.task, request.warm_up), (settings.warm_up_model, "inpaint", True))
        control.set_phase.assert_called_with("loading", "SAM 2.1 Tiny")

    def test_engine_reports_the_warm_up_job_and_models(self) -> None:
        job = main.jobs.submit("load-model", lambda control: None)
        with patch.object(main, "warm_up_job_id", job.id):
            body = TestClient(main.app).get("/api/engine").json()
        self.assertEqual(body["warm_up"]["id"], job.id)
        self.assertIn("flux2-klein-4b-nvfp4", [model["name"] for model in body["models"]])
        with patch.object(main, "warm_up_job_id", None):
            self.assertIsNone(TestClient(main.app).get("/api/engine").json()["warm_up"])


class EngineShutdownTests(unittest.TestCase):
    def test_shutdown_is_hidden_without_a_token_from_the_desktop_app(self) -> None:
        with patch.object(settings, "shutdown_token", None):
            self.assertEqual(TestClient(main.app).post("/api/shutdown").status_code, 404)

    def test_shutdown_needs_the_matching_token_and_then_exits_gracefully(self) -> None:
        with patch.object(settings, "shutdown_token", "secret"), patch("app.main._schedule_exit") as schedule:
            client = TestClient(main.app)
            self.assertEqual(client.post("/api/shutdown").status_code, 403)
            self.assertEqual(client.post("/api/shutdown", headers={"X-MosImage-Token": "wrong"}).status_code, 403)
            schedule.assert_not_called()
            response = client.post("/api/shutdown", headers={"X-MosImage-Token": "secret"})
        self.assertEqual(response.status_code, 202)
        schedule.assert_called_once()


if __name__ == "__main__":
    unittest.main()
