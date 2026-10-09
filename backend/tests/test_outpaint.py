from __future__ import annotations

from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
from pydantic import ValidationError

from app.compositing import prepare_inpaint_workspace
from app.schemas import OutpaintForm, SelectionRequest
from app.services import prepare_outpaint_images
from fastapi.testclient import TestClient
from app.main import app


class OutpaintTests(unittest.TestCase):
    def test_expansion_runs_on_klein_and_rejects_unknown_models(self) -> None:
        request = OutpaintForm(prompt="scene", left=32)
        self.assertEqual((request.model_name, request.steps, request.guidance), ("flux2-klein-9b-nvfp4", 4, 1.0))
        client = TestClient(app)
        for model in ("flux2-klein-4b", "flux2-klein-4b-nvfp4", "flux2-klein-9b-nvfp4"):
            with self.subTest(model=model), patch("app.main.jobs.submit", return_value=SimpleNamespace(id="job", kind="outpaint", status="queued")) as submit:
                response = client.post(
                    "/api/outpaint",
                    files={"image": ("source.png", b"queued, not decoded", "image/png")},
                    data={"parameters": '{"prompt":"scene","left":32,"model_name":"' + model + '"}'},
                )
                self.assertEqual(response.status_code, 200)
                self.assertEqual(submit.call_args.args[0], "outpaint")
        with patch("app.main.jobs.submit") as submit:
            response = client.post(
                "/api/outpaint",
                files={"image": ("source.png", b"never read", "image/png")},
                data={"parameters": '{"prompt":"scene","left":32,"model_name":"z-image-turbo"}'},
            )
            self.assertEqual(response.status_code, 422)
            self.assertIn("Unknown model", response.json()["detail"])
            submit.assert_not_called()

    def test_expansion_preserves_every_original_rgba_pixel(self) -> None:
        source = Image.new("RGBA", (96, 64), (10, 20, 30, 125))
        source.putpixel((25, 20), (255, 200, 0, 0))
        buffer = BytesIO()
        source.save(buffer, "PNG")
        request = OutpaintForm(prompt="continue the scene", left=32, right=64, top=16, bottom=0)
        canvas, mask = prepare_outpaint_images(buffer.getvalue(), request)
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "canvas.png").write_bytes(canvas)
            (root / "mask.png").write_bytes(mask)
            workspace = prepare_inpaint_workspace(root / "canvas.png", root / "mask.png", root, 64, .5, 128, composite_feather=0)
            with Image.open(workspace.model_input_path) as model_input:
                self.assertEqual(model_input.mode, "RGB")
                self.assertEqual(model_input.getpixel((0, 0)), (128, 128, 128))
                Image.new("RGB", model_input.size, (200, 180, 160)).save(workspace.model_output_path)
            workspace.composite_to(root / "result.png")
            with Image.open(root / "result.png") as result:
                self.assertEqual(result.size, (192, 80))
                self.assertEqual(result.crop((32, 16, 128, 80)).tobytes(), source.tobytes())
                self.assertEqual(result.getpixel((0, 0)), (200, 180, 160, 255))

    def test_expansion_hides_the_seam_and_keeps_the_original_beyond_the_band(self) -> None:
        source = Image.new("RGB", (96, 64), (100, 100, 100))
        buffer = BytesIO()
        source.save(buffer, "PNG")
        canvas, mask = prepare_outpaint_images(buffer.getvalue(), OutpaintForm(prompt="scene", left=32))
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "canvas.png").write_bytes(canvas)
            (root / "mask.png").write_bytes(mask)
            workspace = prepare_inpaint_workspace(root / "canvas.png", root / "mask.png", root, 1024, .5, 128, mask_growth=8, composite_feather=0, match_edges=True)
            # The model keeps what it may not touch and draws everything else 10 levels lighter.
            with Image.open(workspace.model_mask_path) as model_mask:
                redrawn = np.asarray(model_mask) > 0
            Image.fromarray(np.where(redrawn[..., None], 110, 100).repeat(3, axis=2).astype(np.uint8)).save(workspace.model_output_path)
            workspace.composite_to(root / "result.png")
            with Image.open(root / "result.png") as result:
                pixels = np.asarray(result.convert("RGB")).astype(int)[32, :, 0]
        self.assertTrue((pixels[40:] == 100).all())  # Untouched beyond the redrawn band.
        self.assertLessEqual(abs(pixels[31] - 100), 2)  # The new area meets the original without a step.
        self.assertLessEqual(np.abs(np.diff(pixels[:48])).max(), 2)
        self.assertGreater(pixels[0], pixels[31])  # Far from the edge the model's own tone returns.

    def test_limits_expansion_and_rejects_unexpected_parameters(self) -> None:
        with self.assertRaises(ValidationError):
            OutpaintForm(prompt="scene")
        with self.assertRaises(ValidationError):
            OutpaintForm(prompt="scene", left=-1)
        with self.assertRaises(ValidationError):
            OutpaintForm(prompt="scene", left=1, unsupported=True)
        image = BytesIO()
        Image.new("RGB", (3000, 2000)).save(image, "PNG")
        with self.assertRaisesRegex(ValueError, "4096"):
            prepare_outpaint_images(image.getvalue(), OutpaintForm(prompt="scene", left=1200))

    def test_selection_requires_bounded_finite_positive_points(self) -> None:
        for payload in ({"points": [{"x": 1, "y": 2, "label": 0}]}, {"points": [{"x": float("inf"), "y": 1, "label": 1}]}, {"points": []}):
            with self.assertRaises(ValidationError):
                SelectionRequest.model_validate(payload)


if __name__ == "__main__":
    unittest.main()
