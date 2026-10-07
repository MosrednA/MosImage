from __future__ import annotations

from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from PIL import Image
from pydantic import ValidationError

from app.compositing import prepare_inpaint_workspace
from app.schemas import OutpaintForm, SelectionRequest
from app.services import prepare_outpaint_images
from fastapi.testclient import TestClient
from app.main import app


class OutpaintTests(unittest.TestCase):
    def test_expansion_uses_a_mask_conditioned_default_and_rejects_klein(self) -> None:
        request = OutpaintForm(prompt="scene", left=32)
        self.assertEqual((request.model_name, request.steps, request.guidance), ("z-image-turbo", 8, 0))
        response = TestClient(app).post(
            "/api/outpaint",
            files={"image": ("source.png", b"not read for an unsupported model", "image/png")},
            data={"parameters": '{"prompt":"scene","left":32,"model_name":"flux2-klein-4b"}'},
        )
        self.assertEqual(response.status_code, 422)
        self.assertIn("mask-conditioned", response.json()["detail"])

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
