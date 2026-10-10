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

from app.compositing import OUTPAINT_BLEND_BAND, prepare_inpaint_workspace, prepare_outpaint_workspace, rebase_expansion
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

    def test_model_overlap_survives_downscaling_without_expanding_final_edits(self) -> None:
        source = Image.new("RGBA", (576, 288), (20, 80, 140, 180))
        buffer = BytesIO()
        source.save(buffer, "PNG")
        canvas, mask = prepare_outpaint_images(
            buffer.getvalue(), OutpaintForm(prompt="scene", left=96, right=96, top=48, bottom=48),
        )
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "canvas.png").write_bytes(canvas)
            (root / "mask.png").write_bytes(mask)
            workspace = prepare_outpaint_workspace(root / "canvas.png", root / "mask.png", root, 256, .5, 128)
            with Image.open(workspace.model_mask_path) as model_mask:
                # An area budget of 256 x 256 keeps more of a wide canvas than a 256 px long side.
                self.assertEqual(model_mask.size, (352, 176))
                # Twenty model pixels inside the old border (44 native px)
                # remain fully editable even after a 2.2x downscale.
                for point in ((64, 88), (288, 88), (176, 42), (176, 134)):
                    self.assertEqual(model_mask.getpixel(point), 255)
                self.assertEqual(model_mask.getpixel((176, 88)), 0)
                Image.new("RGB", model_mask.size, (240, 120, 10)).save(workspace.model_output_path)
            self.assertEqual(workspace.composite_mask.getpixel((156, 192)), 0)
            self.assertIsNotNone(workspace.color_reference_mask)
            self.assertEqual(workspace.color_reference_mask.getpixel((156, 192)), 255)
            workspace.composite_to(root / "result.png")
            with Image.open(root / "result.png") as result:
                core = (96 + OUTPAINT_BLEND_BAND, 48 + OUTPAINT_BLEND_BAND,
                        672 - OUTPAINT_BLEND_BAND, 336 - OUTPAINT_BLEND_BAND)
                original_core = (OUTPAINT_BLEND_BAND, OUTPAINT_BLEND_BAND,
                                 source.width - OUTPAINT_BLEND_BAND, source.height - OUTPAINT_BLEND_BAND)
                self.assertEqual(result.crop(core).tobytes(), source.crop(original_core).tobytes())
                self.assertEqual(result.getpixel((0, 0))[3], 255)

    def test_outpaint_blend_footprint_stays_native_when_model_resolution_changes(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = Image.new("RGB", (384, 256), (100, 100, 100))
            buffer = BytesIO()
            source.save(buffer, "PNG")
            canvas, mask = prepare_outpaint_images(buffer.getvalue(), OutpaintForm(prompt="scene", left=128))
            (root / "canvas.png").write_bytes(canvas)
            (root / "mask.png").write_bytes(mask)
            (root / "native").mkdir()
            (root / "reduced").mkdir()
            native = prepare_outpaint_workspace(root / "canvas.png", root / "mask.png", root / "native", 1024, .5, 128)
            reduced = prepare_outpaint_workspace(root / "canvas.png", root / "mask.png", root / "reduced", 256, .5, 128)
            self.assertEqual(native.composite_mask.tobytes(), reduced.composite_mask.tobytes())
            self.assertTrue(np.all(np.asarray(reduced.composite_mask)[:, 128 + OUTPAINT_BLEND_BAND:] == 0))
            # The overlap redraws the original, so its whole offset is carried.
            self.assertEqual((native.seam_band, native.seam_tolerance), (OUTPAINT_BLEND_BAND, None))

    def _expanded(self, picture: Image.Image, request: OutpaintForm, root: Path) -> tuple[Image.Image, Image.Image]:
        """The engine's result for `picture`, with a model that draws flat grey, and the canvas it saw."""
        buffer = BytesIO()
        picture.save(buffer, "PNG")
        canvas, mask = prepare_outpaint_images(buffer.getvalue(), request)
        root.mkdir()
        (root / "canvas.png").write_bytes(canvas)
        (root / "mask.png").write_bytes(mask)
        workspace = prepare_outpaint_workspace(root / "canvas.png", root / "mask.png", root, 1024, .5, 128)
        with Image.open(workspace.model_input_path) as model_input:
            Image.new("RGB", model_input.size, (150, 150, 150)).save(workspace.model_output_path)
        workspace.composite_to(root / "result.png")
        with Image.open(root / "result.png") as result, Image.open(root / "canvas.png") as seen:
            return result.copy(), seen.copy()

    def test_rebased_expansion_keeps_layers_out_of_the_background(self) -> None:
        x, y = np.meshgrid(np.arange(96), np.arange(64))
        background = Image.fromarray(np.stack([x * 2, y * 3, np.full_like(x, 50)], axis=-1).astype(np.uint8))
        # A layer covers the left of the picture, up to its edge; the right shows the background.
        picture = background.copy()
        picture.paste((200, 30, 30), (0, 0, 40, 64))
        request = OutpaintForm(prompt="scene", left=32, right=32)
        band = OUTPAINT_BLEND_BAND
        with TemporaryDirectory() as directory:
            result, seen = self._expanded(picture, request, Path(directory) / "layered")
        rebased = np.asarray(rebase_expansion(result, seen, background, (32, 0))).astype(int)
        made = np.asarray(result).astype(int)
        flat = np.asarray(background.convert("RGBA")).astype(int)

        self.assertTrue((rebased[:, :32] == made[:, :32]).all())  # The new area continues the picture with its layer.
        self.assertTrue((rebased[:, 128:] == made[:, 128:]).all())
        self.assertTrue((rebased[:, 32 + band:128 - band] == flat[:, band:96 - band]).all())  # No copy of the layer.
        self.assertTrue((rebased[:, 128 - band:128] == made[:, 128 - band:128]).all())  # The blend stays where the background shows.
        # Under the layer the redraw fades into the background, not into the layer.
        self.assertLessEqual(np.abs(made[:, 32 + band - 2, :3] - (200, 30, 30)).max(), 3)
        self.assertLessEqual(np.abs(rebased[:, 32 + band - 2] - flat[:, band - 2]).max(), 3)
        self.assertTrue((rebased[:, 32 + band - 1] == flat[:, band - 1]).all())
        with self.assertRaisesRegex(ValueError, "line up"):
            rebase_expansion(result, seen, background, (100, 0))

    def test_rebasing_an_expansion_of_the_background_itself_changes_nothing(self) -> None:
        background = Image.new("RGB", (96, 64), (90, 110, 70))
        background.paste((20, 40, 60), (0, 0, 30, 64))
        with TemporaryDirectory() as directory:
            result, seen = self._expanded(background, OutpaintForm(prompt="scene", left=32, top=16), Path(directory) / "plain")
        self.assertEqual(rebase_expansion(result, seen, background, (32, 16)).tobytes(), result.tobytes())

    def test_selection_requires_bounded_finite_positive_points(self) -> None:
        for payload in ({"points": [{"x": 1, "y": 2, "label": 0}]}, {"points": [{"x": float("inf"), "y": 1, "label": 1}]}, {"points": []}):
            with self.assertRaises(ValidationError):
                SelectionRequest.model_validate(payload)


if __name__ == "__main__":
    unittest.main()
