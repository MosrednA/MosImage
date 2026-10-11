from __future__ import annotations

import json
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

from app import storage
from app.config import settings
from app.enhance import (ENHANCE_STRENGTH, MAX_TILES, SINGLE_PASS_AREA, TILE_AREA, TILE_OVERLAP, enhance_region, kept_tones,
                         redraw, run_enhance, tile_layout, tile_weights)
from app.main import app
from app.schemas import EnhanceRequest
from tests.test_references import isolated_data, png


class TileLayoutTests(unittest.TestCase):
    def test_a_generated_picture_is_one_pass(self) -> None:
        self.assertEqual(tile_layout(1024, 1024), [(0, 0, 1024, 1024)])
        self.assertEqual(tile_layout(1088, 1024), [(0, 0, 1088, 1024)])
        self.assertLessEqual(1088 * 1024, SINGLE_PASS_AREA)

    def test_large_regions_get_the_fewest_overlapping_tiles_that_cover_them(self) -> None:
        for width, height in ((2048, 2048), (3000, 1200), (1400, 900), (2816, 2816), (700, 2600)):
            with self.subTest(size=(width, height)):
                tiles = tile_layout(width, height)
                self.assertLessEqual(len(tiles), MAX_TILES)
                covered = np.zeros((height, width), dtype=bool)
                for left, top, right, bottom in tiles:
                    self.assertTrue(0 <= left < right <= width and 0 <= top < bottom <= height)
                    self.assertLessEqual((right - left) * (bottom - top), TILE_AREA)
                    covered[top:bottom, left:right] = True
                self.assertTrue(covered.all())
                # Neighbours share at least the overlap, so their seam can be blended.
                starts = sorted({tile[0] for tile in tiles})
                tile_width = tiles[0][2] - tiles[0][0]
                for first, second in zip(starts, starts[1:]):
                    self.assertGreaterEqual(first + tile_width - second, TILE_OVERLAP)
        self.assertEqual(len(tile_layout(2048, 2048)), 6)

    def test_an_area_beyond_nine_tiles_asks_for_a_smaller_selection(self) -> None:
        with self.assertRaisesRegex(ValueError, "Select a part"):
            tile_layout(4000, 3000)

    def test_weights_ramp_across_inner_edges_only(self) -> None:
        weights = tile_weights((0, 0, 600, 400), 1000, 400)
        self.assertEqual(weights[200, 0], 1)
        self.assertEqual(weights[0, 300], 1)
        self.assertLess(weights[200, 599], 0.01)
        self.assertEqual(weights[200, 600 - TILE_OVERLAP - 1], 1)
        # Two overlapping tiles always add up to a positive weight.
        right = tile_weights((400, 0, 1000, 400), 1000, 400)
        self.assertGreater(float((weights[:, 400:] + right[:, :200]).min()), 0.5)


class ToneAndRegionTests(unittest.TestCase):
    def test_kept_tones_restores_the_original_colour_but_keeps_new_detail(self) -> None:
        original = Image.new("RGB", (128, 128), (100, 120, 140))
        noise = np.random.default_rng(1).integers(-20, 21, (128, 128, 1))
        enhanced = Image.fromarray(np.clip(np.full((128, 128, 3), (140, 120, 100)) + noise, 0, 255).astype(np.uint8))
        result = np.asarray(kept_tones(enhanced, original), dtype=np.float32)
        self.assertTrue(np.allclose(result.reshape(-1, 3).mean(0), (100, 120, 140), atol=2))
        self.assertGreater(float(result[..., 0].std()), 8)

    def test_region_is_the_whole_picture_or_the_selection_with_context(self) -> None:
        self.assertEqual(enhance_region(None, 800, 600), (0, 0, 800, 600))
        selection = np.zeros((600, 800), dtype=np.uint8)
        selection[100:300, 200:600] = 255
        self.assertEqual(enhance_region(selection, 800, 600), (168, 68, 632, 332))
        selection[:] = 0
        selection[0:50, 750:800] = 255
        self.assertEqual(enhance_region(selection, 800, 600), (718, 0, 800, 82))
        with self.assertRaisesRegex(ValueError, "empty"):
            enhance_region(np.zeros((600, 800), dtype=np.uint8), 800, 600)


class EnhanceServiceTests(unittest.TestCase):
    def provider(self, noisy: bool = False, strength: float = ENHANCE_STRENGTH, detail: float = .4) -> MagicMock:
        provider = MagicMock()

        def fake_edit(request, input_path, output_path, _control, **kwargs):
            self.assertEqual((request.mode, request.strength, request.detail), ("enhance", strength, detail))
            self.assertEqual(kwargs, {})
            with Image.open(input_path) as tile:
                # Klein renders at its own size; the service scales back.
                rendered = tile.convert("RGB").resize((1024, 1024))
            if noisy:
                rendered = Image.fromarray(np.random.default_rng(3).integers(0, 256, (1024, 1024, 3), dtype=np.uint8))
            rendered.save(output_path)
            return 21, 0

        provider.edit.side_effect = fake_edit
        return provider

    def test_whole_picture_is_enhanced_and_recorded(self) -> None:
        with isolated_data():
            picture = Image.new("RGB", (300, 200), (60, 90, 120))
            provider = self.provider()
            with patch("app.enhance.registry.activate", return_value=provider):
                ids = run_enhance(EnhanceRequest(count=2), png(picture), "photo.png", None, MagicMock())
            self.assertEqual(provider.edit.call_count, 2)
            record = storage.get_generation(ids[0])
            self.assertEqual((record["task"], record["edit_operation"], record["prompt"], record["strength"]), ("edit", "enhance", "Enhance", .4))
            self.assertIsNone(record["mask_image_path"])
            with Image.open(settings.data_dir / record["output_image_path"]) as output:
                self.assertEqual(output.size, (300, 200))
                self.assertEqual(output.getpixel((150, 100)), (60, 90, 120))

    def test_fast_enhances_without_the_detail_lora_and_redraws_more_for_strong(self) -> None:
        # The LoRA was trained on 9B; Fast relies on the enhance prompt and a deeper partial redraw.
        self.assertEqual(redraw(settings.quality_model, .7), (.7, ENHANCE_STRENGTH))
        self.assertEqual(redraw(settings.fast_model, .4), (0, ENHANCE_STRENGTH))
        self.assertAlmostEqual(redraw(settings.fast_model, .7)[1], .75)
        self.assertEqual((redraw(settings.fast_model, .1)[1], redraw(settings.fast_model, 1)[1]), (.5, .875))
        with isolated_data():
            provider = self.provider(strength=redraw(settings.fast_model, .7)[1], detail=0)
            with patch("app.enhance.registry.activate", return_value=provider):
                [generation] = run_enhance(EnhanceRequest(model_name=settings.fast_model, detail=.7), png(Image.new("RGB", (300, 200))),
                                           "photo.png", None, MagicMock())
            record = storage.get_generation(generation)
            self.assertEqual((record["model_name"], record["strength"]), (settings.fast_model, .7))
        provider.edit.assert_called_once()

    def test_selection_changes_only_the_selected_part(self) -> None:
        with isolated_data():
            picture = Image.new("RGB", (400, 300), (60, 90, 120))
            selection = Image.new("L", picture.size, 0)
            selection.paste(255, (100, 100, 200, 200))
            provider = self.provider(noisy=True)
            with patch("app.enhance.registry.activate", return_value=provider):
                [generation] = run_enhance(EnhanceRequest(), png(picture), "photo.png", png(selection), MagicMock())
            record = storage.get_generation(generation)
            self.assertIsNotNone(record["mask_image_path"])
            with Image.open(settings.data_dir / record["output_image_path"]) as output:
                pixels = np.asarray(output, dtype=np.int16)
            self.assertEqual(provider.edit.call_count, 1)
            outside = np.asarray(selection) == 0
            self.assertTrue((pixels[outside] == (60, 90, 120)).all())
            self.assertGreater(float(pixels[120:180, 120:180].std()), 5)

    def test_large_pictures_render_in_tiles_and_blend_without_seams(self) -> None:
        with isolated_data():
            gradient = np.tile(np.linspace(0, 255, 2000, dtype=np.float32)[None, :, None], (1200, 1, 3)).astype(np.uint8)
            provider = self.provider()
            with patch("app.enhance.registry.activate", return_value=provider):
                [generation] = run_enhance(EnhanceRequest(seed=4), png(Image.fromarray(gradient)), "photo.png", None, MagicMock())
            self.assertEqual(provider.edit.call_count, len(tile_layout(2000, 1200)))
            self.assertGreater(provider.edit.call_count, 1)
            self.assertEqual(len({call.args[0].seed for call in provider.edit.call_args_list}), provider.edit.call_count)
            with Image.open(settings.data_dir / storage.get_generation(generation)["output_image_path"]) as output:
                difference = np.abs(np.asarray(output, dtype=np.int16) - gradient.astype(np.int16))
            self.assertLessEqual(int(difference.max()), 3)

    def test_too_large_or_failed_runs_leave_no_files(self) -> None:
        with isolated_data():
            with patch("app.enhance.registry.activate", return_value=MagicMock()):
                with self.assertRaisesRegex(ValueError, "too large"):
                    run_enhance(EnhanceRequest(), png(Image.new("RGB", (4000, 3000))), "photo.png", None, MagicMock())
            self.assertEqual(list(settings.input_dir.iterdir()), [])
            self.assertEqual(list(settings.output_dir.iterdir()), [])


class EnhanceApiTests(unittest.TestCase):
    def test_enhance_queues_an_edit_job_on_either_preset(self) -> None:
        client = TestClient(app)
        picture = ("picture.png", png(Image.new("RGB", (64, 48))), "image/png")
        queued = SimpleNamespace(id="job", kind="edit", status="queued")
        with patch("app.main.jobs.submit", return_value=queued) as submit, patch("app.routes.generation.run_enhance") as run:
            mask = ("mask.png", png(Image.new("L", (64, 48), 255)), "image/png")
            response = client.post("/api/enhance", files={"image": picture, "mask": mask}, data={"parameters": json.dumps({"detail": .7})})
            self.assertEqual(response.status_code, 200)
            self.assertEqual(submit.call_args.args[0], "edit")
            submit.call_args.args[1](MagicMock())
            request, _, _, mask_bytes, _ = run.call_args.args
            self.assertEqual((request.model_name, request.detail), (settings.quality_model, .7))
            self.assertIsNotNone(mask_bytes)
            parameters = json.dumps({"model_name": settings.fast_model})
            self.assertEqual(client.post("/api/enhance", files={"image": picture}, data={"parameters": parameters}).status_code, 200)
            submit.call_args.args[1](MagicMock())
            self.assertEqual(run.call_args.args[0].model_name, settings.fast_model)


if __name__ == "__main__":
    unittest.main()
