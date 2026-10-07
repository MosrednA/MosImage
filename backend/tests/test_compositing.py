from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from PIL import Image

from app.compositing import _grow_mask, _inner_feather, prepare_inpaint_workspace


class InpaintCompositingTests(unittest.TestCase):
    def test_context_crop_preserves_native_size_and_unselected_pixels(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            source_path = root / "source.png"
            mask_path = root / "mask.png"
            destination = root / "result.png"
            work_dir = root / "work"
            work_dir.mkdir()

            source = Image.new("RGB", (120, 80), (10, 20, 30))
            source.save(source_path)
            # Transparent white outside the selection matches browser canvas masks
            # and verifies that alpha, not hidden RGB, controls the edit.
            mask = Image.new("RGBA", source.size, (255, 255, 255, 0))
            for y in range(30, 50):
                for x in range(45, 75):
                    mask.putpixel((x, y), (255, 255, 255, 255))
            mask.save(mask_path)

            workspace = prepare_inpaint_workspace(
                source_path,
                mask_path,
                work_dir,
                max_dimension=64,
                context_ratio=0.25,
                minimum_context=8,
            )
            self.assertLess(workspace.source.crop(workspace.crop_box).width, source.width)
            with Image.open(workspace.model_input_path) as model_input:
                model_size = model_input.size
                self.assertEqual(model_input.width % 32, 0)

            Image.new("RGB", model_size, (200, 0, 0)).save(workspace.model_output_path)
            self.assertEqual(workspace.composite_to(destination), source.size)

            with Image.open(destination) as result:
                self.assertEqual(result.getpixel((0, 0)), (10, 20, 30))
                self.assertEqual(result.getpixel((44, 40)), (10, 20, 30))
                boundary_red = result.getpixel((45, 40))[0]
                self.assertEqual(boundary_red, 10)
                self.assertGreater(result.getpixel((48, 40))[0], 10)
                self.assertLess(result.getpixel((48, 40))[0], 200)
                self.assertEqual(result.getpixel((60, 40)), (200, 0, 0))

    def test_explicit_growth_and_inward_feather_have_separate_footprints(self) -> None:
        mask = Image.new("L", (25, 25), 0)
        for y in range(8, 17):
            for x in range(8, 17):
                mask.putpixel((x, y), 255)

        grown = _grow_mask(mask, 3)
        feathered = _inner_feather(mask, 3)

        self.assertEqual(grown.getbbox(), (5, 5, 20, 20))
        self.assertEqual(feathered.getpixel((7, 12)), 0)
        self.assertEqual(feathered.getpixel((8, 12)), 0)
        self.assertGreater(feathered.getpixel((9, 12)), 0)
        self.assertLess(feathered.getpixel((9, 12)), 255)
        self.assertEqual(feathered.getpixel((12, 12)), 255)

    def test_soft_alpha_is_capped_instead_of_squared(self) -> None:
        mask = Image.new("L", (32, 32), 128)
        self.assertEqual(_inner_feather(mask, 8).getpixel((16, 16)), 128)

    def test_zero_feather_and_thin_selections_remain_editable(self) -> None:
        mask = Image.new("L", (17, 17), 0)
        mask.putpixel((8, 8), 255)
        self.assertEqual(_inner_feather(mask, 0).tobytes(), mask.tobytes())
        self.assertEqual(_inner_feather(mask, 8).getpixel((8, 8)), 255)

    def test_generation_mask_is_not_silently_expanded_and_surroundings_are_included(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = Image.new("RGB", (1024, 1024), (30, 20, 10))
            mask = Image.new("L", source.size, 0)
            for y in range(480, 544):
                for x in range(480, 544):
                    mask.putpixel((x, y), 255)
            source.save(root / "source.png")
            mask.save(root / "mask.png")
            workspace = prepare_inpaint_workspace(root / "source.png", root / "mask.png", root, 1024, .5, 192)
            with Image.open(workspace.model_mask_path) as generated_mask:
                self.assertEqual(generated_mask.getbbox(), (192, 192, 256, 256))
                self.assertEqual(generated_mask.getpixel((191, 220)), 0)
            with Image.open(workspace.model_input_path) as scene:
                self.assertEqual(scene.size, (448, 448))
                self.assertEqual(scene.getpixel((0, 0)), (30, 20, 10))

    def test_rejects_an_empty_mask(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            source_path = root / "source.png"
            mask_path = root / "mask.png"
            Image.new("RGB", (64, 64), "black").save(source_path)
            Image.new("L", (64, 64), 0).save(mask_path)

            with self.assertRaisesRegex(ValueError, "mask is empty"):
                prepare_inpaint_workspace(source_path, mask_path, root, 1024, 0.5, 128)

    def test_context_padding_matches_visible_rectangle_even_for_a_large_selection(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            Image.new("RGB", (1024, 1024)).save(root / "source.png")
            mask = Image.new("L", (1024, 1024), 0)
            mask.paste(255, (200, 200, 600, 600))
            mask.save(root / "mask.png")
            workspace = prepare_inpaint_workspace(root / "source.png", root / "mask.png", root, 1024, 0, 64)
            self.assertEqual(workspace.crop_box, (136, 136, 664, 664))

    def test_generation_feather_retains_soft_values_without_expanding_the_selection(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            Image.new("RGB", (256, 256)).save(root / "source.png")
            mask = Image.new("L", (256, 256), 0)
            mask.paste(255, (64, 64, 192, 192))
            mask.paste(128, (100, 100, 156, 156))
            mask.save(root / "mask.png")
            workspace = prepare_inpaint_workspace(root / "source.png", root / "mask.png", root, 1024, 0, 32, generation_feather=16)
            with Image.open(workspace.model_mask_path) as generated_mask:
                self.assertEqual(generated_mask.getpixel((31, 70)), 0)
                self.assertEqual(generated_mask.getpixel((32, 70)), 0)
                self.assertGreater(generated_mask.getpixel((33, 70)), 0)
                self.assertLess(generated_mask.getpixel((33, 70)), 255)
                self.assertEqual(generated_mask.getpixel((96, 96)), 128)
                self.assertEqual(generated_mask.getpixel((64, 70)), 255)

    def test_zero_generation_feather_preserves_the_users_soft_alpha(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            Image.new("RGB", (64, 64)).save(root / "source.png")
            mask = Image.new("L", (64, 64), 0)
            mask.paste(128, (16, 16, 48, 48))
            mask.save(root / "mask.png")
            workspace = prepare_inpaint_workspace(root / "source.png", root / "mask.png", root, 1024, 0, 32, generation_feather=0)
            with Image.open(workspace.model_mask_path) as generated_mask:
                self.assertEqual(generated_mask.getpixel((32, 32)), 128)


if __name__ == "__main__":
    unittest.main()
