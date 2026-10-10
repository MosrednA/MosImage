from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
from math import ceil

from PIL import Image, ImageChops, ImageDraw, ImageFilter

from app.compositing import (
    DRIFT_LIMIT, GRAIN_LIMIT, INPAINT_GENERATION_OVERLAP, INPAINT_SEAM_TOLERANCE, _grow_mask, _inner_feather,
    _model_size, correct_color_drift, match_grain, match_seam, prepare_inpaint_workspace,
)


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
            # Exact compositing values; colour drift correction is covered below.
            workspace = replace(workspace, color_correction=False)
            self.assertEqual(workspace.composite_to(destination), source.size)

            with Image.open(destination) as result:
                self.assertEqual(result.getpixel((0, 0)), (10, 20, 30))
                self.assertEqual(result.getpixel((44, 40)), (10, 20, 30))
                boundary_red = result.getpixel((45, 40))[0]
                self.assertEqual(boundary_red, 10)
                self.assertGreater(result.getpixel((48, 40))[0], 10)
                self.assertLess(result.getpixel((48, 40))[0], 200)
                self.assertEqual(result.getpixel((60, 40)), (200, 0, 0))

    def test_regenerated_overlap_is_excluded_from_colour_drift_reference(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = Image.new("RGB", (256, 128), (100, 100, 100))
            source.save(root / "source.png")
            selection = Image.new("L", source.size, 0)
            selection.paste(255, (128, 0, 256, 128))
            selection.save(root / "mask.png")
            workspace = prepare_inpaint_workspace(
                root / "source.png", root / "mask.png", root, 256, 0, 128, composite_feather=0,
            )
            generated = source.copy()
            # The model changes an overlap which the final blend protects.
            # Treating it as unchanged context would tint the new area.
            generated.paste((80, 80, 80), (96, 0, 128, 128))
            generated.save(workspace.model_output_path)
            coverage = Image.new("L", source.size, 0)
            coverage.paste(255, (96, 0, 256, 128))
            replace(workspace, color_reference_mask=coverage).composite_to(root / "result.png")
            with Image.open(root / "result.png") as result:
                self.assertEqual(result.tobytes(), source.tobytes())

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

    def test_full_generation_coverage_keeps_the_independent_soft_composite(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = Image.new("RGB", (128, 128), (0, 0, 200))
            source.save(root / "source.png")
            mask = Image.new("L", source.size, 0)
            mask.paste(64, (32, 32, 96, 96))
            mask.paste(255, (48, 48, 80, 80))
            mask.save(root / "mask.png")
            workspace = prepare_inpaint_workspace(
                root / "source.png", root / "mask.png", root, 1024, 0, 32,
                composite_feather=0, center_selection=True,
            )
            with Image.open(workspace.model_mask_path) as generation_mask:
                self.assertEqual(generation_mask.getpixel((36, 64)), 255)
                self.assertEqual(generation_mask.getpixel((31, 64)), 0)
                Image.new("RGB", generation_mask.size, (200, 0, 0)).save(workspace.model_output_path)
            self.assertEqual(workspace.composite_mask.tobytes(), mask.tobytes())
            replace(workspace, color_correction=False).composite_to(root / "result.png")
            with Image.open(root / "result.png") as result:
                # A single 25% final blend, rather than 25% denoising coverage
                # followed by another 25% blend, while protected pixels match.
                self.assertEqual(result.getpixel((36, 64)), (50, 0, 150))
                self.assertEqual(result.getpixel((64, 64)), (200, 0, 0))
                self.assertEqual(result.getpixel((31, 64)), source.getpixel((31, 64)))

    def test_centered_model_mask_continues_selected_canvas_edges_and_corners(self) -> None:
        cases = (
            ((0, 32, 64, 96), (-8, 64), (0, 64)),
            ((64, 32, 128, 96), (135, 64), (127, 64)),
            ((32, 0, 96, 64), (64, -8), (64, 0)),
            ((32, 64, 96, 128), (64, 135), (64, 127)),
            ((0, 0, 64, 64), (-8, -8), (0, 0)),
            ((64, 0, 128, 64), (135, -8), (127, 0)),
            ((0, 64, 64, 128), (-8, 135), (0, 127)),
            ((64, 64, 128, 128), (135, 135), (127, 127)),
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = Image.new("RGB", (128, 128), (10, 20, 30))
            source.save(root / "source.png")
            for box, outside, edge in cases:
                for feather in (0, 8):
                    with self.subTest(box=box, feather=feather):
                        mask = Image.new("L", source.size, 0)
                        mask.paste(255, box)
                        mask.save(root / "mask.png")
                        workspace = prepare_inpaint_workspace(
                            root / "source.png", root / "mask.png", root, 1024, 0, 32,
                            composite_feather=feather, generation_feather=feather, center_selection=True,
                        )
                        assert workspace.model_box is not None
                        model_left, model_top, _, _ = workspace.model_box
                        with Image.open(workspace.model_mask_path) as generated_mask:
                            self.assertEqual(generated_mask.getpixel((outside[0] - model_left, outside[1] - model_top)), 255)
                            # Extending edge coverage must leave the actual surroundings protected.
                            unselected_x = box[2] + 8 if box[2] < 128 else box[0] - 8
                            unselected_y = (box[1] + box[3]) // 2
                            self.assertEqual(generated_mask.getpixel((unselected_x - model_left, unselected_y - model_top)), 0)
                            Image.new("RGB", generated_mask.size, (200, 100, 50)).save(workspace.model_output_path)
                        workspace = replace(workspace, color_correction=False)
                        workspace.composite_to(root / "result.png")
                        with Image.open(root / "result.png") as result:
                            self.assertEqual(result.getpixel(edge), (200, 100, 50))
                            protected = np.asarray(mask) == 0
                            np.testing.assert_array_equal(np.asarray(result)[protected], np.asarray(source)[protected])

    def test_resizing_a_centered_mask_keeps_full_coverage_at_the_canvas_edge(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            Image.new("RGB", (256, 256), (10, 20, 30)).save(root / "source.png")
            mask = Image.new("L", (256, 256), 0)
            mask.paste(255, (0, 64, 128, 192))
            mask.save(root / "mask.png")
            workspace = prepare_inpaint_workspace(
                root / "source.png", root / "mask.png", root, 128, 0, 64,
                composite_feather=0, generation_feather=0, center_selection=True,
            )
            with Image.open(workspace.model_mask_path) as generated_mask:
                self.assertEqual(generated_mask.size, (128, 128))
                # x=32 is the original canvas edge in the downscaled, mirrored crop.
                self.assertEqual([generated_mask.getpixel((x, 64)) for x in range(29, 36)], [255] * 7)

    def test_mask_context_preserves_soft_edge_coverage_and_unselected_padding(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            Image.new("RGB", (128, 128), (10, 20, 30)).save(root / "source.png")
            mask = Image.new("L", (128, 128), 0)
            mask.paste(128, (0, 32, 64, 96))
            mask.save(root / "mask.png")
            workspace = prepare_inpaint_workspace(
                root / "source.png", root / "mask.png", root, 1024, 0, 32,
                composite_feather=0, generation_feather=0, center_selection=True,
            )
            with Image.open(workspace.model_mask_path) as generated_mask:
                self.assertEqual(generated_mask.getpixel((24, 64)), 128)  # Past the canvas edge.
                self.assertEqual(generated_mask.getpixel((32, 64)), 128)  # On the canvas edge.
                self.assertEqual(generated_mask.getpixel((24, 8)), 0)  # Unselected edge stays protected.


    def test_a_selection_near_the_edge_stays_centred_for_the_model(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = Image.new("RGB", (256, 256), (40, 50, 60))
            source.paste((200, 10, 10), (0, 0, 8, 256))
            source.save(root / "source.png")
            mask = Image.new("L", source.size, 0)
            mask.paste(255, (16, 96, 80, 160))
            mask.save(root / "mask.png")
            workspace = prepare_inpaint_workspace(root / "source.png", root / "mask.png", root, 1024, 0, 64, center_selection=True)
            self.assertEqual(workspace.crop_box, (0, 32, 144, 224))
            self.assertEqual(workspace.model_box, (-48, 32, 144, 224))
            with Image.open(workspace.model_mask_path) as model_mask:
                left, _top, right, _bottom = model_mask.getbbox()
                self.assertEqual(left, model_mask.width - right)
            with Image.open(workspace.model_input_path) as model_input:
                # The red edge is mirrored into the missing context.
                self.assertEqual(model_input.getpixel((44, 10)), (200, 10, 10))
                self.assertEqual(model_input.getpixel((52, 10)), (200, 10, 10))
                self.assertEqual(model_input.getpixel((30, 10)), (40, 50, 60))
                Image.new("RGB", model_input.size, (0, 200, 0)).save(workspace.model_output_path)
            workspace = replace(workspace, color_correction=False)
            self.assertEqual(workspace.composite_to(root / "result.png"), source.size)
            with Image.open(root / "result.png") as result:
                self.assertEqual(result.getpixel((48, 128)), (0, 200, 0))
                self.assertEqual(result.getpixel((4, 128)), (200, 10, 10))
                self.assertEqual(result.getpixel((100, 128)), (40, 50, 60))


class MaskMorphologyTests(unittest.TestCase):
    """The array versions must match the repeated 3 x 3 Pillow filters they replace, pixel for pixel."""

    @staticmethod
    def _pillow_grow(mask: Image.Image, radius: int) -> Image.Image:
        for _ in range(radius):
            mask = mask.filter(ImageFilter.MaxFilter(3))
        return mask

    @staticmethod
    def _pillow_feather(mask: Image.Image, radius: float) -> Image.Image:
        steps = min(64, ceil(radius))
        remaining = mask.point(lambda value: 255 if value else 0)
        distance = Image.new("L", mask.size, 0)
        for _ in range(steps):
            remaining = remaining.filter(ImageFilter.MinFilter(3))
            distance = ImageChops.add(distance, remaining.point(lambda value: 1 if value else 0))
        peak = distance.getextrema()[1]
        if peak == 0:
            return mask.copy()
        denominator = min(steps, peak)
        falloff = distance.point(lambda value: round(255 * (min(value, denominator) / denominator) ** 2 * (3 - 2 * min(value, denominator) / denominator)))
        return ImageChops.darker(mask, falloff)

    @staticmethod
    def _masks() -> list[Image.Image]:
        rng = np.random.default_rng(3)
        masks = [Image.fromarray(((rng.random((37, 53)) > .7) * 255).astype(np.uint8)),
                 Image.fromarray(rng.integers(0, 256, (41, 29), dtype=np.uint8)),
                 Image.new("L", (1, 9), 255), Image.new("L", (30, 20), 255)]
        shapes = Image.new("L", (64, 48), 0)
        draw = ImageDraw.Draw(shapes)
        draw.ellipse((-10, -6, 30, 40), fill=255)  # Crosses the picture's edge.
        draw.rectangle((36, 10, 70, 30), fill=140)
        masks.append(shapes)
        hole = Image.new("L", (40, 40), 255)
        hole.putpixel((25, 12), 0)
        masks.append(hole)
        return masks

    def test_growth_matches_repeated_pillow_dilation(self) -> None:
        for index, mask in enumerate(self._masks()):
            for radius in (1, 2, 5, 16, 32):
                with self.subTest(mask=index, radius=radius):
                    self.assertEqual(_grow_mask(mask, radius).tobytes(), self._pillow_grow(mask, radius).tobytes())

    def test_inner_feather_matches_repeated_pillow_erosion(self) -> None:
        for index, mask in enumerate(self._masks()):
            for radius in (.5, 3, 8, 16, 40, 100):
                with self.subTest(mask=index, radius=radius):
                    self.assertEqual(_inner_feather(mask, radius).tobytes(), self._pillow_feather(mask, radius).tobytes())


class ModelSizeTests(unittest.TestCase):
    def test_area_budget_keeps_wide_crops_as_detailed_as_square_ones(self) -> None:
        self.assertEqual(_model_size((2048, 2048), 1024), (1024, 1024))
        width, height = _model_size((3000, 1000), 1024)
        self.assertGreater(width * height, 0.9 * 1024 * 1024)
        self.assertLessEqual(width * height, 1024 * 1024)
        self.assertAlmostEqual(width / height, 3, delta=.1)

    def test_sizes_are_whole_tokens_and_never_enlarged_by_default(self) -> None:
        for native in ((448, 448), (1000, 700), (130, 4000), (17, 17)):
            with self.subTest(native=native):
                width, height = _model_size(native, 1024)
                self.assertEqual((width % 16, height % 16), (0, 0))
                self.assertLessEqual(width, max(16, native[0]))
                self.assertLessEqual(height, max(16, native[1]))
        # A strip stays within twice the budget's side instead of becoming 4096 px long.
        self.assertLessEqual(max(_model_size((8000, 200), 1024)), 2048)

    def test_small_crops_are_enlarged_to_the_minimum(self) -> None:
        self.assertEqual(_model_size((448, 448), 1024, 768), (768, 768))
        width, height = _model_size((600, 300), 1024, 768)
        self.assertGreaterEqual(width * height, 0.95 * 768 * 768)
        # The minimum never exceeds the budget, and large crops are unaffected.
        self.assertEqual(_model_size((448, 448), 512, 768), (512, 512))
        self.assertEqual(_model_size((2048, 2048), 1024, 768), (1024, 1024))


class GenerationGrowthTests(unittest.TestCase):
    def test_generation_covers_partial_tokens_while_the_blend_keeps_the_selection(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = Image.new("RGB", (512, 512), (60, 90, 120))
            source.save(root / "source.png")
            mask = Image.new("L", source.size, 0)
            mask.paste(255, (200, 200, 296, 296))
            mask.save(root / "mask.png")
            workspace = prepare_inpaint_workspace(
                root / "source.png", root / "mask.png", root, 1024, 0, 80,
                composite_feather=0, center_selection=True, generation_growth=INPAINT_GENERATION_OVERLAP,
            )
            assert workspace.model_box is not None
            self.assertEqual(workspace.model_box, (120, 120, 376, 376))
            with Image.open(workspace.model_mask_path) as model_mask:
                # Selection at 80..176 in the 256 px model crop, grown by one token.
                self.assertEqual(model_mask.getbbox(), (64, 64, 192, 192))
                generated = Image.new("RGB", model_mask.size, (60, 90, 120))
                # The model redraws the grown ring differently, as a wider subject would.
                generated.paste((200, 30, 30), (64, 64, 192, 192))
                generated.paste((10, 200, 10), (80, 80, 176, 176))
                generated.save(workspace.model_output_path)
            self.assertEqual(workspace.composite_mask.tobytes(), mask.tobytes())
            # The seam is matched over the regenerated ring: one token in native pixels.
            self.assertEqual((workspace.seam_band, workspace.seam_tolerance), (16, INPAINT_SEAM_TOLERANCE))
            reference = workspace.color_reference_mask
            assert reference is not None
            self.assertEqual(reference.getpixel((190, 250)), 255)  # The regenerated ring is no reference.
            self.assertEqual(reference.getpixel((150, 250)), 0)
            workspace.composite_to(root / "result.png")
            with Image.open(root / "result.png") as result:
                protected = np.asarray(mask) == 0
                np.testing.assert_array_equal(np.asarray(result)[protected], np.asarray(source)[protected])
                # The red ring neither shows nor tints the new content through drift correction.
                self.assertEqual(result.getpixel((250, 250)), (10, 200, 10))


class InpaintSeamTests(unittest.TestCase):
    @staticmethod
    def _scene() -> tuple[Image.Image, Image.Image]:
        source = Image.new("RGB", (400, 400), (120, 120, 120))
        mask = Image.new("L", source.size, 0)
        mask.paste(255, (60, 60, 340, 340))
        return source, mask

    def test_a_tone_offset_at_the_edge_fades_into_the_fill(self) -> None:
        source, mask = self._scene()
        # The fill and the regenerated ring came out lighter, as a removed shadow does.
        generated = Image.new("RGB", source.size, (140, 140, 140))
        matched = match_seam(generated, source, mask, 16, INPAINT_SEAM_TOLERANCE)
        edge, centre = matched.getpixel((62, 200))[0], matched.getpixel((200, 200))[0]
        self.assertLess(edge, 128)
        self.assertEqual(centre, 140)  # Far from the edge the fill keeps its own tone.

    def test_different_content_in_the_ring_is_not_smeared_inward(self) -> None:
        source, mask = self._scene()
        generated = Image.new("RGB", source.size, (20, 40, 200))  # Remains of another object.
        generated.paste((140, 140, 140), (60, 60, 340, 340))
        tolerant = match_seam(generated, source, mask, 16, INPAINT_SEAM_TOLERANCE).getpixel((62, 200))
        self.assertLessEqual(max(abs(value - 140) for value in tolerant), 2)
        # Without the tolerance the full clipped offset would tint the edge.
        plain = match_seam(generated, source, mask, 16).getpixel((62, 200))
        self.assertGreater(max(abs(value - 140) for value in plain), DRIFT_LIMIT / 2)


class GrainMatchingTests(unittest.TestCase):
    @staticmethod
    def _grainy(size: int, level: float, seed: int) -> Image.Image:
        noise = np.random.default_rng(seed).normal(0, level, (size, size, 1))
        return Image.fromarray(np.clip(np.rint(110 + noise), 0, 255).astype(np.uint8).repeat(3, axis=2))

    @staticmethod
    def _reference(size: int, start: int, end: int) -> tuple[Image.Image, Image.Image]:
        mask = Image.new("L", (size, size), 0)
        mask.paste(255, (start, start, end, end))
        reference = Image.new("L", mask.size, 255)
        reference.paste(0, (start - 16, start - 16, end + 16, end + 16))
        return mask, reference

    def test_smoothed_content_gets_the_photos_grain_back(self) -> None:
        source = self._grainy(256, 4, seed=1)
        generated = self._grainy(256, 1, seed=2)  # The model's output is smoother.
        mask, reference = self._reference(256, 64, 192)
        matched = match_grain(generated, source, mask, reference, seed=(1, 2, 3, 4))
        inside = np.asarray(matched, dtype=np.float32)[80:176, 80:176, 0]
        self.assertAlmostEqual(float(inside.std()), 4, delta=.5)
        outside = np.asarray(matched)[:48, :48]
        np.testing.assert_array_equal(outside, np.asarray(generated)[:48, :48])
        again = match_grain(generated, source, mask, reference, seed=(1, 2, 3, 4))
        self.assertEqual(matched.tobytes(), again.tobytes())

    def test_clean_pictures_and_strong_texture_are_capped(self) -> None:
        mask, reference = self._reference(256, 64, 192)
        clean = Image.new("RGB", (256, 256), (110, 110, 110))
        self.assertIs(match_grain(clean, clean, mask, reference, seed=(0,)), clean)
        busy = self._grainy(256, 40, seed=3)
        matched = match_grain(clean, busy, mask, reference, seed=(0,))
        self.assertLessEqual(float(np.asarray(matched, dtype=np.float32)[80:176, 80:176, 0].std()), GRAIN_LIMIT + .3)


class ColorDriftCorrectionTests(unittest.TestCase):
    @staticmethod
    def _square_mask(size: int, start: int, end: int) -> Image.Image:
        mask = Image.new("L", (size, size), 0)
        mask.paste(255, (start, start, end, end))
        return mask

    def test_uniform_drift_inside_the_selection_is_removed(self) -> None:
        source = Image.new("RGB", (160, 160), (100, 120, 140))
        generated = Image.new("RGB", source.size, (110, 112, 146))
        corrected = correct_color_drift(generated, source, self._square_mask(160, 50, 110))
        for point in ((80, 80), (52, 52), (5, 5)):
            for value, expected in zip(corrected.getpixel(point), (100, 120, 140)):
                self.assertLessEqual(abs(value - expected), 1)

    def test_new_detail_is_kept_while_its_colour_cast_is_corrected(self) -> None:
        source = np.full((128, 128, 3), 90, dtype=np.uint8)
        generated = source.astype(np.int16) + 12
        generated[40:88, 40:88:2] += 60  # high-frequency stripes in the new content
        corrected = np.asarray(correct_color_drift(
            Image.fromarray(np.clip(generated, 0, 255).astype(np.uint8)), Image.fromarray(source), self._square_mask(128, 36, 92)
        ), dtype=np.int16)
        self.assertLessEqual(abs(int(corrected[64, 65, 0]) - 90), 1)
        self.assertEqual(int(corrected[64, 64, 0]) - int(corrected[64, 65, 0]), 60)

    def test_correction_is_clamped(self) -> None:
        source = Image.new("RGB", (96, 96), (200, 200, 200))
        generated = Image.new("RGB", source.size, (100, 100, 100))
        corrected = correct_color_drift(generated, source, self._square_mask(96, 32, 64))
        self.assertEqual(corrected.getpixel((48, 48)), (100 + int(DRIFT_LIMIT),) * 3)

    def test_transparent_padding_and_full_masks_are_ignored(self) -> None:
        source = Image.new("RGBA", (96, 96), (0, 0, 0, 0))
        source.paste((50, 60, 70, 255), (0, 0, 48, 96))
        generated = Image.new("RGBA", source.size, (55, 60, 70, 255))
        mask = Image.new("L", source.size, 0)
        mask.paste(255, (48, 0, 96, 96))
        corrected = correct_color_drift(generated, source, mask)
        self.assertEqual(corrected.getpixel((80, 48)), (50, 60, 70, 255))
        full = Image.new("L", source.size, 255)
        self.assertIs(correct_color_drift(generated, source, full), generated)

    def test_workspace_composite_applies_the_correction_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            Image.new("RGB", (128, 128), (80, 90, 100)).save(root / "source.png")
            self._square_mask(128, 48, 80).save(root / "mask.png")
            workspace = prepare_inpaint_workspace(root / "source.png", root / "mask.png", root, 1024, 0, 32, composite_feather=0)
            with Image.open(workspace.model_input_path) as model_input:
                Image.new("RGB", model_input.size, (88, 90, 94)).save(workspace.model_output_path)
            workspace.composite_to(root / "result.png")
            with Image.open(root / "result.png") as result:
                for value, expected in zip(result.getpixel((64, 64)), (80, 90, 100)):
                    self.assertLessEqual(abs(value - expected), 1)


if __name__ == "__main__":
    unittest.main()
