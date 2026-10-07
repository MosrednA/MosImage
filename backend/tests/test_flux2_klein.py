from __future__ import annotations

import unittest
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from PIL import Image

from app.config import settings
from app.jobs import JobCancelled
from app.models.flux2_klein import Flux2KleinProvider, _inpaint_prompt
from app.models.registry import ModelRegistry
from app.models.z_image import ZImageTurboProvider
from app.schemas import InpaintForm


class Flux2KleinPlacementTests(unittest.TestCase):
    def setUp(self) -> None:
        self.provider = Flux2KleinProvider()

    def test_balanced_uses_offload_on_a_16_gb_profile(self) -> None:
        with patch.object(settings, "enable_cpu_offload", True):
            self.assertEqual(self.provider._placement_for("balanced"), "offload")

    def test_performance_can_keep_the_pipeline_gpu_resident(self) -> None:
        with patch.object(settings, "enable_cpu_offload", True):
            self.assertEqual(self.provider._placement_for("performance"), "cuda")

    def test_model_reports_native_editing_capabilities(self) -> None:
        info = self.provider.info()
        self.assertEqual(info.name, "flux2-klein-4b")
        self.assertEqual(info.capabilities, ["text-to-image", "inpaint"])

    def test_switching_to_inpaint_selects_native_mask_pipeline_and_releases_generation(self) -> None:
        generation = MagicMock()
        inpaint = MagicMock()
        vendor = SimpleNamespace(Flux2KleinPipeline=generation, Flux2KleinInpaintPipeline=inpaint)
        with patch.dict("sys.modules", {"diffusers": vendor}), patch.object(self.provider, "_place_pipeline"), patch.object(self.provider, "_collect") as collect:
            self.provider._load_pipe("offload", "text-to-image", MagicMock())
            self.provider._load_pipe("offload", "inpaint", MagicMock())
            collect.assert_called_once()
            inpaint.from_pretrained.assert_called_once()
            inpaint.from_pretrained.return_value.mask_processor.register_to_config.assert_called_once_with(do_binarize=False)
            self.assertEqual(self.provider._task, "inpaint")
            self.assertIs(self.provider._pipe, inpaint.from_pretrained.return_value)

    def test_failed_placement_is_not_reported_as_a_loaded_model(self) -> None:
        with patch.object(self.provider, "_create_pipeline", return_value=MagicMock()), patch.object(self.provider, "_place_pipeline", side_effect=RuntimeError("out of memory")), patch.object(self.provider, "_collect"):
            with self.assertRaisesRegex(RuntimeError, "out of memory"):
                self.provider._load_pipe("offload", "inpaint", MagicMock())
        self.assertFalse(self.provider.info().loaded)
        self.assertEqual(self.provider.info().loaded_pipelines, [])

    def test_cancel_during_placement_does_not_publish_the_pipeline(self) -> None:
        control = MagicMock()
        control.raise_if_cancelled.side_effect = [None, None, JobCancelled()]
        with patch.object(self.provider, "_create_pipeline", return_value=MagicMock()), patch.object(self.provider, "_place_pipeline"), patch.object(self.provider, "_collect"):
            with self.assertRaises(JobCancelled):
                self.provider._load_pipe("offload", "inpaint", control)
        self.assertFalse(self.provider.info().loaded)

    def test_inpaint_passes_mask_and_strength_to_the_model(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = Image.new("RGB", (64, 64), "purple")
            mask = Image.new("L", source.size, 0)
            mask.paste(255, (20, 20, 40, 40))
            source.save(root / "source.png")
            mask.save(root / "mask.png")
            pipe = MagicMock(return_value=SimpleNamespace(images=[source]))
            torch = MagicMock()
            control = MagicMock()
            request = InpaintForm(prompt="glowing mushroom", strength=.75, seed=123)
            with patch("app.models.flux2_klein.require_generation_device"), patch("app.models.flux2_klein.selected_device", return_value="cuda"), patch.object(self.provider, "_load_torch", return_value=torch), patch.object(self.provider, "_load_pipe", return_value=pipe):
                self.provider.inpaint(request, root / "source.png", root / "mask.png", root / "output.png", control)
            arguments = pipe.call_args.kwargs
            self.assertEqual(arguments["mask_image"].tobytes(), mask.tobytes())
            self.assertEqual(arguments["strength"], .75)
            self.assertNotIn("image_reference", arguments)
            self.assertTrue(arguments["prompt"].startswith("glowing mushroom."))
            self.assertTrue((root / "output.png").exists())

    def test_remove_prompt_restores_the_background_without_adding_a_subject(self) -> None:
        prompt = _inpaint_prompt(InpaintForm(prompt="violet flowers", edit_operation="remove", scene_context="soft dusk"))
        self.assertIn("reconstruct the background", prompt)
        self.assertIn("soft dusk", prompt)
        self.assertNotIn("Place one complete subject", prompt)


class ModelRegistryTests(unittest.TestCase):
    def test_registry_lists_quality_fast_and_specialist_models(self) -> None:
        names = [model.name for model in ModelRegistry().list()]
        self.assertEqual(names, ["flux2-klein-4b-nvfp4", "flux2-klein-4b", "z-image-turbo", "flux1-fill-dev"])

    def test_switching_precision_releases_the_other_klein_pipeline(self) -> None:
        registry = ModelRegistry()
        for source, target in (("flux2-klein-4b", "flux2-klein-4b-nvfp4"), ("flux2-klein-4b-nvfp4", "flux2-klein-4b")):
            with self.subTest(source=source):
                provider = registry.get(source)
                provider._pipe = object()
                registry.activate(target)
                self.assertFalse(provider.info().loaded)

    def test_activating_quality_releases_the_fast_pipeline(self) -> None:
        registry = ModelRegistry()
        fast = registry.get("z-image-turbo")
        self.assertIsInstance(fast, ZImageTurboProvider)
        fast._text_pipe = object()
        fast._text_placement = "cuda"

        selected = registry.activate("flux2-klein-4b")

        self.assertIsInstance(selected, Flux2KleinProvider)
        self.assertFalse(fast.info().loaded)


if __name__ == "__main__":
    unittest.main()
