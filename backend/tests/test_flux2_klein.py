from __future__ import annotations

import unittest
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from PIL import Image

from app.config import settings
from app.jobs import JobCancelled
from app.models.base import ModelReference
from app.models.flux2_klein import MULTI_REFERENCE_AREA, REFERENCE_BACKGROUND, Flux2KleinProvider, _inpaint_prompt
from app.models.registry import ModelRegistry
from app.models.z_image import ZImageTurboProvider
from app.schemas import InpaintForm, LoadModelRequest


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

    def test_switching_to_inpaint_shares_the_loaded_weights_without_reloading(self) -> None:
        generation = MagicMock()
        generation.from_pretrained.return_value.config = {"is_distilled": True}
        inpaint = MagicMock()
        vendor = SimpleNamespace(Flux2KleinPipeline=generation, Flux2KleinInpaintPipeline=inpaint)
        control = MagicMock()
        with patch.dict("sys.modules", {"diffusers": vendor}), patch.object(self.provider, "_place_pipeline") as place, patch.object(self.provider, "_collect") as collect:
            text = self.provider._load_pipe("offload", "text-to-image", control)
            control.set_phase.assert_called_once_with("loading", self.provider.display_name)
            edit = self.provider._load_pipe("offload", "inpaint", control)
            collect.assert_not_called()
            control.set_phase.assert_called_once()
            inpaint.from_pretrained.assert_not_called()
            arguments = inpaint.call_args.kwargs
            self.assertIs(arguments["transformer"], text.transformer)
            self.assertIs(arguments["text_encoder"], text.text_encoder)
            self.assertTrue(arguments["is_distilled"])
            edit.mask_processor.register_to_config.assert_called_once_with(do_binarize=False)
            # Offload hooks are re-installed for the new wrapper around the shared modules.
            self.assertEqual(place.call_count, 2)
            self.assertIs(self.provider._load_pipe("offload", "text-to-image", control), text)
            self.assertEqual(sorted(self.provider.info().loaded_pipelines), ["inpaint:offload", "text-to-image:offload"])

    def test_changing_placement_releases_every_pipeline(self) -> None:
        with patch.object(self.provider, "_create_pipeline", return_value=MagicMock()), patch.object(self.provider, "_place_pipeline"), patch.object(self.provider, "_collect") as collect:
            self.provider._load_pipe("offload", "inpaint", MagicMock())
            self.provider._load_pipe("cuda", "inpaint", MagicMock())
        collect.assert_called_once()
        self.assertEqual(self.provider.info().loaded_pipelines, ["inpaint:cuda"])

    def test_warm_up_preload_runs_one_tiny_step_and_readies_both_tasks(self) -> None:
        pipe = MagicMock()
        control = MagicMock()
        with patch("app.models.flux2_klein.require_generation_device"), patch("app.models.flux2_klein.selected_device", return_value="cuda"), \
                patch.object(self.provider, "_load_torch", return_value=MagicMock()), patch.object(self.provider, "_load_pipe", return_value=pipe) as load:
            self.provider.preload(LoadModelRequest(model_name=self.provider.name, task="inpaint", warm_up=True), control)
        arguments = pipe.call_args.kwargs
        self.assertEqual((arguments["width"], arguments["height"], arguments["num_inference_steps"]), (256, 256, 1))
        self.assertIn("mask_image", arguments)
        self.assertEqual([call.args[1] for call in load.call_args_list], ["inpaint", "text-to-image"])
        control.set_phase.assert_called_with("warming", self.provider.display_name)

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
            self.assertTrue(pipe.encode_prompt.call_args.kwargs["prompt"].startswith("glowing mushroom."))
            self.assertTrue((root / "output.png").exists())

    def test_reference_image_is_passed_for_replacement_but_not_for_removal(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = Image.new("RGB", (64, 64), "purple")
            source.save(root / "source.png")
            Image.new("L", source.size, 255).save(root / "mask.png")
            Image.new("RGB", (1200, 2400), "orange").save(root / "reference.png")
            reference = [ModelReference("Anna", root / "reference.png")]
            pipe = MagicMock(return_value=SimpleNamespace(images=[source]))
            with patch("app.models.flux2_klein.require_generation_device"), patch("app.models.flux2_klein.selected_device", return_value="cuda"), patch.object(self.provider, "_load_torch", return_value=MagicMock()), patch.object(self.provider, "_load_pipe", return_value=pipe):
                self.provider.inpaint(InpaintForm(prompt="sitting on a bench"), root / "source.png", root / "mask.png", root / "output.png", MagicMock(), references=reference)
                arguments = pipe.call_args.kwargs
                self.assertEqual(arguments["image_reference"].size, (512, 1024))
                self.assertIn("from the reference image", pipe.encode_prompt.call_args.kwargs["prompt"])
                self.provider.inpaint(InpaintForm(prompt="grass", edit_operation="remove"), root / "source.png", root / "mask.png", root / "output.png", MagicMock(), references=reference)
                self.assertNotIn("image_reference", pipe.call_args.kwargs)

    def test_several_references_share_one_padded_size_and_are_named_in_order(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = Image.new("RGB", (64, 64), "purple")
            source.save(root / "source.png")
            Image.new("L", source.size, 255).save(root / "mask.png")
            Image.new("RGB", (600, 1200), "orange").save(root / "anna.png")
            Image.new("RGB", (900, 900), "blue").save(root / "tom.png")
            references = [ModelReference("Anna", root / "anna.png"), ModelReference("Tom", root / "tom.png")]
            pipe = MagicMock(return_value=SimpleNamespace(images=[source]))
            control = MagicMock()
            with patch("app.models.flux2_klein.require_generation_device"), patch("app.models.flux2_klein.selected_device", return_value="cuda"), patch.object(self.provider, "_load_torch", return_value=MagicMock()), patch.object(self.provider, "_load_pipe", return_value=pipe):
                self.provider.inpaint(InpaintForm(prompt="Anna and Tom on a bench"), root / "source.png", root / "mask.png", root / "output.png", control, references=references)
            images = pipe.call_args.kwargs["image_reference"]
            self.assertEqual(len(images), 2)
            self.assertEqual(images[0].size, images[1].size)
            width, height = images[0].size
            self.assertTrue(width % 16 == 0 and height % 16 == 0)
            self.assertLessEqual(width * height, MULTI_REFERENCE_AREA * 1.1)
            # The padding keeps the whole subject: the centre is the reference, the edge is neutral grey.
            self.assertEqual(images[0].getpixel((width // 2, height // 2)), (255, 165, 0))
            self.assertEqual(images[0].getpixel((0, height // 2)), REFERENCE_BACKGROUND)
            self.assertIn("Anna is reference image 1, Tom is reference image 2", pipe.encode_prompt.call_args.kwargs["prompt"])
            control.set_phase.assert_called_with("generating")

    def test_prompt_is_encoded_once_for_repeated_proposals_until_unload(self) -> None:
        pipe = MagicMock()
        pipe.encode_prompt.return_value = ("embeds", "ids")
        self.assertEqual(self.provider._encode_prompt(pipe, "a cat"), "embeds")
        self.provider._encode_prompt(pipe, "a cat")
        self.assertEqual(pipe.encode_prompt.call_count, 1)
        self.provider._encode_prompt(pipe, "a dog")
        self.assertEqual(pipe.encode_prompt.call_count, 2)
        self.provider.unload()
        self.provider._encode_prompt(pipe, "a cat")
        self.assertEqual(pipe.encode_prompt.call_count, 3)

    def test_cached_prompt_embeddings_do_not_retain_autograd_graphs(self) -> None:
        import torch

        encoder = torch.nn.Linear(4, 4)
        pipe = SimpleNamespace(
            _execution_device="cpu",
            encode_prompt=lambda **_kwargs: (encoder(torch.ones(1, 4)), None),
        )
        with torch.enable_grad():
            first = self.provider._encode_prompt(pipe, "a cat")
            second = self.provider._encode_prompt(pipe, "a dog")
            self.assertTrue(torch.is_grad_enabled())
        for embedding in (first, second):
            self.assertFalse(embedding.requires_grad)
            self.assertIsNone(embedding.grad_fn)
        self.assertIs(self.provider._encode_prompt(pipe, "a cat"), first)

    def test_generic_reference_names_are_not_used_as_names(self) -> None:
        prompt = _inpaint_prompt(InpaintForm(prompt="two people on a bench"), ["Reference 1", "Reference 2"])
        self.assertIn("from the 2 reference images.", prompt)
        self.assertNotIn("Reference 1 is", prompt)

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
                provider._pipes = {"inpaint": object()}
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
