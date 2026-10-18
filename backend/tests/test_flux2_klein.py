from __future__ import annotations

import unittest
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from PIL import Image

from app.config import settings
from app.jobs import JobCancelled, JobControl
from app.models.base import ModelReference
from app.models.flux2_klein import MULTI_REFERENCE_AREA, REFERENCE_BACKGROUND, Flux2KleinProvider, _edit_prompt, _inpaint_prompt
from app.models.registry import ModelRegistry
from app.schemas import EditRequest, InpaintForm, LoadModelRequest, OutpaintForm, TextToImageRequest


class ProgressLog:
    """A job manager that records the progress a job reports."""

    def __init__(self) -> None:
        self.values: list[float] = []
        self.on_progress = lambda: None

    def is_cancel_requested(self, _job_id: str) -> bool:
        return False

    def update_progress(self, _job_id: str, progress: float) -> None:
        self.on_progress()
        self.values.append(progress)

    def update_phase(self, *_args: object) -> None:
        pass

    def update_step(self, *_args: object) -> None:
        pass


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
        self.assertEqual(info.capabilities, ["text-to-image", "edit", "inpaint", "outpaint"])

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

    def test_start_up_progress_moves_with_the_load_and_never_back(self) -> None:
        manager = ProgressLog()

        def create(_task: str, control: JobControl) -> MagicMock:
            control.set_progress(0.5)
            control.set_progress(1.0)
            return MagicMock()

        def place(_pipe: object, _placement: str, control: JobControl) -> None:
            control.set_progress(1.0)

        with patch("app.models.flux2_klein.require_generation_device"), patch.object(self.provider, "_create_pipeline", side_effect=create), \
                patch.object(self.provider, "_derive_pipeline", return_value=MagicMock()), \
                patch.object(self.provider, "_place_pipeline", side_effect=place), patch.object(self.provider, "_warm_up"):
            self.provider.preload(LoadModelRequest(model_name=self.provider.name, task="inpaint", warm_up=True), JobControl(manager, "job"))
        # Reading the weights and copying them to the GPU fill most of the bar; the warm-up step the rest.
        self.assertEqual([round(value, 3) for value in manager.values], [0.315, 0.63, 0.9, 0.9, 0.95, 0.95])
        self.assertEqual(manager.values, sorted(manager.values))

    def test_a_load_during_an_edit_stays_before_the_first_step(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            Image.new("RGB", (64, 64), "purple").save(root / "source.png")
            Image.new("L", (64, 64), 255).save(root / "mask.png")
            pipe = MagicMock(return_value=SimpleNamespace(images=[Image.new("RGB", (64, 64))]))

            def load(_placement: str, _task: str, control: JobControl) -> MagicMock:
                control.set_progress(1.0)
                return pipe

            manager = ProgressLog()
            with patch("app.models.flux2_klein.require_generation_device"), patch("app.models.flux2_klein.selected_device", return_value="cuda"), \
                    patch.object(self.provider, "_load_torch", return_value=MagicMock()), patch.object(self.provider, "_load_pipe", side_effect=load):
                self.provider.inpaint(InpaintForm(prompt="moss"), root / "source.png", root / "mask.png", root / "output.png", JobControl(manager, "job"))
        # Steps start at 15%, so a model load during an edit fills the bar up to there.
        self.assertEqual(manager.values[:2], [0.15, 0.15])
        self.assertEqual(manager.values, sorted(manager.values))

    def test_placement_copies_the_transformer_block_by_block(self) -> None:
        import torch

        transformer = torch.nn.Module()
        transformer.embedder = torch.nn.Linear(4, 12)  # 60 weights
        transformer.blocks = torch.nn.ModuleList(torch.nn.Linear(4, 4) for _ in range(3))  # 20 each
        manager, moved = ProgressLog(), []
        manager.on_progress = lambda: moved.append(sum(parameter.is_meta for parameter in transformer.parameters()))
        with patch.object(self.provider, "_load_torch", return_value=torch):
            self.provider._move_in_parts(transformer, "meta", JobControl(manager, "job"))
        # Each block reports its share by size once it has moved.
        self.assertEqual([round(value, 3) for value in manager.values], [0.5, 0.667, 0.833, 1.0])
        self.assertEqual(moved, [2, 4, 6, 8])

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

    def test_cancelling_during_denoising_keeps_the_model_loaded(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            Image.new("RGB", (64, 64), "purple").save(root / "source.png")
            Image.new("L", (64, 64), 255).save(root / "mask.png")
            pipe = MagicMock(side_effect=JobCancelled())
            self.provider._pipes = {"inpaint": pipe, "text-to-image": pipe}
            self.provider._placement = "cuda"
            with patch("app.models.flux2_klein.require_generation_device"), patch("app.models.flux2_klein.selected_device", return_value="cuda"), \
                    patch.object(self.provider, "_load_torch", return_value=MagicMock()), patch.object(self.provider, "_placement_for", return_value="cuda"):
                with self.assertRaises(JobCancelled):
                    self.provider.inpaint(InpaintForm(prompt="moss"), root / "source.png", root / "mask.png", root / "output.png", MagicMock())
                with self.assertRaises(JobCancelled):
                    self.provider.generate(TextToImageRequest(prompt="moss"), root / "output.png", MagicMock())
        # The loop holds no state between calls, so the next edit reuses the weights and prompt cache.
        self.assertEqual(self.provider.info().loaded_pipelines, ["inpaint:cuda", "text-to-image:cuda"])

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

    def test_whole_image_edit_leads_with_the_picture_and_keeps_its_size(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            Image.new("RGB", (1600, 1200), "green").save(root / "picture.png")
            Image.new("RGB", (600, 1200), "orange").save(root / "anna.png")
            pipe = MagicMock(return_value=SimpleNamespace(images=[Image.new("RGB", (1184, 880))]))
            with patch("app.models.flux2_klein.require_generation_device"), patch("app.models.flux2_klein.selected_device", return_value="cuda"), patch.object(self.provider, "_load_torch", return_value=MagicMock()), patch.object(self.provider, "_load_pipe", return_value=pipe) as load:
                self.provider.edit(EditRequest(prompt="Make it night."), root / "picture.png", root / "output.png", MagicMock(),
                                   references=[ModelReference("Anna", root / "anna.png")])
                self.assertEqual(load.call_args.args[1], "text-to-image")
                arguments = pipe.call_args.kwargs
                picture, reference = arguments["image"]
                # About one megapixel, sides in multiples of 16, the picture's aspect ratio.
                self.assertEqual(picture.size, (arguments["width"], arguments["height"]))
                self.assertEqual((picture.width % 16, picture.height % 16), (0, 0))
                self.assertAlmostEqual(picture.width / picture.height, 4 / 3, delta=0.02)
                self.assertLessEqual(abs(picture.width * picture.height - 1024 * 1024), 1024 * 32)
                self.assertEqual(reference.size, (512, 1024))
                prompt = pipe.encode_prompt.call_args.kwargs["prompt"]
                self.assertTrue(prompt.startswith("Edit image 1: Make it night."))
                self.assertIn("Anna is the person or object from image 2", prompt)
                with Image.open(root / "output.png") as output:
                    self.assertEqual(output.size, (1600, 1200))
                self.provider.edit(EditRequest(prompt="add snow"), root / "picture.png", root / "output.png", MagicMock())
                self.assertEqual(len(pipe.call_args.kwargs["image"]), 1)
                self.assertNotIn("image 2", pipe.encode_prompt.call_args.kwargs["prompt"])

    def test_a_transparent_layer_is_edited_on_neutral_grey(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            layer = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
            layer.paste((200, 30, 30, 255), (16, 16, 48, 48))
            layer.save(root / "layer.png")
            pipe = MagicMock(return_value=SimpleNamespace(images=[Image.new("RGB", (1024, 1024))]))
            with patch("app.models.flux2_klein.require_generation_device"), patch("app.models.flux2_klein.selected_device", return_value="cuda"), patch.object(self.provider, "_load_torch", return_value=MagicMock()), patch.object(self.provider, "_load_pipe", return_value=pipe):
                self.provider.edit(EditRequest(prompt="make it blue"), root / "layer.png", root / "output.png", MagicMock())
                picture = pipe.call_args.kwargs["image"][0]
                self.assertEqual(picture.mode, "RGB")
                self.assertEqual(picture.getpixel((5, 5)), REFERENCE_BACKGROUND)
                self.assertEqual(picture.getpixel((512, 512)), (200, 30, 30))
                with Image.open(root / "output.png") as output:
                    self.assertEqual(output.size, (64, 64))

    def test_edit_prompt_numbers_several_references_after_the_picture(self) -> None:
        prompt = _edit_prompt("Put them on the beach", ["Anna", "Reference 2", "Tom"])
        self.assertIn("Images 2 to 4 show the people or objects to bring in: Anna is image 2, Tom is image 4.", prompt)
        self.assertTrue(prompt.endswith("including each person's face and identity."))

    def test_generation_composes_references_into_a_new_image(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            Image.new("RGB", (600, 1200), "orange").save(root / "anna.png")
            Image.new("RGB", (900, 900), "blue").save(root / "tom.png")
            references = [ModelReference("Anna", root / "anna.png"), ModelReference("Tom", root / "tom.png")]
            pipe = MagicMock(return_value=SimpleNamespace(images=[Image.new("RGB", (64, 48))]))
            request = TextToImageRequest(prompt="Anna and Tom riding a roller coaster", width=1216, height=832)
            with patch("app.models.flux2_klein.require_generation_device"), patch("app.models.flux2_klein.selected_device", return_value="cuda"), patch.object(self.provider, "_load_torch", return_value=MagicMock()), patch.object(self.provider, "_load_pipe", return_value=pipe):
                self.provider.generate(request, root / "output.png", MagicMock(), references=references)
                arguments = pipe.call_args.kwargs
                self.assertEqual(len(arguments["image"]), 2)
                self.assertEqual(arguments["image"][0].size, arguments["image"][1].size)
                self.assertEqual((arguments["width"], arguments["height"]), (1216, 832))
                prompt = pipe.encode_prompt.call_args.kwargs["prompt"]
                self.assertIn("Anna is reference image 1, Tom is reference image 2", prompt)
                self.provider.generate(TextToImageRequest(prompt="a quiet garden"), root / "output.png", MagicMock())
                self.assertNotIn("image", pipe.call_args.kwargs)
                self.assertEqual(pipe.encode_prompt.call_args.kwargs["prompt"], "a quiet garden")

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
        with patch.object(self.provider, "_collect") as collect:
            self.assertEqual(self.provider._encode_prompt(pipe, "a cat"), "embeds")
            # The encoder's freed GPU blocks go back before the denoiser allocates.
            collect.assert_called_once()
            self.provider._encode_prompt(pipe, "a cat")
            collect.assert_called_once()
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

    def test_expansion_prompt_continues_the_scene_into_the_border(self) -> None:
        prompt = _inpaint_prompt(OutpaintForm(prompt="a sunny beach.", left=64))
        self.assertTrue(prompt.startswith("Extend the picture outward into the masked border with a sunny beach."))
        self.assertIn("no frame, border or visible seam", prompt)
        self.assertIn("Match the surrounding lighting", prompt)


class ModelRegistryTests(unittest.TestCase):
    def test_registry_lists_only_klein_models_and_each_does_every_task(self) -> None:
        models = ModelRegistry().list()
        self.assertEqual([model.name for model in models], ["flux2-klein-9b-nvfp4", "flux2-klein-4b-nvfp4", "flux2-klein-4b"])
        for model in models:
            self.assertEqual(set(model.capabilities), {"text-to-image", "edit", "inpaint", "outpaint"})

    def test_switching_precision_releases_the_other_klein_pipeline(self) -> None:
        registry = ModelRegistry()
        for source, target in (("flux2-klein-4b", "flux2-klein-4b-nvfp4"), ("flux2-klein-4b-nvfp4", "flux2-klein-9b-nvfp4"), ("flux2-klein-9b-nvfp4", "flux2-klein-4b")):
            with self.subTest(source=source):
                provider = registry.get(source)
                provider._pipes = {"inpaint": object()}
                registry.activate(target)
                self.assertFalse(provider.info().loaded)



if __name__ == "__main__":
    unittest.main()
