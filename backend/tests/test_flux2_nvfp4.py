from __future__ import annotations

from io import BytesIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

from PIL import Image

from app.config import settings
from app.jobs import JobCancelled
from app.models.flux2_klein_nvfp4 import (
    BASE_REPO, BASE_REVISION, BASE_REPO_9B, BASE_REVISION_9B, QUANT_REPO, QUANT_REPO_9B, QUANT_REVISION, QUANT_FILENAME,
    QUANT_FILENAME_9B, Flux2Klein9bNvfp4Provider, Flux2KleinNvfp4Provider,
)
from app.schemas import InpaintForm, LoadModelRequest, TextToImageRequest
from app.services import run_inpaint


class Nvfp4ProviderTests(unittest.TestCase):
    def test_generation_edit_and_preload_default_to_9b_nvfp4(self) -> None:
        requests = (TextToImageRequest(prompt="scene"), InpaintForm(prompt="scene"), LoadModelRequest())
        self.assertTrue(all(request.model_name == "flux2-klein-9b-nvfp4" for request in requests))
        self.assertEqual(settings.warm_up_model, "flux2-klein-9b-nvfp4")
        self.assertEqual(Flux2KleinNvfp4Provider().info().model_id, QUANT_REPO)
        self.assertEqual(Flux2Klein9bNvfp4Provider().info().model_id, QUANT_REPO_9B)

    def test_9b_streams_its_encoder_and_stays_resident(self) -> None:
        provider = Flux2Klein9bNvfp4Provider()
        generation, inpaint, transformer, encoder = (MagicMock() for _ in range(4))
        inpaint.from_pretrained.return_value.config = {"is_distilled": True}
        vendor = SimpleNamespace(Flux2KleinPipeline=generation, Flux2KleinInpaintPipeline=inpaint)
        runtime = SimpleNamespace(kernel_api=MagicMock(), load_transformer=MagicMock(return_value=transformer))
        hub = SimpleNamespace(hf_hub_download=MagicMock(side_effect=["config.json", "checkpoint.safetensors"]),
                              snapshot_download=MagicMock(return_value="snapshot"))
        loader = SimpleNamespace(load_text_encoder=MagicMock(return_value=encoder))
        control = MagicMock()
        modules = {"diffusers": vendor, "huggingface_hub": hub, "app.models.nvfp4_runtime": runtime, "app.models.klein_text_encoder": loader}
        with patch.dict("sys.modules", modules), patch.object(provider, "_validate_runtime"), patch.object(provider, "_load_torch", return_value=SimpleNamespace(bfloat16="bf16")):
            provider._create_pipeline("inpaint", control)
        self.assertEqual(hub.hf_hub_download.call_args_list[0].args, (BASE_REPO_9B, "transformer/config.json"))
        self.assertEqual(hub.hf_hub_download.call_args_list[1].args, (QUANT_REPO_9B, QUANT_FILENAME_9B))
        loaded = inpaint.from_pretrained.call_args
        self.assertEqual((loaded.args, loaded.kwargs["revision"]), ((BASE_REPO_9B,), BASE_REVISION_9B))
        self.assertIsNone(loaded.kwargs["text_encoder"])
        root, cancelled, progress = loader.load_text_encoder.call_args.args
        self.assertEqual((root, cancelled), (Path("snapshot"), control.raise_if_cancelled))
        # The encoder is most of a 9B start-up, so its share becomes the load's progress.
        progress(0.5)
        self.assertAlmostEqual(control.set_progress.call_args.args[0], 0.525)
        built = inpaint.call_args.kwargs
        self.assertIs(built["text_encoder"], encoder)
        self.assertIs(built["transformer"], inpaint.from_pretrained.return_value.transformer)
        self.assertTrue(built["is_distilled"])
        self.assertEqual({provider._placement_for(mode) for mode in ("balanced", "safe", "performance")}, {"cuda"})

    def test_encoder_weights_report_their_share_once_each_is_handled(self) -> None:
        import torch
        from safetensors.torch import save_file
        from app.models.klein_text_encoder import KEEP_LAYERS, _kept_weights

        shards = {"a.safetensors": {"model.embed_tokens.weight": torch.ones(2, 2), "lm_head.weight": torch.ones(2, 2)},
                  "b.safetensors": {"model.layers.0.mlp.up_proj.weight": torch.ones(2, 2),
                                    f"model.layers.{KEEP_LAYERS}.mlp.up_proj.weight": torch.ones(2, 2), "model.norm.weight": torch.ones(2)}}
        events: list[object] = []
        with TemporaryDirectory() as directory:
            folder = Path(directory)
            for shard, weights in shards.items():
                save_file(weights, str(folder / shard))
            weight_map = {key: shard for shard, weights in shards.items() for key in weights}
            (folder / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}), encoding="utf-8")
            for name, _tensor in _kept_weights(folder, "cpu", lambda: None, lambda share: events.append(round(share, 2))):
                events.append(name)
        # Skipped weights (the LM head, layers past the last one read) count for nothing.
        self.assertEqual(sorted(events[0::2]), ["embed_tokens.weight", "layers.0.mlp.up_proj.weight", "norm.weight"])
        self.assertEqual(events[1::2], [0.33, 0.67, 1.0])

    def test_native_loader_receives_pinned_files_and_pipeline_uses_bf16_components(self) -> None:
        provider = Flux2KleinNvfp4Provider()
        generation, inpaint, transformer, kernel = (MagicMock() for _ in range(4))
        vendor = SimpleNamespace(Flux2KleinPipeline=generation, Flux2KleinInpaintPipeline=inpaint)
        runtime = SimpleNamespace(kernel_api=MagicMock(return_value=kernel), load_transformer=MagicMock(return_value=transformer))
        download = MagicMock(side_effect=["config.json", "checkpoint.safetensors"])
        control = MagicMock()
        torch = SimpleNamespace(bfloat16="bf16")
        with patch.dict("sys.modules", {"diffusers": vendor, "huggingface_hub": SimpleNamespace(hf_hub_download=download), "app.models.nvfp4_runtime": runtime}), patch.object(provider, "_validate_runtime"), patch.object(provider, "_load_torch", return_value=torch):
            result = provider._create_pipeline("inpaint", control)
        self.assertEqual(download.call_args_list[0].args, (BASE_REPO, "transformer/config.json"))
        self.assertEqual(download.call_args_list[0].kwargs["revision"], BASE_REVISION)
        self.assertEqual(download.call_args_list[1].args, (QUANT_REPO, QUANT_FILENAME))
        self.assertEqual(download.call_args_list[1].kwargs["revision"], QUANT_REVISION)
        runtime.load_transformer.assert_called_once_with(Path("config.json"), Path("checkpoint.safetensors"), kernel, control.raise_if_cancelled)
        arguments = inpaint.from_pretrained.call_args
        self.assertEqual(arguments.args, (BASE_REPO,))
        self.assertEqual(arguments.kwargs["revision"], BASE_REVISION)
        self.assertIs(arguments.kwargs["transformer"], transformer)
        self.assertEqual(arguments.kwargs["torch_dtype"], "bf16")
        self.assertIs(result, inpaint.from_pretrained.return_value)
        generation.from_pretrained.assert_not_called()

    def test_cancel_after_config_stops_before_checkpoint_download(self) -> None:
        provider = Flux2KleinNvfp4Provider()
        control = MagicMock()
        control.raise_if_cancelled.side_effect = [None, JobCancelled()]
        download = MagicMock(return_value="config.json")
        runtime = SimpleNamespace(kernel_api=MagicMock(), load_transformer=MagicMock())
        with patch.dict("sys.modules", {"diffusers": MagicMock(), "huggingface_hub": SimpleNamespace(hf_hub_download=download), "app.models.nvfp4_runtime": runtime}), patch.object(provider, "_validate_runtime"):
            with self.assertRaises(JobCancelled):
                provider._create_pipeline("inpaint", control)
        download.assert_called_once()
        runtime.load_transformer.assert_not_called()

    def test_unsupported_runtime_fails_before_any_model_download(self) -> None:
        provider = Flux2KleinNvfp4Provider()
        for capability, cuda_version, dtype, message in (((8, 9), "13.0", "bf16", "Blackwell"), ((12, 0), "12.8", "bf16", "CUDA 13"), ((12, 0), "13.0", "fp16", "bfloat16")):
            with self.subTest(message=message):
                torch = SimpleNamespace(cuda=SimpleNamespace(get_device_capability=lambda: capability), version=SimpleNamespace(cuda=cuda_version), bfloat16="bf16")
                with patch.object(provider, "_load_torch", return_value=torch), patch("app.models.flux2_klein_nvfp4.selected_device", return_value="cuda"), patch("app.models.flux2_klein_nvfp4.selected_dtype", return_value=dtype):
                    with self.assertRaisesRegex(RuntimeError, message):
                        provider._create_pipeline("inpaint", MagicMock())
        self.assertFalse(provider.info().loaded)


class KleinMaskRoutingTests(unittest.TestCase):
    def test_both_precisions_receive_full_generation_coverage_with_a_soft_final_blend(self) -> None:
        source = Image.new("RGB", (64, 64), "purple")
        selection = Image.new("L", source.size, 0)
        selection.paste(255, (16, 16, 48, 48))
        image_buffer, mask_buffer = BytesIO(), BytesIO()
        source.save(image_buffer, "PNG")
        selection.save(mask_buffer, "PNG")
        for model in ("flux2-klein-4b", "flux2-klein-4b-nvfp4", "flux2-klein-9b-nvfp4"):
            with self.subTest(model=model), TemporaryDirectory() as directory:
                root = Path(directory)
                for kind in ("inputs", "masks", "outputs"):
                    (root / "images" / kind).mkdir(parents=True)
                provider = MagicMock()
                masks: list[bytes] = []

                def render(_request, input_path, mask_path, output_path, _control):
                    with Image.open(mask_path) as mask:
                        masks.append(mask.tobytes())
                    with Image.open(input_path) as crop:
                        Image.new("RGB", crop.size, "orange").save(output_path)
                    return 42, 64

                provider.inpaint.side_effect = render
                # Native model size, so the mask stays binary; enlargement is tested in test_compositing.
                with patch.object(settings, "data_dir", root), patch.object(settings, "inpaint_minimum_dimension", 0), patch("app.services.registry.activate", return_value=provider), patch("app.services.storage.to_relative", side_effect=str), patch("app.services.storage.create_generations", return_value=[1]) as save:
                    run_inpaint(InpaintForm(prompt="scene", model_name=model), image_buffer.getvalue(), "source.png", mask_buffer.getvalue(), "mask.png", MagicMock())
                self.assertEqual(set(masks[0]), {0, 255})
                record = save.call_args.args[0][0]
                with Image.open(record["output_image_path"]) as result:
                    self.assertEqual(result.getpixel((0, 0))[:3], source.getpixel((0, 0)))
                    blended = result.getpixel((18, 32))[:3]
                    self.assertNotEqual(blended, source.getpixel((18, 32)))
                    self.assertNotEqual(blended, Image.new("RGB", (1, 1), "orange").getpixel((0, 0)))
                self.assertEqual(record["model_name"], model)


if __name__ == "__main__":
    unittest.main()
