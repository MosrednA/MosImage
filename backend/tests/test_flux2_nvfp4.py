from __future__ import annotations

from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

from PIL import Image

from app.config import settings
from app.jobs import JobCancelled
from app.models.flux2_klein_nvfp4 import (
    BASE_REPO, BASE_REVISION, QUANT_REPO, QUANT_REVISION, QUANT_FILENAME,
    Flux2KleinNvfp4Provider,
)
from app.schemas import InpaintForm, LoadModelRequest, TextToImageRequest
from app.services import run_inpaint


class Nvfp4ProviderTests(unittest.TestCase):
    def test_generation_edit_and_preload_default_to_nvfp4(self) -> None:
        requests = (TextToImageRequest(prompt="scene"), InpaintForm(prompt="scene"), LoadModelRequest())
        self.assertTrue(all(request.model_name == "flux2-klein-4b-nvfp4" for request in requests))
        self.assertEqual(Flux2KleinNvfp4Provider().info().model_id, QUANT_REPO)

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


class KleinSoftMaskRoutingTests(unittest.TestCase):
    def test_both_precisions_receive_soft_sampling_masks_and_preserve_protected_pixels(self) -> None:
        source = Image.new("RGB", (64, 64), "purple")
        selection = Image.new("L", source.size, 0)
        selection.paste(255, (16, 16, 48, 48))
        image_buffer, mask_buffer = BytesIO(), BytesIO()
        source.save(image_buffer, "PNG")
        selection.save(mask_buffer, "PNG")
        for model in ("flux2-klein-4b", "flux2-klein-4b-nvfp4"):
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
                with patch.object(settings, "data_dir", root), patch("app.services.registry.activate", return_value=provider), patch("app.services.storage.to_relative", side_effect=str), patch("app.services.storage.create_generations", return_value=[1]) as save:
                    run_inpaint(InpaintForm(prompt="scene", model_name=model), image_buffer.getvalue(), "source.png", mask_buffer.getvalue(), "mask.png", MagicMock())
                self.assertTrue(any(0 < value < 255 for value in masks[0]))
                record = save.call_args.args[0][0]
                with Image.open(record["output_image_path"]) as result:
                    self.assertEqual(result.getpixel((0, 0))[:3], source.getpixel((0, 0)))
                self.assertEqual(record["model_name"], model)


if __name__ == "__main__":
    unittest.main()
