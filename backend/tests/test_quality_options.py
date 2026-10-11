from __future__ import annotations

from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image
from pydantic import ValidationError

from app import storage
from app.config import settings
from app.jobs import JobCancelled
from app.main import app
from app.models.base import ModelReference
from app.models.flux2_klein_nvfp4 import Flux2Klein9bNvfp4Provider, Flux2KleinNvfp4Provider
from app.models.klein_lora import AdapterCheckpoint, resolve_checkpoint
from app.schemas import EditRequest, EnhanceRequest, InpaintForm, OutpaintForm
from app.services import run_inpaint, run_outpaint


def png(image: Image.Image) -> bytes:
    buffer = BytesIO()
    image.save(buffer, "PNG")
    return buffer.getvalue()


class QualityOptionTests(unittest.TestCase):
    def test_unsupported_combinations_are_rejected(self) -> None:
        for schema, options in (
            (EditRequest, {"consistency_lora": True, "model_name": "flux2-klein-4b-nvfp4"}),
            (EditRequest, {"consistency_lora": True, "mode": "sketch"}),
            (EditRequest, {"detail": .4}),
            (EditRequest, {"detail": .4, "mode": "enhance", "model_name": "flux2-klein-4b-nvfp4"}),
            (EditRequest, {"strength": .6}),
            (EditRequest, {"strength": .6, "mode": "sketch"}),
            (EditRequest, {"strength": 0, "mode": "face"}),
            (EnhanceRequest, {"reference_ids": [1]}),
            (EnhanceRequest, {"detail": 0}),
            (EnhanceRequest, {"detail": 1.5}),
            (InpaintForm, {"consistency_lora": True, "edit_operation": "remove"}),
            (InpaintForm, {"detail_crop": True, "edit_operation": "sketch"}),
            (InpaintForm, {"detail_crop": True, "strength": .6}),
            (OutpaintForm, {"left": 32, "detail_crop": True}),
            (OutpaintForm, {"left": 32, "specialized_outpaint": True}),
            (OutpaintForm, {"left": 32, "specialized_outpaint": True, "model_name": "flux2-klein-4b-nvfp4", "steps": 50}),
            (OutpaintForm, {"left": 32, "specialized_outpaint": True, "model_name": "flux2-klein-4b-nvfp4", "guidance": 4}),
            (OutpaintForm, {"left": 32, "specialized_outpaint": True, "model_name": "flux2-klein-4b-nvfp4", "strength": .6}),
        ):
            with self.subTest(options=options), self.assertRaises(ValidationError):
                schema(prompt="texture", **options)

    def test_inpaint_flags_reach_the_queued_request(self) -> None:
        queued = SimpleNamespace(id="job", kind="inpaint", status="queued")
        with patch("app.main.jobs.submit", return_value=queued) as submit, patch("app.routes.generation.run_inpaint") as run:
            response = TestClient(app).post("/api/inpaint", data={"prompt": "texture", "detail_crop": "true", "consistency_lora": "true"},
                files={field: (field + ".png", b"queued", "image/png") for field in ("image", "mask")})
            self.assertEqual(response.status_code, 200)
            submit.call_args.args[1](MagicMock())
            request = run.call_args.args[0]
            self.assertTrue(request.detail_crop)
            self.assertTrue(request.consistency_lora)

    def test_invalid_flags_are_rejected_before_queueing(self) -> None:
        with patch("app.main.jobs.submit") as submit:
            response = TestClient(app).post("/api/inpaint", data={"prompt": "texture", "detail_crop": "true", "edit_operation": "remove"},
                files={field: (field + ".png", b"never decoded", "image/png") for field in ("image", "mask")})
            self.assertEqual(response.status_code, 422)
            submit.assert_not_called()

    def test_consistency_attaches_once_and_disables_exactly(self) -> None:
        provider = Flux2Klein9bNvfp4Provider()
        adapter, pipe, control = MagicMock(), MagicMock(), MagicMock()
        with patch("app.models.klein_lora.resolve_checkpoint", return_value=Path("v2")) as resolve, \
                patch("app.models.klein_lora.KleinLora.attach", return_value=adapter) as attach, patch.object(provider, "_collect"):
            provider._configure_edit_adapters(pipe, False, control)
            resolve.assert_not_called()
            provider._configure_edit_adapters(pipe, True, control)
            provider._configure_edit_adapters(pipe, False, control)
            provider._configure_edit_adapters(pipe, True, control)
            self.assertEqual([call.args[0] for call in adapter.set_scale.call_args_list], [.6, 0, .6])
            attach.assert_called_once_with(pipe.transformer, Path("v2"), control.raise_if_cancelled)
            provider.unload()
            self.assertIsNone(provider._consistency_adapter)

    def test_consistency_and_detail_take_turns_on_the_shared_layers(self) -> None:
        provider = Flux2Klein9bNvfp4Provider()
        consistency, detail, pipe, control = MagicMock(), MagicMock(), MagicMock(), MagicMock()
        with patch("app.models.klein_lora.resolve_checkpoint", side_effect=lambda checkpoint, _: Path(checkpoint.filename)) as resolve,                 patch("app.models.klein_lora.KleinLora.attach", side_effect=[detail, consistency]) as attach,                 patch.object(provider, "_collect"):
            provider._configure_edit_adapters(pipe, False, control, .4)
            self.assertIs(provider._detail_adapter, detail)
            provider._configure_edit_adapters(pipe, False, control, .7)
            provider._configure_edit_adapters(pipe, False, control)
            self.assertEqual([call.args[0] for call in detail.set_scale.call_args_list], [.4, .7, 0])
            # Consistency needs the same layers, so the detail LoRA goes first.
            provider._configure_edit_adapters(pipe, True, control)
            detail.detach.assert_called_once_with()
            self.assertIsNone(provider._detail_adapter)
            self.assertIs(provider._consistency_adapter, consistency)
            consistency.set_scale.assert_called_with(.6)
            self.assertEqual([call.args[0].name for call in resolve.call_args_list], ["Detail enhancer", "Consistency V2"])
            self.assertEqual(attach.call_count, 2)
            with self.assertRaisesRegex(ValueError, "cannot be combined"):
                provider._configure_edit_adapters(pipe, True, control, .4)
            provider.unload()
            self.assertIsNone(provider._detail_adapter)

    def test_detail_requires_the_quality_model(self) -> None:
        with patch("app.models.klein_lora.resolve_checkpoint") as resolve:
            with self.assertRaisesRegex(ValueError, "Quality 9B"):
                Flux2KleinNvfp4Provider()._configure_edit_adapters(MagicMock(), False, MagicMock(), .4)
            resolve.assert_not_called()

    def test_regular_fast_operations_disable_fal_and_unload_releases_it(self) -> None:
        provider = Flux2KleinNvfp4Provider()
        adapter = MagicMock()
        provider._outpaint_adapter = adapter
        provider._want_outpaint_adapter = True
        with patch.object(provider, "_collect"):
            provider._configure_edit_adapters(MagicMock(), False, MagicMock())
            adapter.set_scale.assert_called_once_with(0)
            provider.unload()
            self.assertIsNone(provider._outpaint_adapter)
            self.assertFalse(provider._want_outpaint_adapter)

    def test_failed_initial_placement_releases_attached_adapters(self) -> None:
        provider = Flux2KleinNvfp4Provider()
        def create(*_):
            provider._outpaint_adapter = MagicMock()
            return MagicMock()
        with patch.object(provider, "_create_pipeline", side_effect=create), patch.object(provider, "_place_pipeline", side_effect=RuntimeError("placement")), patch.object(provider, "_collect"):
            with self.assertRaisesRegex(RuntimeError, "placement"):
                provider._load_pipe("offload", "text-to-image", MagicMock())
            self.assertIsNone(provider._outpaint_adapter)
            self.assertFalse(provider.info().loaded)

    def test_specialized_runtime_change_reattaches_before_new_placement(self) -> None:
        provider = Flux2KleinNvfp4Provider()
        provider._pipes = {"text-to-image": MagicMock()}
        provider._placement = "offload"
        old, attached = MagicMock(), MagicMock()
        provider._outpaint_adapter = old
        pipe = MagicMock(return_value=SimpleNamespace(images=[Image.new("RGB", (64, 64))]))
        def load(placement, task, control):
            self.assertEqual((placement, task), ("cuda", "text-to-image"))
            self.assertTrue(provider._want_outpaint_adapter)
            self.assertIsNone(provider._outpaint_adapter)
            provider._outpaint_adapter = attached
            return pipe
        with TemporaryDirectory() as directory, patch("app.models.flux2_klein_nvfp4.require_generation_device"), \
                patch.object(provider, "_collect"), patch.object(provider, "_load_pipe", side_effect=load), \
                patch.object(provider, "_load_torch", return_value=MagicMock()), patch.object(provider, "_encode_prompt", return_value=MagicMock()):
            root = Path(directory)
            Image.new("RGB", (64, 64)).save(root/"input.png")
            provider.outpaint(OutpaintForm(prompt="continue", left=32, specialized_outpaint=True,
                model_name=provider.name, runtime_mode="performance", seed=7), root/"input.png", root/"output.png", MagicMock())
            attached.set_scale.assert_called_once_with(1.1)
            old.set_scale.assert_not_called()
            self.assertTrue((root/"output.png").is_file())

    def test_cached_checkpoint_requires_checksum_and_honours_cancellation(self) -> None:
        with TemporaryDirectory() as directory, patch.object(settings, "project_root", Path(directory)), patch("huggingface_hub.hf_hub_download") as download:
            path = Path(directory) / "models/adapters/test/adapter.safetensors"
            path.parent.mkdir(parents=True)
            path.write_bytes(b"checkpoint")
            checkpoint = AdapterCheckpoint("test", "maker/test", "revision", path.name, sha256(b"checkpoint").hexdigest())
            self.assertEqual(resolve_checkpoint(checkpoint, MagicMock()), path)
            download.assert_not_called()
            path.write_bytes(b"corrupt")
            with self.assertRaisesRegex(ValueError, "checksum"):
                resolve_checkpoint(checkpoint, MagicMock())
            control = MagicMock()
            control.raise_if_cancelled.side_effect = JobCancelled()
            with self.assertRaises(JobCancelled):
                resolve_checkpoint(checkpoint, control)

    def test_detail_routes_crop_edit_and_preserves_protected_rgba(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.object(settings, "data_dir", root), patch.object(settings, "image_dir", root / "images"), patch.object(settings, "db_path", root / "app.db"), patch.object(settings, "inpaint_minimum_dimension", 0):
                storage.ensure_storage()
                source = Image.new("RGBA", (192, 160), (5, 30, 70, 119))
                mask = Image.new("L", source.size, 0)
                mask.paste(128, (75, 60, 115, 100))
                provider = MagicMock()
                reference = ModelReference("fabric", root / "reference.png")
                def render(request, input_path, output_path, control, *, references):
                    self.assertEqual(request.prompt, "texture Scene context: daylight")
                    self.assertEqual(request.seed, 7)
                    self.assertTrue(request.consistency_lora)
                    self.assertEqual(references, [reference])
                    with Image.open(input_path) as crop:
                        self.assertLess(crop.width, source.width)
                        Image.new("RGB", crop.size, (210, 120, 20)).save(output_path)
                    return 7, 128
                provider.edit.side_effect = render
                with patch("app.services.registry.activate", return_value=provider), patch("app.services._model_references", return_value=[reference]):
                    ids = run_inpaint(InpaintForm(prompt="texture", seed=7, context_padding=32, blend_feather=0, detail_crop=True,
                        consistency_lora=True, scene_context="daylight", reference_ids=[1]), png(source), "source.png", png(mask), "mask.png", MagicMock())
                provider.inpaint.assert_not_called()
                record = storage.get_generation(ids[0])
                self.assertIs(record["detail_crop"], True)
                self.assertIs(record["consistency_lora"], True)
                with Image.open(storage.absolute_path(record["output_image_path"])) as result:
                    protected = np.asarray(mask) == 0
                    self.assertTrue(np.array_equal(np.asarray(result)[protected], np.asarray(source)[protected]))
                    self.assertNotEqual(result.getpixel((90, 80)), source.getpixel((90, 80)))

    def test_specialized_outpaint_receives_full_green_canvas_and_preserves_core(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.object(settings, "data_dir", root), patch.object(settings, "image_dir", root / "images"), patch.object(settings, "db_path", root / "app.db"):
                storage.ensure_storage()
                source = Image.new("RGBA", (192, 128), (50, 60, 70, 125))
                provider = MagicMock()
                def render(request, input_path, output_path, control):
                    with Image.open(input_path) as green:
                        self.assertEqual(green.size, (256, 160))
                        self.assertEqual(green.getpixel((0, 0)), (0, 255, 0))
                        self.assertEqual(green.getpixel((100, 80)), (50, 60, 70))
                        Image.new("RGB", green.size, (180, 100, 80)).save(output_path)
                    return 42, 256
                provider.outpaint.side_effect = render
                with patch("app.services.registry.activate", return_value=provider):
                    ids = run_outpaint(OutpaintForm(prompt="continue", left=64, top=32, specialized_outpaint=True,
                        model_name="flux2-klein-4b-nvfp4", seed=42), png(source), MagicMock())
                provider.inpaint.assert_not_called()
                record = storage.get_generation(ids[0])
                self.assertIs(record["specialized_outpaint"], True)
                self.assertIs(record["detail_crop"], False)
                with Image.open(storage.absolute_path(record["output_image_path"])) as result:
                    self.assertEqual(result.size, (256, 160))
                    self.assertEqual(result.crop((80, 48, 240, 144)).tobytes(), source.crop((16, 16, 176, 112)).tobytes())
                # API history retains the experiment choice as booleans.
                response = TestClient(app).get(f"/api/history/{ids[0]}")
                self.assertEqual(response.status_code, 200)
                self.assertIs(response.json()["specialized_outpaint"], True)


if __name__ == "__main__":
    unittest.main()
