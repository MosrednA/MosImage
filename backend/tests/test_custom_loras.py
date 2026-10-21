from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

import torch
from fastapi.testclient import TestClient
from PIL import Image
from pydantic import ValidationError
from safetensors.torch import save_file

from app import loras, storage
from app.config import settings
from app.main import app
from app.models.flux2_klein_nvfp4 import Flux2Klein9bNvfp4Provider
from app.schemas import LoraChoice, OutpaintForm, TextToImageRequest
from app.services import run_text_to_image

# One output projection: four features wide on "4B", eight on "9B".
LAYER = "transformer_blocks.0.attn.to_out.0"
LINEARS = {"klein-4b": {LAYER: (4, 4)}, "klein-9b": {LAYER: (8, 8)}}


def write_lora(folder: Path, name: str, width: int, rank: int = 2) -> Path:
    path = folder / f"{name}.safetensors"
    save_file({"diffusion_model.double_blocks.0.img_attn.proj.lora_A.weight": torch.ones(rank, width),
               "diffusion_model.double_blocks.0.img_attn.proj.lora_B.weight": torch.ones(width, rank)}, str(path))
    return path


class LoraFolder(unittest.TestCase):
    """A temporary LoRA folder with the layer shapes of two tiny models."""

    def setUp(self) -> None:
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.folder = Path(directory.name)
        for patcher in (patch.object(settings, "lora_dir", self.folder),
                        patch("app.loras._linears", side_effect=LINEARS.__getitem__)):
            patcher.start()
            self.addCleanup(patcher.stop)


class LoraLibraryTests(LoraFolder):
    def test_library_lists_each_file_with_the_models_it_fits(self) -> None:
        write_lora(self.folder, "Small", 4)
        write_lora(self.folder, "large", 8)
        (self.folder / "Broken.safetensors").write_bytes(b"not a safetensors file")
        (self.folder / ".hidden.safetensors").write_bytes(b"")
        (self.folder / "notes.txt").write_text("not a LoRA")
        response = TestClient(app).get("/api/loras")
        self.assertEqual(response.status_code, 200)
        library = response.json()
        self.assertEqual(library["folder"], str(self.folder))
        by_name = {item["name"]: item for item in library["loras"]}
        self.assertEqual([item["name"] for item in library["loras"]], ["Broken", "large", "Small"])
        self.assertEqual(by_name["Small"]["models"], ["flux2-klein-4b-nvfp4", "flux2-klein-4b"])
        self.assertEqual(by_name["large"]["models"], ["flux2-klein-9b-nvfp4"])
        self.assertIsNone(by_name["Small"]["problem"])
        self.assertEqual(by_name["Broken"]["models"], [])
        self.assertIn("safetensors", by_name["Broken"]["problem"])

    def test_inspection_is_cached_until_the_file_changes(self) -> None:
        path = write_lora(self.folder, "Style", 4)
        self.assertEqual(loras.inspect(path).fits, ("klein-4b",))
        with patch("app.loras._inspect") as inspect:
            loras.inspect(path)
            inspect.assert_not_called()
        write_lora(self.folder, "Style", 8, rank=3)
        self.assertEqual(loras.inspect(path).fits, ("klein-9b",))

    def test_jobs_refuse_missing_and_mismatched_loras_before_queueing(self) -> None:
        write_lora(self.folder, "Large", 8)
        client = TestClient(app)
        with patch("app.main.jobs.submit") as submit:
            for name, message in (("Large", "made for FLUX.2 Klein 9B"), ("Gone", "not in the LoRA folder")):
                response = client.post("/api/generate", json={"prompt": "a cat", "model_name": "flux2-klein-4b-nvfp4",
                                                              "loras": [{"name": name}]})
                self.assertEqual(response.status_code, 422)
                self.assertIn(message, response.json()["detail"])
            submit.assert_not_called()

    def test_inpaint_reads_its_loras_from_json(self) -> None:
        write_lora(self.folder, "Large", 8)
        queued = SimpleNamespace(id="job", kind="inpaint", status="queued")
        with patch("app.main.jobs.submit", return_value=queued) as submit, patch("app.routes.generation.run_inpaint") as run:
            response = TestClient(app).post("/api/inpaint", data={"prompt": "texture", "loras": json.dumps([{"name": "Large", "scale": 0.7}])},
                files={field: (field + ".png", b"queued", "image/png") for field in ("image", "mask")})
            self.assertEqual(response.status_code, 200)
            submit.call_args.args[1](MagicMock())
        self.assertEqual(run.call_args.args[0].loras, [LoraChoice(name="Large", scale=0.7)])

    def test_choices_are_file_names_with_a_bounded_strength(self) -> None:
        for choice in ({"name": "../secrets"}, {"name": "C:style"}, {"name": ".hidden"}, {"name": "a", "scale": 0},
                       {"name": "a", "scale": 2.5}, {"name": "a", "extra": 1}):
            with self.subTest(choice=choice), self.assertRaises(ValidationError):
                LoraChoice(**choice)
        with self.assertRaises(ValidationError):
            TextToImageRequest(prompt="x", loras=[{"name": "a"}, {"name": "a"}])
        with self.assertRaises(ValidationError):
            TextToImageRequest(prompt="x", loras=[{"name": name} for name in "abcd"])
        with self.assertRaises(ValidationError):
            OutpaintForm(prompt="x", left=32, specialized_outpaint=True, model_name="flux2-klein-4b-nvfp4", loras=[{"name": "a"}])

    def test_generation_history_keeps_the_loras(self) -> None:
        write_lora(self.folder, "Large", 8)
        with TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.object(settings, "data_dir", root), patch.object(settings, "image_dir", root / "images"), \
                    patch.object(settings, "db_path", root / "app.db"):
                storage.ensure_storage()
                provider = MagicMock()
                provider.generate.side_effect = lambda request, path, control: (Image.new("RGB", (8, 8)).save(path) or (1, 8))
                request = TextToImageRequest(prompt="a cat", loras=[{"name": "Large", "scale": 0.5}])
                with patch("app.services.registry.activate", return_value=provider):
                    run_text_to_image(request, MagicMock())
                record = storage.list_generations()[0]
        self.assertEqual(record["loras"], [{"name": "Large", "scale": 0.5}])


class ProviderLoraTests(LoraFolder):
    def setUp(self) -> None:
        super().setUp()
        self.provider = Flux2Klein9bNvfp4Provider()
        self.pipe, self.control = MagicMock(), MagicMock()
        self.adapters: dict[str, list[MagicMock]] = {}
        attach = patch("app.models.klein_lora.KleinLora.attach", side_effect=self.attach)
        attach.start()
        self.addCleanup(attach.stop)

    def attach(self, model: object, path: Path, check_cancelled: object) -> MagicMock:
        adapter = MagicMock(name=path.stem)
        self.adapters.setdefault(path.stem, []).append(adapter)
        return adapter

    def configure(self, *choices: tuple[str, float]) -> None:
        self.provider._configure_loras(self.pipe, [LoraChoice(name=name, scale=scale) for name, scale in choices], self.control)

    def test_chosen_loras_stay_attached_and_the_rest_come_off(self) -> None:
        for name in ("Ink", "Film"):
            write_lora(self.folder, name, 8)
        self.configure(("Ink", .8), ("Film", 1.2))
        ink, film = self.adapters["Ink"][0], self.adapters["Film"][0]
        ink.set_scale.assert_called_with(.8)
        film.set_scale.assert_called_with(1.2)
        self.configure(("Ink", .5))
        self.assertEqual(len(self.adapters["Ink"]), 1)  # not read again
        ink.set_scale.assert_called_with(.5)
        film.detach.assert_called_once()
        self.configure()
        ink.detach.assert_called_once()

    def test_a_rewritten_file_is_read_again(self) -> None:
        write_lora(self.folder, "Ink", 8)
        self.configure(("Ink", 1))
        write_lora(self.folder, "Ink", 8, rank=4)
        self.configure(("Ink", 1))
        self.assertEqual(len(self.adapters["Ink"]), 2)
        self.adapters["Ink"][0].detach.assert_called_once()

    def test_own_loras_leave_consistency_attached(self) -> None:
        write_lora(self.folder, "Ink", 8)
        consistency = MagicMock()
        with patch("app.models.klein_lora.resolve_checkpoint", return_value=Path("v2")), \
                patch("app.models.klein_lora.KleinLora.attach", return_value=consistency):
            self.provider._configure_edit_adapters(self.pipe, True, self.control)
        self.configure(("Ink", 1))
        self.configure()
        consistency.detach.assert_not_called()
        consistency.set_scale.assert_called_with(.6)

    def test_a_missing_file_is_a_clear_error_and_unload_forgets_the_loras(self) -> None:
        with self.assertRaisesRegex(ValueError, "not in the LoRA folder"):
            self.configure(("Gone", 1))
        write_lora(self.folder, "Ink", 8)
        self.configure(("Ink", 1))
        self.provider.unload()
        self.configure(("Ink", 1))
        self.assertEqual(len(self.adapters["Ink"]), 2)


if __name__ == "__main__":
    unittest.main()
