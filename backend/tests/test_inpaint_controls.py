from __future__ import annotations

from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient
from PIL import Image
from pydantic import ValidationError

from app import storage
from app.config import settings
from app.main import app
from app.schemas import InpaintForm
from app.services import run_inpaint


def image_bytes(image: Image.Image) -> bytes:
    buffer = BytesIO()
    image.save(buffer, "PNG")
    return buffer.getvalue()


class InpaintControlTests(unittest.TestCase):
    def test_default_edit_includes_more_scene_context_and_a_gradual_boundary(self) -> None:
        request = InpaintForm(prompt="mushroom")
        self.assertEqual((request.context_padding, request.blend_feather), (320, 16))

    def test_rejects_invalid_blend_context_operation_and_notes(self) -> None:
        for fields in (
            {"blend_feather": -1}, {"blend_feather": 65}, {"blend_feather": float("nan")},
            {"context_padding": 31}, {"context_padding": 1025},
            {"edit_operation": "invented"}, {"scene_context": "x" * 1501},
        ):
            with self.subTest(fields=fields), self.assertRaises(ValidationError):
                InpaintForm(prompt="mushroom", **fields)

    def test_invalid_blend_context_and_operation_are_rejected_before_queueing(self) -> None:
        for fields in ({"context_padding": "0"}, {"blend_feather": "65"}, {"edit_operation": "invented"}):
            with self.subTest(fields=fields), patch("app.main.jobs.submit") as submit:
                response = TestClient(app).post(
                    "/api/inpaint",
                    data={"prompt": "mushroom", **fields},
                    files={field: (field + ".png", b"never decoded", "image/png") for field in ("image", "mask")},
                )
                self.assertEqual(response.status_code, 422)
                submit.assert_not_called()

    def test_failed_edit_cleans_up_uploaded_source_files(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.object(settings, "data_dir", root), patch.object(settings, "image_dir", root / "images"), patch.object(settings, "db_path", root / "app.db"):
                storage.ensure_storage()
                provider = MagicMock()
                provider.inpaint.side_effect = RuntimeError("inference failed")
                source = Image.new("RGB", (64, 64), "purple")
                mask = Image.new("L", source.size, 0)
                mask.paste(255, (16, 16, 48, 48))
                with patch("app.services.registry.activate", return_value=provider), self.assertRaisesRegex(RuntimeError, "inference failed"):
                    run_inpaint(InpaintForm(prompt="mushroom"), image_bytes(source), "source.png", image_bytes(mask), "mask.png", MagicMock())
                provider.unload.assert_called_once()
                for path in (settings.input_dir, settings.mask_dir, settings.output_dir):
                    self.assertEqual(list(path.iterdir()), [])

    def test_service_generates_fully_inside_a_soft_selection(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.object(settings, "data_dir", root), patch.object(settings, "image_dir", root / "images"), patch.object(settings, "db_path", root / "app.db"):
                storage.ensure_storage()
                provider = MagicMock()

                def render(request, source_path, mask_path, output_path, control):
                    with Image.open(mask_path) as model_mask:
                        self.assertEqual(model_mask.getpixel((32, 32)), 255)
                        self.assertEqual(model_mask.getpixel((15, 32)), 0)
                        Image.new("RGB", model_mask.size, (200, 0, 0)).save(output_path)
                    return 42, 64

                provider.inpaint.side_effect = render
                source = Image.new("RGB", (64, 64), "blue")
                mask = Image.new("L", source.size, 0)
                mask.paste(128, (16, 16, 48, 48))
                with patch("app.services.registry.activate", return_value=provider), patch("app.services.storage.create_generations", return_value=[123]):
                    result = run_inpaint(
                        InpaintForm(prompt="red", blend_feather=0, context_padding=32),
                        image_bytes(source), "source.png", image_bytes(mask), "mask.png", MagicMock(),
                    )
                self.assertEqual(result, [123])
                provider.inpaint.assert_called_once()

    def test_existing_generation_survives_migration_with_optional_new_metadata(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.object(settings, "data_dir", root), patch.object(settings, "image_dir", root / "images"), patch.object(settings, "db_path", root / "app.db"):
                with storage.connect() as db:
                    db.execute("""CREATE TABLE generations (
                        id INTEGER PRIMARY KEY, task TEXT NOT NULL, prompt TEXT NOT NULL,
                        negative_prompt TEXT, model_name TEXT NOT NULL, seed INTEGER NOT NULL,
                        steps INTEGER NOT NULL, guidance REAL NOT NULL, width INTEGER,
                        height INTEGER, strength REAL, input_image_path TEXT, mask_image_path TEXT,
                        output_image_path TEXT NOT NULL, created_at TEXT NOT NULL DEFAULT (datetime('now'))
                    )""")
                    db.execute("INSERT INTO generations (id, task, prompt, model_name, seed, steps, guidance, output_image_path) VALUES (7, 'inpaint', 'old mushroom', 'flux2-klein-4b', 123, 4, 1, 'images/outputs/old.png')")
                    db.commit()
                storage.ensure_storage()
                storage.ensure_storage()  # Migration remains safe across restarts.
                record = storage.get_generation(7)
                self.assertIsNotNone(record)
                self.assertEqual(record["prompt"], "old mushroom")
                self.assertEqual(record["runtime_mode"], "balanced")
                for field in ("consistency_lora", "detail_crop", "specialized_outpaint"):
                    self.assertIs(record[field], False)
                for field in ("edit_operation", "blend_feather", "context_padding", "scene_context"):
                    self.assertIsNone(record[field])

    def test_delete_generation_removes_record_and_exclusive_files(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.object(settings, "data_dir", root), patch.object(settings, "image_dir", root / "images"), patch.object(settings, "db_path", root / "app.db"):
                storage.ensure_storage()
                out_path = root / "images" / "outputs" / "sample.png"
                out_path.parent.mkdir(parents=True, exist_ok=True)
                out_path.write_bytes(b"image")
                gen_id = storage.create_generation({
                    "task": "text-to-image", "prompt": "test", "model_name": "flux",
                    "seed": 1, "steps": 4, "guidance": 1.0,
                    "output_image_path": storage.to_relative(out_path),
                })
                self.assertTrue(out_path.exists())
                storage.delete_generation(gen_id)
                self.assertIsNone(storage.get_generation(gen_id))
                self.assertFalse(out_path.exists())


if __name__ == "__main__":
    unittest.main()
