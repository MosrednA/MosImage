from __future__ import annotations

import json
from types import SimpleNamespace

from contextlib import ExitStack, contextmanager
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Iterator
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image
from pydantic import ValidationError

from app import documents, references, storage
from app.config import settings
from app.main import app
from app.models.flux2_klein import _inpaint_prompt
from app.schemas import EditRequest, InpaintForm, OutpaintForm, TextToImageRequest
from app.services import run_image_edit, run_inpaint, run_text_to_image


def png(image: Image.Image) -> bytes:
    buffer = BytesIO()
    image.save(buffer, "PNG")
    return buffer.getvalue()


@contextmanager
def isolated_data() -> Iterator[Path]:
    with TemporaryDirectory() as directory, ExitStack() as stack:
        root = Path(directory)
        stack.enter_context(patch.object(settings, "data_dir", root))
        stack.enter_context(patch.object(settings, "image_dir", root / "images"))
        stack.enter_context(patch.object(settings, "db_path", root / "app.db"))
        storage.ensure_storage()
        documents.ensure_document_storage()
        references.ensure_reference_storage()
        yield root


def new_document() -> int:
    return documents.create_from_upload("garden", png(Image.new("RGB", (64, 64), "green")), "garden.png")["id"]


class ReferenceStorageTests(unittest.TestCase):
    def test_references_belong_to_a_document_and_files_are_removed_on_delete(self) -> None:
        with isolated_data():
            document_id = new_document()
            created = references.create_reference(document_id, png(Image.new("RGB", (40, 80), "orange")), "Anna.jpg")
            self.assertEqual((created["name"], created["width"], created["height"]), ("Anna", 40, 80))
            self.assertIsNone(created["subject_url"])
            self.assertEqual([item["id"] for item in references.list_references(document_id)], [created["id"]])
            self.assertEqual(references.list_references(document_id + 1), [])
            stored = references.original_image_path(created["id"])
            self.assertTrue(stored.is_file())
            references.delete_reference(created["id"])
            self.assertFalse(stored.exists())
            self.assertIsNone(references.get_reference(created["id"]))

    def test_pasted_images_get_a_numbered_name_and_unknown_documents_are_rejected(self) -> None:
        with isolated_data():
            document_id = new_document()
            self.assertEqual(references.create_reference(document_id, png(Image.new("RGB", (8, 8))), "image.png")["name"], "Reference 1")
            with self.assertRaises(LookupError):
                references.create_reference(document_id + 99, png(Image.new("RGB", (8, 8))), "x.png")

    def test_code_like_file_names_become_numbered_references(self) -> None:
        machine = ["T5S8DAP28-U5SUC4UTC-7b1f1113fc48-512", "IMG_1234", "PXL_20261009_142211", "Screenshot 2026-10-09 at 14.22.11",
                   "WhatsApp Image 2026-10-09 at 14.22.11", "20261009_142211", "3f2b9c1e-8a4d-4f6e-9b0a-1c2d3e4f5a6b", "Gemini_Generated_Image_k2j4h5"]
        human = ["Anna", "Kees en Mark", "anna-2", "Tom (2)", "Oma 2024", "Mr Bean"]
        for name in machine:
            with self.subTest(name=name):
                self.assertTrue(references._is_machine_name(name))
        for name in human:
            with self.subTest(name=name):
                self.assertFalse(references._is_machine_name(name))
        with isolated_data():
            document_id = new_document()
            self.assertEqual(references.create_reference(document_id, png(Image.new("RGB", (8, 8))), "Anna.jpg")["name"], "Anna")
            self.assertEqual(references.create_reference(document_id, png(Image.new("RGB", (8, 8))), "IMG_1234.jpg")["name"], "Reference 2")

    def test_existing_code_like_names_are_numbered_once_per_holder(self) -> None:
        with isolated_data():
            document_id = new_document()
            ids = [references.create_reference(holder, png(Image.new("RGB", (8, 8))), name)["id"]
                   for holder, name in ((document_id, "Anna.png"), (document_id, "x.png"), (None, "y.png"))]
            with storage.connect() as db:
                db.executemany("UPDATE document_references SET name = ? WHERE id = ?", [("T5S8DAP28-7b1f1113fc48", ids[1]), ("PXL_20261009_142211", ids[2])])
                db.execute("DELETE FROM schema_migrations WHERE name = 'references.0003_machine_names'")
                db.commit()
            references.ensure_reference_storage()
            self.assertEqual([references.get_reference(item)["name"] for item in ids], ["Anna", "Reference 2", "Reference 1"])

    def test_loose_references_wait_on_the_start_screen_and_copy_into_a_document(self) -> None:
        with isolated_data():
            loose = references.create_reference(None, png(Image.new("RGB", (20, 40), "orange")), "Anna.png")
            self.assertIsNone(loose["document_id"])
            references.set_subject(loose["id"], Image.new("RGB", (10, 10), "red"))
            document_id = new_document()
            self.assertEqual([item["id"] for item in references.list_references(None)], [loose["id"]])
            self.assertEqual(references.list_references(document_id), [])
            [copy] = references.copy_references([loose["id"]], document_id)
            self.assertEqual((copy["document_id"], copy["name"]), (document_id, "Anna"))
            self.assertNotEqual(references.model_image_path(copy["id"]), references.model_image_path(loose["id"]))
            # Deleting the copy keeps the loose original and its files.
            references.delete_reference(copy["id"])
            self.assertTrue(references.model_image_path(loose["id"]).is_file())
            self.assertTrue(references.original_image_path(loose["id"]).is_file())
            with self.assertRaises(ValueError):
                references.copy_references([references.create_reference(document_id, png(Image.new("RGB", (8, 8))), "t.png")["id"]], document_id)
            with self.assertRaises(LookupError):
                references.copy_references([loose["id"]], document_id + 99)

    def test_older_databases_are_migrated_to_allow_loose_references(self) -> None:
        with isolated_data():
            reference_id = references.create_reference(new_document(), png(Image.new("RGB", (8, 8))), "p.png")["id"]
            with storage.connect() as db:
                db.executescript(
                    "CREATE TABLE old AS SELECT * FROM document_references; DROP TABLE document_references;"
                    + references.REFERENCE_SCHEMA
                    + "INSERT INTO document_references SELECT * FROM old; DROP TABLE old;"
                    " DELETE FROM schema_migrations WHERE name = 'references.0002_loose';"
                )
                self.assertEqual({row["name"]: row["notnull"] for row in db.execute("PRAGMA table_info(document_references)")}["document_id"], 1)
            references.ensure_reference_storage()
            self.assertIsNotNone(references.get_reference(reference_id))
            self.assertIsNone(references.create_reference(None, png(Image.new("RGB", (8, 8))), "q.png")["document_id"])

    def test_model_uses_the_isolated_subject_until_the_background_is_restored(self) -> None:
        with isolated_data():
            reference_id = references.create_reference(new_document(), png(Image.new("RGB", (32, 32), "blue")), "p.png")["id"]
            original = references.model_image_path(reference_id)
            isolated = references.set_subject(reference_id, Image.new("RGB", (16, 16), "red"))
            self.assertIsNotNone(isolated["subject_url"])
            subject = references.model_image_path(reference_id)
            self.assertNotEqual(subject, original)
            references.clear_subject(reference_id)
            self.assertFalse(subject.exists())
            self.assertEqual(references.model_image_path(reference_id), original)

    def test_extracted_subject_sits_on_neutral_grey_with_a_margin(self) -> None:
        image = Image.new("RGB", (200, 200), (10, 200, 10))
        mask = np.zeros((200, 200), dtype=np.uint8)
        mask[50:150, 80:120] = 1
        subject = references.extract_subject(image, mask)
        self.assertEqual(subject.size, (52, 112))  # 6% of the 100 px subject on each side
        self.assertEqual(subject.getpixel((0, 0)), references.SUBJECT_BACKGROUND)
        self.assertEqual(subject.getpixel((26, 56)), (10, 200, 10))
        with self.assertRaises(ValueError):
            references.extract_subject(image, np.zeros((200, 200), dtype=np.uint8))


class ReferenceApiTests(unittest.TestCase):
    def test_upload_list_and_delete_through_the_api(self) -> None:
        with isolated_data():
            document_id = new_document()
            client = TestClient(app)
            response = client.post(f"/api/v2/documents/{document_id}/references", files={"image": ("concept.png", png(Image.new("RGB", (30, 60))), "image/png")})
            self.assertEqual(response.status_code, 200)
            reference_id = response.json()["id"]
            self.assertEqual(len(client.get(f"/api/v2/documents/{document_id}/references").json()), 1)
            self.assertEqual(client.post("/api/v2/documents/999/references", files={"image": ("c.png", png(Image.new("RGB", (8, 8))), "image/png")}).status_code, 404)
            self.assertEqual(client.post(f"/api/v2/documents/{document_id}/references", files={"image": ("c.png", b"not an image", "image/png")}).status_code, 422)
            self.assertEqual(client.delete(f"/api/v2/references/{reference_id}").status_code, 200)
            self.assertEqual(client.delete(f"/api/v2/references/{reference_id}").status_code, 404)

    def test_loose_references_and_copying_through_the_api(self) -> None:
        with isolated_data():
            client = TestClient(app)
            response = client.post("/api/v2/references", files={"image": ("Anna.png", png(Image.new("RGB", (30, 60))), "image/png")})
            self.assertEqual(response.status_code, 200)
            loose_id = response.json()["id"]
            self.assertEqual([item["id"] for item in client.get("/api/v2/references").json()], [loose_id])
            document_id = new_document()
            copied = client.post(f"/api/v2/documents/{document_id}/references/copy", json={"reference_ids": [loose_id]})
            self.assertEqual(copied.status_code, 200)
            self.assertEqual(copied.json()[0]["document_id"], document_id)
            self.assertEqual(len(client.get(f"/api/v2/documents/{document_id}/references").json()), 1)
            self.assertEqual(client.post("/api/v2/documents/999/references/copy", json={"reference_ids": [loose_id]}).status_code, 404)
            self.assertEqual(client.post(f"/api/v2/documents/{document_id}/references/copy", json={"reference_ids": [copied.json()[0]["id"]]}).status_code, 422)

    def test_isolation_reports_when_no_subject_is_found(self) -> None:
        with isolated_data():
            reference_id = references.create_reference(new_document(), png(Image.new("RGB", (32, 32))), "p.png")["id"]
            with patch("app.main.selector.subject_mask", return_value=None):
                response = TestClient(app).post(f"/api/v2/references/{reference_id}/subject")
            self.assertEqual(response.status_code, 422)
            mask = np.zeros((32, 32), dtype=np.uint8)
            mask[8:24, 8:24] = 1
            with patch("app.main.selector.subject_mask", return_value=mask):
                response = TestClient(app).post(f"/api/v2/references/{reference_id}/subject")
            self.assertEqual(response.status_code, 200)
            self.assertIsNotNone(response.json()["subject_url"])

    def test_inpaint_rejects_missing_references_and_unknown_models_before_queueing(self) -> None:
        with isolated_data():
            reference_id = references.create_reference(new_document(), png(Image.new("RGB", (8, 8))), "p.png")["id"]
            for fields in ({"reference_ids": [str(reference_id), str(reference_id + 5)]}, {"reference_ids": [str(reference_id)], "model_name": "not-a-model"}):
                with self.subTest(fields=fields), patch("app.main.jobs.submit") as submit:
                    response = TestClient(app).post(
                        "/api/inpaint",
                        data={"prompt": "sitting on a bench", **fields},
                        files={field: (field + ".png", b"never decoded", "image/png") for field in ("image", "mask")},
                    )
                    self.assertEqual(response.status_code, 422)
                    submit.assert_not_called()


    def test_generation_rejects_missing_references_and_unknown_models_before_queueing(self) -> None:
        with isolated_data():
            reference_id = references.create_reference(None, png(Image.new("RGB", (8, 8))), "p.png")["id"]
            for fields in ({"reference_ids": [reference_id, reference_id + 5]}, {"reference_ids": [reference_id], "model_name": "not-a-model"}):
                with self.subTest(fields=fields), patch("app.main.jobs.submit") as submit:
                    response = TestClient(app).post("/api/generate", json={"prompt": "on a roller coaster", **fields})
                    self.assertEqual(response.status_code, 422)
                    submit.assert_not_called()


    def test_whole_image_edits_need_a_known_model_and_existing_references_before_queueing(self) -> None:
        with isolated_data():
            reference_id = references.create_reference(None, png(Image.new("RGB", (8, 8))), "p.png")["id"]
            client = TestClient(app)
            picture = ("picture.png", png(Image.new("RGB", (64, 48))), "image/png")
            for fields in ({"model_name": "not-a-model"}, {"reference_ids": [reference_id + 5]}, {"prompt": ""}):
                with self.subTest(fields=fields), patch("app.main.jobs.submit") as submit:
                    parameters = json.dumps({"prompt": "make it night", **fields})
                    self.assertEqual(client.post("/api/edit", files={"image": picture}, data={"parameters": parameters}).status_code, 422)
                    submit.assert_not_called()
            with patch("app.main.jobs.submit", return_value=SimpleNamespace(id="job", kind="edit", status="queued")) as submit:
                parameters = json.dumps({"prompt": "make it night", "reference_ids": [reference_id]})
                response = client.post("/api/edit", files={"image": picture}, data={"parameters": parameters})
                self.assertEqual(response.status_code, 200)
                self.assertEqual(submit.call_args.args[0], "edit")


class ReferenceGenerationTests(unittest.TestCase):
    def test_whole_image_edits_keep_the_picture_and_record_their_inputs(self) -> None:
        with isolated_data():
            anna = references.create_reference(None, png(Image.new("RGB", (20, 40), "orange")), "Anna.png")["id"]
            provider = MagicMock()

            def fake_edit(_request, input_path, output_path, _control, **kwargs):
                with Image.open(input_path) as picture:
                    Image.new("RGB", picture.size).save(output_path)
                return 7, picture.width

            provider.edit.side_effect = fake_edit
            with patch("app.services.registry.activate", return_value=provider):
                ids = run_image_edit(EditRequest(prompt="make it night", reference_ids=[anna], count=2), png(Image.new("RGB", (90, 60))), "photo.png", MagicMock())
                self.assertEqual([item.name for item in provider.edit.call_args.kwargs["references"]], ["Anna"])
                record = storage.get_generation(ids[1])
                self.assertEqual((record["task"], record["width"], record["height"]), ("edit", 90, 60))
                self.assertTrue(record["input_image_path"])
                self.assertEqual(len(record["reference_image_paths"]), 1)
                provider.edit.side_effect = RuntimeError("out of memory")
                outputs = len(list(settings.output_dir.iterdir()))
                with self.assertRaises(RuntimeError):
                    run_image_edit(EditRequest(prompt="add snow"), png(Image.new("RGB", (90, 60))), "photo.png", MagicMock())
                self.assertEqual(len(list(settings.output_dir.iterdir())), outputs)

    def test_generation_references_reach_the_provider_in_order_and_are_recorded(self) -> None:
        with isolated_data():
            anna = references.create_reference(None, png(Image.new("RGB", (20, 40), "orange")), "Anna.png")["id"]
            tom = references.create_reference(None, png(Image.new("RGB", (30, 30), "blue")), "Tom.png")["id"]
            provider = MagicMock()

            def fake_generate(_request, output_path, _control, **kwargs):
                Image.new("RGB", (32, 32)).save(output_path)
                return 1, 32

            provider.generate.side_effect = fake_generate
            with patch("app.services.registry.activate", return_value=provider):
                ids = run_text_to_image(TextToImageRequest(prompt="on a roller coaster", reference_ids=[anna, tom], count=2), MagicMock())
                passed = provider.generate.call_args.kwargs["references"]
                self.assertEqual([item.name for item in passed], ["Anna", "Tom"])
                self.assertEqual(len(storage.get_generation(ids[1])["reference_image_paths"]), 2)
                provider.generate.reset_mock()
                run_text_to_image(TextToImageRequest(prompt="a garden"), MagicMock())
                self.assertNotIn("references", provider.generate.call_args.kwargs)

    def test_references_reach_the_provider_in_order_and_are_recorded(self) -> None:
        with isolated_data():
            document_id = new_document()
            reference_id = references.create_reference(document_id, png(Image.new("RGB", (20, 40), "orange")), "Anna.png")["id"]
            second_id = references.create_reference(document_id, png(Image.new("RGB", (30, 30), "blue")), "Tom.png")["id"]
            source = Image.new("RGB", (64, 64), "purple")
            mask = Image.new("L", source.size, 0)
            mask.paste(255, (16, 16, 48, 48))
            provider = MagicMock()

            def fake_inpaint(_request, input_path, _mask_path, output_path, _control, **kwargs):
                with Image.open(input_path) as model_input:
                    model_input.save(output_path)
                return 1, 64

            provider.inpaint.side_effect = fake_inpaint
            with patch("app.services.registry.activate", return_value=provider):
                ids = run_inpaint(InpaintForm(prompt="on a bench", reference_ids=[second_id, reference_id]), png(source), "s.png", png(mask), "m.png", MagicMock())
                passed = provider.inpaint.call_args.kwargs["references"]
                self.assertEqual([item.name for item in passed], ["Tom", "Anna"])
                self.assertEqual(passed[1].path, references.model_image_path(reference_id))
                self.assertEqual(len(storage.get_generation(ids[0])["reference_image_paths"]), 2)

                provider.inpaint.reset_mock()
                run_inpaint(InpaintForm(prompt="a mushroom"), png(source), "s.png", png(mask), "m.png", MagicMock())
                self.assertNotIn("references", provider.inpaint.call_args.kwargs)

    def test_prompt_names_the_reference_only_for_replacement(self) -> None:
        with_reference = _inpaint_prompt(InpaintForm(prompt="sitting on a bench."), ["Anna"])
        self.assertTrue(with_reference.startswith("sitting on a bench. The subject is the person or object from the reference image"))
        self.assertNotIn("reference", _inpaint_prompt(InpaintForm(prompt="sitting on a bench")))
        self.assertNotIn("reference image", _inpaint_prompt(InpaintForm(prompt="grass", edit_operation="remove"), ["Anna"]))

    def test_reference_ids_are_validated(self) -> None:
        with self.assertRaises(ValidationError):
            OutpaintForm(prompt="sky", left=64, reference_ids=[3])
        for invalid in ([0], [1, 2, 3, 4]):
            with self.subTest(invalid=invalid), self.assertRaises(ValidationError):
                InpaintForm(prompt="x", reference_ids=invalid)
        self.assertEqual(InpaintForm(prompt="x", reference_ids=[2, 1, 2]).reference_ids, [2, 1])


if __name__ == "__main__":
    unittest.main()
