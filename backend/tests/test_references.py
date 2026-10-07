from __future__ import annotations

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
from app.schemas import InpaintForm, OutpaintForm
from app.services import run_inpaint


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

    def test_inpaint_rejects_missing_references_and_non_klein_models_before_queueing(self) -> None:
        with isolated_data():
            reference_id = references.create_reference(new_document(), png(Image.new("RGB", (8, 8))), "p.png")["id"]
            for fields in ({"reference_ids": [str(reference_id), str(reference_id + 5)]}, {"reference_ids": [str(reference_id)], "model_name": "z-image-turbo"}):
                with self.subTest(fields=fields), patch("app.main.jobs.submit") as submit:
                    response = TestClient(app).post(
                        "/api/inpaint",
                        data={"prompt": "sitting on a bench", **fields},
                        files={field: (field + ".png", b"never decoded", "image/png") for field in ("image", "mask")},
                    )
                    self.assertEqual(response.status_code, 422)
                    submit.assert_not_called()


class ReferenceGenerationTests(unittest.TestCase):
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
