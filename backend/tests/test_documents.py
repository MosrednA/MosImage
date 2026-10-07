from __future__ import annotations

import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from app import documents, storage
from app.config import settings


def png_bytes(color: tuple[int, int, int], size: tuple[int, int] = (96, 64)) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", size, color).save(buffer, "PNG")
    return buffer.getvalue()


class DocumentPersistenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        root = Path(self.temporary.name)
        self.patches = [
            patch.object(settings, "data_dir", root),
            patch.object(settings, "db_path", root / "app.db"),
            patch.object(settings, "image_dir", root / "images"),
        ]
        for active_patch in self.patches:
            active_patch.start()
        storage.ensure_storage()
        documents.ensure_document_storage()

    def tearDown(self) -> None:
        for active_patch in reversed(self.patches):
            active_patch.stop()
        self.temporary.cleanup()

    def test_document_reopens_and_metadata_autosaves(self) -> None:
        created = documents.create_from_upload("Landscape", png_bytes((10, 20, 30)), "source.png")
        renamed = documents.update_document(created["id"], "Reopened landscape")
        reopened = documents.get_document(created["id"])

        self.assertEqual(renamed["name"], "Reopened landscape")
        self.assertEqual(reopened["name"], "Reopened landscape")
        self.assertEqual((reopened["canvas_width"], reopened["canvas_height"]), (96, 64))
        self.assertTrue(reopened["current_image_url"].startswith("/files/images/documents/"))

    def test_accepts_an_alternative_then_undoes_and_redoes(self) -> None:
        document = documents.create_from_upload("Base", png_bytes((10, 10, 10)), "base.png")
        generation_ids = [
            self._create_generation((200, 30, 30), 101),
            self._create_generation((30, 200, 30), 102),
        ]

        proposal = documents.create_alternative_proposal(document["id"], generation_ids, "inpaint")
        unchanged = documents.get_document(document["id"])
        accepted = documents.accept_alternative(
            document["id"], proposal["id"], proposal["alternatives"][1]["id"]
        )
        undone = documents.undo(document["id"])
        redone = documents.redo(document["id"])

        self.assertEqual(unchanged["current_revision_id"], document["current_revision_id"])
        self.assertTrue(accepted["can_undo"])
        self.assertTrue(undone["can_redo"])
        self.assertEqual(redone["current_revision_id"], accepted["current_revision_id"])
        selected = documents.get_revision(proposal["id"])
        self.assertEqual(selected["selected_alternative_id"], proposal["alternatives"][1]["id"])

    def _create_generation(self, color: tuple[int, int, int], seed: int) -> int:
        output = settings.output_dir / f"generation-{seed}.png"
        output.write_bytes(png_bytes(color))
        return storage.create_generation(
            {
                "task": "inpaint",
                "prompt": "replace object",
                "negative_prompt": None,
                "model_name": "z-image-turbo",
                "runtime_mode": "balanced",
                "seed": seed,
                "steps": 8,
                "guidance": 0,
                "width": 96,
                "height": 64,
                "strength": 1,
                "input_image_path": None,
                "mask_image_path": None,
                "output_image_path": storage.to_relative(output),
            }
        )

    def test_outpaint_accept_undo_redo_restores_canvas_dimensions(self) -> None:
        document = documents.create_from_upload("Base", png_bytes((10, 10, 10)), "base.png")
        generation_id = self._create_generation((200, 30, 30), 201)
        record = storage.get_generation(generation_id)
        (settings.data_dir / record["output_image_path"]).write_bytes(png_bytes((200, 30, 30), (160, 96)))
        with storage.connect() as db:
            db.execute("UPDATE generations SET task = 'outpaint', width = 160, height = 96 WHERE id = ?", (generation_id,))
            db.commit()
        proposal = documents.create_alternative_proposal(document["id"], [generation_id], "outpaint", document["current_revision_id"])
        self.assertEqual(documents.get_document(document["id"])["canvas_width"], 96)
        accepted = documents.accept_alternative(document["id"], proposal["id"], proposal["alternatives"][0]["id"])
        self.assertEqual((accepted["canvas_width"], accepted["canvas_height"]), (160, 96))
        undone = documents.undo(document["id"])
        self.assertEqual((undone["canvas_width"], undone["canvas_height"]), (96, 64))
        redone = documents.redo(document["id"])
        self.assertEqual((redone["canvas_width"], redone["canvas_height"]), (160, 96))
        with self.assertRaisesRegex(ValueError, "changed"):
            documents.create_alternative_proposal(document["id"], [generation_id], "outpaint", document["current_revision_id"])


if __name__ == "__main__":
    unittest.main()
