from __future__ import annotations

import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient
from PIL import Image

from app import documents, layers, main, storage
from app.config import settings


def image_bytes(mode: str, size: tuple[int, int], color: object, kind: str = "PNG") -> bytes:
    buffer = io.BytesIO()
    Image.new(mode, size, color).save(buffer, kind)
    return buffer.getvalue()


class LayerTests(unittest.TestCase):
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
        self.document = documents.create_from_upload("Garden", image_bytes("RGB", (96, 64), (10, 20, 30)), "garden.png")

    def tearDown(self) -> None:
        for active_patch in reversed(self.patches):
            active_patch.stop()
        self.temporary.cleanup()

    def layer(self, asset: dict, **values: object) -> dict:
        return {"id": "anna", "name": "Anna", "image_path": asset["path"], "x": 48.0, "y": 32.0, **values}

    def test_layer_changes_are_undoable_revisions_over_the_same_background(self) -> None:
        asset = layers.save_asset(image_bytes("RGBA", (20, 10), (255, 0, 0, 128)), "image")
        added = documents.save_layers(self.document["id"], self.document["current_revision_id"], [self.layer(asset)], "Add layer")
        moved = documents.save_layers(added["id"], added["current_revision_id"], [self.layer(asset, x=10.0)], "Move layer")

        self.assertEqual(moved["current_image_url"], self.document["current_image_url"])
        self.assertEqual((moved["layers"][0]["x"], moved["layers"][0]["width"]), (10.0, 20))
        self.assertTrue(moved["layers"][0]["image_url"].startswith("/files/images/layers/layer-"))
        self.assertEqual(documents.undo(moved["id"])["layers"][0]["x"], 48.0)
        self.assertEqual(documents.undo(moved["id"])["layers"], [])
        self.assertEqual(documents.redo(moved["id"])["layers"][0]["x"], 48.0)

    def test_deleting_document_cleans_layer_assets_from_its_entire_history(self) -> None:
        asset = layers.save_asset(image_bytes("RGBA", (20, 10), (255, 0, 0, 128)), "image")
        mask = layers.save_asset(image_bytes("L", (20, 10), 255), "mask")
        added = documents.save_layers(
            self.document["id"], self.document["current_revision_id"],
            [self.layer(asset, mask_path=mask["path"])], "Add layer",
        )
        documents.save_layers(added["id"], added["current_revision_id"], [], "Remove layer")

        documents.delete_document(self.document["id"])

        self.assertIsNone(documents.get_document(self.document["id"]))
        self.assertFalse(storage.absolute_path(asset["path"]).exists())
        self.assertFalse(storage.absolute_path(mask["path"]).exists())

    def test_deleting_document_preserves_assets_in_another_documents_history(self) -> None:
        asset = layers.save_asset(image_bytes("RGBA", (20, 10), (255, 0, 0, 128)), "image")
        mask = layers.save_asset(image_bytes("L", (20, 10), 255), "mask")
        shared = self.layer(asset, mask_path=mask["path"])
        documents.save_layers(
            self.document["id"], self.document["current_revision_id"], [shared], "Add layer",
        )
        other = documents.create_from_upload("Other", image_bytes("RGB", (96, 64), (20, 30, 40)), "other.png")
        added = documents.save_layers(other["id"], other["current_revision_id"], [shared], "Add shared layer")
        documents.save_layers(other["id"], added["current_revision_id"], [], "Remove shared layer")

        documents.delete_document(self.document["id"])

        self.assertTrue(storage.absolute_path(asset["path"]).exists())
        self.assertTrue(storage.absolute_path(mask["path"]).exists())
        restored = documents.undo(other["id"])
        self.assertEqual(restored["layers"][0]["image_path"], asset["path"])

        documents.delete_document(other["id"])
        self.assertFalse(storage.absolute_path(asset["path"]).exists())
        self.assertFalse(storage.absolute_path(mask["path"]).exists())

    def test_background_edits_keep_layers_and_expand_moves_them_with_the_content(self) -> None:
        asset = layers.save_asset(image_bytes("RGBA", (20, 10), (255, 0, 0, 255)), "image")
        document = documents.save_layers(self.document["id"], self.document["current_revision_id"], [self.layer(asset)], "Add layer")
        generation = storage.create_generation({
            "task": "outpaint", "prompt": "more garden", "negative_prompt": None, "model_name": "z-image-turbo",
            "runtime_mode": "balanced", "seed": 1, "steps": 4, "guidance": 1.0, "width": 160, "height": 64, "strength": 1.0,
            "input_image_path": None, "mask_image_path": None, "output_image_path": self._output((160, 64)),
        })
        proposal = documents.create_alternative_proposal(document["id"], [generation], "outpaint", layer_offset=(32, 0))
        self.assertEqual(proposal["layers"][0]["x"], 80.0)
        accepted = documents.accept_alternative(document["id"], proposal["id"], proposal["alternatives"][0]["id"])
        self.assertEqual((accepted["layers"][0]["x"], accepted["layers"][0]["y"]), (80.0, 32.0))

    def test_rejects_unknown_files_mismatched_masks_and_stale_bases(self) -> None:
        asset = layers.save_asset(image_bytes("RGBA", (20, 10), (255, 0, 0, 255)), "image")
        document_image = {"path": f"images/documents/{Path(self.document['current_image_url']).name}"}
        with self.assertRaisesRegex(ValueError, "missing"):
            documents.save_layers(self.document["id"], self.document["current_revision_id"], [self.layer(document_image)], "Add")
        wrong_mask = layers.save_asset(image_bytes("L", (8, 8), 255), "mask")
        with self.assertRaisesRegex(ValueError, "match"):
            documents.save_layers(self.document["id"], self.document["current_revision_id"], [self.layer(asset, mask_path=wrong_mask["path"])], "Mask")
        with self.assertRaisesRegex(ValueError, "changed"):
            documents.save_layers(self.document["id"], self.document["current_revision_id"] + 99, [], "Clear")

    def test_masks_take_coverage_from_alpha_and_large_layers_are_scaled_down(self) -> None:
        mask = layers.save_asset(image_bytes("RGBA", (20, 10), (255, 255, 255, 77)), "mask")
        with Image.open(settings.data_dir / mask["path"]) as stored:
            self.assertEqual((stored.mode, stored.getpixel((3, 3))), ("L", 77))
        large = layers.save_asset(image_bytes("RGB", (5000, 2500), (0, 0, 0), "JPEG"), "image")
        self.assertEqual((large["width"], large["height"]), (4096, 2048))

    def test_layer_api_uploads_assets_and_reports_conflicts(self) -> None:
        client = TestClient(main.app)
        url = f"/api/v2/documents/{self.document['id']}"
        asset = client.post(f"{url}/layer-assets", files={"image": ("anna.png", image_bytes("RGBA", (20, 10), (0, 0, 0, 0)), "image/png")}).json()
        body = {"base_revision_id": self.document["current_revision_id"], "label": "Add layer", "layers": [self.layer(asset)]}
        saved = client.put(f"{url}/layers", json=body)
        self.assertEqual(saved.status_code, 200)
        self.assertEqual(saved.json()["layers"][0]["name"], "Anna")
        self.assertEqual(client.put(f"{url}/layers", json=body).status_code, 409)
        flat = {**body, "base_revision_id": saved.json()["current_revision_id"], "layers": [self.layer(asset, scale_x=0)]}
        self.assertEqual(client.put(f"{url}/layers", json=flat).status_code, 422)

    def _output(self, size: tuple[int, int]) -> str:
        path = settings.output_dir / "output-wide.png"
        Image.new("RGB", size, (1, 2, 3)).save(path)
        return storage.to_relative(path)


if __name__ == "__main__":
    unittest.main()
