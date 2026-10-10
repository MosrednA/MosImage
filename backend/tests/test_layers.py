from __future__ import annotations

import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

from app import documents, layers, main, services, storage
from app.compositing import OUTPAINT_BLEND_BAND
from app.config import settings
from app.schemas import OutpaintForm


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

    def background(self, **values: object) -> dict:
        """The document's picture as its bottom layer, as the layer state the app saves."""
        state = {key: value for key, value in self.document["layers"][0].items() if key not in {"image_url", "mask_url", "variant_urls"}}
        return {**state, **values}

    def test_a_new_document_starts_with_its_picture_as_the_background_layer(self) -> None:
        [background] = self.document["layers"]
        self.assertEqual((background["id"], background["name"]), ("background", "Background"))
        self.assertEqual(background["image_url"], self.document["current_image_url"])
        self.assertEqual((background["x"], background["y"], background["width"], background["height"]), (48.0, 32.0, 96, 64))
        # It is an ordinary layer: hidden, moved or gone, the picture follows the stack.
        hidden = documents.save_layers(self.document["id"], self.document["current_revision_id"], [self.background(visible=False, x=10.0)], "Hide layer")
        self.assertEqual((hidden["layers"][0]["visible"], hidden["layers"][0]["x"]), (False, 10.0))
        empty = documents.save_layers(hidden["id"], hidden["current_revision_id"], [], "Delete layer")
        self.assertEqual((empty["layers"], empty["canvas_width"]), ([], 96))

    def test_older_revisions_get_their_picture_as_the_bottom_layer(self) -> None:
        asset = layers.save_asset(image_bytes("RGBA", (20, 10), (255, 0, 0, 255)), "image")
        document = documents.save_layers(self.document["id"], self.document["current_revision_id"], [self.background(), self.layer(asset, visible=False)], "Add layer")
        generation = storage.create_generation({
            "task": "outpaint", "prompt": "more garden", "negative_prompt": None, "model_name": "flux2-klein-9b-nvfp4",
            "runtime_mode": "balanced", "seed": 1, "steps": 4, "guidance": 1.0, "width": 160, "height": 64, "strength": 1.0,
            "input_image_path": None, "mask_image_path": None, "output_image_path": self._output((160, 64)),
        })
        proposal = documents.create_alternative_proposal(document["id"], [generation], "outpaint", layer_offset=(32, 0))
        # Revisions from before the Background was a layer: the picture under a stack of their own.
        with storage.connect() as db:
            for revision in db.execute("SELECT id, layers, operation_parameters FROM revisions").fetchall():
                stack = [layer for layer in json.loads(revision["layers"]) if layer["id"] != "background"]
                parameters = {key: value for key, value in json.loads(revision["operation_parameters"]).items() if key != "result_layer"}
                db.execute("UPDATE revisions SET layers = ?, operation_parameters = ? WHERE id = ?", (json.dumps(stack), json.dumps(parameters), revision["id"]))
            db.execute("DELETE FROM schema_migrations WHERE name = 'documents.0003_background_layers'")
            db.commit()

        documents.ensure_document_storage()
        documents.ensure_document_storage()

        current = documents.get_document(document["id"])
        self.assertEqual([layer["id"] for layer in current["layers"]], ["background", "anna"])
        self.assertEqual(current["layers"][0]["image_url"], current["current_image_url"])
        first = documents.get_revision(self.document["current_revision_id"])
        self.assertEqual([layer["id"] for layer in first["layers"]], ["background"])
        # A proposal still waiting shows its chosen alternative in that bottom layer.
        self.assertEqual(documents.get_revision(proposal["id"])["operation_parameters"]["result_layer"], "background")

    def test_layer_changes_are_undoable_revisions_over_the_same_background(self) -> None:
        asset = layers.save_asset(image_bytes("RGBA", (20, 10), (255, 0, 0, 128)), "image")
        added = documents.save_layers(self.document["id"], self.document["current_revision_id"], [self.background(), self.layer(asset)], "Add layer")
        moved = documents.save_layers(added["id"], added["current_revision_id"], [self.background(), self.layer(asset, x=10.0)], "Move layer")

        self.assertEqual(moved["current_image_url"], self.document["current_image_url"])
        self.assertEqual((moved["layers"][1]["x"], moved["layers"][1]["width"]), (10.0, 20))
        self.assertTrue(moved["layers"][1]["image_url"].startswith("/files/images/layers/layer-"))
        self.assertEqual(documents.undo(moved["id"])["layers"][1]["x"], 48.0)
        self.assertEqual([layer["id"] for layer in documents.undo(moved["id"])["layers"]], ["background"])
        self.assertEqual(documents.redo(moved["id"])["layers"][1]["x"], 48.0)

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

    def test_expand_continues_the_visible_picture_and_moves_layers_over_their_own_background(self) -> None:
        asset = layers.save_asset(image_bytes("RGBA", (20, 10), (255, 0, 0, 255)), "image")
        document = documents.save_layers(self.document["id"], self.document["current_revision_id"], [self.background(), self.layer(asset)], "Add layer")
        # The engine saw the background with the red layer on it and drew the new area beside it.
        seen = Image.new("RGBA", (160, 64), (128, 128, 128, 0))
        seen.paste((10, 20, 30, 255), (32, 0, 128, 64))
        seen.paste((255, 0, 0, 255), (70, 27, 90, 37))
        output = seen.copy()
        output.paste((1, 2, 3, 255), (0, 0, 32, 64))
        output.paste((1, 2, 3, 255), (128, 0, 160, 64))
        generation = storage.create_generation({
            "task": "outpaint", "prompt": "more garden", "negative_prompt": None, "model_name": "flux2-klein-9b-nvfp4",
            "runtime_mode": "balanced", "seed": 1, "steps": 4, "guidance": 1.0, "width": 160, "height": 64, "strength": 1.0,
            "input_image_path": self._save(seen, settings.input_dir / "input-wide.png"), "mask_image_path": None,
            "output_image_path": self._save(output, settings.output_dir / "output-wide.png"),
        })
        proposal = documents.create_alternative_proposal(document["id"], [generation], "outpaint", layer_offset=(32, 0))
        self.assertEqual(proposal["layers"][1]["x"], 80.0)
        with Image.open(settings.data_dir / proposal["alternatives"][0]["output_image_path"]) as alternative:
            self.assertEqual(alternative.getpixel((80, 32)), (10, 20, 30, 255))  # The background, not a copy of the layer.
            self.assertEqual(alternative.getpixel((4, 32)), (1, 2, 3, 255))
        accepted = documents.accept_alternative(document["id"], proposal["id"], proposal["alternatives"][0]["id"])
        # The Background grows with the canvas and the layer moves along with the content.
        background, layer = accepted["layers"]
        self.assertEqual((background["id"], background["x"], background["width"], background["height"]), ("background", 80.0, 160, 64))
        self.assertEqual(background["image_path"], proposal["alternatives"][0]["output_image_path"])
        self.assertEqual((layer["x"], layer["y"]), (80.0, 32.0))
        self.assertEqual((accepted["canvas_width"], accepted["canvas_height"]), (160, 64))

    def test_expanding_a_layered_picture_from_engine_to_accepted_background(self) -> None:
        # A layer covers the left of the picture, up to its edge, as an applied AI result often does.
        asset = layers.save_asset(image_bytes("RGBA", (40, 64), (200, 30, 30, 255)), "image")
        document = documents.save_layers(self.document["id"], self.document["current_revision_id"], [self.background(), self.layer(asset, x=20.0)], "Add layer")
        picture = Image.new("RGB", (96, 64), (10, 20, 30))
        picture.paste((200, 30, 30), (0, 0, 40, 64))
        buffer = io.BytesIO()
        picture.save(buffer, "PNG")
        provider = MagicMock()

        def draw_grey(request, source_path, mask_path, output_path, control):
            with Image.open(mask_path) as model_mask:
                Image.new("RGB", model_mask.size, (150, 150, 150)).save(output_path)
            return 7, model_mask.width

        provider.inpaint.side_effect = draw_grey
        with patch("app.services.registry.activate", return_value=provider):
            generation_ids = services.run_outpaint(OutpaintForm(prompt="more garden", left=32, right=32), buffer.getvalue(), MagicMock())
        proposal = documents.create_alternative_proposal(document["id"], generation_ids, "outpaint", document["current_revision_id"], (32, 0))
        accepted = documents.accept_alternative(document["id"], proposal["id"], proposal["alternatives"][0]["id"])

        self.assertEqual((accepted["layers"][1]["x"], accepted["canvas_width"]), (52.0, 160))
        self.assertEqual(accepted["layers"][0]["image_url"], accepted["current_image_url"])
        with Image.open(settings.data_dir / accepted["current_image_url"].removeprefix("/files/")) as background:
            pixels = np.asarray(background.convert("RGBA")).astype(int)
        # The new edge continues the layer on the left and the background on the right.
        self.assertTrue((pixels[:, 31, 0] - pixels[:, 31, 1] > 30).all())
        self.assertTrue((np.abs(pixels[:, 128, 0] - pixels[:, 128, 1]) < 5).all())
        # Under the layer the background keeps its own pixels: moving the layer shows no copy of it.
        self.assertTrue((pixels[:, 32 + OUTPAINT_BLEND_BAND:128 - OUTPAINT_BLEND_BAND] == (10, 20, 30, 255)).all())

    def test_a_layered_expansion_needs_the_picture_the_engine_continued(self) -> None:
        asset = layers.save_asset(image_bytes("RGBA", (20, 10), (255, 0, 0, 255)), "image")
        document = documents.save_layers(self.document["id"], self.document["current_revision_id"], [self.background(), self.layer(asset)], "Add layer")
        generation = storage.create_generation({
            "task": "outpaint", "prompt": "more garden", "negative_prompt": None, "model_name": "flux2-klein-9b-nvfp4",
            "runtime_mode": "balanced", "seed": 1, "steps": 4, "guidance": 1.0, "width": 160, "height": 64, "strength": 1.0,
            "input_image_path": None, "mask_image_path": None, "output_image_path": self._output((160, 64)),
        })
        with self.assertRaisesRegex(ValueError, "picture it continued"):
            documents.create_alternative_proposal(document["id"], [generation], "outpaint", layer_offset=(32, 0))
        self.assertEqual(list(settings.revision_dir.glob("alternative-*")), [])

    def test_expand_puts_the_new_picture_under_the_layers_when_the_bottom_one_does_not_fill_the_canvas(self) -> None:
        moved = documents.save_layers(self.document["id"], self.document["current_revision_id"], [self.background(x=40.0)], "Move layer")
        outputs = [self._save(Image.new("RGB", (160, 64), color), settings.output_dir / f"output-{index}.png") for index, color in enumerate(((1, 2, 3), (4, 5, 6)))]
        generations = [storage.create_generation({
            "task": "outpaint", "prompt": "more garden", "negative_prompt": None, "model_name": "flux2-klein-9b-nvfp4",
            "runtime_mode": "balanced", "seed": index, "steps": 4, "guidance": 1.0, "width": 160, "height": 64, "strength": 1.0,
            "input_image_path": None, "mask_image_path": None, "output_image_path": output,
        }) for index, output in enumerate(outputs)]
        proposal = documents.create_alternative_proposal(moved["id"], generations, "outpaint", layer_offset=(32, 0))
        accepted = documents.accept_alternative(moved["id"], proposal["id"], proposal["alternatives"][1]["id"])

        expanded, background = accepted["layers"]
        self.assertEqual((expanded["name"], expanded["width"], expanded["x"]), ("Expanded", 160, 80.0))
        self.assertEqual(expanded["image_path"], proposal["alternatives"][1]["output_image_path"])
        self.assertEqual((background["id"], background["x"], background["width"]), ("background", 72.0, 96))

    def test_crop_keeps_the_background_layer_whole(self) -> None:
        cropped = documents.crop(self.document["id"], self.document["current_revision_id"], 16, 8, 48, 40)
        self.assertEqual((cropped["canvas_width"], cropped["canvas_height"]), (48, 40))
        [background] = cropped["layers"]
        self.assertEqual((background["x"], background["y"], background["width"]), (32.0, 24.0, 96))
        self.assertEqual(background["image_url"], self.document["current_image_url"])

    def test_rejects_unknown_files_mismatched_masks_and_stale_bases(self) -> None:
        asset = layers.save_asset(image_bytes("RGBA", (20, 10), (255, 0, 0, 255)), "image")
        output = {"path": self._output((20, 10))}
        with self.assertRaisesRegex(ValueError, "missing"):
            documents.save_layers(self.document["id"], self.document["current_revision_id"], [self.layer(output)], "Add")
        with self.assertRaisesRegex(ValueError, "cannot be used"):
            documents.save_layers(self.document["id"], self.document["current_revision_id"], [self.layer(layers.save_asset(image_bytes("L", (20, 10), 255), "mask"))], "Add")
        wrong_mask = layers.save_asset(image_bytes("L", (8, 8), 255), "mask")
        with self.assertRaisesRegex(ValueError, "match"):
            documents.save_layers(self.document["id"], self.document["current_revision_id"], [self.layer(asset, mask_path=wrong_mask["path"])], "Mask")
        with self.assertRaisesRegex(ValueError, "changed"):
            documents.save_layers(self.document["id"], self.document["current_revision_id"] + 99, [], "Clear")

    def test_masks_take_coverage_from_alpha_and_large_layers_are_scaled_down(self) -> None:
        mask = layers.save_asset(image_bytes("RGBA", (20, 10), (255, 255, 255, 77)), "mask")
        with Image.open(settings.data_dir / mask["path"]) as stored:
            self.assertEqual((stored.mode, stored.getpixel((3, 3))), ("L", 77))
        large = layers.save_asset(image_bytes("RGB", (9000, 90), (0, 0, 0), "JPEG"), "image")
        self.assertEqual((large["width"], large["height"]), (8192, 82))

    def test_masks_can_be_switched_off_and_older_stacks_keep_them_on(self) -> None:
        asset = layers.save_asset(image_bytes("RGBA", (20, 10), (255, 0, 0, 255)), "image")
        mask = layers.save_asset(image_bytes("L", (20, 10), 0), "mask")
        older = documents.save_layers(self.document["id"], self.document["current_revision_id"], [self.layer(asset, mask_path=mask["path"])], "Add mask")
        self.assertTrue(older["layers"][0]["mask_enabled"])

        off = documents.save_layers(older["id"], older["current_revision_id"], [self.layer(asset, mask_path=mask["path"], mask_enabled=False)], "Disable layer mask")
        self.assertEqual((off["layers"][0]["mask_enabled"], off["layers"][0]["mask_url"]), (False, storage.image_url(mask["path"])))
        self.assertTrue(documents.undo(off["id"])["layers"][0]["mask_enabled"])

    def test_variants_switch_within_one_layer_and_their_files_stay_until_the_document_goes(self) -> None:
        first, second = (layers.save_asset(image_bytes("RGBA", (20, 10), (value, 0, 0, 255)), "image") for value in (255, 128))
        variants = [first["path"], second["path"]]
        saved = documents.save_layers(self.document["id"], self.document["current_revision_id"], [self.layer(first, variant_paths=variants)], "Add AI result")
        self.assertEqual(saved["layers"][0]["variant_urls"], [storage.image_url(path) for path in variants])
        switched = documents.save_layers(saved["id"], saved["current_revision_id"], [self.layer(second, variant_paths=variants)], "Switch variant")
        self.assertEqual(documents.undo(switched["id"])["layers"][0]["image_path"], first["path"])

        other_size = layers.save_asset(image_bytes("RGBA", (8, 8), (0, 0, 0, 255)), "image")
        with self.assertRaisesRegex(ValueError, "same size|match"):
            documents.save_layers(saved["id"], saved["current_revision_id"], [self.layer(first, variant_paths=[first["path"], other_size["path"]])], "Mixed")
        with self.assertRaisesRegex(ValueError, "one of its variants"):
            documents.save_layers(saved["id"], saved["current_revision_id"], [self.layer(other_size, variant_paths=variants)], "Outside")

        # Only the first variant is shown, yet deleting the document removes both files.
        documents.delete_document(saved["id"])
        self.assertFalse(any((settings.data_dir / path).exists() for path in variants))

    def test_older_stacks_have_no_variants(self) -> None:
        asset = layers.save_asset(image_bytes("RGBA", (20, 10), (255, 0, 0, 255)), "image")
        record = layers.records(f'[{{"id": "anna", "name": "Anna", "image_path": "{asset["path"]}", "x": 1, "y": 2}}]')[0]
        self.assertEqual((record["variant_paths"], record["variant_urls"]), ([], []))

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
        return self._save(Image.new("RGB", size, (1, 2, 3)), settings.output_dir / "output-wide.png")

    def _save(self, image: Image.Image, path: Path) -> str:
        image.save(path)
        return storage.to_relative(path)


if __name__ == "__main__":
    unittest.main()
