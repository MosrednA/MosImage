from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

from app import storage
from app.config import settings
from app.detection import Detection, coverage, detection_query, iou, strongest
from app.faces import FACE_QUERY, face_blend_mask, face_candidates, face_crop_box, faces_to_fix, matched_colours, picked_faces, run_face_fix
from app.main import app
from app.matting import UNKNOWN, edge_radius, mask_box, trimap
from app.models.flux2_klein import ENHANCE_TRIGGER, FACE_AREA, Flux2KleinProvider, _face_prompt, _inpaint_prompt, _mode_prompt, _sketch_prompt
from app.schemas import EditRequest, FaceFixRequest, InpaintForm
from app.selection import main_subjects
from tests.test_references import isolated_data, png


def detected(faces: list[Detection]):
    """Stands in for the detector: these faces, and one person around them all."""
    return lambda _image, query, *_args, **_kwargs: faces if query == FACE_QUERY else [Detection("person", 0.8, (0, 0, 10_000, 10_000))]


class DetectionTests(unittest.TestCase):
    def test_queries_become_lowercase_phrases_with_full_stops(self) -> None:
        self.assertEqual(detection_query("The Hat"), "the hat.")
        self.assertEqual(detection_query("dog, red car; tree."), "dog. red car. tree.")
        with self.assertRaises(ValueError):
            detection_query(" . , ")

    def test_strongest_drops_weak_matches_and_duplicate_boxes(self) -> None:
        found = [Detection("hat", 0.8, (0, 0, 10, 10)), Detection("hat", 0.75, (1, 1, 10, 10)),
                 Detection("hat", 0.6, (50, 50, 60, 60)), Detection("hat", 0.3, (80, 80, 90, 90))]
        self.assertEqual([item.score for item in strongest(found)], [0.8, 0.6])
        self.assertEqual(strongest([]), [])
        self.assertAlmostEqual(iou((0, 0, 10, 10), (5, 0, 15, 10)), 1 / 3)
        self.assertAlmostEqual(coverage((0, 0, 10, 10), (5, 0, 30, 30)), 1 / 2)

    def test_main_subjects_keep_a_group_but_not_small_background_people(self) -> None:
        found = [Detection("person", 0.8, (0, 0, 500, 1000)), Detection("animal", 0.6, (400, 300, 700, 800)),
                 Detection("person", 0.7, (900, 300, 980, 450)), Detection("person", 0.3, (0, 0, 1000, 1000))]
        self.assertEqual([item.label for item in main_subjects(found)], ["person", "animal"])
        self.assertEqual(main_subjects([Detection("person", 0.2, (0, 0, 9, 9))]), [])


class MattingTests(unittest.TestCase):
    def test_trimap_marks_a_band_of_unknown_along_the_edge(self) -> None:
        mask = np.zeros((40, 40), dtype=np.uint8)
        mask[10:30, 10:30] = 255
        result = trimap(mask, 3)
        self.assertEqual(result[20, 20], 255)
        self.assertEqual(result[0, 0], 0)
        self.assertEqual(result[10, 20], UNKNOWN)
        self.assertEqual(result[8, 20], UNKNOWN)
        self.assertEqual(result[5, 20], 0)

    def test_edge_band_and_box_follow_the_subject(self) -> None:
        self.assertEqual(edge_radius(100, 50), 3)
        self.assertEqual(edge_radius(1000, 400), 15)
        mask = np.zeros((20, 30), dtype=np.uint8)
        mask[5:8, 10:12] = 255
        self.assertEqual(mask_box(mask, 2), (8, 3, 14, 10))
        self.assertIsNone(mask_box(np.zeros((4, 4), dtype=np.uint8), 2))


class FaceGeometryTests(unittest.TestCase):
    def test_crop_is_a_square_around_the_face_inside_the_picture(self) -> None:
        crop = face_crop_box((100, 100, 140, 150), 400, 300)
        self.assertEqual(crop[2] - crop[0], crop[3] - crop[1])
        self.assertEqual(crop[2] - crop[0], 110)
        self.assertLessEqual(crop[0], 100)
        edge = face_crop_box((0, 0, 60, 60), 100, 80)
        self.assertEqual(edge, (0, 0, 80, 80))

    def test_blend_mask_covers_the_face_and_fades_out(self) -> None:
        face, crop = (40, 40, 80, 90), (0, 0, 120, 120)
        mask = face_blend_mask(face, crop)
        self.assertEqual(mask.size, (120, 120))
        self.assertEqual(mask.getpixel((60, 65)), 255)
        self.assertEqual(mask.getpixel((2, 2)), 0)

    def test_colours_are_shifted_back_to_the_original_average(self) -> None:
        restored = matched_colours(Image.new("RGB", (8, 8), (120, 100, 80)), Image.new("RGB", (8, 8), (100, 100, 100)))
        self.assertEqual(restored.getpixel((3, 3)), (100, 100, 100))

    def test_selection_and_size_pick_the_faces_to_fix(self) -> None:
        faces = [Detection("face", 0.6, (10, 10, 40, 40)), Detection("face", 0.5, (60, 10, 100, 50)), Detection("face", 0.5, (0, 60, 8, 68))]
        people = [Detection("person", 0.8, (0, 0, 120, 100))]
        self.assertEqual([item.box for item in faces_to_fix(faces, people, None)], [(60, 10, 100, 50), (10, 10, 40, 40)])
        selection = np.zeros((100, 120), dtype=np.uint8)
        selection[:, :50] = 255
        self.assertEqual([item.box for item in faces_to_fix(faces, people, selection)], [(10, 10, 40, 40)])

    def test_only_faces_inside_a_bigger_person_are_fixed(self) -> None:
        # Measured on a street picture: a pastry on a sign among cherry blossoms
        # scored 0.33 as a face, no more than small blurry real faces do.
        woman, pastry = Detection("human face", 0.53, (673, 303, 704, 346)), Detection("human face", 0.33, (486, 193, 532, 233))
        blurry = Detection("human face", 0.32, (557, 674, 578, 698))
        people = [Detection("person", 0.74, (629, 292, 740, 704)), Detection("person", 0.76, (478, 377, 542, 541)),
                  Detection("person", 0.44, (515, 657, 600, 824))]
        self.assertEqual(faces_to_fix([woman, pastry, blurry], people, None), [woman, blurry])
        self.assertEqual(faces_to_fix([woman], [], None), [])
        # Without people, one object tends to come back as both face and person.
        ball = Detection("human face", 0.51, (400, 300, 600, 500))
        self.assertEqual(faces_to_fix([ball], [Detection("person", 0.31, (398, 301, 601, 502))], None), [])

    def test_candidates_show_every_face_and_suggest_the_real_ones(self) -> None:
        picture = Image.new("RGB", (100, 80))
        real, cake, tiny = (Detection("human face", 0.5, (10.4, -2.0, 40.6, 30.0)), Detection("human face", 0.34, (60, 40, 100.8, 80)),
                            Detection("human face", 0.4, (50, 5, 60, 15)))
        people = [Detection("person", 0.8, (0, 0, 50, 80)), Detection("person", 0.3, (60, 40, 100, 80))]
        detect = lambda _image, query, *_args, **_kwargs: [real, cake, tiny] if query == FACE_QUERY else people
        with patch("app.faces.detector.detect", side_effect=detect):
            candidates = face_candidates(picture)
        self.assertEqual([(item.box, item.suggested) for item in candidates], [((60, 40, 100, 80), False), ((10, 0, 41, 30), True)])

    def test_picked_faces_stay_inside_the_picture(self) -> None:
        self.assertEqual([item.box for item in picked_faces([(10, 10, 30, 30), (90, 0, 140, 60), (120, 0, 130, 10)], 100, 50)],
                         [(90, 0, 100, 50), (10, 10, 30, 30)])
        for faces in ([], [(0, 0, 10, 10)] * 9, [(10, 0, 10, 10)]):
            with self.assertRaises(ValueError):
                FaceFixRequest(faces=faces)


class FaceFixServiceTests(unittest.TestCase):
    def test_each_face_is_restored_and_pasted_back_with_one_shared_mask(self) -> None:
        with isolated_data():
            provider = MagicMock()

            def fake_edit(request, input_path, output_path, _control, **kwargs):
                self.assertEqual(request.mode, "face")
                with Image.open(input_path) as crop:
                    Image.new("RGB", crop.size, (250, 250, 250)).save(output_path)
                return 11, 0

            provider.edit.side_effect = fake_edit
            picture = Image.new("RGB", (200, 120), (40, 40, 40))
            faces = [Detection("human face", 0.7, (30, 30, 60, 70)), Detection("human face", 0.6, (130, 30, 160, 70))]
            with patch("app.faces.registry.activate", return_value=provider), patch("app.faces.detector.detect", side_effect=detected(faces)):
                ids = run_face_fix(FaceFixRequest(count=2), png(picture), "photo.png", None, MagicMock())
                self.assertEqual(provider.edit.call_count, 4)
                first, second = (storage.get_generation(item) for item in ids)
                self.assertEqual((first["task"], first["edit_operation"], first["prompt"]), ("edit", "faces", "Fix faces"))
                self.assertEqual(first["mask_image_path"], second["mask_image_path"])
                with Image.open(settings.data_dir / first["output_image_path"]) as output:
                    # Colours are matched, so the flat crop keeps its tone; outside the faces nothing changes.
                    self.assertEqual(output.getpixel((45, 50)), (40, 40, 40))
                    self.assertEqual(output.getpixel((100, 110)), (40, 40, 40))
                selection = Image.new("L", picture.size, 0)
                selection.paste(255, (0, 0, 100, 120))
                provider.edit.reset_mock()
                run_face_fix(FaceFixRequest(), png(picture), "photo.png", png(selection), MagicMock())
                self.assertEqual(provider.edit.call_count, 1)

    def test_faithful_is_a_partial_redraw_with_the_detail_lora_on_quality_only(self) -> None:
        with isolated_data():
            requests = []

            def fake_edit(request, input_path, output_path, _control, **kwargs):
                requests.append(request)
                with Image.open(input_path) as crop:
                    crop.save(output_path)
                return 5, 0

            provider = MagicMock()
            provider.edit.side_effect = fake_edit
            picture = png(Image.new("RGB", (120, 120), (90, 60, 40)))
            faces = [Detection("human face", 0.7, (40, 40, 80, 80))]
            with patch("app.faces.registry.activate", return_value=provider), patch("app.faces.detector.detect", side_effect=detected(faces)):
                [faithful] = run_face_fix(FaceFixRequest(), picture, "photo.png", None, MagicMock())
                run_face_fix(FaceFixRequest(faithful=False), picture, "photo.png", None, MagicMock())
                run_face_fix(FaceFixRequest(model_name="flux2-klein-4b-nvfp4"), picture, "photo.png", None, MagicMock())
            self.assertEqual([(r.steps, r.strength, r.detail) for r in requests], [(8, .625, .4), (4, 1, 0), (8, .625, 0)])
            record = storage.get_generation(faithful)
            self.assertEqual((record["steps"], record["strength"]), (8, .625))

    def test_picked_faces_replace_detection_and_the_selection(self) -> None:
        with isolated_data():
            provider = MagicMock()

            def fake_edit(request, input_path, output_path, _control, **kwargs):
                with Image.open(input_path) as crop:
                    crop.save(output_path)
                return 3, 0

            provider.edit.side_effect = fake_edit
            picture = png(Image.new("RGB", (200, 120), (40, 40, 40)))
            selection = png(Image.new("L", (200, 120), 0))
            with patch("app.faces.registry.activate", return_value=provider), patch("app.faces.detector.detect") as detect:
                run_face_fix(FaceFixRequest(faces=[(30, 30, 60, 70), (130, 30, 160, 70)]), picture, "photo.png", selection, MagicMock())
            detect.assert_not_called()
            self.assertEqual(provider.edit.call_count, 2)

    def test_no_faces_is_an_error_that_leaves_no_files(self) -> None:
        with isolated_data():
            with patch("app.faces.registry.activate", return_value=MagicMock()), patch("app.faces.detector.detect", return_value=[]):
                with self.assertRaisesRegex(ValueError, "No faces found"):
                    run_face_fix(FaceFixRequest(), png(Image.new("RGB", (64, 64))), "photo.png", None, MagicMock())
            self.assertEqual(list(settings.input_dir.iterdir()), [])
            self.assertEqual(list(settings.mask_dir.iterdir()), [])


class SketchAndFacePromptTests(unittest.TestCase):
    def setUp(self) -> None:
        self.provider = Flux2KleinProvider()

    def test_prompts_describe_the_sketch_and_the_face(self) -> None:
        self.assertIn("rough coloured sketch in the masked area into a deer.", _inpaint_prompt(InpaintForm(prompt="a deer", edit_operation="sketch")))
        self.assertIn("finished, detailed image: a cottage.", _sketch_prompt("a cottage"))
        self.assertNotIn("Also:", _face_prompt(""))
        self.assertIn("Also: smiling.", _face_prompt("smiling"))
        self.assertIn("Anna is the person in image 2", _face_prompt("", ["Anna"]))
        self.assertIn("This person is the person in image 2", _face_prompt("", ["Reference 1"]))
        # Naming teeth turned neutral and closed-mouth faces into grins.
        self.assertNotIn("teeth", _face_prompt(""))

    def test_detail_and_enhance_prompts_lead_with_the_lora_trigger(self) -> None:
        face = EditRequest(prompt="x", mode="face", detail=.4)
        self.assertTrue(_mode_prompt(face, []).startswith(ENHANCE_TRIGGER + " Restore image 1"))
        self.assertTrue(_mode_prompt(face.model_copy(update={"detail": 0}), []).startswith("Restore image 1"))
        enhance = EditRequest(prompt="sharper bark", mode="enhance", detail=.4)
        self.assertTrue(_mode_prompt(enhance, []).startswith(ENHANCE_TRIGGER))
        self.assertTrue(_mode_prompt(enhance, []).endswith(" Also: sharper bark."))
        self.assertNotIn("Also:", _mode_prompt(enhance.model_copy(update={"prompt": " "}), []))

    def test_partial_redraw_noises_the_picture_and_keeps_references_after_it(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            Image.new("RGB", (120, 120), "tan").save(root / "face.png")
            Image.new("RGB", (40, 40), "orange").save(root / "anna.png")
            pipe = MagicMock(return_value=SimpleNamespace(images=[Image.new("RGB", (768, 768))]))
            with patch("app.models.flux2_klein.require_generation_device"), patch("app.models.flux2_klein.selected_device", return_value="cuda"),                     patch.object(self.provider, "_load_torch", return_value=MagicMock()), patch.object(self.provider, "_load_pipe", return_value=pipe) as load:
                from app.models.base import ModelReference
                anna = [ModelReference("Anna", root / "anna.png")]
                self.provider.edit(EditRequest(prompt="x", mode="face", steps=8, strength=.625), root / "face.png", root / "out.png", MagicMock(), references=anna)
                self.assertEqual(load.call_args.args[1], "inpaint")
                arguments = pipe.call_args.kwargs
                self.assertEqual((arguments["strength"], arguments["num_inference_steps"]), (.625, 8))
                self.assertEqual(arguments["mask_image"].getextrema(), (255, 255))
                self.assertEqual(arguments["mask_image"].size, arguments["image"].size)
                self.assertEqual(arguments["image_reference"].size, (40, 40))
                self.assertIn("Anna is the person in image 2", pipe.encode_prompt.call_args.kwargs["prompt"])
                # Enhance looks at the picture only.
                self.provider.edit(EditRequest(prompt="x", mode="enhance", strength=.625), root / "face.png", root / "out.png", MagicMock(), references=anna)
                self.assertNotIn("image_reference", pipe.call_args.kwargs)
                self.provider.edit(EditRequest(prompt="x", mode="face"), root / "face.png", root / "out.png", MagicMock())
                self.assertEqual(load.call_args.args[1], "text-to-image")
                self.assertNotIn("strength", pipe.call_args.kwargs)

    def test_sketch_inpaint_uses_its_own_crop_as_reference(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            Image.new("RGB", (64, 64), "purple").save(root / "source.png")
            Image.new("L", (64, 64), 255).save(root / "mask.png")
            pipe = MagicMock(return_value=SimpleNamespace(images=[Image.new("RGB", (64, 64))]))
            with patch("app.models.flux2_klein.require_generation_device"), patch("app.models.flux2_klein.selected_device", return_value="cuda"), patch.object(self.provider, "_load_torch", return_value=MagicMock()), patch.object(self.provider, "_load_pipe", return_value=pipe):
                self.provider.inpaint(InpaintForm(prompt="a deer", edit_operation="sketch"), root / "source.png", root / "mask.png", root / "out.png", MagicMock())
                arguments = pipe.call_args.kwargs
                self.assertIs(arguments["image_reference"], arguments["image"])
                self.assertIn("pose and colours", pipe.encode_prompt.call_args.kwargs["prompt"])

    def test_sketch_inpaint_puts_references_after_the_sketch_at_one_size(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            Image.new("RGB", (96, 64), "purple").save(root / "source.png")
            Image.new("L", (96, 64), 255).save(root / "mask.png")
            Image.new("RGB", (40, 60), "orange").save(root / "anna.png")
            pipe = MagicMock(return_value=SimpleNamespace(images=[Image.new("RGB", (96, 64))]))
            with patch("app.models.flux2_klein.require_generation_device"), patch("app.models.flux2_klein.selected_device", return_value="cuda"), patch.object(self.provider, "_load_torch", return_value=MagicMock()), patch.object(self.provider, "_load_pipe", return_value=pipe):
                from app.models.base import ModelReference
                self.provider.inpaint(InpaintForm(prompt="Anna on the bench", edit_operation="sketch"), root / "source.png", root / "mask.png", root / "out.png", MagicMock(),
                                      references=[ModelReference("Anna", root / "anna.png")])
                arguments = pipe.call_args.kwargs
                sketch, anna = arguments["image_reference"]
                # The pipeline crops later references to the first one's size.
                self.assertEqual(sketch.size, anna.size)
                self.assertEqual(sketch.getpixel((sketch.width // 2, sketch.height // 2)), (128, 0, 128))
                prompt = pipe.encode_prompt.call_args.kwargs["prompt"]
                self.assertIn("Reference image 1 shows the sketch. Follow its shape, position, size and pose", prompt)
                self.assertIn("Anna is the person or object from reference image 2", prompt)
                self.assertIn("The sketch only marks where Anna goes and in what pose: draw Anna from reference image 2 there,"
                              " with Anna's own face, hair, clothing and colours instead of the sketch's flat colours.", prompt)
                self.assertNotIn("pose and colours", prompt)
                # Removal still has nothing to place.
                self.provider.inpaint(InpaintForm(prompt="x", edit_operation="remove"), root / "source.png", root / "mask.png", root / "out.png", MagicMock(),
                                      references=[ModelReference("Anna", root / "anna.png")])
                self.assertNotIn("image_reference", pipe.call_args.kwargs)

    def test_whole_picture_sketch_numbers_references_after_the_picture(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            Image.new("RGB", (120, 120), "tan").save(root / "picture.png")
            Image.new("RGB", (40, 60), "orange").save(root / "anna.png")
            pipe = MagicMock(return_value=SimpleNamespace(images=[Image.new("RGB", (1024, 1024))]))
            with patch("app.models.flux2_klein.require_generation_device"), patch("app.models.flux2_klein.selected_device", return_value="cuda"), patch.object(self.provider, "_load_torch", return_value=MagicMock()), patch.object(self.provider, "_load_pipe", return_value=pipe):
                from app.models.base import ModelReference
                self.provider.edit(EditRequest(prompt="Anna on the bench", mode="sketch"), root / "picture.png", root / "out.png", MagicMock(),
                                   references=[ModelReference("Anna", root / "anna.png", "subject")])
                picture, anna = pipe.call_args.kwargs["image"]
                self.assertEqual(anna.size, (40, 60))
                prompt = pipe.encode_prompt.call_args.kwargs["prompt"]
                self.assertTrue(prompt.startswith("Image 1 contains a rough coloured sketch"))
                self.assertIn("Follow the sketch's layout, shapes, sizes and poses closely.", prompt)
                self.assertIn("Anna is the person or object from image 2", prompt)
                self.assertTrue(prompt.endswith("Keep every part of image 1 that is not sketched as it is."))

    def test_sketch_prompts_leave_the_pose_to_a_pose_reference(self) -> None:
        from app.models.base import ModelReference
        references = [ModelReference("Anna", Path("anna.png")), ModelReference("Dancer", Path("dancer.png"), "pose")]
        whole = _sketch_prompt("Anna dancing", references)
        self.assertIn("Follow the sketch's layout, shapes and sizes closely.", whole)
        self.assertIn("follow the body pose, posture, gesture and framing from image 3", whole)
        self.assertIn("The sketch only marks where Anna goes: draw Anna from image 2 there", whole)
        crop = _inpaint_prompt(InpaintForm(prompt="Anna dancing", edit_operation="sketch"), references)
        self.assertIn("Follow its shape, position and size,", crop)
        self.assertIn("from reference image 3", crop)

    def test_sketch_prompts_tie_every_subject_to_its_reference(self) -> None:
        from app.models.base import ModelReference
        two = _sketch_prompt("Anna and a friend on the bench", [ModelReference("Anna", Path("a.png")), ModelReference("Reference 2", Path("b.png"))])
        self.assertIn("The sketch only marks where Anna and the subject from image 3 go and in what pose: draw them there,"
                      " each from their own image, with their own face", two)
        one = _sketch_prompt("someone on the bench", [ModelReference("Reference 1", Path("a.png"))])
        self.assertIn("where the subject goes and in what pose: draw the subject from image 2 there, with its own face", one)
        # An outfit has no one to name: its colours simply win over the sketch's.
        outfit = _sketch_prompt("a coat", [ModelReference("Coat", Path("c.png"), "clothing")])
        self.assertIn("use that reference's real colours", outfit)

    def test_face_mode_renders_at_the_smaller_face_area(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            Image.new("RGB", (120, 120), "tan").save(root / "face.png")
            pipe = MagicMock(return_value=SimpleNamespace(images=[Image.new("RGB", (768, 768))]))
            with patch("app.models.flux2_klein.require_generation_device"), patch("app.models.flux2_klein.selected_device", return_value="cuda"), patch.object(self.provider, "_load_torch", return_value=MagicMock()), patch.object(self.provider, "_load_pipe", return_value=pipe):
                self.provider.edit(EditRequest(prompt="x", mode="face").model_copy(update={"prompt": ""}), root / "face.png", root / "out.png", MagicMock())
                arguments = pipe.call_args.kwargs
                self.assertEqual(arguments["width"] * arguments["height"], FACE_AREA)
                self.assertTrue(pipe.encode_prompt.call_args.kwargs["prompt"].startswith("Restore image 1"))
                with Image.open(root / "out.png") as output:
                    self.assertEqual(output.size, (120, 120))
                self.provider.edit(EditRequest(prompt="a cottage", mode="sketch"), root / "face.png", root / "out.png", MagicMock())
                self.assertTrue(pipe.encode_prompt.call_args.kwargs["prompt"].startswith("Image 1 contains a rough coloured sketch"))


class SmartToolApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(app)
        self.picture = ("picture.png", png(Image.new("RGB", (64, 48))), "image/png")

    def queued(self, kind: str):
        return patch("app.main.jobs.submit", return_value=SimpleNamespace(id="job", kind=kind, status="queued"))

    def test_text_selection_and_cutout_queue_selection_jobs(self) -> None:
        with self.queued("select") as submit:
            response = self.client.post("/api/select-text", files={"image": self.picture}, data={"parameters": json.dumps({"text": "the hat"})})
            self.assertEqual(response.status_code, 200)
            self.assertEqual(submit.call_args.args[0], "select")
            self.assertEqual(self.client.post("/api/cutout", files={"image": self.picture}).status_code, 200)
        with patch("app.main.jobs.submit") as submit:
            self.assertEqual(self.client.post("/api/select-text", files={"image": self.picture}, data={"parameters": json.dumps({"text": ""})}).status_code, 422)
            submit.assert_not_called()

    def test_fix_faces_needs_klein_and_takes_an_optional_selection(self) -> None:
        with patch("app.main.jobs.submit") as submit:
            parameters = json.dumps({"model_name": "not-a-model"})
            self.assertEqual(self.client.post("/api/faces", files={"image": self.picture}, data={"parameters": parameters}).status_code, 422)
            submit.assert_not_called()
        with self.queued("edit") as submit:
            mask = ("mask.png", png(Image.new("L", (64, 48), 255)), "image/png")
            response = self.client.post("/api/faces", files={"image": self.picture, "mask": mask}, data={"parameters": "{}"})
            self.assertEqual(response.status_code, 200)
            self.assertEqual(submit.call_args.args[0], "edit")
            picked = json.dumps({"faces": [[4, 4, 30, 40]]})
            self.assertEqual(self.client.post("/api/faces", files={"image": self.picture}, data={"parameters": picked}).status_code, 200)
            flat = json.dumps({"faces": [[4, 4, 30, 4]]})
            self.assertEqual(self.client.post("/api/faces", files={"image": self.picture}, data={"parameters": flat}).status_code, 422)

    def test_finding_faces_answers_with_the_candidates(self) -> None:
        faces = [Detection("human face", 0.61, (10, 8, 30, 30))]
        with patch("app.faces.detector.detect", side_effect=detected(faces)):
            response = self.client.post("/api/faces/detect", files={"image": self.picture})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"faces": [{"box": [10, 8, 30, 30], "score": 0.61, "suggested": True}]})
        broken = ("picture.png", b"not an image", "image/png")
        self.assertEqual(self.client.post("/api/faces/detect", files={"image": broken}).status_code, 422)

    def test_sketches_need_a_known_model(self) -> None:
        with patch("app.main.jobs.submit") as submit:
            mask = ("mask.png", png(Image.new("L", (64, 48), 255)), "image/png")
            data = {"prompt": "a deer", "edit_operation": "sketch", "model_name": "not-a-model"}
            self.assertEqual(self.client.post("/api/inpaint", files={"image": self.picture, "mask": mask}, data=data).status_code, 422)
            submit.assert_not_called()


if __name__ == "__main__":
    unittest.main()
