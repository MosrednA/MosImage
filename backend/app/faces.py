from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from . import storage
from .config import settings
from .detection import Box, Detection, detector, strongest
from .jobs import JobControl
from .models.registry import registry
from .schemas import EditRequest, FaceFixRequest
from .services import _alternative_seed, _model_references, _new_image_path, _save_bytes

FACE_QUERY = "human face."
MIN_FACE_SIDE = 16
MAX_FACES = 8
# The crop shows hair, chin and some background, so the face sits in context.
CROP_FACTOR = 2.2


def face_crop_box(face: Box, width: int, height: int) -> tuple[int, int, int, int]:
    """A square around the face, slightly above its centre for the hair, kept inside the picture."""
    left, top, right, bottom = face
    side = min(round(max(right - left, bottom - top) * CROP_FACTOR), width, height)
    centre_x = (left + right) / 2
    centre_y = (top + bottom) / 2 - (bottom - top) * 0.08
    x = min(max(0, round(centre_x - side / 2)), width - side)
    y = min(max(0, round(centre_y - side / 2)), height - side)
    return x, y, x + side, y + side


def face_blend_mask(face: Box, crop: tuple[int, int, int, int]) -> Image.Image:
    """A feathered ellipse over the face, in crop coordinates; hair and background stay original."""
    left, top, right, bottom = face
    width, height = crop[2] - crop[0], crop[3] - crop[1]
    centre_x, centre_y = (left + right) / 2 - crop[0], (top + bottom) / 2 - crop[1]
    radius_x, radius_y = (right - left) * 0.62, (bottom - top) * 0.68
    feather = max(2.0, max(width, height) * 0.05)
    mask = Image.new("L", (width, height), 0)
    # Shrink by the feather so the blurred edge still fits the ellipse.
    ImageDraw.Draw(mask).ellipse((centre_x - radius_x + feather, centre_y - radius_y + feather,
                                  centre_x + radius_x - feather, centre_y + radius_y - feather), fill=255)
    return mask.filter(ImageFilter.GaussianBlur(feather))


def matched_colours(restored: Image.Image, original: Image.Image) -> Image.Image:
    """Shifts the restored crop's average colour back to the original's, so the face does not change tone."""
    restored_pixels = np.asarray(restored.convert("RGB"), dtype=np.float32)
    original_pixels = np.asarray(original.convert("RGB"), dtype=np.float32)
    shift = original_pixels.reshape(-1, 3).mean(0) - restored_pixels.reshape(-1, 3).mean(0)
    return Image.fromarray(np.clip(restored_pixels + shift, 0, 255).astype(np.uint8), mode="RGB")


def faces_to_fix(detections: list[Detection], selection: np.ndarray | None) -> list[Detection]:
    """Real faces, largest first; with a selection, only those centred inside it."""
    faces = [item for item in strongest(detections, relative=0, limit=32)
             if min(item.box[2] - item.box[0], item.box[3] - item.box[1]) >= MIN_FACE_SIDE]
    if selection is not None:
        height, width = selection.shape
        faces = [item for item in faces
                 if selection[min(height - 1, int(item.centre[1])), min(width - 1, int(item.centre[0]))] > 127]
    return sorted(faces, key=lambda item: item.area, reverse=True)[:MAX_FACES]


def run_face_fix(request: FaceFixRequest, image_bytes: bytes, image_name: str | None, mask_bytes: bytes | None, control: JobControl) -> list[int]:
    """Redraws each face sharper at Klein's resolution and pastes it back at the picture's own."""
    provider = registry.activate(request.model_name)
    input_path = _save_bytes(image_bytes, settings.input_dir, "input", image_name)
    mask_path: Path | None = None
    output_paths: list[Path] = []
    try:
        with Image.open(input_path) as uploaded:
            picture = uploaded.convert("RGB")
        selection = None
        if mask_bytes is not None:
            with Image.open(BytesIO(mask_bytes)) as uploaded_mask:
                # Selections arrive with their coverage in alpha when they have one.
                band = uploaded_mask.getchannel("A") if "A" in uploaded_mask.getbands() else uploaded_mask.convert("L")
                selection = np.asarray(band.resize(picture.size))
        faces = faces_to_fix(detector.detect(picture, FACE_QUERY, control, activity="Finding faces"), selection)
        if not faces:
            raise ValueError("No faces found in the selection" if selection is not None else "No faces found in the picture")
        crops = [face_crop_box(face.box, picture.width, picture.height) for face in faces]
        blend = Image.new("L", picture.size, 0)
        for face, crop in zip(faces, crops):
            blend.paste(255, crop, face_blend_mask(face.box, crop))
        mask_path = _new_image_path(settings.mask_dir, "faces")
        blend.save(mask_path)
        model_references = _model_references(request.reference_ids)
        extra = {"references": model_references} if model_references else {}
        values: list[dict[str, object]] = []
        with TemporaryDirectory(prefix="faces-", dir=settings.data_dir) as temporary:
            work = Path(temporary)
            for index in range(request.count):
                proposal = control.slice(index, request.count)
                seed = _alternative_seed(request.seed, index)
                result = picture.copy()
                used_seed = 0
                for number, (face, crop) in enumerate(zip(faces, crops)):
                    step = proposal.subrange(number / len(faces), (number + 1) / len(faces))
                    original = result.crop(crop)
                    crop_in, crop_out = work / f"face-{number}.png", work / f"face-{number}-out.png"
                    original.save(crop_in)
                    # The face prompt is built in; the user's words only add to it, so they may be empty.
                    face_request = EditRequest(prompt="restore the face", model_name=request.model_name, runtime_mode=request.runtime_mode,
                                               steps=request.steps, guidance=request.guidance, seed=None if seed is None else (seed + number) % (2**32),
                                               reference_ids=request.reference_ids, mode="face").model_copy(update={"prompt": request.prompt})
                    face_seed, _ = provider.edit(face_request, crop_in, crop_out, step, **extra)
                    used_seed = used_seed or face_seed
                    with Image.open(crop_out) as restored:
                        fixed = matched_colours(restored.convert("RGB").resize(original.size, Image.Resampling.LANCZOS), original)
                    result.paste(fixed, crop, face_blend_mask(face.box, crop))
                output_path = _new_image_path(settings.output_dir, "output")
                output_paths.append(output_path)
                result.save(output_path)
                control.raise_if_cancelled()
                values.append({
                    "task": "edit",
                    "prompt": request.prompt or "Fix faces",
                    "negative_prompt": request.negative_prompt,
                    "model_name": request.model_name,
                    "runtime_mode": request.runtime_mode,
                    "seed": used_seed,
                    "steps": request.steps,
                    "guidance": request.guidance,
                    "width": picture.width,
                    "height": picture.height,
                    "strength": None,
                    "edit_operation": "faces",
                    "input_image_path": storage.to_relative(input_path),
                    "mask_image_path": storage.to_relative(mask_path),
                    "output_image_path": storage.to_relative(output_path),
                    "reference_image_paths": json.dumps([storage.to_relative(reference.path) for reference in model_references]) if model_references else None,
                })
        control.raise_if_cancelled()
        return storage.create_generations(values)
    except BaseException:
        provider.unload()
        for path in (input_path, mask_path, *output_paths):
            if path is not None:
                path.unlink(missing_ok=True)
        raise
