from __future__ import annotations

from hashlib import sha256
from io import BytesIO
from threading import Lock
from typing import TYPE_CHECKING
from uuid import uuid4

import numpy as np
from PIL import Image

from . import storage
from .config import settings
from .detection import Detection, detector, strongest
from .jobs import JobControl, SelectionResult
from .matting import matter
from .schemas import CutoutRequest, SelectionRequest, TextSelectionRequest

if TYPE_CHECKING:
    from torch import Tensor
    from transformers import Sam2Model, Sam2Processor


# People and animals are what most cut-outs are about; anything else falls back
# to the centred-subject guess or a typed name.
SUBJECT_QUERY = "person. animal."
SUBJECT_MIN_SCORE = 0.35


def main_subjects(detections: list[Detection]) -> list[Detection]:
    """The biggest confident person or animal, plus others of comparable size (a group photo)."""
    candidates = strongest([item for item in detections if item.score >= SUBJECT_MIN_SCORE], relative=0)
    if not candidates:
        return []
    main = max(candidates, key=lambda item: item.area * item.score)
    return [item for item in candidates if item is main or item.area >= main.area * 0.25]


class ObjectSelector:
    """One CPU model and one image embedding; GPU generation stays resident."""

    model_id = "facebook/sam2.1-hiera-tiny"

    def __init__(self) -> None:
        self._model: Sam2Model | None = None
        self._processor: Sam2Processor | None = None
        self._image_key: str | None = None
        self._embeddings: list[Tensor] | None = None
        # Selection jobs and reference uploads share one model and embedding cache.
        self._lock = Lock()

    def select(self, content: bytes, request: SelectionRequest, control: JobControl) -> SelectionResult:
        with Image.open(BytesIO(content)) as uploaded:
            image = uploaded.convert("RGB")
        if image.width * image.height > 16_777_216:
            raise ValueError("Object selection supports images up to 16 megapixels")
        if any(point.x >= image.width or point.y >= image.height for point in request.points):
            raise ValueError("Selection points must lie inside the image")
        control.raise_if_cancelled()
        mask_pixels = self._predict(
            image,
            sha256(content).hexdigest(),
            [(point.x, point.y) for point in request.points],
            [point.label for point in request.points],
            # One click is ambiguous (part, object, group); pick the best of
            # three. Refined clicks are not, and one mask follows them better.
            multimask=len(request.points) == 1,
            control=control,
        )
        control.set_progress(0.95)
        if not mask_pixels.any():
            raise ValueError("No object found. Try another point or use the brush")
        return self._save(mask_pixels, control)

    def select_text(self, content: bytes, request: TextSelectionRequest, control: JobControl) -> SelectionResult:
        """Selects everything that matches a short description, such as "the hat"."""
        image = self._open(content)
        control.set_progress(0.05)
        found = strongest(detector.detect(image, request.text, control, activity=f"Looking for “{request.text.strip()}”"))
        if not found:
            raise ValueError(f"Nothing found for “{request.text.strip()}”. Try other English words, or click the object")
        control.set_progress(0.5)
        mask_pixels = self._predict_boxes(image, sha256(content).hexdigest(), [item.box for item in found], control)
        control.set_progress(0.95)
        if not mask_pixels.any():
            raise ValueError("No object found. Try other words or click the object")
        return self._save(mask_pixels, control)

    def cutout(self, content: bytes, request: CutoutRequest, control: JobControl) -> SelectionResult:
        """The main subject, or what the text names, with soft edges for hair and fur."""
        image = self._open(content)
        key = sha256(content).hexdigest()
        control.set_progress(0.05)
        text = request.text.strip()
        if text:
            found = strongest(detector.detect(image, text, control, activity=f"Looking for “{text}”"))
        else:
            found = main_subjects(detector.detect(image, SUBJECT_QUERY, control, activity="Finding the subject"))
        control.set_progress(0.35)
        if found:
            mask_pixels = self._predict_boxes(image, key, [item.box for item in found], control)
        elif text:
            raise ValueError(f"Nothing found for “{text}”. Try other English words")
        else:
            subject = self.subject_mask(image)
            if subject is None:
                raise ValueError("No clear subject found. Type what to keep, such as “the car”")
            mask_pixels = subject
        if not mask_pixels.any():
            raise ValueError("No clear subject found. Type what to keep, such as “the car”")
        control.set_progress(0.6)
        control.set_phase("finishing", "Refining edges")
        mask_pixels = matter.refine(image, mask_pixels, control)
        control.set_progress(0.95)
        return self._save(mask_pixels, control)

    @staticmethod
    def _open(content: bytes) -> Image.Image:
        with Image.open(BytesIO(content)) as uploaded:
            image = uploaded.convert("RGB")
        if image.width * image.height > 16_777_216:
            raise ValueError("Object selection supports images up to 16 megapixels")
        return image

    @staticmethod
    def _save(mask_pixels: np.ndarray, control: JobControl) -> SelectionResult:
        path = settings.mask_dir / f"selection-{uuid4().hex}.png"
        try:
            Image.fromarray(mask_pixels, mode="L").save(path)
            control.raise_if_cancelled()
        except BaseException:
            path.unlink(missing_ok=True)
            raise
        return SelectionResult(mask_image_url=storage.image_url(storage.to_relative(path)))

    def warm_up(self) -> None:
        """Load SAM ahead of the first click; it downloads on first use."""
        with self._lock:
            self._ensure_model()

    def subject_mask(self, image: Image.Image) -> np.ndarray | None:
        """Mask of the main subject of a concept image, or None when unclear.

        Concept images usually centre their subject, so three points on the
        vertical centre line (head, body, legs for a person) prompt one mask.
        """
        rgb = image.convert("RGB")
        width, height = rgb.size
        points = [(width * 0.5, height * fraction) for fraction in (0.3, 0.5, 0.7)]
        mask = self._predict(rgb, sha256(rgb.tobytes()).hexdigest(), points, [1, 1, 1], multimask=False)
        coverage = float((mask > 0).mean())
        # Nearly empty or nearly full masks mean SAM found no distinct subject.
        return mask if 0.03 <= coverage <= 0.85 else None

    def _predict_boxes(self, image: Image.Image, key: str, boxes: list[tuple[float, float, float, float]], control: JobControl | None = None) -> np.ndarray:
        """One mask per box, joined: every matching object ends up selected."""
        import torch

        with self._lock:
            if self._model is None and control:
                control.set_phase("loading", "SAM 2.1 Tiny")
            self._ensure_model()
            processor, model = self._processor, self._model
            if processor is None or model is None:
                raise RuntimeError("Selection processor was not loaded")
            inputs = processor(images=image, input_boxes=[[list(box) for box in boxes]], return_tensors="pt")
            with torch.inference_mode():
                if key != self._image_key or self._embeddings is None:
                    self._embeddings = model.get_image_embeddings(inputs["pixel_values"])
                    self._image_key = key
                outputs = model(image_embeddings=self._embeddings, input_boxes=inputs["input_boxes"], multimask_output=False)
                masks = processor.post_process_masks(outputs.pred_masks.cpu(), inputs["original_sizes"])[0]
        return (masks[:, 0].any(dim=0).numpy().astype("uint8")) * 255

    def _ensure_model(self) -> None:
        if self._model is None:
            from transformers import Sam2Model, Sam2Processor

            self._processor = Sam2Processor.from_pretrained(self.model_id, token=settings.huggingface_token)
            self._model = Sam2Model.from_pretrained(self.model_id, token=settings.huggingface_token, use_safetensors=True, trust_remote_code=False).eval()

    def _predict(
        self,
        image: Image.Image,
        key: str,
        points: list[tuple[float, float]],
        labels: list[int],
        *,
        multimask: bool,
        control: JobControl | None = None,
    ) -> np.ndarray:
        import torch

        with self._lock:
            if self._model is None and control:
                control.set_phase("loading", "SAM 2.1 Tiny")
            self._ensure_model()
            processor, model = self._processor, self._model
            if processor is None:
                raise RuntimeError("Selection processor was not loaded")
            if control:
                control.set_progress(0.2)
            inputs = processor(
                images=image,
                input_points=[[[[x, y] for x, y in points]]],
                input_labels=[[labels]],
                return_tensors="pt",
            )
            with torch.inference_mode():
                if key != self._image_key or self._embeddings is None:
                    self._embeddings = model.get_image_embeddings(inputs["pixel_values"])
                    self._image_key = key
                if control:
                    control.set_progress(0.65)
                outputs = model(
                    image_embeddings=self._embeddings,
                    input_points=inputs["input_points"],
                    input_labels=inputs["input_labels"],
                    multimask_output=multimask,
                )
                masks = processor.post_process_masks(outputs.pred_masks.cpu(), inputs["original_sizes"])[0]
                best = int(outputs.iou_scores[0, 0].argmax())
                return masks[0, best].numpy().astype("uint8") * 255


selector = ObjectSelector()
