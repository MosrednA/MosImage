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
from .jobs import JobControl, SelectionResult
from .schemas import SelectionRequest

if TYPE_CHECKING:
    from torch import Tensor
    from transformers import Sam2Model, Sam2Processor


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
        path = settings.mask_dir / f"selection-{uuid4().hex}.png"
        try:
            Image.fromarray(mask_pixels, mode="L").save(path)
            control.raise_if_cancelled()
        except BaseException:
            path.unlink(missing_ok=True)
            raise
        return SelectionResult(mask_image_url=storage.image_url(storage.to_relative(path)))

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
        from transformers import Sam2Model, Sam2Processor

        with self._lock:
            if self._model is None:
                self._processor = Sam2Processor.from_pretrained(self.model_id, token=settings.huggingface_token)
                self._model = Sam2Model.from_pretrained(self.model_id, token=settings.huggingface_token, use_safetensors=True, trust_remote_code=False).eval()
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
