from __future__ import annotations

from threading import Lock
from typing import TYPE_CHECKING

import numpy as np
from PIL import Image, ImageFilter

from .config import settings
from .jobs import JobControl

if TYPE_CHECKING:
    from transformers import VitMatteForImageMatting, VitMatteImageProcessor

# ViTMatte runs on the CPU; this keeps one cut-out to a few seconds.
MATTE_MAX_SIDE = 1024
UNKNOWN = 128


def trimap(mask: np.ndarray, radius: int) -> np.ndarray:
    """Certain foreground (255), certain background (0) and a band of unknown (128) along the edge."""
    binary = Image.fromarray(np.where(mask > 127, 255, 0).astype(np.uint8), mode="L")
    size = 2 * max(1, radius) + 1
    inner = np.asarray(binary.filter(ImageFilter.MinFilter(size)))
    outer = np.asarray(binary.filter(ImageFilter.MaxFilter(size)))
    result = np.full(mask.shape, UNKNOWN, dtype=np.uint8)
    result[outer == 0] = 0
    result[inner == 255] = 255
    return result


def edge_radius(width: int, height: int) -> int:
    """About 1.5 % of the subject's size: wide enough for loose hair and fur."""
    return max(3, round(max(width, height) * 0.015))


def mask_box(mask: np.ndarray, margin: int) -> tuple[int, int, int, int] | None:
    rows, columns = np.nonzero(mask > 127)
    if not len(rows):
        return None
    height, width = mask.shape
    return (max(0, int(columns.min()) - margin), max(0, int(rows.min()) - margin),
            min(width, int(columns.max()) + 1 + margin), min(height, int(rows.max()) + 1 + margin))


class Matter:
    """Turns a hard object mask into soft alpha, so hair and fur keep their strands."""

    model_id = "hustvl/vitmatte-small-composition-1k"

    def __init__(self) -> None:
        self._model: VitMatteForImageMatting | None = None
        self._processor: VitMatteImageProcessor | None = None
        self._lock = Lock()

    def refine(self, image: Image.Image, mask: np.ndarray, control: JobControl | None = None) -> np.ndarray:
        import torch

        box = mask_box(mask, 0)
        if box is None:
            return mask
        radius = edge_radius(box[2] - box[0], box[3] - box[1])
        box = mask_box(mask, radius * 3)
        assert box is not None
        left, top, right, bottom = box
        crop = image.convert("RGB").crop(box)
        crop_map = trimap(mask[top:bottom, left:right], radius)
        scale = min(1.0, MATTE_MAX_SIDE / max(crop.size))
        work_size = (max(1, round(crop.width * scale)), max(1, round(crop.height * scale)))
        work_image = crop.resize(work_size, Image.Resampling.LANCZOS) if scale < 1 else crop
        work_map = Image.fromarray(crop_map, mode="L")
        if scale < 1:
            work_map = work_map.resize(work_size, Image.Resampling.NEAREST)
        with self._lock:
            if self._model is None and control:
                control.set_phase("loading", "ViTMatte")
            self._ensure_model()
            processor, model = self._processor, self._model
            if processor is None or model is None:
                raise RuntimeError("Matting model was not loaded")
            inputs = processor(images=work_image, trimaps=work_map, return_tensors="pt")
            with torch.inference_mode():
                alphas = model(**inputs).alphas
        predicted = (alphas[0, 0, :work_size[1], :work_size[0]].clamp(0, 1).numpy() * 255).astype(np.uint8)
        alpha = Image.fromarray(predicted, mode="L")
        if scale < 1:
            alpha = alpha.resize(crop.size, Image.Resampling.BILINEAR)
        # The model only decides the unknown band; certain areas stay exact.
        soft = np.where(crop_map == UNKNOWN, np.asarray(alpha), crop_map)
        result = np.zeros_like(mask, dtype=np.uint8)
        result[top:bottom, left:right] = soft
        return result

    def _ensure_model(self) -> None:
        if self._model is None:
            from transformers import VitMatteForImageMatting, VitMatteImageProcessor

            self._processor = VitMatteImageProcessor.from_pretrained(self.model_id, token=settings.huggingface_token)
            self._model = VitMatteForImageMatting.from_pretrained(
                self.model_id, token=settings.huggingface_token, use_safetensors=True, trust_remote_code=False,
            ).eval()


matter = Matter()
