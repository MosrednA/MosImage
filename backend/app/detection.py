from __future__ import annotations

import re
from dataclasses import dataclass
from threading import Lock
from typing import TYPE_CHECKING, Sequence

from PIL import Image

from .config import settings
from .jobs import JobControl

if TYPE_CHECKING:
    from transformers import GroundingDinoForObjectDetection, GroundingDinoProcessor


Box = tuple[float, float, float, float]


@dataclass(frozen=True)
class Detection:
    label: str
    score: float
    box: Box

    @property
    def area(self) -> float:
        left, top, right, bottom = self.box
        return max(0.0, right - left) * max(0.0, bottom - top)

    @property
    def centre(self) -> tuple[float, float]:
        left, top, right, bottom = self.box
        return (left + right) / 2, (top + bottom) / 2


def detection_query(text: str) -> str:
    """Grounding DINO's format: lowercase phrases, each ending in a full stop."""
    phrases = [phrase.strip(" .") for phrase in re.split(r"[,;.\n]+", text.lower())]
    phrases = [phrase for phrase in phrases if phrase]
    if not phrases:
        raise ValueError("Type what to find, such as “the hat” or “dog”")
    return " ".join(f"{phrase}." for phrase in phrases)


def iou(first: Box, second: Box) -> float:
    left, top = max(first[0], second[0]), max(first[1], second[1])
    right, bottom = min(first[2], second[2]), min(first[3], second[3])
    overlap = max(0.0, right - left) * max(0.0, bottom - top)
    union = Detection("", 0, first).area + Detection("", 0, second).area - overlap
    return overlap / union if union > 0 else 0.0


def coverage(inner: Box, outer: Box) -> float:
    """The share of the inner box that lies inside the outer one."""
    left, top = max(inner[0], outer[0]), max(inner[1], outer[1])
    right, bottom = min(inner[2], outer[2]), min(inner[3], outer[3])
    area = Detection("", 0, inner).area
    return max(0.0, right - left) * max(0.0, bottom - top) / area if area > 0 else 0.0


def strongest(detections: Sequence[Detection], *, relative: float = 0.5, overlap: float = 0.6, limit: int = 12) -> list[Detection]:
    """The confident detections, best first, without near-duplicate boxes.

    Weak matches next to a strong one are usually other things that share a
    word with the query, so anything far below the best score is dropped.
    """
    if not detections:
        return []
    best = max(detection.score for detection in detections)
    kept: list[Detection] = []
    for detection in sorted(detections, key=lambda item: item.score, reverse=True):
        if detection.score < best * relative:
            break
        if all(iou(detection.box, other.box) < overlap for other in kept):
            kept.append(detection)
        if len(kept) == limit:
            break
    return kept


class TextDetector:
    """Finds things by name. Runs on the CPU, so the GPU model stays resident."""

    model_id = "IDEA-Research/grounding-dino-tiny"

    def __init__(self) -> None:
        self._model: GroundingDinoForObjectDetection | None = None
        self._processor: GroundingDinoProcessor | None = None
        self._lock = Lock()

    def detect(self, image: Image.Image, text: str, control: JobControl | None = None, threshold: float = 0.3,
               activity: str = "Finding objects") -> list[Detection]:
        import torch

        query = detection_query(text)
        rgb = image.convert("RGB")
        with self._lock:
            if self._model is None and control:
                control.set_phase("loading", "Grounding DINO")
            self._ensure_model()
            if control:
                control.set_phase("preparing", activity)
            processor, model = self._processor, self._model
            if processor is None or model is None:
                raise RuntimeError("Text detection model was not loaded")
            inputs = processor(images=rgb, text=query, return_tensors="pt")
            with torch.inference_mode():
                outputs = model(**inputs)
            result = processor.post_process_grounded_object_detection(
                outputs, inputs.input_ids, threshold=threshold, text_threshold=0.25, target_sizes=[(rgb.height, rgb.width)],
            )[0]
        labels = result.get("text_labels") or result.get("labels") or []
        detections = []
        for label, score, box in zip(labels, result["scores"], result["boxes"]):
            left, top, right, bottom = (float(value) for value in box)
            left, top = max(0.0, left), max(0.0, top)
            right, bottom = min(float(rgb.width), right), min(float(rgb.height), bottom)
            if right - left >= 2 and bottom - top >= 2:
                detections.append(Detection(str(label), float(score), (left, top, right, bottom)))
        return detections

    def _ensure_model(self) -> None:
        if self._model is None:
            from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

            self._processor = AutoProcessor.from_pretrained(self.model_id, token=settings.huggingface_token)
            self._model = AutoModelForZeroShotObjectDetection.from_pretrained(
                self.model_id, token=settings.huggingface_token, use_safetensors=True, trust_remote_code=False,
            ).eval()


detector = TextDetector()
