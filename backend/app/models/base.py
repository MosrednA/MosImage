from __future__ import annotations

from pathlib import Path
from typing import Protocol

from app.jobs import JobControl
from app.schemas import InpaintForm, LoadModelRequest, ModelInfo, TextToImageRequest


class ImageModelProvider(Protocol):
    name: str

    def info(self) -> ModelInfo:
        ...

    def generate(self, request: TextToImageRequest, output_path: Path, control: JobControl) -> tuple[int, int]:
        ...

    def inpaint(
        self,
        request: InpaintForm,
        input_path: Path,
        mask_path: Path,
        output_path: Path,
        control: JobControl,
    ) -> tuple[int, int]:
        ...

    def preload(self, request: LoadModelRequest, control: JobControl) -> None:
        ...

    def unload(self) -> None:
        ...
