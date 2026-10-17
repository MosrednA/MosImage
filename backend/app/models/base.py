from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Sequence

from app.jobs import JobControl
from app.schemas import EditRequest, InpaintForm, LoadModelRequest, ModelInfo, OutpaintForm, ReferenceRole, TextToImageRequest


@dataclass(frozen=True)
class ModelReference:
    """A concept image the user named, as handed to reference-aware providers."""

    name: str
    path: Path
    role: ReferenceRole = "subject"


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
        references: Sequence[ModelReference] = (),
    ) -> tuple[int, int]:
        ...

    # Only providers with the "edit" capability implement whole-image edits.
    def edit(
        self,
        request: EditRequest,
        input_path: Path,
        output_path: Path,
        control: JobControl,
        references: Sequence[ModelReference] = (),
    ) -> tuple[int, int]:
        ...

    def outpaint(self, request: OutpaintForm, input_path: Path, output_path: Path,
                 control: JobControl) -> tuple[int, int]:
        ...

    def preload(self, request: LoadModelRequest, control: JobControl) -> None:
        ...

    def unload(self) -> None:
        ...
