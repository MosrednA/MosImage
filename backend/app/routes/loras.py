"""The user's own LoRA files and the models each one fits."""
from __future__ import annotations

from fastapi import APIRouter
from pydantic import ValidationError

from ..config import settings
from ..loras import inspect, lora_files
from ..models.registry import registry
from ..schemas import LoraChoice, LoraFile, LoraLibrary
from .common import http_errors

router = APIRouter(prefix="/api")


@router.get("/loras", response_model=LoraLibrary)
def list_loras() -> LoraLibrary:
    models = registry.list()
    files: list[LoraFile] = []
    # The layer shapes come from the models' configs, which a fresh install downloads.
    with http_errors({OSError: 503}):
        for path in lora_files():
            try:
                LoraChoice(name=path.stem)
            except ValidationError:
                files.append(LoraFile(name=path.stem, size_bytes=path.stat().st_size, models=[],
                                      problem='Rename the file: a LoRA name cannot hold / \\ : * ? " < > |.'))
                continue
            inspection = inspect(path)
            files.append(LoraFile(name=path.stem, size_bytes=path.stat().st_size, problem=inspection.problem,
                                  models=[model.name for model in models if model.architecture in inspection.fits]))
    return LoraLibrary(folder=str(settings.lora_dir), loras=files)
