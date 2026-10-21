"""The user's own LoRAs: ``.safetensors`` files in one folder, matched to the Klein models.

A LoRA only fits the transformer it was trained on, so each file is compared
with the 4B and 9B layer shapes before it is offered. The check reads tensor
shapes only, no weights, and its answer is kept until the file changes.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from math import prod
from pathlib import Path
from typing import TYPE_CHECKING

from app.config import settings
from app.models.flux2_klein_nvfp4 import BASE_REPO, BASE_REPO_9B, BASE_REVISION, BASE_REVISION_9B
from app.schemas import LoraArchitecture

if TYPE_CHECKING:
    from torch import Tensor

SUFFIX = ".safetensors"
# The transformer config each architecture's layer shapes come from.
ARCHITECTURES: dict[LoraArchitecture, tuple[str, str]] = {
    "klein-4b": (BASE_REPO, BASE_REVISION),
    "klein-9b": (BASE_REPO_9B, BASE_REVISION_9B),
}
ARCHITECTURE_NAMES: dict[LoraArchitecture, str] = {"klein-4b": "FLUX.2 Klein 4B", "klein-9b": "FLUX.2 Klein 9B"}


@dataclass(frozen=True)
class Inspection:
    """Which transformers a file fits, or why it fits none."""

    fits: tuple[LoraArchitecture, ...]
    problem: str | None


_inspections: dict[tuple[Path, int, int], Inspection] = {}


def lora_file(name: str) -> Path:
    """The file behind a LoRA name, its file name without ``.safetensors``."""
    path = settings.lora_dir / (name + SUFFIX)
    if not path.is_file():
        raise ValueError(f"The LoRA '{name}' is not in the LoRA folder.")
    return path


def signature(path: Path) -> tuple[int, int]:
    """Changes whenever the file is replaced or rewritten."""
    stat = path.stat()
    return stat.st_mtime_ns, stat.st_size


def lora_files() -> list[Path]:
    settings.lora_dir.mkdir(parents=True, exist_ok=True)
    return sorted((path for path in settings.lora_dir.iterdir()
                   if path.is_file() and path.suffix.lower() == SUFFIX and not path.name.startswith(".")),
                  key=lambda path: path.stem.lower())


def inspect(path: Path) -> Inspection:
    key = (path, *signature(path))
    inspection = _inspections.get(key)
    if inspection is None:
        inspection = _inspect(path)
        _inspections[key] = inspection
    return inspection


def _inspect(path: Path) -> Inspection:
    from safetensors import SafetensorError

    try:
        pairs = _pairs(path)
    except (ValueError, OSError, SafetensorError) as exc:
        return Inspection((), str(exc) if isinstance(exc, ValueError) else "Not a readable safetensors file.")
    fits = tuple(architecture for architecture in ARCHITECTURES if _fits(_linears(architecture), pairs))
    if not fits:
        return Inspection((), "Made for another model: it fits neither FLUX.2 Klein 4B nor 9B.")
    return Inspection(fits, None)


def _pairs(path: Path) -> dict[str, dict[str, Tensor]]:
    from app.models.klein_lora import lora_pairs, read_lora

    try:
        return lora_pairs(*_shapes(path))
    except (RuntimeError, NotImplementedError):
        # A converter that needs the values (kohya's sparse QKV test) reads them all.
        return read_lora(path)


def _shapes(path: Path) -> tuple[dict[str, Tensor], dict[str, str] | None]:
    """The file's tensors as meta tensors of the same shape; alphas, being scalars, stay real."""
    import torch
    from safetensors import safe_open

    with safe_open(str(path), framework="pt", device="cpu") as file:
        state = {}
        for key in file.keys():
            shape = file.get_slice(key).get_shape()
            state[key] = file.get_tensor(key) if prod(shape) <= 1 else torch.empty(shape, device="meta")
        return state, file.metadata()


def _fits(linears: dict[str, tuple[int, int]], pairs: dict[str, dict[str, Tensor]]) -> bool:
    for name, pair in pairs.items():
        if name not in linears:
            return False
        in_features, out_features = linears[name]
        rank = pair["A"].shape[0]
        if tuple(pair["A"].shape) != (rank, in_features) or tuple(pair["B"].shape) != (out_features, rank):
            return False
    return True


@lru_cache(maxsize=None)
def _linears(architecture: LoraArchitecture) -> dict[str, tuple[int, int]]:
    """(in, out) features of every linear in the transformer, from an empty skeleton of it."""
    from accelerate import init_empty_weights
    from diffusers import Flux2Transformer2DModel
    from huggingface_hub import hf_hub_download
    from torch import nn

    repo, revision = ARCHITECTURES[architecture]
    config = hf_hub_download(repo, "transformer/config.json", revision=revision, token=settings.huggingface_token)
    with init_empty_weights():
        model = Flux2Transformer2DModel.from_config(json.loads(Path(config).read_text(encoding="utf-8")))
    return {name: (module.in_features, module.out_features)
            for name, module in model.named_modules() if isinstance(module, nn.Linear)}
