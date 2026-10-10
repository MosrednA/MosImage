"""Image layers stacked above a document's background.

Each revision stores its layer stack as JSON that points to immutable PNG
files, so moving or masking a layer is a cheap revision and undo covers it.
"""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any, Iterable, Literal
from uuid import uuid4

from PIL import Image, ImageOps

from . import storage
from .config import settings

MAX_LAYERS = 32
# Larger uploads are scaled down; layers are placed on canvases of a few megapixels.
MAX_LAYER_DIMENSION = 4096
LAYER_ASSET_PREFIXES = ("layer-", "layer-mask-")


def ensure_layer_storage() -> None:
    settings.layer_dir.mkdir(parents=True, exist_ok=True)


def save_asset(content: bytes, kind: Literal["image", "mask"]) -> dict[str, Any]:
    """Stores an uploaded layer image (RGBA) or mask (grayscale, from alpha when present)."""
    try:
        with Image.open(io.BytesIO(content)) as uploaded:
            uploaded.load()
            image = ImageOps.exif_transpose(uploaded)
    except (OSError, ValueError) as exc:
        raise ValueError("Layers can be PNG, JPG or WebP images.") from exc
    if kind == "image":
        image = image.convert("RGBA")
        image.thumbnail((MAX_LAYER_DIMENSION, MAX_LAYER_DIMENSION), Image.Resampling.LANCZOS)
        prefix = "layer-"
    else:
        # Masks painted in the browser keep their coverage in alpha.
        image = image.getchannel("A") if "A" in image.getbands() else image.convert("L")
        if max(image.size) > MAX_LAYER_DIMENSION:
            raise ValueError("Layer masks must match their layer.")
        prefix = "layer-mask-"
    ensure_layer_storage()
    path = settings.layer_dir / f"{prefix}{uuid4().hex}.png"
    image.save(path, "PNG")
    width, height = image.size
    return {"path": storage.to_relative(path), "url": storage.image_url(storage.to_relative(path)), "width": width, "height": height}


def validate(layers: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Checks that every layer points to stored layer files of matching size."""
    if len(layers) > MAX_LAYERS:
        raise ValueError(f"A document can have up to {MAX_LAYERS} layers.")
    if len({layer["id"] for layer in layers}) != len(layers):
        raise ValueError("Layer IDs must be unique.")
    checked = []
    for layer in layers:
        width, height = _size(_layer_file(layer["image_path"], "layer-"))
        if layer.get("mask_path"):
            if _size(_layer_file(layer["mask_path"], "layer-mask-")) != (width, height):
                raise ValueError("A layer mask must match its layer size.")
        variants = layer.get("variant_paths") or []
        if variants:
            # Variants share the layer's mask and transform, so they must be the same size.
            if layer["image_path"] not in variants:
                raise ValueError("A layer must show one of its variants.")
            if any(_size(_layer_file(path, "layer-")) != (width, height) for path in variants):
                raise ValueError("Layer variants must match the layer size.")
        checked.append({**layer, "width": width, "height": height})
    return checked


def records(layers_json: str | None) -> list[dict[str, Any]]:
    """Layer state plus file URLs, bottom layer first."""
    return [
        {
            **layer,
            # Stacks saved before masks could be switched off have them on.
            "mask_enabled": layer.get("mask_enabled", True),
            "image_url": storage.image_url(layer["image_path"]),
            "mask_url": storage.image_url(layer["mask_path"]) if layer.get("mask_path") else None,
            "variant_paths": layer.get("variant_paths", []),
            "variant_urls": [storage.image_url(path) for path in layer.get("variant_paths", [])],
        }
        for layer in parse(layers_json)
    ]


def parse(layers_json: str | None) -> list[dict[str, Any]]:
    return json.loads(layers_json or "[]")


def shifted(layers: Iterable[dict[str, Any]], dx: float, dy: float) -> list[dict[str, Any]]:
    """Keeps layers over the same content when the canvas grows to the left or top."""
    return [{**layer, "x": layer["x"] + dx, "y": layer["y"] + dy} for layer in layers]


def _layer_file(relative: str, prefix: str) -> Path:
    path = (settings.data_dir / relative).resolve()
    layer_dir = settings.layer_dir.resolve()
    if path.parent != layer_dir or not path.name.startswith(prefix) or not path.is_file():
        raise ValueError("A layer image is missing. Add it again.")
    if prefix == "layer-" and path.name.startswith("layer-mask-"):
        raise ValueError("A layer mask cannot be used as a layer image.")
    return path


def _size(path: Path) -> tuple[int, int]:
    with Image.open(path) as image:
        return image.size
