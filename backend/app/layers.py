"""The image layers that make up a document's picture.

Each revision stores its layer stack as JSON that points to immutable image
files, so moving or masking a layer is a cheap revision and undo covers it.
The picture a document starts from is its bottom layer, the Background, as
an ordinary layer: it can be hidden, moved, masked, merged or deleted.
"""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any, Iterable, Literal
from uuid import uuid4

from PIL import Image, ImageOps

from . import storage
from .compositing import FAST_PNG
from .config import settings

# Room for 32 layers over the background that older documents kept apart.
MAX_LAYERS = 40
# Larger uploads are scaled down; a merged background keeps a large photo's size.
MAX_LAYER_DIMENSION = 8192
LAYER_ASSET_PREFIXES = ("layer-", "layer-mask-")
BACKGROUND_ID = "background"


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
    # Fast compression: merging a large background uploads tens of megapixels.
    image.save(path, "PNG", **FAST_PNG)
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


def background(image_path: str, size: tuple[int, int], layer_id: str = BACKGROUND_ID, name: str = "Background") -> dict[str, Any]:
    """A layer that fills the canvas with `image_path`, as a new document's picture does."""
    width, height = size
    return {
        "id": layer_id, "name": name, "image_path": image_path, "mask_path": None, "mask_enabled": True, "variant_paths": [],
        "x": width / 2, "y": height / 2, "scale_x": 1.0, "scale_y": 1.0, "rotation": 0.0, "opacity": 1.0, "visible": True,
        "width": width, "height": height,
    }


def fills_canvas(layer: dict[str, Any], size: tuple[int, int]) -> bool:
    """True for a visible, unmasked layer whose pixels are the canvas's own, one to one."""
    width, height = size
    return (
        layer.get("visible", True) and layer.get("opacity", 1.0) == 1.0
        and not (layer.get("mask_path") and layer.get("mask_enabled", True))
        and (layer.get("width"), layer.get("height")) == (width, height)
        and (layer["x"], layer["y"]) == (width / 2, height / 2)
        and (layer.get("scale_x", 1.0), layer.get("scale_y", 1.0), layer.get("rotation", 0.0)) == (1.0, 1.0, 0.0)
    )


def with_result_layer(layers: list[dict[str, Any]], image_path: str, size: tuple[int, int], name: str, replace_bottom: bool) -> tuple[list[dict[str, Any]], str]:
    """Shows a full-canvas result in the bottom layer, or in a new layer under every layer.

    Returns the stack and the id of the layer that shows the result.
    """
    if layers and replace_bottom:
        bottom = {**background(image_path, size, layers[0]["id"], layers[0]["name"]), "original_id": layers[0].get("original_id")}
        return [bottom, *layers[1:]], bottom["id"]
    added = background(image_path, size, new_id(), name)
    return [added, *layers], added["id"]


def new_id() -> str:
    return f"layer-{uuid4().hex[:16]}"


def records(layers_json: str | None) -> list[dict[str, Any]]:
    """Layer state plus file URLs, bottom layer first."""
    return [
        {
            **layer,
            # Stacks saved before masks could be switched off have them on.
            "mask_enabled": layer.get("mask_enabled", True),
            "original_id": layer.get("original_id"),
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
    # A background shows the document's own picture, or an Expand result, without a copy.
    picture = prefix == "layer-" and path.parent in (settings.document_dir.resolve(), settings.revision_dir.resolve())
    if not path.is_file() or not (picture or (path.parent == settings.layer_dir.resolve() and path.name.startswith(prefix))):
        raise ValueError("A layer image is missing. Add it again.")
    if prefix == "layer-" and path.name.startswith("layer-mask-"):
        raise ValueError("A layer mask cannot be used as a layer image.")
    return path


def _size(path: Path) -> tuple[int, int]:
    with Image.open(path) as image:
        return image.size
