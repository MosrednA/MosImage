"""Enhance: adds fine detail (with the detail-enhancer LoRA on Quality), in overlapping tiles on large pictures."""
from __future__ import annotations

import json
import math
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from PIL import Image, ImageFilter

from . import storage
from .compositing import FAST_PNG
from .config import settings
from .jobs import JobControl
from .models.registry import registry
from .schemas import EditRequest, EnhanceRequest
from .services import _alternative_seed, _new_image_path, _save_bytes, release_after_failure

# Each tile renders at Klein's native size or larger, so no tile loses pixels.
TILE_AREA = 1024 * 1024
# A picture slightly over the budget still renders in one pass.
SINGLE_PASS_AREA = int(TILE_AREA * 1.1)
TILE_OVERLAP = 128
MAX_TILES = 9
# Three by three tiles of 1024 px with their overlaps.
MAX_REGION_SIDE = 3 * 1024 - 2 * TILE_OVERLAP
# Context around a selection, so the edge of the enhanced area has surroundings.
SELECTION_MARGIN = 0.08
MIN_SELECTION_MARGIN = 32
# Partial redraw, as Faithful Fix faces: five model passes from noise level
# 0.93. It keeps each tile's layout, so neighbouring tiles agree where they overlap.
ENHANCE_STEPS = 8
ENHANCE_STRENGTH = 0.625
# Fast has no detail-enhancer LoRA (it was trained on 9B). The enhance prompt
# alone adds detail there, and more detail means redrawing more: Subtle (0.4)
# at ENHANCE_STRENGTH, Strong (0.7) at 0.75.
FAST_STRENGTH_PER_DETAIL = 5 / 12

Box = tuple[int, int, int, int]


def redraw(model_name: str, detail: float) -> tuple[float, float]:
    """The detail LoRA scale and redraw strength of Enhance on this model."""
    if model_name == settings.quality_model:
        return detail, ENHANCE_STRENGTH
    return 0.0, min(0.875, max(0.5, ENHANCE_STRENGTH + (detail - 0.4) * FAST_STRENGTH_PER_DETAIL))


def tile_layout(width: int, height: int) -> list[Box]:
    """The fewest overlapping tiles of at most TILE_AREA that cover the region, squarest first."""
    if width * height <= SINGLE_PASS_AREA:
        return [(0, 0, width, height)]
    # Long strips would mean fewer tiles, but Klein draws best near square.
    longest_ratio = max(2.0, max(width, height) / min(width, height))
    best: tuple[int, float, int, int, int, int] | None = None
    for columns in range(1, MAX_TILES + 1):
        for rows in range(1, MAX_TILES // columns + 1):
            tile_width = min(width, math.ceil((width + (columns - 1) * TILE_OVERLAP) / columns))
            tile_height = min(height, math.ceil((height + (rows - 1) * TILE_OVERLAP) / rows))
            if tile_width * tile_height > TILE_AREA:
                continue
            ratio = max(tile_width, tile_height) / min(tile_width, tile_height)
            if ratio > longest_ratio:
                continue
            candidate = (columns * rows, ratio, columns, rows, tile_width, tile_height)
            if best is None or candidate[:2] < best[:2]:
                best = candidate
    if best is None:
        raise ValueError(f"This area is too large to enhance at once. Select a part of at most about {MAX_REGION_SIDE} × {MAX_REGION_SIDE} pixels.")
    _, _, columns, rows, tile_width, tile_height = best

    def starts(count: int, total: int, size: int) -> list[int]:
        return [0] if count == 1 else [round(index * (total - size) / (count - 1)) for index in range(count)]

    return [(x, y, x + tile_width, y + tile_height)
            for y in starts(rows, height, tile_height) for x in starts(columns, width, tile_width)]


def tile_weights(tile: Box, width: int, height: int) -> np.ndarray:
    """Linear ramps across the overlap on inner edges, flat on the picture's own edges."""
    left, top, right, bottom = tile

    def ramp(size: int, start_inner: bool, end_inner: bool) -> np.ndarray:
        values = np.ones(size, dtype=np.float32)
        span = min(TILE_OVERLAP, size // 2)
        rising = (np.arange(span, dtype=np.float32) + 0.5) / span
        if start_inner:
            values[:span] = np.minimum(values[:span], rising)
        if end_inner:
            values[size - span:] = np.minimum(values[size - span:], rising[::-1])
        return values

    horizontal = ramp(right - left, left > 0, right < width)
    vertical = ramp(bottom - top, top > 0, bottom < height)
    return vertical[:, None] * horizontal[None, :]


def kept_tones(enhanced: Image.Image, original: Image.Image) -> Image.Image:
    """The enhanced fine detail over the original's broad tones, so colour and light do not drift."""
    radius = max(4.0, math.sqrt(original.width * original.height) / 40)
    blur = ImageFilter.GaussianBlur(radius)
    detail = np.asarray(enhanced, dtype=np.float32) - np.asarray(enhanced.filter(blur), dtype=np.float32)
    base = np.asarray(original.filter(blur), dtype=np.float32)
    return Image.fromarray(np.clip(base + detail, 0, 255).round().astype(np.uint8), mode="RGB")


def enhance_region(selection: np.ndarray | None, width: int, height: int) -> Box:
    """The whole picture, or the selection's bounds with some context around them."""
    if selection is None:
        return 0, 0, width, height
    rows = np.flatnonzero(selection.max(axis=1) > 8)
    columns = np.flatnonzero(selection.max(axis=0) > 8)
    if not len(rows):
        raise ValueError("The selection is empty")
    left, right, top, bottom = int(columns[0]), int(columns[-1]) + 1, int(rows[0]), int(rows[-1]) + 1
    margin = max(MIN_SELECTION_MARGIN, round(max(right - left, bottom - top) * SELECTION_MARGIN))
    return max(0, left - margin), max(0, top - margin), min(width, right + margin), min(height, bottom + margin)


def run_enhance(request: EnhanceRequest, image_bytes: bytes, image_name: str | None, mask_bytes: bytes | None, control: JobControl) -> list[int]:
    """Redraws the picture or selection tile by tile (with the detail LoRA on Quality) and blends the tiles back."""
    provider = registry.activate(request.model_name)
    detail, strength = redraw(request.model_name, request.detail)
    input_path = _save_bytes(image_bytes, settings.input_dir, "input", image_name)
    mask_path: Path | None = None
    output_paths: list[Path] = []
    try:
        with Image.open(input_path) as uploaded:
            picture = uploaded.convert("RGB")
        selection = None
        if mask_bytes is not None:
            with Image.open(BytesIO(mask_bytes)) as uploaded_mask:
                # Selections arrive with their coverage in alpha when they have one.
                band = uploaded_mask.getchannel("A") if "A" in uploaded_mask.getbands() else uploaded_mask.convert("L")
                selection = band.resize(picture.size)
        region = enhance_region(None if selection is None else np.asarray(selection), picture.width, picture.height)
        source = picture.crop(region)
        tiles = tile_layout(source.width, source.height)
        if selection is not None:
            mask_path = _new_image_path(settings.mask_dir, "enhance")
            selection.save(mask_path)
        values: list[dict[str, object]] = []
        with TemporaryDirectory(prefix="enhance-", dir=settings.data_dir) as temporary:
            work = Path(temporary)
            for index in range(request.count):
                proposal = control.slice(index, request.count)
                seed = _alternative_seed(request.seed, index)
                total = np.zeros((source.height, source.width, 3), dtype=np.float32)
                weight = np.zeros((source.height, source.width, 1), dtype=np.float32)
                used_seed = 0
                for number, tile in enumerate(tiles):
                    step = proposal.subrange(number / len(tiles), (number + 1) / len(tiles))
                    original = source.crop(tile)
                    tile_in, tile_out = work / f"tile-{number}.png", work / f"tile-{number}-out.png"
                    original.save(tile_in, **FAST_PNG)
                    tile_request = EditRequest(prompt="enhance", model_name=request.model_name, runtime_mode=request.runtime_mode,
                                               steps=ENHANCE_STEPS, guidance=request.guidance,
                                               seed=None if seed is None else (seed + number) % (2**32),
                                               mode="enhance", detail=detail, strength=strength,
                                               ).model_copy(update={"prompt": request.prompt})
                    tile_seed, _ = provider.edit(tile_request, tile_in, tile_out, step)
                    used_seed = used_seed or tile_seed
                    with Image.open(tile_out) as rendered:
                        enhanced = kept_tones(rendered.convert("RGB").resize(original.size, Image.Resampling.LANCZOS), original)
                    tile_weight = tile_weights(tile, source.width, source.height)[..., None]
                    left, top, right, bottom = tile
                    total[top:bottom, left:right] += np.asarray(enhanced, dtype=np.float32) * tile_weight
                    weight[top:bottom, left:right] += tile_weight
                blended = Image.fromarray(np.clip(total / np.maximum(weight, 1e-6), 0, 255).round().astype(np.uint8), mode="RGB")
                result = picture.copy()
                if selection is None:
                    result.paste(blended, region)
                else:
                    result.paste(blended, region, selection.crop(region))
                output_path = _new_image_path(settings.output_dir, "output")
                output_paths.append(output_path)
                result.save(output_path, **FAST_PNG)
                control.raise_if_cancelled()
                values.append({
                    "task": "edit",
                    "prompt": request.prompt or "Enhance",
                    "negative_prompt": request.negative_prompt,
                    "model_name": request.model_name,
                    "runtime_mode": request.runtime_mode,
                    "seed": used_seed,
                    "steps": ENHANCE_STEPS,
                    "guidance": request.guidance,
                    "width": picture.width,
                    "height": picture.height,
                    "strength": request.detail,
                    "edit_operation": "enhance",
                    "input_image_path": storage.to_relative(input_path),
                    "mask_image_path": storage.to_relative(mask_path) if mask_path is not None else None,
                    "output_image_path": storage.to_relative(output_path),
                    "reference_image_paths": None,
                })
        control.raise_if_cancelled()
        return storage.create_generations(values)
    except BaseException as error:
        release_after_failure(provider, error)
        for path in (input_path, mask_path, *output_paths):
            if path is not None:
                path.unlink(missing_ok=True)
        raise
