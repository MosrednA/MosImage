from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from pathlib import Path
from typing import Literal

import numpy as np
from PIL import Image, ImageChops, ImageFilter


CropBox = tuple[int, int, int, int]

# Colour drift correction: the generated crop is compared with the source in
# the untouched ring around the selection, and the smooth difference is added
# to the generated pixels. This removes the faint "patch" left by VAE colour
# drift without touching detail, because only very low frequencies are moved.
DRIFT_LIMIT = 24.0
DRIFT_GUARD = 12
DRIFT_WORKING_SIZE = 128
# Seam matching for expansion: the model also redraws a thin band inside the
# original, which the result keeps original. The difference there is the new
# area's tone offset right at the edge; it fades out over a few band widths.
SEAM_WORKING_SIZE = 512


@dataclass(frozen=True)
class InpaintWorkspace:
    source: Image.Image
    composite_mask: Image.Image
    crop_box: CropBox
    model_input_path: Path
    model_mask_path: Path
    model_output_path: Path
    color_correction: bool = True
    # The area the model saw; it may reach past the picture (mirrored there).
    model_box: CropBox | None = None
    # Expansion: the selection before growing, and the width of the band the
    # model redrew inside the original. Its offset is carried into the new area.
    seam_mask: Image.Image | None = None
    seam_band: int = 0

    def composite_to(self, destination: Path) -> tuple[int, int]:
        left, top, right, bottom = self.crop_box
        model_left, model_top, model_right, model_bottom = self.model_box or self.crop_box
        with Image.open(self.model_output_path) as generated_file:
            generated = generated_file.convert(self.source.mode).resize(
                (model_right - model_left, model_bottom - model_top), Image.Resampling.LANCZOS
            )
        generated = generated.crop((left - model_left, top - model_top, right - model_left, bottom - model_top))

        source_crop = self.source.crop(self.crop_box)
        mask_crop = self.composite_mask.crop(self.crop_box)
        if self.color_correction:
            generated = correct_color_drift(generated, source_crop, mask_crop)
        if self.seam_mask is not None:
            generated = match_seam(generated, source_crop, self.seam_mask.crop(self.crop_box), self.seam_band)
        protected_crop = Image.composite(generated, source_crop, mask_crop)
        result = self.source.copy()
        result.paste(protected_crop, (left, top))
        result.save(destination)
        return result.size


def prepare_inpaint_workspace(
    source_path: Path,
    mask_path: Path,
    work_dir: Path,
    max_dimension: int,
    context_ratio: float,
    minimum_context: int,
    mask_growth: int = 0,
    composite_feather: float = 8.0,
    generation_feather: float | None = None,
    center_selection: bool = False,
    match_edges: bool = False,
) -> InpaintWorkspace:
    with Image.open(source_path) as source_file:
        source = source_file.convert("RGBA" if "A" in source_file.getbands() or "transparency" in source_file.info else "RGB")
    with Image.open(mask_path) as mask_file:
        mask = _selection_alpha(mask_file)

    if mask.size != source.size:
        raise ValueError(f"Mask dimensions {mask.size} do not match source image dimensions {source.size}.")
    selection_box = mask.getbbox()
    if selection_box is None:
        raise ValueError("The inpaint mask is empty.")

    crop_box = _context_box(selection_box, source.size, context_ratio, minimum_context)
    # Models place a new subject near the middle of what they see. A selection
    # near the picture's edge would sit off-centre in a clamped crop, so the
    # subject drifted away from it; mirroring the missing context keeps the
    # selection in the middle.
    model_box = _context_box(selection_box, None, context_ratio, minimum_context) if center_selection else crop_box
    source_crop = _padded_crop(source, model_box, "symmetric")
    # Pipelines supporting soft latent masks retain the user's alpha as well
    # as the inward transition. Other providers receive a binary footprint.
    # Continue the mask's edge coverage into the source's mirrored context. Padding
    # it with black would invent a protected boundary at the canvas edge,
    # letting the model retain the original there even for a full selection.
    # Replicate the edge instead of mirroring interior selections into new islands.
    generation_mask = _padded_crop(mask, model_box, "edge")
    if generation_feather is None:
        generation_mask = generation_mask.point(lambda value: 255 if value else 0)
    model_mask_crop = _grow_mask(generation_mask, mask_growth)
    if generation_feather is not None:
        model_mask_crop = _inner_feather(model_mask_crop, generation_feather)
    # Expansion crossfades over the band the model redrew inside the original,
    # so the new area meets the original without a visible line.
    composite_mask = _inner_feather(_grow_mask(mask, mask_growth), mask_growth) if match_edges else _inner_feather(mask, composite_feather)
    model_size = _model_size(source_crop.size, max_dimension)
    # Flatten before resizing: Pillow premultiplies RGBA during resampling,
    # turning fully transparent padding into black RGB pixels otherwise.
    if source_crop.mode == "RGBA":
        model_input = Image.new("RGB", source_crop.size, (128, 128, 128))
        model_input.paste(source_crop, (0, 0), source_crop.getchannel("A"))
    else:
        model_input = source_crop
    if source_crop.size != model_size:
        model_input = model_input.resize(model_size, Image.Resampling.LANCZOS)
        model_mask_crop = model_mask_crop.resize(model_size, Image.Resampling.LANCZOS)

    model_input_path = work_dir / "input.png"
    model_mask_path = work_dir / "mask.png"
    model_output_path = work_dir / "output.png"
    model_input.save(model_input_path)
    model_mask_crop.save(model_mask_path)
    return InpaintWorkspace(
        source=source,
        composite_mask=composite_mask,
        crop_box=crop_box,
        model_input_path=model_input_path,
        model_mask_path=model_mask_path,
        model_output_path=model_output_path,
        model_box=model_box,
        seam_mask=mask if match_edges and mask_growth else None,
        seam_band=mask_growth,
    )


def correct_color_drift(generated: Image.Image, source: Image.Image, mask: Image.Image) -> Image.Image:
    """Match the generated crop's low-frequency colour to the source around the mask."""
    # Reference pixels: unselected, a few pixels clear of the soft edge, and
    # opaque in the source (outpaint padding carries no colour information).
    # A box blur spreads any visible alpha over the guard radius cheaply.
    reference = mask.filter(ImageFilter.BoxBlur(DRIFT_GUARD)).point(lambda value: 0 if value else 255)
    if source.mode == "RGBA":
        reference = ImageChops.darker(reference, source.getchannel("A").point(lambda value: 255 if value == 255 else 0))
    if reference.getbbox() is None or mask.getbbox() is None:
        return generated

    bands = 3
    generated_pixels = np.asarray(generated, dtype=np.float32)
    source_pixels = np.asarray(source, dtype=np.float32)[..., :bands]
    weight = np.asarray(reference, dtype=np.float32) / 255
    difference = (source_pixels - generated_pixels[..., :bands]) * weight[..., None]

    # Normalized convolution at a small working size: blur(weight * diff) /
    # blur(weight) extrapolates the ring's drift smoothly into the selection.
    height, width = weight.shape
    scale = min(1.0, DRIFT_WORKING_SIZE / max(width, height))
    small = (max(1, round(width * scale)), max(1, round(height * scale)))
    small_weight = _resize_float(weight, small)
    small_difference = np.stack([_resize_float(difference[..., band], small) for band in range(bands)], axis=-1)
    sigma = max(2.0, max(small) / 10)
    blur_x, blur_y = _gaussian_matrix(small[0], sigma), _gaussian_matrix(small[1], sigma)
    blurred_weight = blur_y @ small_weight @ blur_x.T
    blurred_difference = np.stack([blur_y @ small_difference[..., band] @ blur_x.T for band in range(bands)], axis=-1)
    # A tiny share of the global mean fills regions far from any reference.
    epsilon = 1e-3
    global_drift = difference.sum(axis=(0, 1)) / max(float(weight.sum()), 1.0)
    correction = (blurred_difference + epsilon * global_drift) / (blurred_weight[..., None] + epsilon)
    correction = np.clip(correction, -DRIFT_LIMIT, DRIFT_LIMIT)
    full = np.stack([_resize_float(correction[..., band], (width, height)) for band in range(bands)], axis=-1)

    corrected = generated_pixels.copy()
    corrected[..., :bands] = np.clip(generated_pixels[..., :bands] + full, 0, 255)
    return Image.fromarray(np.rint(corrected).astype(np.uint8), mode=generated.mode)


def match_seam(generated: Image.Image, source: Image.Image, mask: Image.Image, band: int) -> Image.Image:
    """Carry the offset measured just inside the source's edge into the selection, fading out."""
    selected = mask.point(lambda value: 255 if value else 0)
    if selected.getbbox() is None:
        return generated
    # The band: unselected, within `band` pixels of the selection, opaque in the source.
    ring = ImageChops.subtract(selected.filter(ImageFilter.BoxBlur(band)).point(lambda value: 255 if value else 0), selected)
    if source.mode == "RGBA":
        ring = ImageChops.darker(ring, source.getchannel("A").point(lambda value: 255 if value == 255 else 0))
    if ring.getbbox() is None:
        return generated

    bands = 3
    generated_pixels = np.asarray(generated, dtype=np.float32)
    weight = np.asarray(ring, dtype=np.float32) / 255
    difference = (np.asarray(source, dtype=np.float32)[..., :bands] - generated_pixels[..., :bands]) * weight[..., None]

    height, width = weight.shape
    scale = min(1.0, SEAM_WORKING_SIZE / max(width, height))
    small = (max(1, round(width * scale)), max(1, round(height * scale)))
    sigma = max(1.5, 2 * band * scale)
    blur_x, blur_y = _gaussian_matrix(small[0], sigma), _gaussian_matrix(small[1], sigma)
    blurred_weight = blur_y @ _resize_float(weight, small) @ blur_x.T
    blurred_difference = np.stack([blur_y @ _resize_float(difference[..., index], small) @ blur_x.T for index in range(bands)], axis=-1)
    # Exact next to the band; further out the weight thins and the offset fades to zero.
    floor = 0.25 * float(blurred_weight.max())
    correction = np.clip(blurred_difference / np.maximum(blurred_weight, floor)[..., None], -DRIFT_LIMIT, DRIFT_LIMIT)
    full = np.stack([_resize_float(correction[..., index], (width, height)) for index in range(bands)], axis=-1)

    corrected = generated_pixels.copy()
    corrected[..., :bands] = np.clip(generated_pixels[..., :bands] + full, 0, 255)
    return Image.fromarray(np.rint(corrected).astype(np.uint8), mode=generated.mode)


def _resize_float(values: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    image = Image.fromarray(values.astype(np.float32), mode="F")
    if image.size == size:
        return values.astype(np.float32)
    resample = Image.Resampling.BOX if size[0] < image.width else Image.Resampling.BILINEAR
    return np.asarray(image.resize(size, resample), dtype=np.float32)


def _gaussian_matrix(length: int, sigma: float) -> np.ndarray:
    positions = np.arange(length, dtype=np.float32)
    return np.exp(-((positions[:, None] - positions[None, :]) ** 2) / (2 * sigma * sigma))


def _grow_mask(mask: Image.Image, radius: int) -> Image.Image:
    if radius < 0:
        raise ValueError("Inpaint mask growth cannot be negative.")
    if radius == 0:
        return mask.copy()
    return mask.filter(ImageFilter.MaxFilter((radius * 2) + 1))


def _inner_feather(mask: Image.Image, radius: float) -> Image.Image:
    if radius < 0:
        raise ValueError("Inpaint composite feather cannot be negative.")
    if radius == 0:
        return mask.copy()
    # Count inward distances with Pillow's native erosion. A Gaussian clipped
    # to the selection starts at roughly 50% opacity and leaves a visible seam.
    # This smoothstep starts at zero and reaches full opacity inside the mask.
    steps = min(64, ceil(radius))
    remaining = mask.point(lambda value: 255 if value else 0)
    distance = Image.new("L", mask.size, 0)
    for _ in range(steps):
        remaining = remaining.filter(ImageFilter.MinFilter(3))
        distance = ImageChops.add(distance, remaining.point(lambda value: 1 if value else 0))
        if remaining.getbbox() is None:
            break
    _minimum, peak = distance.getextrema()
    if peak == 0:
        return mask.copy()  # Preserve selections too thin for an inward ramp.
    denominator = min(steps, peak)
    falloff = distance.point(lambda value: round(255 * (min(value, denominator) / denominator) ** 2 * (3 - 2 * min(value, denominator) / denominator)))
    # Cap existing soft alpha instead of multiplying it a second time.
    return ImageChops.darker(mask, falloff)


def _selection_alpha(mask: Image.Image) -> Image.Image:
    if "A" in mask.getbands():
        alpha = mask.getchannel("A")
        minimum_alpha, _maximum_alpha = alpha.getextrema()
        if minimum_alpha < 255:
            return alpha.copy()
    return mask.convert("L")


def _context_box(
    selection: CropBox,
    image_size: tuple[int, int] | None,
    context_ratio: float,
    minimum_context: int,
) -> CropBox:
    """The selection plus its context, kept within the picture unless `image_size` is None."""
    left, top, right, bottom = selection
    selection_size = max(right - left, bottom - top)
    padding = max(minimum_context, round(selection_size * context_ratio))
    if image_size is None:
        return (left - padding, top - padding, right + padding, bottom + padding)
    image_width, image_height = image_size
    return (
        max(left - padding, 0),
        max(top - padding, 0),
        min(right + padding, image_width),
        min(bottom + padding, image_height),
    )


def _padded_crop(image: Image.Image, box: CropBox, padding_mode: Literal["symmetric", "edge"]) -> Image.Image:
    """Crops `box` with explicit padding for picture context or mask coverage."""
    left, top, right, bottom = box
    width, height = image.size
    if left >= 0 and top >= 0 and right <= width and bottom <= height:
        return image.crop(box)
    pixels = np.asarray(image)
    before_x, before_y = max(0, -left), max(0, -top)
    padding = [(before_y, max(0, bottom - height)), (before_x, max(0, right - width))] + [(0, 0)] * (pixels.ndim - 2)
    padded = np.pad(pixels, padding, mode=padding_mode)
    x, y = left + before_x, top + before_y
    return Image.fromarray(padded[y:y + bottom - top, x:x + right - left], mode=image.mode)


def _model_size(native_size: tuple[int, int], max_dimension: int) -> tuple[int, int]:
    if max_dimension < 32:
        raise ValueError("Inpaint maximum dimension must be at least 32 pixels.")
    width, height = native_size
    scale = min(1.0, max_dimension / max(width, height))
    return (_multiple_of_32(width * scale), _multiple_of_32(height * scale))


def _multiple_of_32(value: float) -> int:
    return max(32, int(value) // 32 * 32)
