from __future__ import annotations

from dataclasses import dataclass, replace
from math import ceil, sqrt
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
OUTPAINT_GENERATION_OVERLAP = 32  # Pixels at inference resolution, independent of canvas size.
OUTPAINT_BLEND_BAND = 16  # Native pixels the final composite may change inside the original.
# Klein blends the original back in per 16-pixel token, sampled at the token's
# centre, so a token only partly inside the selection kept old content: the
# blue fringe around a recoloured shirt. One token of hidden growth regenerates
# every token the selection touches; the final blend still stops at the
# visible selection.
INPAINT_GENERATION_OVERLAP = 16
# The regenerated ring also shows how new content meets what stays. Offsets
# well under this RMS difference (8-bit levels), such as a removed object's
# shadow, are carried inward; larger ones are different content, such as
# leftovers of the old object, and smearing them would leave a ghost.
INPAINT_SEAM_TOLERANCE = 32.0
# Grain matching: the model and resizing smooth away a photo's finest grain,
# which makes new content look plastic. The loss is measured where the model
# reproduced the original and added back as monochrome noise.
GRAIN_LIMIT = 5.0  # Largest added standard deviation in 8-bit levels; texture is not noise.
GRAIN_SAMPLES = 250_000
# Immerkaer's noise operator ignores flat areas and linear gradients; unit
# white noise gives a response with standard deviation 6.
NOISE_KERNEL = ((1, -2, 1), (-2, 4, -2), (1, -2, 1))
NOISE_KERNEL_GAIN = 6.0
# Intermediate files are read once and outputs are written at every edit, so
# fast compression beats small files.
FAST_PNG = {"compress_level": 1}


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
    seam_tolerance: float | None = None
    # A wider generation footprint must not become reference pixels for drift
    # correction merely because the final blend protects those pixels.
    color_reference_mask: Image.Image | None = None
    grain_matching: bool = True

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
        reference = _reference_pixels((self.color_reference_mask or self.composite_mask).crop(self.crop_box), source_crop)
        if self.color_correction:
            generated = _correct_color_drift(generated, source_crop, reference)
        if self.seam_mask is not None:
            generated = match_seam(generated, source_crop, self.seam_mask.crop(self.crop_box), self.seam_band, self.seam_tolerance)
        if self.grain_matching:
            generated = match_grain(generated, source_crop, mask_crop, reference, seed=self.crop_box)
        protected_crop = Image.composite(generated, source_crop, mask_crop)
        result = self.source.copy()
        result.paste(protected_crop, (left, top))
        result.save(destination, **FAST_PNG)
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
    generation_growth: int = 0,
    minimum_dimension: int = 0,
) -> InpaintWorkspace:
    """Crop the selection with its context and save what the model sees.

    `generation_growth` widens the generated area by that many model pixels
    after resizing; the final blend keeps the visible selection. Crops smaller
    than `minimum_dimension` squared are rendered larger and scaled back.
    """
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
    # Production inpaint gives the model full coverage inside the selection,
    # while the compositor keeps its alpha. The optional sampling feather is
    # retained for controlled comparisons with older mask recipes.
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
    model_size = _model_size(source_crop.size, max_dimension, minimum_dimension)
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
    color_reference_mask = None
    seam_mask, seam_band, seam_tolerance = (mask, mask_growth, None) if match_edges and mask_growth else (None, 0, None)
    if generation_growth:
        # Grow after resizing, so the overlap is the same in model pixels for
        # every canvas size. The regenerated ring is no colour reference.
        model_mask_crop = _grow_mask(model_mask_crop, generation_growth)
        color_reference_mask = _native_coverage(model_mask_crop, model_box, crop_box, source.size)
        # The ring shows how the new content meets what stays: carrying its
        # offset inward joins a removed object's shadow or a lighter fill to
        # the surroundings instead of leaving the selection's outline.
        if seam_mask is None:
            seam_mask, seam_band = mask, max(1, int(generation_growth * source_crop.width / model_size[0]))
            seam_tolerance = INPAINT_SEAM_TOLERANCE

    model_input_path = work_dir / "input.png"
    model_mask_path = work_dir / "mask.png"
    model_output_path = work_dir / "output.png"
    model_input.save(model_input_path, **FAST_PNG)
    model_mask_crop.save(model_mask_path, **FAST_PNG)
    return InpaintWorkspace(
        source=source,
        composite_mask=composite_mask,
        crop_box=crop_box,
        model_input_path=model_input_path,
        model_mask_path=model_mask_path,
        model_output_path=model_output_path,
        model_box=model_box,
        seam_mask=seam_mask,
        seam_band=seam_band,
        seam_tolerance=seam_tolerance,
        color_reference_mask=color_reference_mask,
    )


def prepare_outpaint_workspace(
    source_path: Path,
    mask_path: Path,
    work_dir: Path,
    max_dimension: int,
    context_ratio: float,
    minimum_context: int,
    minimum_dimension: int = 0,
) -> InpaintWorkspace:
    """Redraw an overlap at model resolution; blend only the native edge band."""
    with Image.open(mask_path) as uploaded_mask:
        selection = _selection_alpha(uploaded_mask)
    # 32 inference pixels must not become four on a 4K canvas.
    workspace = prepare_inpaint_workspace(
        source_path, mask_path, work_dir, max_dimension, context_ratio,
        minimum_context, composite_feather=0, generation_growth=OUTPAINT_GENERATION_OVERLAP,
        minimum_dimension=minimum_dimension,
    )
    # The overlap redraws the original itself, so every offset there is tone.
    return replace(workspace, composite_mask=outpaint_composite_mask(selection), seam_mask=selection, seam_band=OUTPAINT_BLEND_BAND, seam_tolerance=None)


def expansion_mask(size: tuple[int, int], original: CropBox) -> Image.Image:
    """What an expansion redraws: everything around the original at `original`."""
    mask = Image.new("L", size, 255)
    mask.paste(0, original)
    return mask


def outpaint_composite_mask(selection: Image.Image) -> Image.Image:
    """Where an expansion's result shows: the new area, fading out over a band inside the original."""
    return _inner_feather(_grow_mask(selection, OUTPAINT_BLEND_BAND), OUTPAINT_BLEND_BAND)


def rebase_expansion(result: Image.Image, seen: Image.Image, background: Image.Image, offset: tuple[int, int]) -> Image.Image:
    """An expansion of `seen` with `background` under the original instead.

    The model continued `seen`, the background with its visible layers, so the
    new area matches what you see. The layers stay separate, so inside the
    original the result keeps the background's own pixels; in the band where
    the model's redraw fades in, it fades into the background instead of the
    layers: result + (background - seen) * (1 - blend). Where the background
    itself was visible, the result is unchanged.
    """
    left, top = offset
    original = (left, top, left + background.width, top + background.height)
    if seen.size != result.size or left < 0 or top < 0 or original[2] > result.width or original[3] > result.height:
        raise ValueError("The expansion does not line up with the document.")
    rebased = result.convert("RGBA")
    blend = np.asarray(outpaint_composite_mask(expansion_mask(result.size, original)).crop(original))
    band = blend > 0
    inside = np.array(background.convert("RGBA"))
    if band.any():
        made = np.asarray(rebased.crop(original))[band].astype(np.float32)
        kept = np.asarray(seen.convert("RGBA").crop(original))[band].astype(np.float32)
        share = 1 - blend[band][:, None].astype(np.float32) / 255
        inside[band] = np.rint(np.clip(made + (inside[band] - kept) * share, 0, 255)).astype(np.uint8)
    rebased.paste(Image.fromarray(inside), (left, top))
    return rebased


def correct_color_drift(generated: Image.Image, source: Image.Image, mask: Image.Image) -> Image.Image:
    """Match the generated crop's low-frequency colour to the source around the mask."""
    return _correct_color_drift(generated, source, _reference_pixels(mask, source))


def _reference_pixels(mask: Image.Image, source: Image.Image) -> Image.Image:
    """Pixels the model reproduced: unselected, clear of the soft edge, opaque in the source."""
    if mask.getbbox() is None:
        return Image.new("L", mask.size, 0)  # Nothing was generated, so nothing needs matching.
    # Outpaint padding carries no colour information. A box blur spreads any
    # visible alpha over the guard radius cheaply.
    reference = mask.filter(ImageFilter.BoxBlur(DRIFT_GUARD)).point(lambda value: 0 if value else 255)
    if source.mode == "RGBA":
        reference = ImageChops.darker(reference, source.getchannel("A").point(lambda value: 255 if value == 255 else 0))
    return reference


def _correct_color_drift(generated: Image.Image, source: Image.Image, reference: Image.Image) -> Image.Image:
    if reference.getbbox() is None:
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


def match_seam(generated: Image.Image, source: Image.Image, mask: Image.Image, band: int, tolerance: float | None = None) -> Image.Image:
    """Carry the offset measured just inside the source's edge into the selection, fading out.

    With a `tolerance`, ring pixels whose difference is large compared with it
    count less: a tone offset is carried, different content is not.
    """
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
    difference = np.asarray(source, dtype=np.float32)[..., :bands] - generated_pixels[..., :bands]

    height, width = weight.shape
    scale = min(1.0, SEAM_WORKING_SIZE / max(width, height))
    small = (max(1, round(width * scale)), max(1, round(height * scale)))
    sigma = max(1.5, 2 * band * scale)
    blur_x, blur_y = _gaussian_matrix(small[0], sigma), _gaussian_matrix(small[1], sigma)
    # Exact next to the band; further out the weight thins and the offset fades
    # to zero. The floor comes from the band's shape alone, so ring pixels the
    # tolerance discounts weaken the correction instead of being renormalised.
    floor = 0.25 * float((blur_y @ _resize_float(weight, small) @ blur_x.T).max())
    if tolerance is not None:
        weight = weight * np.exp(-np.mean(difference * difference, axis=-1) / (tolerance * tolerance))
    difference *= weight[..., None]
    blurred_weight = blur_y @ _resize_float(weight, small) @ blur_x.T
    blurred_difference = np.stack([blur_y @ _resize_float(difference[..., index], small) @ blur_x.T for index in range(bands)], axis=-1)
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


def match_grain(
    generated: Image.Image, source: Image.Image, mask: Image.Image, reference: Image.Image, seed: tuple[int, ...]
) -> Image.Image:
    """Add back the fine grain the model and resizing smoothed away inside the mask.

    Both crops are measured on the reference pixels, where the model reproduced
    the original, so the difference is the grain lost, not a difference in
    content. The noise is monochrome and seeded, so a composite is repeatable.
    """
    selected = np.asarray(mask) > 0
    rows, columns = np.flatnonzero(selected.any(axis=1)), np.flatnonzero(selected.any(axis=0))
    if rows.size == 0:
        return generated
    weight = np.asarray(reference) == 255
    source_level = _noise_level(source, weight)
    generated_level = _noise_level(generated, weight)
    if source_level is None or generated_level is None:
        return generated
    amplitude = min(GRAIN_LIMIT, sqrt(max(0.0, source_level ** 2 - generated_level ** 2)) / NOISE_KERNEL_GAIN)
    if amplitude < 0.5:
        return generated  # Below one level after rounding.
    top, bottom, left, right = rows[0], rows[-1] + 1, columns[0], columns[-1] + 1
    noise = np.random.default_rng(seed).standard_normal((bottom - top, right - left), dtype=np.float32) * amplitude
    pixels = np.array(generated, dtype=np.float32)
    pixels[top:bottom, left:right, :3] += noise[..., None]
    return Image.fromarray(np.rint(np.clip(pixels, 0, 255)).astype(np.uint8))


def _noise_level(image: Image.Image, weight: np.ndarray) -> float | None:
    """Robust standard deviation of the noise operator's luma response on the weighted pixels."""
    # Only pixels whose whole 3 x 3 neighbourhood is reference count.
    inner = weight[1:-1, 1:-1] & weight[:-2, 1:-1] & weight[2:, 1:-1] & weight[1:-1, :-2] & weight[1:-1, 2:]
    positions = np.flatnonzero(inner)
    if positions.size < 1024:
        return None
    positions = positions[:: max(1, positions.size // GRAIN_SAMPLES)]
    rgb = np.asarray(image, dtype=np.float32)[..., :3]
    luma = rgb @ np.array((0.299, 0.587, 0.114), dtype=np.float32)
    width = inner.shape[1]
    y, x = positions // width + 1, positions % width + 1
    response = np.zeros(positions.size, dtype=np.float32)
    for dy, row in enumerate(NOISE_KERNEL):
        for dx, factor in enumerate(row):
            response += factor * luma[y + dy - 1, x + dx - 1]
    return float(np.median(np.abs(response))) / 0.6745


def _grow_mask(mask: Image.Image, radius: int) -> Image.Image:
    """Square dilation by `radius`, like repeated 3 x 3 MaxFilters with Pillow's edge handling."""
    if radius < 0:
        raise ValueError("Inpaint mask growth cannot be negative.")
    if radius == 0:
        return mask.copy()
    pixels = np.asarray(mask)
    grown = _row_extreme(_row_extreme(pixels, radius, np.maximum, 0).T, radius, np.maximum, 0).T
    return Image.fromarray(np.ascontiguousarray(grown))


def _row_extreme(values: np.ndarray, radius: int, reduce: np.ufunc, outside: int) -> np.ndarray:
    """Max or min over 2 * radius + 1 neighbours in each row; positions past the edge do not count.

    Windows double in size each pass, so a 65-pixel window takes seven array
    operations instead of 32 filter passes.
    """
    window = 2 * radius + 1
    windows = np.pad(values, ((0, 0), (radius, radius)), constant_values=outside)
    span = 1
    while span * 2 <= window:
        windows = reduce(windows[:, :-span], windows[:, span:])
        span *= 2
    width = values.shape[1]
    return reduce(windows[:, :width], windows[:, window - span:window - span + width])


def _inner_feather(mask: Image.Image, radius: float) -> Image.Image:
    if radius < 0:
        raise ValueError("Inpaint composite feather cannot be negative.")
    if radius == 0:
        return mask.copy()
    # A Gaussian clipped to the selection starts at roughly 50% opacity and
    # leaves a visible seam. This smoothstep starts at zero on the edge and
    # reaches full opacity `radius` pixels inside the mask.
    steps = min(64, ceil(radius))
    pixels = np.asarray(mask)
    # How many 3 x 3 erosions each pixel survives, up to `steps`.
    distance = np.clip(_chessboard_distance(pixels > 0, steps + 1).astype(np.int16) - 1, 0, steps)
    peak = int(distance.max())
    if peak == 0:
        return mask.copy()  # Preserve selections too thin for an inward ramp.
    denominator = min(steps, peak)
    falloff = np.array(
        [round(255 * (min(value, denominator) / denominator) ** 2 * (3 - 2 * min(value, denominator) / denominator)) for value in range(steps + 1)],
        dtype=np.uint8,
    )
    # Cap existing soft alpha instead of multiplying it a second time.
    return Image.fromarray(np.minimum(pixels, falloff[distance]))


def _chessboard_distance(selected: np.ndarray, cap: int) -> np.ndarray:
    """Chessboard distance from each pixel to the nearest unselected one, at most `cap`.

    Pixels past the picture's edge count as selected, which matches Pillow's
    edge replication in repeated erosion.
    """
    height, width = selected.shape
    columns = np.arange(width, dtype=np.int32)
    far = width + cap
    last_unselected = np.maximum.accumulate(np.where(selected, -far, columns), axis=1)
    next_unselected = np.minimum.accumulate(np.where(selected, far, columns)[:, ::-1], axis=1)[:, ::-1]
    across = np.minimum(np.minimum(columns - last_unselected, next_unselected - columns), cap).astype(np.uint8)
    # Combine rows: distance = min over row offsets t of max(|t|, distance across that row).
    distance = across.copy()
    for offset in range(1, min(cap, height)):
        if int(distance.max()) <= offset:
            break
        floor = np.uint8(offset)
        np.minimum(distance[offset:], np.maximum(across[:-offset], floor), out=distance[offset:])
        np.minimum(distance[:-offset], np.maximum(across[offset:], floor), out=distance[:-offset])
    return distance


def _native_coverage(model_mask: Image.Image, model_box: CropBox, crop_box: CropBox, size: tuple[int, int]) -> Image.Image:
    """The generation mask in picture coordinates, zero outside the crop."""
    model_left, model_top, model_right, model_bottom = model_box
    left, top, right, bottom = crop_box
    coverage = model_mask.resize((model_right - model_left, model_bottom - model_top), Image.Resampling.LANCZOS)
    native = Image.new("L", size, 0)
    native.paste(coverage.crop((left - model_left, top - model_top, right - model_left, bottom - model_top)), (left, top))
    return native


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


def _model_size(native_size: tuple[int, int], max_dimension: int, minimum_dimension: int = 0) -> tuple[int, int]:
    """The crop's size at inference: at most `max_dimension` squared in area.

    An area budget gives a long, narrow crop as many pixels as a square one; the
    long side stays within twice `max_dimension`. Crops smaller than
    `minimum_dimension` squared are enlarged, so small edits get the model's
    detail rather than its blur at low resolution.
    """
    if max_dimension < 32:
        raise ValueError("Inpaint maximum dimension must be at least 32 pixels.")
    width, height = native_size
    side = sqrt(width * height)
    scale = min(1.0, max_dimension / side, 2 * max_dimension / max(width, height))
    if minimum_dimension:
        scale = max(scale, min(minimum_dimension, max_dimension) / side)
        scale = min(scale, 2 * max_dimension / max(width, height))
    return (_multiple_of_16(width * scale), _multiple_of_16(height * scale))


def _multiple_of_16(value: float) -> int:
    # Klein packs 16 x 16 pixel tokens and floors other sizes itself.
    return max(16, int(value) // 16 * 16)
