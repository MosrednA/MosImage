"""Bounded GPU/visual comparison; optional dependencies stay outside the app env.

Run from backend, one variant per process with the API unloaded. Checkpoints are
verified before weights-only loading. Outputs and timing records remain in data/.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
from hashlib import sha256
from importlib.metadata import version
import json
from pathlib import Path
import sys
import time
from typing import TYPE_CHECKING

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
from app.compositing import prepare_inpaint_workspace
from benchmark_inpaint_quality import assert_api_unloaded

if TYPE_CHECKING:
    from spandrel import ImageModelDescriptor


@dataclass(frozen=True)
class Checkpoint:
    name: str
    release: str
    digest: str


CHECKPOINTS = (
    Checkpoint("RealESRGAN_x2plus", "v0.2.1", "49fafd45f8fd7aa8d31ab2a22d14d91b536c34494a5cfe31eb5d89c2fa266abb"),
    Checkpoint("realesr-general-x4v3", "v0.2.5.0", "8dc7edb9ac80ccdc30c3a5dca6616509367f05fbc184ad95b731f05bece96292"),
    Checkpoint("realesr-general-wdn-x4v3", "v0.2.5.0", "1641f8c4464b9f097c9fdda5589273713f67cf59f3d909e0bd688f0cee269dca"),
)


@dataclass(frozen=True)
class Measurement:
    case: str
    input_size: tuple[int, int]
    output_scale: int
    seconds: tuple[float, ...]
    peak_allocated_gib: float
    peak_reserved_gib: float
    png_save_seconds: float | None = None
    reference_psnr: float | None = None
    lanczos_psnr: float | None = None


@dataclass(frozen=True)
class TileComparison:
    halo: int
    max_delta: int
    mean_absolute_delta: float


@dataclass
class Metadata:
    variant: str
    gpu: str
    torch: str
    cuda: str | None
    spandrel: str
    einops: str
    precision: str
    tile: int
    halo: int
    model_scale: int
    load_seconds: float
    first_small_seconds: float
    checkpoints: tuple[Checkpoint, ...]
    gpu_residency: str
    tile_comparison_max_delta: int | None = None
    tile_comparison_mae: float | None = None
    protected_changes: int | None = None
    alpha_changes_against_geometric_resize: int | None = None
    tile_comparisons: tuple[TileComparison, ...] = ()


def checkpoint_state(index: int) -> dict[str, torch.Tensor]:
    checkpoint = CHECKPOINTS[index]
    path = ROOT / "models/research-cache/Real-ESRGAN" / f"{checkpoint.name}.pth"
    if sha256(path.read_bytes()).hexdigest() != checkpoint.digest:
        raise ValueError(f"Checkpoint hash mismatch: {path}")
    state = torch.load(path, map_location="cpu", weights_only=True)
    weights = state.get("params_ema", state.get("params"))
    if not isinstance(weights, dict) or not all(isinstance(k, str) and isinstance(v, torch.Tensor) for k, v in weights.items()):
        raise ValueError(f"Expected named tensor weights: {path}")
    return weights


def load_model(variant: str) -> ImageModelDescriptor:
    from spandrel import ImageModelDescriptor, ModelLoader
    state = checkpoint_state(0 if variant == "x2plus" else 1)
    if variant == "x4v3-dn05":
        weak = checkpoint_state(2)
        if state.keys() != weak.keys():
            raise ValueError("Denoise checkpoint keys differ")
        state = {key: 0.5 * value + 0.5 * weak[key] for key, value in state.items()}
    descriptor = ModelLoader().load_from_state_dict(state)
    if not isinstance(descriptor, ImageModelDescriptor) or not descriptor.supports_half:
        raise ValueError("Expected an FP16 image upscaler")
    return descriptor.cuda().half().eval()


def upscale(model: ImageModelDescriptor, image: Image.Image, output_scale: int, tile: int, halo: int) -> Image.Image:
    """CPU stitch with overlapped input tiles; alpha uses geometric resampling."""
    rgb = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    width, height = image.size
    scale = model.scale
    output = np.empty((height * scale, width * scale, 3), dtype=np.uint8)
    with torch.inference_mode():
        for top in range(0, height, tile):
            for left in range(0, width, tile):
                right, bottom = min(left + tile, width), min(top + tile, height)
                x0, y0 = max(0, left - halo), max(0, top - halo)
                x1, y1 = min(width, right + halo), min(height, bottom + halo)
                tensor = torch.from_numpy(rgb[y0:y1, x0:x1].copy()).permute(2, 0, 1).unsqueeze(0).to(device=model.device, dtype=model.dtype)
                # Shifted-window masks can remain FP32; autocast keeps BF16
                # attention matmuls compatible without patching vendor code.
                with torch.autocast(device_type=model.device.type, dtype=torch.bfloat16, enabled=model.dtype == torch.bfloat16):
                    generated = model(tensor)
                generated = generated[:, :, (top-y0)*scale:(bottom-y0)*scale, (left-x0)*scale:(right-x0)*scale]
                if not bool(generated.isfinite().all()):
                    raise ValueError("Nonfinite upscaler output")
                pixels = generated.squeeze(0).permute(1, 2, 0).clamp(0, 1).mul(255).round().byte().cpu().numpy()
                output[top*scale:bottom*scale, left*scale:right*scale] = pixels
                del tensor, generated
    result = Image.fromarray(output)
    if output_scale != scale:
        result = result.resize((width * output_scale, height * output_scale), Image.Resampling.LANCZOS)
    if "A" in image.getbands():
        alpha = image.getchannel("A").resize(result.size, Image.Resampling.LANCZOS)
        result.putalpha(alpha)
    return result


def timed(model: ImageModelDescriptor, image: Image.Image, scale: int, tile: int, halo: int, repeats: int) -> tuple[Image.Image, Measurement]:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    samples: list[float] = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        start = time.perf_counter()
        result = upscale(model, image, scale, tile, halo)
        torch.cuda.synchronize()
        samples.append(round(time.perf_counter() - start, 4))
    return result, Measurement("", image.size, scale, tuple(samples),
                              round(torch.cuda.max_memory_allocated() / 2**30, 4),
                              round(torch.cuda.max_memory_reserved() / 2**30, 4))


def psnr(image: Image.Image, reference: Image.Image) -> float:
    error = np.mean((np.asarray(image.convert("RGB"), dtype=np.float64) - np.asarray(reference.convert("RGB"), dtype=np.float64)) ** 2)
    return round(float(20 * np.log10(255 / np.sqrt(max(error, 1e-12)))), 3)


def compare_tiles(model: ImageModelDescriptor, sample: Image.Image, output: Path) -> tuple[TileComparison, ...]:
    full = upscale(model, sample, 2, 512, 32)
    full.save(output / "tile-full.png")
    comparisons: list[TileComparison] = []
    for halo in (32, 64, 128):
        tiled = upscale(model, sample, 2, 192, halo)
        tiled.save(output / ("tile-split.png" if halo == 32 else f"tile-split-halo{halo}.png"))
        delta = np.abs(np.asarray(full, dtype=np.int16) - np.asarray(tiled, dtype=np.int16))
        comparisons.append(TileComparison(halo, int(delta.max()), round(float(delta.mean()), 5)))
    return tuple(comparisons)


def quality_sources(output: Path) -> list[tuple[str, Image.Image]]:
    fixtures = ROOT / "benchmarks/fixtures"
    with Image.open(fixtures / "v1/garden.png") as source:
        garden = source.convert("RGB").crop((115, 115, 1139, 1139))
    with Image.open(fixtures / "v2/man.png") as source:
        man = source.convert("RGB").copy()
    with Image.open(fixtures / "v1/portrait.png") as source:
        denim = source.convert("RGB").crop((610, 630, 1122, 1142))
    text = Image.new("RGB", (1024, 1024), "#eeeeeb")
    draw = ImageDraw.Draw(text)
    draw.rectangle((40, 45, 984, 360), fill="#204843")
    font_path = "C:/Windows/Fonts/segoeui.ttf"
    for y, size, words, fill in ((72, 56, "MosImage Export", "white"), (170, 38, "Texture / Detail / Natural", "white"),
                                 (270, 24, "Aa Bb 0123456789  €  ñ  é", "white"), (415, 30, "Fine lines and small text", "#222222"),
                                 (465, 20, "Preserve letters, spacing and punctuation.", "#222222"),
                                 (510, 14, "Small labels: Quality 2x / Fast 4x / Apply / Undo", "#222222")):
        draw.text((65, y), words, font=ImageFont.truetype(font_path, size), fill=fill)
    for index in range(50):
        draw.line((60+index*17, 630, 60+index*17, 870), fill=(index*5, 70, 130), width=1+index%3)
    cases = [("garden", garden), ("face", man), ("denim", denim), ("text", text)]
    assets = output / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    for name, reference in cases:
        reference.save(assets / f"{name}-reference.png")
        reference.resize((reference.width//2, reference.height//2), Image.Resampling.LANCZOS).save(assets / f"{name}-input.png")
    return cases


def protected_probe(model: ImageModelDescriptor, output: Path, tile: int, halo: int) -> int:
    fixtures = ROOT / "benchmarks/fixtures/v1"
    (output / "protected").mkdir(parents=True, exist_ok=True)
    workspace = prepare_inpaint_workspace(fixtures / "architecture.png", fixtures / "architecture.mask.png",
                                         output / "protected", 512, 0, 96, composite_feather=16, center_selection=True)
    with Image.open(workspace.model_input_path) as image:
        result = upscale(model, image, 2, tile, halo)
    result.save(workspace.model_output_path)
    destination = output / "protected/composite.png"
    workspace.composite_to(destination)
    source = np.asarray(workspace.source)
    with Image.open(destination) as image:
        changed = np.any(source != np.asarray(image), axis=2)
    protected = np.asarray(workspace.composite_mask) == 0
    count = int(np.count_nonzero(changed & protected))
    if count:
        raise AssertionError(f"Protected pixels changed: {count}")
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=("x2plus", "x4v3-dn05", "x4v3-dn10"), required=True)
    parser.add_argument("--dependency-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tile", type=int, default=512)
    parser.add_argument("--halo", type=int, default=32)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.tile < 64 or args.halo < 0 or args.repeats < 1:
        parser.error("Use tile >=64, halo >=0 and repeats >=1")
    sys.path.insert(0, str(args.dependency_path.resolve()))
    assert_api_unloaded()
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    model = load_model(args.variant)
    torch.cuda.synchronize()
    load_seconds = round(time.perf_counter() - start, 4)
    with Image.open(ROOT / "benchmarks/fixtures/v2/man.png") as image:
        photo = image.convert("RGB")
    start = time.perf_counter()
    upscale(model, photo.resize((128, 128)), 2, args.tile, args.halo)
    torch.cuda.synchronize()
    first_small_seconds = round(time.perf_counter() - start, 4)
    records: list[Measurement] = []
    metadata = Metadata(args.variant, torch.cuda.get_device_name(), torch.__version__, torch.version.cuda,
                        version("spandrel"), version("einops"), "float16", args.tile, args.halo, model.scale,
                        load_seconds, first_small_seconds, CHECKPOINTS, "upscaler only; Klein unloaded")
    def save_records() -> None:
        (output / "results.json").write_text(json.dumps({"metadata": asdict(metadata), "measurements": [asdict(item) for item in records]}, indent=2)+"\n", encoding="utf-8")
    for width, height in ((512, 512), (1024, 1024), (2048, 1536)):
        for scale in ((2,) if model.scale == 2 else (2, 4)):
            image = photo.resize((width, height), Image.Resampling.LANCZOS)
            result, measurement = timed(model, image, scale, args.tile, args.halo, args.repeats)
            png_seconds = None
            if width >= 1024:
                start = time.perf_counter()
                result.save(output / f"export-{width}x{height}-{scale}x.png")
                png_seconds = round(time.perf_counter() - start, 4)
            measurement = replace(measurement, case=f"speed-{width}x{height}", png_save_seconds=png_seconds)
            records.append(measurement)
            save_records()
            print(json.dumps(asdict(measurement)), flush=True)
            del result
    for name, reference in quality_sources(output):
        image = reference.resize((reference.width//2, reference.height//2), Image.Resampling.LANCZOS)
        result, measurement = timed(model, image, 2, args.tile, args.halo, 1)
        lanczos = image.resize(reference.size, Image.Resampling.LANCZOS)
        start = time.perf_counter()
        result.save(output / f"{name}-2x.png")
        png_save_seconds = round(time.perf_counter() - start, 4)
        lanczos.save(output / f"{name}-lanczos.png")
        measurement = replace(measurement, case=f"quality-{name}", png_save_seconds=png_save_seconds,
                              reference_psnr=psnr(result, reference), lanczos_psnr=psnr(lanczos, reference))
        records.append(measurement)
        save_records()
        print(json.dumps(asdict(measurement)), flush=True)
    # Compare tiled boundaries against a full-frame pass; numerical differences
    # alone do not prove visible seams. Both images survive for 100% inspection.
    sample = photo.resize((384, 384), Image.Resampling.LANCZOS)
    metadata.tile_comparisons = compare_tiles(model, sample, output)
    metadata.tile_comparison_max_delta = metadata.tile_comparisons[0].max_delta
    metadata.tile_comparison_mae = metadata.tile_comparisons[0].mean_absolute_delta
    metadata.protected_changes = protected_probe(model, output, args.tile, args.halo)
    transparent = sample.convert("RGBA")
    alpha = Image.new("L", sample.size)
    ImageDraw.Draw(alpha).ellipse((30, 30, 354, 354), fill=255)
    transparent.putalpha(alpha)
    rgba = upscale(model, transparent, 2, args.tile, args.halo)
    expected_alpha = alpha.resize(rgba.size, Image.Resampling.LANCZOS)
    metadata.alpha_changes_against_geometric_resize = int(np.count_nonzero(np.asarray(rgba.getchannel("A")) != np.asarray(expected_alpha)))
    rgba.save(output / "alpha-2x.png")
    save_records()
    print(json.dumps(asdict(metadata)), flush=True)


if __name__ == "__main__":
    main()
