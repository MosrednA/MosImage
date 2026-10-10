"""Run from backend: native 2x exports and reconstruction without GAN restoration."""
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
from PIL import Image
import torch

from benchmark_upscalers import ROOT, Measurement, assert_api_unloaded, psnr, quality_sources, timed, upscale, protected_probe

if TYPE_CHECKING:
    from spandrel import ImageModelDescriptor


@dataclass(frozen=True)
class Candidate:
    name: str
    digest: str
    source_url: str


CANDIDATES = (
    Candidate("swinir-classical2", "2032ebf8f401dd3ce2fae5f3852117cb72101ec6ed8358faa64c2a3fa09ed4ac", "https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/001_classicalSR_DF2K_s64w8_SwinIR-M_x2.pth"),
    Candidate("swinir-light2", "193b229909ca89cd8b55de9c9e7fce146ae759d59dfcd78d8feb9dd1d6fa0fd7", "https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/002_lightweightSR_DIV2K_s64w8_SwinIR-S_x2.pth"),
    Candidate("swin2sr-classical2", "04834dbf863f00168ece32f9fc34ad1e3ad48ff1bf8098812573ed5810b59fa0", "https://github.com/mv-lab/swin2sr/releases/download/v0.0.1/Swin2SR_ClassicalSR_X2_64.pth"),
)


@dataclass(frozen=True)
class SourceRecord:
    case: str
    reference_sha256: str
    reference_size: tuple[int, int]


@dataclass
class Metadata:
    candidate: Candidate
    architecture: str
    precision: str
    gpu: str
    torch: str
    spandrel: str
    tile: int
    halo: int
    load_seconds: float
    first_small_seconds: float
    sources: tuple[SourceRecord, ...]
    protected_changes: int | None = None
    alpha_changes: int | None = None
    tile_max_delta: int | None = None
    tile_mean_delta: float | None = None


def load_candidate(candidate: Candidate, precision: str) -> ImageModelDescriptor:
    from spandrel import ImageModelDescriptor, ModelLoader
    path = ROOT / "models/research-cache/fidelity-upscalers" / f"{candidate.name}.pth"
    if sha256(path.read_bytes()).hexdigest() != candidate.digest:
        raise ValueError(f"Checkpoint hash mismatch: {path}")
    state = torch.load(path, map_location="cpu", weights_only=True)
    model = ModelLoader().load_from_state_dict(state)
    if not isinstance(model, ImageModelDescriptor) or model.scale != 2:
        raise ValueError("Expected a native 2x image model")
    if precision == "bfloat16" and not model.supports_bfloat16:
        raise ValueError("Descriptor does not support BF16")
    return model.to(device="cuda", dtype=torch.bfloat16 if precision == "bfloat16" else torch.float32).eval()


def frozen_sources(output: Path) -> list[tuple[str, Image.Image]]:
    cases = quality_sources(output)
    path = ROOT / "data/upscale-fidelity-2026-10-10/assets/astronaut.png"
    if sha256(path.read_bytes()).hexdigest() != "88431cd9653ccd539741b555fb0a46b61558b301d4110412b5bc28b5e3ea6cb5":
        raise ValueError("Astronaut fixture changed")
    with Image.open(path) as source:
        astronaut = source.convert("RGB")
    with Image.open(ROOT / "benchmarks/fixtures/v1/portrait.png") as source:
        portrait = source.convert("RGB").crop((350, 80, 862, 592))
    cases.extend((("astronaut", astronaut), ("portrait", portrait)))
    for name, source in cases:
        source.save(output / "assets" / f"{name}-reference.png")
        source.resize((source.width//2, source.height//2), Image.Resampling.LANCZOS).save(output / "assets" / f"{name}-input.png")
    return cases


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", choices=tuple(item.name for item in CANDIDATES), required=True)
    parser.add_argument("--dependency-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--precision", choices=("float32", "bfloat16"), default="float32")
    parser.add_argument("--tile", type=int, default=256)
    parser.add_argument("--halo", type=int, default=64)
    args = parser.parse_args()
    if args.tile < 64 or args.tile % 8 or args.halo < 0 or args.halo % 8:
        parser.error("Tile >=64 and halo >=0 must be multiples of 8")
    sys.path.insert(0, str(args.dependency_path.resolve()))
    assert_api_unloaded()
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    cases = frozen_sources(output)
    candidate = next(item for item in CANDIDATES if item.name == args.candidate)
    start = time.perf_counter()
    model = load_candidate(candidate, args.precision)
    torch.cuda.synchronize()
    load_seconds = round(time.perf_counter()-start, 4)
    start = time.perf_counter()
    upscale(model, cases[1][1].resize((64, 64)), 2, args.tile, args.halo)
    torch.cuda.synchronize()
    first_seconds = round(time.perf_counter()-start, 4)
    metadata = Metadata(candidate, model.architecture.id, args.precision, torch.cuda.get_device_name(), torch.__version__,
                        version("spandrel"), args.tile, args.halo, load_seconds, first_seconds,
                        tuple(SourceRecord(name, sha256((output/"assets"/f"{name}-reference.png").read_bytes()).hexdigest(), image.size) for name, image in cases))
    records: list[Measurement] = []
    def save() -> None:
        (output/"results.json").write_text(json.dumps({"metadata": asdict(metadata), "measurements": [asdict(item) for item in records]}, indent=2)+"\n", encoding="utf-8")
    for size in (256, 1024):
        image = cases[1][1].resize((size, size), Image.Resampling.LANCZOS)
        result, record = timed(model, image, 2, args.tile, args.halo, 3)
        start = time.perf_counter()
        result.save(output/f"speed-{size}-2x.png")
        record = replace(record, case=f"speed-{size}", png_save_seconds=round(time.perf_counter()-start, 4))
        records.append(record)
        save()
        print(json.dumps(asdict(record)), flush=True)
    for name, reference in cases:
        for operation in ("reconstruct", "native"):
            image = reference.resize((reference.width//2, reference.height//2), Image.Resampling.LANCZOS) if operation == "reconstruct" else reference
            result, record = timed(model, image, 2, args.tile, args.halo, 1)
            lanczos = image.resize(result.size, Image.Resampling.LANCZOS)
            start = time.perf_counter()
            result.save(output/f"{name}-{operation}.png")
            png_seconds = round(time.perf_counter()-start, 4)
            lanczos.save(output/f"{name}-{operation}-lanczos.png")
            # Native export has no 2x ground truth; its PSNR measures only
            # consistency after resizing back, not generated high-res detail.
            comparison = result if operation == "reconstruct" else result.resize(reference.size, Image.Resampling.LANCZOS)
            baseline = lanczos if operation == "reconstruct" else lanczos.resize(reference.size, Image.Resampling.LANCZOS)
            record = replace(record, case=f"{name}-{operation}", png_save_seconds=png_seconds,
                             reference_psnr=psnr(comparison, reference), lanczos_psnr=psnr(baseline, reference))
            records.append(record)
            save()
            print(json.dumps(asdict(record)), flush=True)
    sample = cases[1][1].resize((256, 256), Image.Resampling.LANCZOS)
    full = upscale(model, sample, 2, 256, 64)
    tiled = upscale(model, sample, 2, 128, 64)
    full.save(output/"tile-full.png")
    tiled.save(output/"tile-split.png")
    delta = np.abs(np.asarray(full, dtype=np.int16)-np.asarray(tiled, dtype=np.int16))
    metadata.tile_max_delta, metadata.tile_mean_delta = int(delta.max()), round(float(delta.mean()), 5)
    metadata.protected_changes = protected_probe(model, output, args.tile, args.halo)
    alpha = Image.new("L", sample.size, 0)
    alpha.paste(255, (32, 32, 224, 224))
    sample.putalpha(alpha)
    rgba = upscale(model, sample, 2, args.tile, args.halo)
    expected = alpha.resize(rgba.size, Image.Resampling.LANCZOS)
    metadata.alpha_changes = int(np.count_nonzero(np.asarray(rgba.getchannel("A")) != np.asarray(expected)))
    if metadata.alpha_changes:
        raise AssertionError("Alpha differs from geometric resize")
    rgba.save(output/"alpha-2x.png")
    save()
    print(json.dumps(asdict(metadata)), flush=True)


if __name__ == "__main__":
    main()
