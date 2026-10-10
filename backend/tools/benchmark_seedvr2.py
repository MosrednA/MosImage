"""Run the pinned standalone SeedVR2 3B recipe against the frozen SR cohort.

Run from backend. Vendor source, dependencies and model files stay isolated in
data/ and models/research-cache; this is not an application provider.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from hashlib import file_digest, sha256
import importlib
import json
import os
from pathlib import Path
import sys
from threading import Event, Thread
import time
from typing import Protocol, cast

# The vendor CLI sets its allocator before importing torch when used directly.
# This wrapper must also select it before torch is loaded.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "backend:native")

import numpy as np
from PIL import Image
import psutil
import torch

from benchmark_upscale_fidelity import frozen_sources, SourceRecord
from benchmark_upscalers import ROOT, assert_api_unloaded, psnr

VENDOR_REVISION = "4490bd1f482e026674543386bb2a4d176da245b9"
MODEL_NAME = "seedvr2_ema_3b_fp16.safetensors"
MODEL_DIGEST = "2fd0e03a3dad24e07086750360727ca437de4ecd456f769856e960ae93e2b304"
VAE_DIGEST = "20678548f420d98d26f11442d3528f8b8c94e57ee046ef93dbb7633da8612ca1"


class VendorDebug(Protocol):
    def __init__(self, enabled: bool) -> None: ...


class VendorCLI(Protocol):
    Debug: type[VendorDebug]

    def parse_arguments(self) -> argparse.Namespace: ...

    def _process_frames_core(self, frames_tensor: torch.Tensor, args: argparse.Namespace,
                             device_id: str, debug: VendorDebug,
                             runner_cache: dict[str, object]) -> torch.Tensor: ...


@dataclass(frozen=True)
class Record:
    case: str
    operation: str
    seed: int
    repeat: int
    input_size: tuple[int, int]
    output_size: tuple[int, int]
    seconds: float
    png_save_seconds: float
    observed_peak_allocated_gib: float
    observed_peak_reserved_gib: float
    sampled_peak_rss_gib: float
    reference_psnr: float
    lanczos_psnr: float
    output_file: str


@dataclass(frozen=True)
class Failure:
    case: str
    operation: str
    seed: int
    repeat: int
    seconds: float
    error_type: str
    error: str


@dataclass(frozen=True)
class Recipe:
    vendor_revision: str
    model: str
    model_sha256: str
    vae_sha256: str
    gpu: str
    torch: str
    import_seconds: float
    sources: tuple[SourceRecord, ...]
    allocator: str
    attention: str
    color_correction: str
    vae_tile: int
    vae_overlap: int
    blocks_to_swap: int
    offload_device: str
    input_noise: float
    latent_noise: float


class MemorySampler:
    """Observe phase peaks because vendor debug resets CUDA statistics itself."""

    def __init__(self) -> None:
        self.allocated = 0
        self.reserved = 0
        self.rss = 0
        self.stop = Event()
        self.thread = Thread(target=self.sample, daemon=True)

    def sample(self) -> None:
        process = psutil.Process()
        while not self.stop.is_set():
            self.allocated = max(self.allocated, torch.cuda.max_memory_allocated())
            self.reserved = max(self.reserved, torch.cuda.max_memory_reserved())
            self.rss = max(self.rss, process.memory_info().rss)
            self.stop.wait(0.05)

    def finish(self) -> None:
        self.allocated = max(self.allocated, torch.cuda.max_memory_allocated())
        self.reserved = max(self.reserved, torch.cuda.max_memory_reserved())
        self.stop.set()
        self.thread.join()


def verify_weights() -> None:
    folder = ROOT / "models/research-cache/SeedVR2-3B"
    for name, digest in ((MODEL_NAME, MODEL_DIGEST), ("ema_vae_fp16.safetensors", VAE_DIGEST)):
        with (folder / name).open("rb") as source:
            if file_digest(source, "sha256").hexdigest() != digest:
                raise ValueError(f"Checkpoint hash mismatch: {name}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vendor-path", type=Path, required=True)
    parser.add_argument("--dependency-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cases", nargs="+", choices=("face", "portrait", "astronaut", "denim", "garden", "text"),
                        default=("face", "portrait", "astronaut", "denim", "garden", "text"))
    parser.add_argument("--operations", nargs="+", choices=("reconstruct", "native"), default=("reconstruct", "native"))
    parser.add_argument("--seeds", nargs="+", type=int, default=(7,))
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--blocks-to-swap", type=int, default=0)
    parser.add_argument("--color-correction", choices=("lab", "none"), default="lab")
    args = parser.parse_args()
    if args.repeats < 1 or not 0 <= args.blocks_to_swap <= 32:
        parser.error("Repeats >=1 and swapped blocks in 0..32 required")
    if not args.vendor_path.name.endswith(VENDOR_REVISION):
        parser.error("Use the pinned vendor revision")
    sys.path.insert(0, str(args.dependency_path.resolve()))
    sys.path.insert(0, str(args.vendor_path.resolve()))
    assert_api_unloaded()
    verify_weights()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    all_sources = frozen_sources(output)
    sources = {name: source for name, source in all_sources}
    start = time.perf_counter()
    cli = cast(VendorCLI, importlib.import_module("inference_cli"))
    import_seconds = round(time.perf_counter() - start, 4)
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.cuda.synchronize()
    recipe = Recipe(VENDOR_REVISION, MODEL_NAME, MODEL_DIGEST, VAE_DIGEST, torch.cuda.get_device_name(), torch.__version__,
                    import_seconds, tuple(SourceRecord(name, sha256((output / "assets" / f"{name}-reference.png").read_bytes()).hexdigest(),
                                                        source.size) for name, source in all_sources),
                    torch.cuda.memory.get_allocator_backend(), "sdpa", args.color_correction, 512, 64, args.blocks_to_swap, "cpu", 0.0, 0.0)
    records: list[Record] = []
    failures: list[Failure] = []
    cache: dict[str, object] = {}
    original_argv = sys.argv
    sys.argv = [str(args.vendor_path / "inference_cli.py"), str(output / "assets/face-reference.png"),
                "--dit_model", MODEL_NAME, "--model_dir", str(ROOT / "models/research-cache/SeedVR2-3B"),
                "--batch_size", "1", "--resolution", "512", "--color_correction", args.color_correction,
                "--dit_offload_device", "cpu", "--vae_offload_device", "cpu", "--tensor_offload_device", "cpu",
                "--cache_dit", "--cache_vae", "--attention_mode", "sdpa", "--blocks_to_swap", str(args.blocks_to_swap),
                "--vae_encode_tiled", "--vae_encode_tile_size", "512", "--vae_encode_tile_overlap", "64",
                "--vae_decode_tiled", "--vae_decode_tile_size", "512", "--vae_decode_tile_overlap", "64"]
    try:
        vendor_args = cli.parse_arguments()
    finally:
        sys.argv = original_argv

    def save() -> None:
        (output / "results.json").write_text(json.dumps({"recipe": asdict(recipe), "compute_dtype": str(cache.get("ctx", {}).get("compute_dtype")) if isinstance(cache.get("ctx"), dict) else None,
                                                       "measurements": [asdict(item) for item in records],
                                                       "failures": [asdict(item) for item in failures]}, indent=2) + "\n", encoding="utf-8")

    save()
    for name in args.cases:
        reference = sources[name]
        for operation in args.operations:
            image = reference.resize((reference.width // 2, reference.height // 2), Image.Resampling.LANCZOS) if operation == "reconstruct" else reference
            expected = (image.width * 2, image.height * 2)
            baseline = image.resize(expected, Image.Resampling.LANCZOS)
            baseline.save(output / f"{name}-{operation}-lanczos.png")
            frames = torch.from_numpy(np.asarray(image, dtype=np.float32).copy() / 255.0).unsqueeze(0)
            for seed in args.seeds:
                vendor_args.seed = seed
                vendor_args.resolution = min(expected)
                for repeat in range(args.repeats):
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats()
                    sampler = MemorySampler()
                    sampler.thread.start()
                    start = time.perf_counter()
                    print(f"START {name} {operation} seed={seed} repeat={repeat} input={image.size} output={expected}", flush=True)
                    try:
                        result = cli._process_frames_core(frames, vendor_args, "0", cli.Debug(enabled=False), cache)
                        torch.cuda.synchronize()
                        seconds = round(time.perf_counter() - start, 4)
                        if result.shape != (1, expected[1], expected[0], 3) or not bool(torch.isfinite(result).all()):
                            raise ValueError(f"Invalid output shape/values: {tuple(result.shape)}")
                        sampler.finish()
                        pixels = result[0].clamp(0, 1).mul(255).round().byte().numpy()
                        generated = Image.fromarray(pixels)
                        filename = f"{name}-{operation}-seed{seed}-r{repeat}.png"
                        start_save = time.perf_counter()
                        generated.save(output / filename)
                        png_seconds = round(time.perf_counter() - start_save, 4)
                        comparison = generated if operation == "reconstruct" else generated.resize(reference.size, Image.Resampling.LANCZOS)
                        comparison_baseline = baseline if operation == "reconstruct" else baseline.resize(reference.size, Image.Resampling.LANCZOS)
                        record = Record(name, operation, seed, repeat, image.size, generated.size, seconds, png_seconds,
                                        round(sampler.allocated / 2**30, 4), round(sampler.reserved / 2**30, 4), round(sampler.rss / 2**30, 4),
                                        psnr(comparison, reference), psnr(comparison_baseline, reference), filename)
                        records.append(record)
                        save()
                        print(json.dumps(asdict(record)), flush=True)
                        del result
                    except Exception as exc:
                        sampler.finish()
                        failures.append(Failure(name, operation, seed, repeat, round(time.perf_counter() - start, 4), type(exc).__name__, str(exc)))
                        save()
                        raise


if __name__ == "__main__":
    main()
