"""Run from backend: check upscaling between two real Quality edits, no reload."""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import sys
import time

import torch
import numpy as np
from PIL import Image

from benchmark_upscalers import ROOT, assert_api_unloaded, load_model, upscale
from benchmark_upscale_fidelity import CANDIDATES, load_candidate
from app.jobs import JobControl, JobManager
from app.models.flux2_klein_nvfp4 import Flux2Klein9bNvfp4Provider
from app.schemas import EditRequest


@dataclass(frozen=True)
class Result:
    stage: str
    seconds: float
    peak_allocated_gib: float
    peak_reserved_gib: float
    transformer_identity: int


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dependency-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--variant", nargs="+", choices=("x4v3-dn05", "x2plus", "swinir-light2"), default=("x4v3-dn05", "x2plus"))
    parser.add_argument("--tile", type=int, default=512)
    parser.add_argument("--halo", type=int, default=32)
    parser.add_argument("--release-unused-cache", action="store_true")
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
    (output / "recipe.json").write_text(json.dumps({"variants": args.variant, "tile": args.tile, "halo": args.halo,
                                                   "release_unused_cache": args.release_unused_cache,
                                                   "gpu": torch.cuda.get_device_name(), "torch": torch.__version__}, indent=2) + "\n", encoding="utf-8")
    provider = Flux2Klein9bNvfp4Provider()
    manager = JobManager()
    request = EditRequest(prompt="Keep the same man, face, red jacket and studio composition. Make the jacket a natural dark blue.", seed=7)
    source = ROOT / "benchmarks/fixtures/v2/man.png"
    results: list[Result] = []

    def record(stage: str, start: float) -> None:
        torch.cuda.synchronize()
        transformer = provider._pipes["text-to-image"].transformer
        result = Result(stage, round(time.perf_counter()-start, 4),
                        round(torch.cuda.max_memory_allocated()/2**30, 4),
                        round(torch.cuda.max_memory_reserved()/2**30, 4), id(transformer))
        results.append(result)
        (output / "results.json").write_text(json.dumps([asdict(item) for item in results], indent=2)+"\n", encoding="utf-8")
        print(json.dumps(asdict(result)), flush=True)

    def edit(stage: str) -> None:
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        def run(control: JobControl) -> None:
            provider.edit(request, source, output / f"{stage}.png", control)
        job = manager.submit("edit", run)
        while not manager.wait_for_completion(job.id, 5):
            status = manager.status(job.id)
            if status is not None:
                print(stage, status.phase, flush=True)
        status = manager.status(job.id)
        if status is None or status.status != "completed":
            raise RuntimeError(status.error if status else "Missing job")
        record(stage, start)

    try:
        edit("klein-first-edit")
        with Image.open(output / "klein-first-edit.png") as picture:
            image = picture.convert("RGB").resize((1024, 1024), Image.Resampling.LANCZOS)
        for variant in args.variant:
            if args.release_unused_cache:
                torch.cuda.reset_peak_memory_stats()
                start = time.perf_counter()
                torch.cuda.empty_cache()
                record("release-unused-cuda-cache", start)
            torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            if variant == "swinir-light2":
                candidate = next(item for item in CANDIDATES if item.name == variant)
                model = load_candidate(candidate, "float32")
            else:
                model = load_model(variant)
            record(f"{variant}-load-beside-klein", start)
            for index in range(3):
                torch.cuda.reset_peak_memory_stats()
                start = time.perf_counter()
                result = upscale(model, image, 2, args.tile, args.halo)
                record(f"{variant}-upscale-{index}", start)
            result.save(output / f"{variant}-2x.png")
            # A real second edit with the small upscaler resident tests both
            # memory pressure and unchanged Klein pipeline/transformer identity.
            edit(f"klein-edit-with-{variant}")
            with Image.open(output / "klein-first-edit.png") as first, Image.open(output / f"klein-edit-with-{variant}.png") as later:
                changed = int(np.count_nonzero(np.any(np.asarray(first) != np.asarray(later), axis=2)))
            (output / f"{variant}-pixel-equivalence.json").write_text(json.dumps({"changed_pixels": changed}) + "\n", encoding="utf-8")
            if changed:
                raise AssertionError(f"Klein output changed with {variant} resident: {changed} pixels")
            model.cpu()
            del model, result
            torch.cuda.empty_cache()
        if len({item.transformer_identity for item in results}) != 1:
            raise AssertionError("Klein transformer was replaced")
    finally:
        provider.unload()
        manager.shutdown()


if __name__ == "__main__":
    main()
