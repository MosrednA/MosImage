"""Measure larger tiles on the frozen SwinIR Light 2x input, one GPU process."""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image
import torch

from benchmark_upscale_fidelity import CANDIDATES, load_candidate
from benchmark_upscalers import ROOT, Measurement, assert_api_unloaded, timed, upscale


@dataclass(frozen=True)
class TileDifference:
    tile: int
    max_delta: int
    mean_delta: float


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dependency-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--tiles", type=int, nargs="+", default=(256, 512, 1024))
    args = parser.parse_args()
    if args.width < 64 or args.height < 64 or any(tile < 64 or tile % 8 for tile in args.tiles):
        parser.error("Use dimensions >=64 and tiles >=64 divisible by 8")
    sys.path.insert(0, str(args.dependency_path.resolve()))
    assert_api_unloaded()
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    candidate = next(item for item in CANDIDATES if item.name == "swinir-light2")
    model = load_candidate(candidate, "float32")
    with Image.open(ROOT / "benchmarks/fixtures/v2/man.png") as source:
        image = source.convert("RGB").resize((args.width, args.height), Image.Resampling.LANCZOS)
    upscale(model, image.resize((64, 64)), 2, 256, 64)
    records: list[Measurement] = []
    comparisons: list[TileDifference] = []
    previous: Image.Image | None = None
    for tile in args.tiles:
        generated, record = timed(model, image, 2, tile, 64, 3)
        start = time.perf_counter()
        generated.save(output / f"tile-{tile}.png")
        record = replace(record, case=f"tile-{tile}", png_save_seconds=round(time.perf_counter() - start, 4))
        records.append(record)
        if previous is not None:
            delta = np.abs(np.asarray(previous, dtype=np.int16) - np.asarray(generated, dtype=np.int16))
            difference = TileDifference(tile, int(delta.max()), float(delta.mean()))
            comparisons.append(difference)
            print(json.dumps(asdict(difference)), flush=True)
        else:
            previous = generated
        payload = {"candidate": asdict(candidate), "precision": "float32", "halo": 64,
                   "gpu": torch.cuda.get_device_name(), "torch": torch.__version__,
                   "measurements": [asdict(item) for item in records],
                   "comparisons_against_256": [asdict(item) for item in comparisons]}
        (output / "results.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(asdict(record)), flush=True)


if __name__ == "__main__":
    main()
