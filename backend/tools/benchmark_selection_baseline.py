"""Freeze and measure the current CPU selection routes before a SAM 3 comparison."""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import statistics
import sys
from time import perf_counter
from typing import Literal

import numpy as np
from PIL import Image, ImageDraw
import psutil
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))


@dataclass(frozen=True)
class Case:
    name: str
    source: str
    text: str
    point: tuple[int, int]
    criteria: str


CASES = (
    Case("sphere", "benchmarks/fixtures/v1/garden.png", "terracotta sphere", (375, 865), "Sphere, with flowers and leaves excluded"),
    Case("mug", "benchmarks/fixtures/v1/kitchen.png", "yellow mug", (795, 780), "Body and handle, with the handle hole excluded"),
    Case("brooch", "benchmarks/fixtures/v1/portrait.png", "brooch", (795, 711), "Small coral brooch, without selecting the shirt"),
    Case("poster", "benchmarks/fixtures/v1/architecture.png", "poster", (420, 488), "Paper artwork, excluding brick and frame"),
    Case("man", "benchmarks/fixtures/v2/man.png", "person", (416, 610), "Complete person, including hands and shoes"),
    Case("phone", "benchmarks/fixtures/v2/phone.png", "smartphone", (631, 590), "Phone, excluding the hand and street"),
    Case("astronaut", "data/upscale-fidelity-2026-10-10/assets/astronaut.png", "person", (245, 330), "Person, separating the background and helmet"),
)


@dataclass(frozen=True)
class Source:
    case: Case
    sha256: str
    size: tuple[int, int]


@dataclass(frozen=True)
class Measurement:
    case: str
    route: Literal["point", "text"]
    uncached_seconds: float
    cached_seconds: tuple[float, ...]
    selected_pixels: int
    mask_sha256: str
    repeats_identical: bool
    detections: int | None
    error: str | None


def overlay(image: Image.Image, mask: np.ndarray) -> Image.Image:
    pixels = np.array(image.convert("RGB"))
    selected = mask > 0
    pixels[selected] = (pixels[selected] * 0.55 + np.array([40, 190, 255]) * 0.45).astype(np.uint8)
    return Image.fromarray(pixels)


def cached_revision(model_id: str) -> str:
    from huggingface_hub import try_to_load_from_cache
    config_path = try_to_load_from_cache(model_id, "config.json")
    if not isinstance(config_path, str):
        raise RuntimeError(f"Loaded model config missing from cache: {model_id}")
    return Path(config_path).parent.name


def run(output: Path, repeats: int) -> None:
    # Do not import app.main or initialize the editor database. Configuration,
    # generated masks and temporary files belong to this benchmark only.
    output.mkdir(parents=True, exist_ok=False)
    for key, value in {
        "MOSIMAGE_DATA_DIR": str(output),
        "MOSIMAGE_DB_PATH": str(output / "unused.db"),
        "MOSIMAGE_IMAGE_DIR": str(output / "images"),
        "MOSIMAGE_WARM_UP_ON_START": "false",
    }.items():
        os.environ[key] = value
    # Settings reads .env relative to cwd. Research must not consume an unrelated
    # workspace/global .env or print its validation errors (which can contain keys).
    os.chdir(output)

    from app.detection import detector, strongest
    from app.selection import ObjectSelector

    sources = []
    for case in CASES:
        path = ROOT / case.source
        with Image.open(path) as uploaded:
            size = uploaded.size
        if not (0 <= case.point[0] < size[0] and 0 <= case.point[1] < size[1]):
            raise ValueError(f"Point outside {case.name}")
        sources.append(Source(case, sha256(path.read_bytes()).hexdigest(), size))
    (output / "manifest.json").write_text(json.dumps([asdict(item) for item in sources], indent=2), encoding="utf-8")
    selector = ObjectSelector()
    load_start = perf_counter()
    selector.warm_up()
    sam_load = perf_counter() - load_start
    sam_revision = cached_revision(selector.model_id)
    load_start = perf_counter()
    detector._ensure_model()
    dino_load = perf_counter() - load_start
    print(json.dumps({"sam_load_seconds": sam_load, "dino_load_seconds": dino_load}), flush=True)
    with Image.open(ROOT / CASES[0].source) as uploaded:
        warm_image = uploaded.convert("RGB")
    # One warm-up per route, excluded from all timing and visual records.
    selector._predict(warm_image, "warmup", [CASES[0].point], [1], multimask=True)
    detector.detect(warm_image, CASES[0].text)

    records: list[Measurement] = []
    for source in sources:
        case = source.case
        with Image.open(ROOT / case.source) as uploaded:
            image = uploaded.convert("RGB")
        image.save(output / f"{case.name}-source.png")
        for route in ("point", "text"):
            selector._image_key = None
            times: list[float] = []
            masks: list[np.ndarray] = []
            detection_count = None
            error = None
            for _ in range(repeats + 1):
                start = perf_counter()
                if route == "point":
                    mask = selector._predict(image, source.sha256, [case.point], [1], multimask=True)
                else:
                    found = strongest(detector.detect(image, case.text))
                    detection_count = len(found)
                    if found:
                        mask = selector._predict_boxes(image, source.sha256, [item.box for item in found])
                    else:
                        # Preserve a production no-match result in the gallery.
                        mask = np.zeros((image.height, image.width), dtype=np.uint8)
                        error = "No detection at production thresholds"
                times.append(perf_counter() - start)
                masks.append(mask)
            stem = f"{case.name}-{route}"
            Image.fromarray(masks[0]).save(output / f"{stem}-mask.png")
            overlay(image, masks[0]).save(output / f"{stem}-overlay.png")
            records.append(Measurement(case.name, route, times[0], tuple(times[1:]), int((masks[0] > 0).sum()),
                                       sha256(masks[0].tobytes()).hexdigest(), all(np.array_equal(masks[0], mask) for mask in masks[1:]), detection_count, error))
            print(json.dumps(asdict(records[-1])), flush=True)
            write_results(output, records, sam_load, dino_load, sam_revision)
    sheet = Image.new("RGB", (960, len(sources) * 350), "#20242c")
    draw = ImageDraw.Draw(sheet)
    for row, source in enumerate(sources):
        for col, suffix in enumerate(("source", "point-overlay", "text-overlay")):
            image = Image.open(output / f"{source.case.name}-{suffix}.png").convert("RGB")
            image.thumbnail((310, 315))
            sheet.paste(image, (col * 320, row * 350 + 30))
            draw.text((col * 320 + 6, row * 350 + 8), f"{source.case.name}: {suffix}", fill="white")
    sheet.save(output / "contact-sheet.jpg", quality=95)
    for route in ("point", "text"):
        subset = [item for item in records if item.route == route]
        print(route, "uncached median", statistics.median(item.uncached_seconds for item in subset),
              "cached median", statistics.median(value for item in subset for value in item.cached_seconds), flush=True)


def write_results(output: Path, records: list[Measurement], sam_load: float, dino_load: float, sam_revision: str | None) -> None:
    from app.detection import detector
    from transformers import __version__ as transformers_version
    memory = psutil.Process().memory_info()
    payload = {
        "status": "baseline_complete" if len(records) == len(CASES) * 2 else "running",
        "measured_at_utc": datetime.now(timezone.utc).isoformat(),
        "torch": torch.__version__, "transformers": transformers_version,
        "cpu_threads": torch.get_num_threads(), "device": "cpu", "precision": "float32",
        "sam_model": "facebook/sam2.1-hiera-tiny", "dino_model": detector.model_id,
        "sam_revision": sam_revision,
        "dino_revision": cached_revision(detector.model_id),
        "selection_code_sha256": sha256((ROOT / "backend/app/selection.py").read_bytes()).hexdigest(),
        "detection_code_sha256": sha256((ROOT / "backend/app/detection.py").read_bytes()).hexdigest(),
        "sam_load_seconds": sam_load, "dino_load_seconds": dino_load,
        "rss_bytes": memory.rss, "process_peak_working_set_bytes": getattr(memory, "peak_wset", None),
        "timing_scope": "Production prediction plus preprocessing/postprocessing; excludes image decoding, model loading, PNG encoding, API and job queue",
        "records": [asdict(item) for item in records],
    }
    (output / "results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    run(args.output.resolve(), args.repeats)
