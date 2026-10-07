"""Build paired review sheets, an interactive local viewer and timing summary."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import statistics

from PIL import Image, ImageDraw, ImageFont
from pydantic import BaseModel

from benchmark_klein_precision import Measurement
from create_benchmark_fixtures import FIXTURES, Suite, digest


class Timing(BaseModel):
    median_seconds: float
    minimum_seconds: float
    maximum_seconds: float
    maximum_cuda_allocated_gib: float
    maximum_cuda_reserved_gib: float
    maximum_process_rss_gib: float


def timings(measure: Measurement, name: str | None = None) -> Timing:
    outcomes = [item for item in measure.results if name is None or item.fixture == name]
    if not outcomes or any(item.error or item.elapsed_seconds is None for item in outcomes):
        raise ValueError("Cannot summarize an incomplete or failed precision cohort.")
    seconds = [item.elapsed_seconds for item in outcomes if item.elapsed_seconds is not None]
    return Timing(median_seconds=statistics.median(seconds), minimum_seconds=min(seconds), maximum_seconds=max(seconds),
        maximum_cuda_allocated_gib=max(item.cuda_peak_allocated_gib or 0 for item in outcomes),
        maximum_cuda_reserved_gib=max(item.cuda_peak_reserved_gib or 0 for item in outcomes),
        maximum_process_rss_gib=max(item.process_peak_rss_gib or 0 for item in outcomes))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path(__file__).resolve().parents[2] / "data" / "nvfp4-benchmark")
    args = parser.parse_args()
    root = args.results.resolve()
    suite = Suite.model_validate_json((FIXTURES / "manifest.json").read_text(encoding="utf-8"))
    measures = {mode: Measurement.model_validate_json((root / mode / "results.json").read_text(encoding="utf-8")) for mode in ("bf16", "nvfp4")}
    expected = {(fixture.name, seed) for fixture in suite.fixtures for seed in suite.seeds}
    for mode, measure in measures.items():
        observed = [(result.fixture, result.seed) for result in measure.results]
        if set(observed) != expected or len(observed) != len(expected):
            raise ValueError(f"{mode} did not contain the complete, unique seed cohort.")
    if any(measure.manifest_sha256 != digest(FIXTURES / "manifest.json") for measure in measures.values()):
        raise ValueError("The precision runs do not match the current frozen fixture manifest.")
    baseline = {(result.fixture, result.seed): result for result in measures["bf16"].results}
    for result in measures["nvfp4"].results:
        if result.prompt != baseline[(result.fixture, result.seed)].prompt:
            raise ValueError("Paired precision outputs used different rendered prompts.")
        if result.nvfp4_kernel_calls != 400:
            raise ValueError("The NVFP4 cohort lacks the expected 100 native linears per step.")
    pairs = {name: {mode: timings(measure, name).model_dump() for mode, measure in measures.items()} for name in (fixture.name for fixture in suite.fixtures)}
    bf16, nvfp4 = timings(measures["bf16"]), timings(measures["nvfp4"])
    summary = {"outputs_per_precision": len(expected), "seeds": suite.seeds,
        "bf16": bf16.model_dump(), "nvfp4": nvfp4.model_dump(), "by_fixture": pairs,
        "median_time_reduction_percent": 100 * (1 - nvfp4.median_seconds / bf16.median_seconds),
        "median_speedup": bf16.median_seconds / nvfp4.median_seconds,
        "transformer_storage_gib": {mode: measure.transformer_storage_gib for mode, measure in measures.items()},
        "protected_pixel_changes": sum(item.changed_protected_pixels or 0 for measure in measures.values() for item in measure.results),
        "nvfp4_kernel_calls": sum(item.nvfp4_kernel_calls for item in measures["nvfp4"].results),
        "interpretation": "Four synthetic scenes, three seeds each; whole warmed component-offloaded pipeline, not pure kernel timing or a representative quality rate."}
    (root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    assets = root / "review-assets"
    assets.mkdir(exist_ok=True)
    font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 23)
    for fixture in suite.fixtures:
        shutil.copyfile(FIXTURES / fixture.image, assets / fixture.image)
        shutil.copyfile(FIXTURES / fixture.mask, assets / fixture.mask)
        left, top, right, bottom = fixture.mask_bbox
        box = (max(0, left - 90), max(0, top - 90), min(fixture.width, right + 90), min(fixture.height, bottom + 90))
        for source_name in (fixture.image, fixture.mask):
            with Image.open(assets / source_name) as image:
                image.crop(box).save(assets / (Path(source_name).stem + ".crop.png"))
        sheet = Image.new("RGB", (1800, 1280), (25, 30, 37))
        for row, mode in enumerate(("bf16", "nvfp4")):
            for column, seed in enumerate(suite.seeds):
                with Image.open(root / mode / f"{fixture.name}-{seed}.png") as full:
                    crop = full.crop(box)
                crop.save(assets / f"{mode}-{fixture.name}-{seed}.crop.png")
                crop.thumbnail((580, 580), Image.Resampling.LANCZOS)
                x, y = column * 600, row * 640
                sheet.paste(crop, (x + (600 - crop.width) // 2, y + 50 + (580 - crop.height) // 2))
                ImageDraw.Draw(sheet).text((x + 15, y + 15), f"{mode.upper()} · {fixture.name} · {seed}", font=font, fill="white")
        sheet.save(root / f"{fixture.name}-comparison.jpg", quality=95)
    payload = {"suite": suite.model_dump(mode="json"), "summary": summary,
               "runs": {mode: measure.model_dump(mode="json") for mode, measure in measures.items()}}
    template = Path(__file__).with_name("precision_review.html").read_text(encoding="utf-8")
    (root / "index.html").write_text(template.replace("__BENCHMARK_DATA__", json.dumps(payload).replace("</", "<\\/")), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
