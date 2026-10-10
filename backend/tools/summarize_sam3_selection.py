"""Validate all retained masks and build a paired, local SAM 3 review gallery."""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from hashlib import sha256
from html import escape
import json
from pathlib import Path
import statistics

import numpy as np
from PIL import Image, ImageDraw
from pydantic import BaseModel, ConfigDict

from benchmark_selection_baseline import Measurement


class Results(BaseModel):
    # Readers intentionally tolerate future measurement metadata.
    model_config = ConfigDict(extra="ignore")
    status: str
    device: str
    precision: str
    process_peak_working_set_bytes: int
    records: list[Measurement]


@dataclass(frozen=True)
class Summary:
    label: str
    route: str
    device: str
    precision: str
    uncached_median_seconds: float
    cached_median_seconds: float
    peak_process_ram_gib: float
    cuda_allocated_peak_gib: float
    cuda_reserved_peak_gib: float
    all_repeats_identical: bool


@dataclass(frozen=True)
class Agreement:
    case: str
    route: str
    cpu_vs_current_mask_iou: float
    cpu_vs_gpu_mask_iou: float
    cpu_vs_gpu_changed_pixels: int


def mask_agreement(first: np.ndarray, second: np.ndarray) -> float:
    union = np.logical_or(first, second).sum()
    return float(np.logical_and(first, second).sum() / union) if union else 1.0


def summarize(root: Path) -> None:
    baseline = root / "baseline-run2"
    current = Results.model_validate_json((baseline / "results.json").read_text())
    if current.status != "baseline_complete":
        raise ValueError("Incomplete baseline")
    names = [item.case for item in current.records if item.route == "point"]
    output = root / "review"
    output.mkdir(exist_ok=True)
    summaries: list[Summary] = []
    agreements: list[Agreement] = []
    body: list[str] = []
    masks_checked = 0
    for route in ("point", "text"):
        runs = [("Current CPU", baseline, current)]
        gpu_peaks: dict[str, tuple[int, int]] = {"Current CPU": (0, 0)}
        for label, directory in (("SAM 3 CPU", f"sam3-{route}-cpu"), ("SAM 3 GPU", f"sam3-{route}-gpu")):
            path = root / directory
            data = json.loads((path / "results.json").read_text())
            parsed = Results.model_validate(data)
            if parsed.status != "complete":
                raise ValueError(f"Incomplete run: {path}")
            runs.append((label, path, parsed))
            gpu_peaks[label] = (max(item["cuda_allocated_peak_bytes"] for item in data["records"]),
                                max(item["cuda_reserved_peak_bytes"] for item in data["records"]))
        for label, path, data in runs:
            records = [item for item in data.records if item.route == route]
            if [item.case for item in records] != names:
                raise ValueError(f"Case cohort differs: {path}")
            for record in records:
                with Image.open(path / f"{record.case}-{route}-mask.png") as opened:
                    mask = np.array(opened)
                with Image.open(baseline / f"{record.case}-source.png") as source:
                    if mask.shape != (source.height, source.width):
                        raise ValueError("Mask dimension mismatch")
                if not set(np.unique(mask)).issubset({0, 255}) or sha256(mask.tobytes()).hexdigest() != record.mask_sha256:
                    raise ValueError("Mask hash or binary coverage mismatch")
                masks_checked += 1
            allocated, reserved = gpu_peaks[label]
            summaries.append(Summary(label, route, data.device, data.precision,
                                     statistics.median(item.uncached_seconds for item in records),
                                     statistics.median(value for item in records for value in item.cached_seconds),
                                     data.process_peak_working_set_bytes / 1024**3,
                                     allocated / 1024**3, reserved / 1024**3,
                                     all(item.repeats_identical for item in records)))
        body.append(f"<h2>{escape(route.title())} selection</h2>")
        for page in range(3):
            subset = names[page*3:page*3+3]
            sheet = Image.new("RGB", (1600, len(subset)*440), "#20242c")
            draw = ImageDraw.Draw(sheet)
            for row, name in enumerate(subset):
                body.append(f"<section><h3>{escape(name)}</h3><div class='images'>")
                columns = [("Source", baseline / f"{name}-source.png")]
                columns.extend((label, path / f"{name}-{route}-overlay.png") for label, path, _ in runs)
                for col, (label, path) in enumerate(columns):
                    url = "../" + path.relative_to(root).as_posix()
                    mask_url = url.replace("-overlay.png", "-mask.png")
                    body.append(f"<figure><figcaption>{escape(label)}</figcaption><a href='{escape(url)}' target='_blank'><img src='{escape(url)}' data-overlay='{escape(url)}' data-mask='{escape(mask_url)}' loading='lazy'></a></figure>")
                    with Image.open(path) as opened:
                        image = opened.convert("RGB")
                    image.thumbnail((390, 400))
                    sheet.paste(image, (col*400, row*440+30))
                    draw.text((col*400+5, row*440+8), f"{name}: {label}", fill="white")
                masks = [np.array(Image.open(path / f"{name}-{route}-mask.png")) > 0 for _, path, _ in runs]
                agreements.append(Agreement(name, route, mask_agreement(masks[0], masks[1]),
                                            mask_agreement(masks[1], masks[2]), int(np.count_nonzero(masks[1] != masks[2]))))
                body.append("</div></section>")
            sheet.save(output / f"{route}-comparison-{page+1}.jpg", quality=95)
    rows = []
    for item in summaries:
        rows.append(f"<tr><td>{escape(item.label)} / {item.route}</td><td>{item.uncached_median_seconds:.3f} s</td><td>{item.cached_median_seconds:.3f} s</td><td>{item.peak_process_ram_gib:.2f} GiB</td><td>{item.cuda_allocated_peak_gib:.2f} GiB</td></tr>")
    html = """<!doctype html><html lang='en'><meta charset='utf-8'><title>SAM 3 selection comparison</title>
<style>body{background:#15191f;color:#e4eaf1;font:16px system-ui;margin:28px auto;max-width:1600px;padding:0 20px}h1{margin-bottom:8px}p{color:#b5c0cf;max-width:1050px;line-height:1.6}table{border-collapse:collapse;margin:24px 0}td,th{text-align:left;padding:10px 18px;border-bottom:1px solid #364252}button{background:#263b4e;color:white;border:1px solid #61829e;padding:10px 20px;border-radius:6px;cursor:pointer}.images{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}figure{margin:0;background:#20242c}figcaption{padding:10px}img{width:100%;height:440px;object-fit:contain}section{margin:26px 0}@media(max-width:900px){.images{grid-template-columns:repeat(2,1fr)}img{height:340px}}</style>
<h1>SAM 3 against current MosImage selection</h1><p>Seven frozen images and identical point/text prompts. Current: SAM 2.1 Tiny and Grounding DINO Tiny on CPU FP32. SAM 3: CPU FP32, then a separate GPU BF16 option. One new-image prediction plus three cached repeats. No object-mask ground truth: inspect the contours; timing and repeatability alone do not prove quality. Click an image to open its native-resolution overlay.</p>
<button id='toggle'>Show masks</button><table><thead><tr><th>Route</th><th>New image</th><th>Cached image</th><th>Peak process RAM</th><th>Peak CUDA allocation</th></tr></thead><tbody>"""
    html += "".join(rows) + "</tbody></table>" + "".join(body)
    html += """<script>let masks=false;document.querySelector('#toggle').onclick=()=>{masks=!masks;document.querySelectorAll('img[data-overlay]').forEach(img=>{img.src=masks?img.dataset.mask:img.dataset.overlay;img.parentElement.href=img.src;});document.querySelector('#toggle').textContent=masks?'Show overlays':'Show masks';};</script></html>"""
    (output / "index.html").write_text(html, encoding="utf-8")
    payload = {"masks_validated": masks_checked, "summaries": [asdict(item) for item in summaries],
               "mask_agreement_is_not_ground_truth_accuracy": True, "agreements": [asdict(item) for item in agreements]}
    (output / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    args = parser.parse_args()
    summarize(args.input.resolve())
