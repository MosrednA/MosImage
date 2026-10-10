"""Verify retained masks against official labels and create a local review gallery."""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from hashlib import sha256
from html import escape
import json
import os
from pathlib import Path
import statistics

import numpy as np
from PIL import Image, ImageDraw
from pydantic import BaseModel, ConfigDict, TypeAdapter

from benchmark_selection_baseline import ROOT, overlay
from benchmark_selection_refinement import Results, Selection, mask_iou
from prepare_selection_refinement import load_cases
from probe_selection_refinement import Probe, ProbeRecord


@dataclass(frozen=True)
class RouteSummary:
    model: str
    route: str
    count: int
    mean_iou: float
    median_iou: float
    minimum_iou: float
    mean_precision: float
    mean_recall: float
    first_median_seconds: float
    repeat_median_seconds: float | None
    nonempty_selections: int
    exact_repeats: int


@dataclass(frozen=True)
class PairedComparison:
    route: str
    mean_paired_iou_change: float
    sam3_better_by_over_0_02: int
    current_better_by_over_0_02: int
    within_0_02: int


@dataclass(frozen=True)
class ResourceSummary:
    peak_ram_gib: float
    cuda_allocated_gib: float
    cuda_reserved_gib: float
    load_seconds: float
    warmup_seconds: float


class ProbeResults(BaseModel):
    model_config = ConfigDict(extra="ignore")
    status: str
    model: str
    manifest_sha256: str
    records: list[ProbeRecord]


def validate_mask(path: Path, expected_size: tuple[int, int], expected_hash: str) -> np.ndarray:
    pixels = np.array(Image.open(path))
    if pixels.shape != (expected_size[1], expected_size[0]) or not set(np.unique(pixels)) <= {0, 255}:
        raise ValueError(f"Invalid binary mask: {path}")
    if sha256(pixels.tobytes()).hexdigest() != expected_hash:
        raise ValueError(f"Mask changed: {path}")
    return pixels > 0


def local_ref(path: Path, output: Path) -> str:
    return Path(os.path.relpath(path, output)).as_posix()


def tile(label: str, path: Path, output: Path, mask_path: Path | None = None) -> str:
    ref = escape(local_ref(path, output), quote=True)
    mask = escape(local_ref(mask_path or path, output), quote=True)
    return f'<figure><a href="{ref}" target="_blank"><img loading="lazy" src="{ref}" data-overlay="{ref}" data-mask="{mask}"></a><figcaption>{escape(label)}</figcaption></figure>'


def contact_sheet(output: Path, name: str, rows: list[list[tuple[str, Path]]]) -> None:
    columns = max(len(row) for row in rows)
    for page in range(0, len(rows), 3):
        subset = rows[page:page + 3]
        sheet = Image.new("RGB", (360 * columns, 390 * len(subset)), "#20242c")
        draw = ImageDraw.Draw(sheet)
        for row, cells in enumerate(subset):
            for col, (label, path) in enumerate(cells):
                image = Image.open(path).convert("RGB")
                image.thumbnail((350, 340))
                left, top = col * 360, row * 390
                sheet.paste(image, (left, top + 45))
                draw.text((left + 4, top + 6), label[:48], fill="white")
        sheet.save(output / f"{name}-{page // 3 + 1}.jpg", quality=95)


def summarize(root: Path) -> None:
    cases = load_cases(root / "manifest.json")
    for case in cases:
        if sha256((ROOT / case.source).read_bytes()).hexdigest() != case.source_sha256 or sha256((ROOT / case.annotation).read_bytes()).hexdigest() != case.annotation_sha256:
            raise ValueError(f"Frozen source or annotation changed: {case.name}")
    by_name = {case.name: case for case in cases}
    output = root / "review"
    output.mkdir(exist_ok=True)
    selected = Selection.model_validate_json((root / "threshold-selection.json").read_text())
    selection_hash = sha256((root / "threshold-selection.json").read_bytes()).hexdigest()
    manifest_hash = sha256((root / "manifest.json").read_bytes()).hexdigest()
    summaries: list[RouteSummary] = []
    loaded: dict[str, Results] = {}
    checked = 0
    for directory in ("current-calibration", "sam3-text-calibration", "current-holdout", "sam3-text-holdout", "sam3-point-holdout"):
        data = Results.model_validate_json((root / directory / "results.json").read_text())
        if data.status != "complete" or data.manifest_sha256 != manifest_hash:
            raise ValueError(f"Incomplete or mismatched run: {directory}")
        if data.split == "holdout" and data.threshold_selection_sha256 != selection_hash:
            raise ValueError(f"Holdout did not use the frozen thresholds: {directory}")
        expected_count = 72 if data.split == "calibration" else 120 if data.model == "current" else 108 if data.model == "sam3-text" else 48
        if len(data.records) != expected_count:
            raise ValueError(f"Wrong record count: {directory}")
        for record in data.records:
            if Path(record.artifact).name != record.artifact:
                raise ValueError("Mask artifact must be a local basename")
            case = by_name[record.case]
            mask = validate_mask(root / directory / record.artifact, case.size, record.mask_sha256)
            annotation = np.array(Image.open(ROOT / case.annotation))
            target = annotation == case.target_id if record.target == "largest_person" else annotation > 0 if record.target == "all_people" else np.zeros_like(mask)
            if abs(mask_iou(mask, target) - record.iou) > 1e-12:
                raise ValueError(f"Metric mismatch: {directory}/{record.artifact}")
            if record.split != data.split or case.split != data.split:
                raise ValueError("Calibration and holdout mixed")
            checked += 1
        if data.split == "holdout":
            loaded[data.model] = data
            for route in dict.fromkeys(record.route for record in data.records):
                records = [record for record in data.records if record.route == route]
                repeats = [record.repeat_seconds for record in records if record.repeat_seconds is not None]
                summaries.append(RouteSummary(data.model, route, len(records), statistics.mean(record.iou for record in records),
                                               statistics.median(record.iou for record in records), min(record.iou for record in records),
                                               statistics.mean(record.precision for record in records), statistics.mean(record.recall for record in records),
                                               statistics.median(record.seconds for record in records), statistics.median(repeats) if repeats else None,
                                               sum(record.selected_pixels > 0 for record in records), sum(record.repeat_identical is True for record in records)))
    body = ['<h1>SAM 3 refinement and real-photo holdout</h1><p>Six calibration images; twelve untouched evaluation images. Official Penn-Fudan masks. '
            'Point routes target the largest person; text routes target all annotated people. Clicks and boxes come from annotations and simulate ideal user input. '
            'The dataset tests people, not general editor accuracy.</p><p><strong>Annotation limits:</strong> some visible people are unlabelled, some carried bags are included, '
            'and overlapping people can have ambiguous instance boundaries. A correct extra person detection can therefore lower text IoU. '
            'These scores measure agreement with the provided labels, not complete semantic accuracy.</p><button id="toggle">Show masks</button>']
    body.append('<h2>Measured holdout results</h2><table><tr><th>Model / route</th><th>Mean IoU</th><th>Min IoU</th><th>First / repeat seconds</th><th>Nonempty</th></tr>')
    for summary in summaries:
        body.append(f'<tr><td>{summary.model} / {summary.route}</td><td>{summary.mean_iou:.3f}</td><td>{summary.minimum_iou:.3f}</td>'
                    f'<td>{summary.first_median_seconds:.3f} / {summary.repeat_median_seconds:.3f}</td><td>{summary.nonempty_selections}/{summary.count}</td></tr>')
    body.append('</table>')
    lookup = {(model, record.case, record.route): record for model, result in loaded.items() for record in result.records}
    comparisons: list[PairedComparison] = []
    for route in ("point", "point_refined", "point_feedback", "box", "text_default", "text_tuned"):
        candidate = "sam3-text" if route.startswith("text") else "sam3-point"
        differences = [lookup[candidate, case.name, route].iou - lookup["current", case.name, route].iou for case in cases if case.split == "holdout"]
        comparisons.append(PairedComparison(route, statistics.mean(differences), sum(value > 0.02 for value in differences),
                                            sum(value < -0.02 for value in differences), sum(abs(value) <= 0.02 for value in differences)))
    sections = (
        ("Single click", "largest_person", (("current", "point"), ("sam3-point", "point"))),
        ("Three positive clicks plus background click", "largest_person", (("current", "point_refined"), ("sam3-point", "point_refined"))),
        ("Refinement with previous mask logits", "largest_person", (("current", "point_feedback"), ("sam3-point", "point_feedback"))),
        ("Same user box", "largest_person", (("current", "box"), ("sam3-point", "box"))),
        ("Default text thresholds", "all_people", (("current", "text_default"), ("sam3-text", "text_default"))),
        ("Calibrated text thresholds", "all_people", (("current", "text_tuned"), ("sam3-text", "text_tuned"))),
        ("SAM 3 visual concept prompts", "all_people", (("sam3-text", "text_tuned"), ("sam3-text", "text_positive_box"), ("sam3-text", "text_negative_box"), ("sam3-text", "text_native_resize"))),
        ("Absent zebra at calibrated thresholds", "empty", (("current", "absent_tuned"), ("sam3-text", "absent_tuned"))),
        ("Absent astronaut at calibrated thresholds", "empty", (("current", "absent_hard_tuned"), ("sam3-text", "absent_hard_tuned"))),
    )
    for section_index, (title, target_name, routes) in enumerate(sections):
        body.append(f'<h2>{escape(title)}</h2>')
        sheet_rows: list[list[tuple[str, Path]]] = []
        for case in cases:
            if case.split != "holdout":
                continue
            source_path = ROOT / case.source
            annotation = np.array(Image.open(ROOT / case.annotation))
            target = annotation == case.target_id if target_name == "largest_person" else annotation > 0 if target_name == "all_people" else np.zeros_like(annotation, dtype=bool)
            gt_stem = f"{case.name}-{target_name}"
            gt_mask, gt_overlay = output / f"{gt_stem}-mask.png", output / f"{gt_stem}-overlay.png"
            Image.fromarray(target.astype(np.uint8) * 255).save(gt_mask)
            overlay(Image.open(source_path), target).save(gt_overlay)
            cells = [(f"{case.name}: source", source_path), ("Official target", gt_overlay)]
            body.append(f'<h3>{case.name}</h3><div class="row">' + tile("Source", source_path, output) + tile("Official target", gt_overlay, output, gt_mask))
            for model, route in routes:
                record = lookup[model, case.name, route]
                path = root / f"{model}-holdout" / record.artifact
                preview = path.with_name(path.name.replace("-mask", "-overlay"))
                label = f"{model} / {route}: IoU {record.iou:.3f}"
                cells.append((label, preview))
                body.append(tile(label, preview, output, path))
            body.append('</div>')
            sheet_rows.append(cells)
        contact_sheet(output, f"holdout-{section_index + 1}", sheet_rows)
    body.append('<h2>Original seven-image diagnostics</h2><p>Exploratory prompts and manual corrections. These have no ground-truth accuracy score. '
                'Every variation is retained below; no per-image winning recipe is treated as a production result.</p>')
    probes = TypeAdapter(list[Probe], config=ConfigDict(extra="forbid")).validate_json((root / "diagnostic-manifest.json").read_text())
    probe_by_name = {probe.case.name: probe.case for probe in probes}
    diagnostic_loaded: dict[str, ProbeResults] = {}
    for model in ("current", "sam3-point", "sam3-text"):
        data = ProbeResults.model_validate_json((root / f"{model}-diagnostic/results.json").read_text())
        expected_records = 76 if model == "current" else 28 if model == "sam3-point" else 69
        if data.status != "complete" or len(data.records) != expected_records or data.manifest_sha256 != sha256((root / "diagnostic-manifest.json").read_bytes()).hexdigest():
            raise ValueError(f"Diagnostic run incomplete: {model}")
        diagnostic_loaded[model] = data
        for record in data.records:
            case = probe_by_name[record.case]
            validate_mask(root / f"{model}-diagnostic" / record.artifact, case.size, record.mask_sha256)
            checked += 1
    point_sheet_rows: list[list[tuple[str, Path]]] = []
    text_sheet_rows: list[list[tuple[str, Path]]] = []
    for item in probes:
        name, source = item.case.name, ROOT / item.case.source
        body.append(f'<h3>{name}</h3>' + tile("Source", source, output))
        for model, data in diagnostic_loaded.items():
            body.append(f'<details><summary>{model} — all variations</summary><div class="row">')
            for record in data.records:
                if record.case != name:
                    continue
                mask_path = root / f"{model}-diagnostic" / record.artifact
                preview = mask_path.with_name(mask_path.name.replace("-mask", "-overlay"))
                label = f"{record.route} | {record.text or 'visual'} | {record.threshold} | {record.selected_pixels} px"
                body.append(tile(label, preview, output, mask_path))
            body.append('</div></details>')
        point_cells = [(f"{name}: source", source)]
        for model, route in (("current", "point_refined"), ("sam3-point", "point_refined"), ("current", "point_feedback"), ("sam3-point", "point_feedback")):
            record = next(record for record in diagnostic_loaded[model].records if record.case == name and record.route == route)
            path = root / f"{model}-diagnostic" / record.artifact.replace("-mask", "-overlay")
            point_cells.append((f"{model}: {route}", path))
        point_sheet_rows.append(point_cells)
        text_cells = [(f"{name}: source", source)]
        for model, route, text, threshold in (("current", "text", item.texts[0], 0.3), ("sam3-text", "text", item.texts[0], 0.5),
                                             ("sam3-text", "text", item.texts[0], 0.1), ("sam3-text", "text", item.texts[1], 0.5),
                                             ("sam3-text", "text_positive_box", item.texts[0], 0.5)):
            record = next(record for record in diagnostic_loaded[model].records if record.case == name and record.route == route and record.text == text and record.threshold == threshold)
            path = root / f"{model}-diagnostic" / record.artifact.replace("-mask", "-overlay")
            text_cells.append((f"{model}: {text} / {threshold} / {route}", path))
        text_sheet_rows.append(text_cells)
    contact_sheet(output, "diagnostic-point", point_sheet_rows)
    contact_sheet(output, "diagnostic-text", text_sheet_rows)
    script = """let masks=false; document.querySelector('#toggle').addEventListener('click',()=>{
      masks=!masks; document.querySelector('#toggle').textContent=masks?'Show overlays':'Show masks';
      for(const image of document.querySelectorAll('img[data-mask]')){
        image.src=masks?image.dataset.mask:image.dataset.overlay; image.parentElement.href=image.src;
      }
    });"""
    html = '<!doctype html><html lang="en"><meta charset="utf-8"><title>SAM 3 refinement</title><style>' \
           'body{background:#171b23;color:#eef2f8;font:15px system-ui;margin:24px}h2{margin-top:40px}button{position:sticky;top:12px;padding:10px 20px;z-index:1}' \
           '.row{display:flex;gap:12px;overflow:auto}figure{margin:0 0 16px;min-width:250px;width:290px}img{width:100%;height:300px;object-fit:contain;background:#252b35}' \
           'figcaption{font-size:12px;padding:6px}table{border-collapse:collapse}td,th{padding:8px;border:1px solid #505968;text-align:left}summary{cursor:pointer;padding:10px}</style>' \
           + ''.join(body) + '<script>' + script + '</script></html>'
    (output / "index.html").write_text(html, encoding="utf-8")
    (output / "gallery-controller.js").write_text(script, encoding="utf-8")
    payload = {"masks_verified": checked, "threshold_selection": selected.model_dump(), "holdout": [asdict(summary) for summary in summaries],
               "paired_comparisons": [asdict(comparison) for comparison in comparisons],
               "resources": {model: asdict(ResourceSummary(data.peak_ram_bytes / 2 ** 30, data.cuda_allocated_bytes / 2 ** 30,
                                                           data.cuda_reserved_bytes / 2 ** 30, data.load_seconds, data.warmup_seconds))
                             for model, data in loaded.items()}}
    (output / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    summarize(parser.parse_args().root.resolve())
