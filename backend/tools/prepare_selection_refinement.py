"""Freeze a Penn-Fudan calibration/holdout cohort before any model inference."""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path
import shutil
from typing import Literal
from zipfile import ZipFile

import numpy as np
from PIL import Image, ImageDraw
from pydantic import ConfigDict, TypeAdapter
import requests

from benchmark_selection_baseline import ROOT, overlay

DATASET_URL = "https://www.cis.upenn.edu/~jshi/ped_html/PennFudanPed.zip"
SPLIT_SEED = "mosimage-sam3-refinement-v1"
THRESHOLDS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6)


@dataclass(frozen=True)
class RefinementCase:
    name: str
    split: Literal["calibration", "holdout", "diagnostic"]
    source: str
    source_sha256: str
    annotation: str
    annotation_sha256: str
    size: tuple[int, int]
    target_id: int
    points: tuple[tuple[int, int], ...]
    labels: tuple[int, ...]
    box: tuple[int, int, int, int]
    negative_box: tuple[int, int, int, int]


def load_cases(path: Path) -> list[RefinementCase]:
    return TypeAdapter(list[RefinementCase], config=ConfigDict(extra="forbid")).validate_json(path.read_text())


def row_point(mask: np.ndarray, fraction: float) -> tuple[int, int]:
    ys, _ = np.where(mask)
    target_y = int(ys.min() + (ys.max() - ys.min()) * fraction)
    rows = np.flatnonzero(mask.any(axis=1))
    y = int(rows[np.argmin(abs(rows - target_y))])
    columns = np.flatnonzero(mask[y])
    # A median *foreground pixel*, so gaps between legs never become a positive.
    x = int(columns[len(columns) // 2])
    return x, y


def empty_box(mask: np.ndarray) -> tuple[int, int, int, int]:
    height, width = mask.shape
    side = min(24, height // 6, width // 6)
    for y in range(0, height - side, side):
        for x in range(0, width - side, side):
            if not mask[y:y + side, x:x + side].any():
                return x, y, x + side, y + side
    raise ValueError("No annotated-background patch found")


def prepare(output: Path) -> None:
    output.mkdir(parents=True, exist_ok=False)
    archive = output / "PennFudanPed.zip"
    with requests.get(DATASET_URL, stream=True, timeout=(15, 60)) as response:
        response.raise_for_status()
        with archive.open("wb") as stream:
            for chunk in response.iter_content(1024 * 1024):
                stream.write(chunk)
    with ZipFile(archive) as dataset:
        names = sorted(name for name in dataset.namelist() if name.startswith("PennFudanPed/PNGImages/") and name.endswith(".png"))
        selected: list[tuple[str, str]] = []
        for prefix in ("FudanPed", "PennPed"):
            group = [name for name in names if Path(name).name.startswith(prefix)]
            group.sort(key=lambda name: sha256(f"{SPLIT_SEED}:{Path(name).name}".encode()).hexdigest())
            if len(group) < 9:
                raise ValueError(f"Missing dataset group: {prefix}")
            selected.extend((name, "calibration" if index < 3 else "holdout") for index, name in enumerate(group[:9]))
        cases: list[RefinementCase] = []
        for name, split in selected:
            stem = Path(name).stem
            annotation_name = f"PennFudanPed/PedMasks/{stem}_mask.png"
            paths: list[Path] = []
            for member in (name, annotation_name):
                # Only known PNG basenames are copied. Never extract arbitrary ZIP paths.
                path = output / "sources" / Path(member).name
                path.parent.mkdir(exist_ok=True)
                with dataset.open(member) as source, path.open("wb") as destination:
                    shutil.copyfileobj(source, destination)
                paths.append(path)
            image = Image.open(paths[0]).convert("RGB")
            labels = np.array(Image.open(paths[1]))
            if labels.ndim != 2 or labels.shape != (image.height, image.width):
                raise ValueError(f"Unexpected annotation: {stem}")
            ids = [int(value) for value in np.unique(labels) if value]
            target_id = max(ids, key=lambda value: int((labels == value).sum()))
            target = labels == target_id
            y, x = np.where(target)
            box = int(x.min()), int(y.min()), int(x.max() + 1), int(y.max() + 1)
            background = empty_box(labels > 0)
            negative = ((background[0] + background[2]) // 2, (background[1] + background[3]) // 2)
            points = (row_point(target, 0.45), row_point(target, 0.18), row_point(target, 0.8), negative)
            cases.append(RefinementCase(stem, split, str(paths[0].relative_to(ROOT)), sha256(paths[0].read_bytes()).hexdigest(),
                                        str(paths[1].relative_to(ROOT)), sha256(paths[1].read_bytes()).hexdigest(), image.size,
                                        target_id, points, (1, 1, 1, 0), box, background))
    (output / "manifest.json").write_text(json.dumps([asdict(case) for case in cases], indent=2), encoding="utf-8")
    (output / "protocol.json").write_text(json.dumps({
        "dataset_url": DATASET_URL, "archive_sha256": sha256(archive.read_bytes()).hexdigest(),
        "split_seed": SPLIT_SEED, "selection": "SHA256(seed:filename) rank; first 3/group calibration, next 6/group holdout",
        "thresholds": THRESHOLDS, "positive_text": "person", "negative_text": "zebra",
        "additional_holdout_negative": "astronaut; frozen before holdout inference, not used to fit thresholds",
        "calibration_rule": "Highest mean positive-mask IoU with zero nonempty zebra selections; ties prefer higher threshold",
        "point_target": "Largest annotated person; GT-derived clicks and boxes simulate equal ideal user input",
        "text_target": "Union of all annotated people; absent zebra target is empty",
        "holdout_policy": "Freeze selected thresholds before holdout inference; never select a per-image threshold or mask using GT",
        "limitations": "Person-only public dataset, not a representative editor cohort or a model-training holdout",
    }, indent=2), encoding="utf-8")
    sheet = Image.new("RGB", (1200, 300 * 6), "#20242c")
    draw = ImageDraw.Draw(sheet)
    for index, case in enumerate(cases):
        image = Image.open(ROOT / case.source).convert("RGB")
        labels = np.array(Image.open(ROOT / case.annotation))
        preview = overlay(image, labels > 0)
        marks = ImageDraw.Draw(preview)
        for (x, y), label in zip(case.points, case.labels):
            marks.ellipse((x - 4, y - 4, x + 4, y + 4), fill="lime" if label else "red")
        marks.rectangle(case.box, outline="yellow", width=2)
        preview.thumbnail((390, 260))
        left, top = index % 3 * 400, index // 3 * 300
        sheet.paste(preview, (left, top + 30))
        draw.text((left + 5, top + 8), f"{case.name} / {case.split}", fill="white")
    sheet.save(output / "frozen-cohort.jpg", quality=95)
    print(json.dumps({"cases": len(cases), "calibration": 6, "holdout": 12, "archive_bytes": archive.stat().st_size}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    prepare(parser.parse_args().output.resolve())
