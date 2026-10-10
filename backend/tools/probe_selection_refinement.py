"""Retain exploratory prompt/refinement results on the original seven sources."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from PIL import Image
from pydantic import ConfigDict, TypeAdapter
import torch

from benchmark_selection_baseline import ROOT, Source, overlay
from benchmark_selection_refinement import CurrentPredictor, SamPointPredictor, SamTextPredictor, ModelName, isolate, sync
from prepare_selection_refinement import RefinementCase


@dataclass(frozen=True)
class Probe:
    case: RefinementCase
    texts: tuple[str, ...]


@dataclass(frozen=True)
class ProbeRecord:
    case: str
    route: str
    text: str | None
    threshold: float | None
    selected_pixels: int
    scores: tuple[float, ...]
    seconds: float
    mask_sha256: str
    artifact: str


def freeze(root: Path, baseline: Path) -> None:
    sources = TypeAdapter(list[Source], config=ConfigDict(extra="forbid")).validate_json((baseline / "manifest.json").read_text())
    guidance = {
        "sphere": ((350, 825), (435, 865), (550, 860), (260, 770, 485, 1000), ("terracotta sphere", "ball")),
        "mug": ((745, 850), (1004, 726), (1100, 820), (625, 525, 1055, 920), ("yellow mug", "cup")),
        "brooch": ((801, 718), (788, 710), (850, 775), (770, 685, 825, 745), ("brooch", "coral brooch", "flower pin")),
        "poster": ((330, 320), (450, 650), (700, 400), (275, 175, 560, 755), ("poster", "framed poster", "picture")),
        "man": ((416, 200), (345, 1090), (720, 610), (220, 90, 650, 1155), ("person", "man")),
        "phone": ((630, 520), (625, 665), (760, 700), (565, 485, 690, 730), ("smartphone", "phone")),
        "astronaut": ((215, 140), (190, 355), (465, 140), (30, 60, 430, 512), ("person", "astronaut")),
    }
    probes: list[Probe] = []
    for source in sources:
        head, feet, negative, box, texts = guidance[source.case.name]
        case = RefinementCase(source.case.name, "diagnostic", source.case.source, source.sha256, "", "", source.size, 0,
                              (source.case.point, head, feet, negative), (1, 1, 1, 0), box, (negative[0] - 8, negative[1] - 8, negative[0] + 8, negative[1] + 8))
        if any(not (0 <= x < case.size[0] and 0 <= y < case.size[1]) for x, y in case.points):
            raise ValueError(f"Invalid diagnostic coordinates: {case.name}")
        probes.append(Probe(case, texts))
    with (root / "diagnostic-manifest.json").open("x", encoding="utf-8") as stream:
        stream.write(json.dumps([asdict(probe) for probe in probes], indent=2))


def run(root: Path, output: Path, model_name: ModelName) -> None:
    probes = TypeAdapter(list[Probe], config=ConfigDict(extra="forbid")).validate_json((root / "diagnostic-manifest.json").read_text())
    isolate(output)
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    predictor = CurrentPredictor() if model_name == "current" else SamPointPredictor(output) if model_name == "sam3-point" else SamTextPredictor(output)
    records: list[ProbeRecord] = []
    context = torch.autocast("cuda", dtype=torch.bfloat16) if model_name != "current" else nullcontext()
    with torch.inference_mode(), context:
        for probe in probes:
            case = probe.case
            if sha256((ROOT / case.source).read_bytes()).hexdigest() != case.source_sha256:
                raise ValueError(f"Frozen diagnostic source changed: {case.name}")
            image = Image.open(ROOT / case.source).convert("RGB")
            print(json.dumps({"model": model_name, "case": case.name}), flush=True)
            predictor.reset()
            if isinstance(predictor, (CurrentPredictor, SamPointPredictor)):
                previous: torch.Tensor | None = None
                for route in ("point", "point_refined", "point_feedback", "box"):
                    start = perf_counter()
                    result, logits = predictor.point(image, case, route, previous)
                    sync(model_name)
                    if route == "point":
                        previous = logits
                    artifact = f"{case.name}-{route}-mask.png"
                    seconds = perf_counter() - start
                    Image.fromarray(result.mask).save(output / artifact)
                    overlay(image, result.mask).save(output / artifact.replace("-mask", "-overlay"))
                    records.append(ProbeRecord(case.name, route, None, None, int((result.mask > 0).sum()), result.scores, seconds, sha256(result.mask.tobytes()).hexdigest(), artifact))
            if isinstance(predictor, (CurrentPredictor, SamTextPredictor)):
                variants = [("text", text, threshold) for text in probe.texts for threshold in (0.1, 0.3, 0.5)]
                if isinstance(predictor, SamTextPredictor):
                    variants += [(route, probe.texts[0], 0.5) for route in ("text_positive_box", "text_negative_box", "text_native_resize")]
                for index, (route, text, threshold) in enumerate(variants):
                    start = perf_counter()
                    result = predictor.text(image, case, text, threshold, route) if isinstance(predictor, SamTextPredictor) else predictor.text(image, case, text, threshold)
                    sync(model_name)
                    artifact = f"{case.name}-{route}-{index}-{threshold:.1f}-mask.png"
                    seconds = perf_counter() - start
                    Image.fromarray(result.mask).save(output / artifact)
                    overlay(image, result.mask).save(output / artifact.replace("-mask", "-overlay"))
                    records.append(ProbeRecord(case.name, route, text, threshold, int((result.mask > 0).sum()), result.scores, seconds, sha256(result.mask.tobytes()).hexdigest(), artifact))
            (output / "results.json").write_text(json.dumps({"status": "complete" if probe == probes[-1] else "running", "model": model_name,
                                                               "interpretation": "Exploratory; no ground-truth scores, no per-image production recipe chosen",
                                                               "manifest_sha256": sha256((root / "diagnostic-manifest.json").read_bytes()).hexdigest(),
                                                               "records": [asdict(record) for record in records]}, indent=2), encoding="utf-8")
    print(json.dumps({"status": "complete", "records": len(records), "model": model_name}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    freezer = commands.add_parser("freeze")
    freezer.add_argument("--root", type=Path, required=True)
    freezer.add_argument("--baseline", type=Path, required=True)
    runner = commands.add_parser("run")
    runner.add_argument("--root", type=Path, required=True)
    runner.add_argument("--output", type=Path, required=True)
    runner.add_argument("--model", choices=("current", "sam3-point", "sam3-text"), required=True)
    args = parser.parse_args()
    if args.command == "freeze":
        freeze(args.root.resolve(), args.baseline.resolve())
    else:
        run(args.root.resolve(), args.output.resolve(), args.model)
