"""Measure equal-input refinement and calibrate text thresholds on a frozen split."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import statistics
from time import perf_counter
from typing import Literal

import numpy as np
from PIL import Image
import psutil
from pydantic import BaseModel, ConfigDict
import torch
import torch.nn.functional as functional

from benchmark_selection_baseline import ROOT, overlay
from benchmark_sam3_selection import PointPredictor, TextPredictor, REVISION
from prepare_selection_refinement import RefinementCase, THRESHOLDS, load_cases

ModelName = Literal["current", "sam3-point", "sam3-text"]


@dataclass(frozen=True)
class MaskResult:
    mask: np.ndarray
    scores: tuple[float, ...]


@dataclass(frozen=True)
class Record:
    case: str
    split: str
    route: str
    threshold: float | None
    target: Literal["largest_person", "all_people", "empty"]
    iou: float
    precision: float
    recall: float
    selected_pixels: int
    scores: tuple[float, ...]
    seconds: float
    repeat_seconds: float | None
    repeat_identical: bool | None
    mask_sha256: str
    artifact: str


class Results(BaseModel):
    model_config = ConfigDict(extra="forbid")
    status: str
    model: ModelName
    split: str
    measured_at_utc: str
    manifest_sha256: str
    checkpoint_revision: str
    torch_version: str
    transformers_version: str
    device: str
    precision: str
    load_seconds: float
    warmup_seconds: float
    peak_ram_bytes: int
    cuda_allocated_bytes: int
    cuda_reserved_bytes: int
    threshold_selection_sha256: str | None
    records: list[Record]


class ThresholdChoice(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model: ModelName
    threshold: float
    mean_calibration_iou: float
    absent_false_positives: int


class Selection(BaseModel):
    model_config = ConfigDict(extra="forbid")
    manifest_sha256: str
    rule: str
    choices: list[ThresholdChoice]


def mask_iou(mask: np.ndarray, target: np.ndarray) -> float:
    union = np.logical_or(mask, target).sum()
    return float(np.logical_and(mask, target).sum() / union) if union else 1.0


def measure(output: Path, case: RefinementCase, route: str, threshold: float | None,
            target_name: Literal["largest_person", "all_people", "empty"],
            target: np.ndarray, result: MaskResult, seconds: float,
            repeated: MaskResult | None = None, repeat_seconds: float | None = None) -> Record:
    mask = result.mask > 0
    if mask.shape != (case.size[1], case.size[0]):
        raise ValueError(f"Wrong mask size: {case.name}/{route}")
    stem = f"{case.name}-{route}" + (f"-{threshold:.1f}" if threshold is not None else "")
    artifact = f"{stem}-mask.png"
    pixels = mask.astype(np.uint8) * 255
    Image.fromarray(pixels).save(output / artifact)
    with Image.open(ROOT / case.source) as source:
        overlay(source, pixels).save(output / f"{stem}-overlay.png")
    intersection = int(np.logical_and(mask, target).sum())
    selected = int(mask.sum())
    expected = int(target.sum())
    return Record(case.name, case.split, route, threshold, target_name, mask_iou(mask, target),
                  intersection / selected if selected else (1.0 if not expected else 0.0),
                  intersection / expected if expected else (1.0 if not selected else 0.0),
                  selected, result.scores, seconds, repeat_seconds,
                  bool(np.array_equal(result.mask, repeated.mask)) if repeated is not None else None,
                  sha256(pixels.tobytes()).hexdigest(), artifact)


def isolate(output: Path) -> None:
    output.mkdir(parents=True, exist_ok=False)
    for key, value in {
        "MOSIMAGE_DATA_DIR": output, "MOSIMAGE_DB_PATH": output / "unused.db",
        "MOSIMAGE_IMAGE_DIR": output / "images", "TEMP": output, "TMP": output,
        "MOSIMAGE_WARM_UP_ON_START": "false",
    }.items():
        os.environ[key] = str(value)
    os.chdir(output)


class CurrentPredictor:
    def __init__(self) -> None:
        from app.detection import detector, strongest
        from app.selection import ObjectSelector
        self.selector = ObjectSelector()
        self.detector, self.strongest = detector, strongest
        self.selector.warm_up()
        self.detector._ensure_model()

    def reset(self) -> None:
        self.selector._image_key = None
        self.selector._embeddings = None

    def text(self, image: Image.Image, case: RefinementCase, text: str, threshold: float) -> MaskResult:
        found = self.strongest(self.detector.detect(image, text, threshold=threshold))
        mask = (self.selector._predict_boxes(image, case.source_sha256, [item.box for item in found])
                if found else np.zeros((image.height, image.width), dtype=np.uint8))
        return MaskResult(mask, tuple(item.score for item in found))

    def point(self, image: Image.Image, case: RefinementCase, route: str,
              previous: torch.Tensor | None) -> tuple[MaskResult, torch.Tensor]:
        processor, model = self.selector._processor, self.selector._model
        if processor is None or model is None:
            raise RuntimeError("Current SAM not loaded")
        if route == "box":
            inputs = processor(images=image, input_boxes=[[list(case.box)]], return_tensors="pt")
        else:
            points = case.points[:1] if route == "point" else case.points
            labels = case.labels[:1] if route == "point" else case.labels
            inputs = processor(images=image, input_points=[[[list(point) for point in points]]], input_labels=[[list(labels)]], return_tensors="pt")
        if self.selector._image_key != case.source_sha256 or self.selector._embeddings is None:
            self.selector._embeddings = model.get_image_embeddings(inputs["pixel_values"])
            self.selector._image_key = case.source_sha256
        outputs = model(image_embeddings=self.selector._embeddings, input_points=inputs.get("input_points"),
                        input_labels=inputs.get("input_labels"), input_boxes=inputs.get("input_boxes"),
                        input_masks=previous if route == "point_feedback" else None, multimask_output=route == "point")
        masks = processor.post_process_masks(outputs.pred_masks.cpu().float(), inputs["original_sizes"].cpu())[0]
        best = int(outputs.iou_scores[0, 0].argmax())
        result = MaskResult(masks[0, best].numpy().astype(np.uint8) * 255,
                            tuple(float(value) for value in outputs.iou_scores[0, 0]))
        return result, outputs.pred_masks[0, 0, best].unsqueeze(0).unsqueeze(0)


class SamPointPredictor:
    def __init__(self, output: Path) -> None:
        self.predictor = PointPredictor(output, "cuda", torch.bfloat16)

    def reset(self) -> None:
        self.predictor.clear_cache()

    def point(self, image: Image.Image, case: RefinementCase, route: str,
              previous: torch.Tensor | None) -> tuple[MaskResult, torch.Tensor]:
        processor, model = self.predictor.processor, self.predictor.model
        if route == "box":
            inputs = processor(images=image, input_boxes=[[list(case.box)]], return_tensors="pt").to("cuda")
        else:
            points = case.points[:1] if route == "point" else case.points
            labels = case.labels[:1] if route == "point" else case.labels
            inputs = processor(images=image, input_points=[[[list(point) for point in points]]], input_labels=[[list(labels)]], return_tensors="pt").to("cuda")
        if self.predictor.embeddings is None:
            self.predictor.embeddings = model.get_image_embeddings(inputs["pixel_values"].to(torch.bfloat16))
        outputs = model(image_embeddings=self.predictor.embeddings, input_points=inputs.get("input_points"),
                        input_labels=inputs.get("input_labels"), input_boxes=inputs.get("input_boxes"),
                        input_masks=previous if route == "point_feedback" else None, multimask_output=route == "point")
        masks = processor.post_process_masks(outputs.pred_masks.cpu().float(), inputs["original_sizes"].cpu())[0]
        best = int(outputs.iou_scores[0, 0].argmax())
        result = MaskResult(masks[0, best].numpy().astype(np.uint8) * 255,
                            tuple(float(value) for value in outputs.iou_scores[0, 0].cpu()))
        return result, outputs.pred_masks[0, 0, best].unsqueeze(0).unsqueeze(0)


class SamTextPredictor:
    def __init__(self, output: Path) -> None:
        self.predictor = TextPredictor(output, "cuda", torch.bfloat16)

    def reset(self) -> None:
        self.predictor.clear_cache()

    def text(self, image: Image.Image, case: RefinementCase, text: str, threshold: float,
             route: str = "text") -> MaskResult:
        processor, model = self.predictor.processor, self.predictor.model
        if route == "text_positive_box":
            inputs = processor(images=image, text=text, input_boxes=[[list(case.box)]], input_boxes_labels=[[1]], return_tensors="pt").to("cuda")
        elif route == "text_negative_box":
            inputs = processor(images=image, text=text, input_boxes=[[list(case.negative_box)]], input_boxes_labels=[[0]], return_tensors="pt").to("cuda")
        else:
            inputs = processor(images=image, text=text, return_tensors="pt").to("cuda")
        if self.predictor.embeddings is None:
            self.predictor.embeddings = model.get_vision_features(inputs["pixel_values"].to(torch.bfloat16))
        outputs = model(vision_embeds=self.predictor.embeddings, input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"],
                        input_boxes=inputs.get("input_boxes"), input_boxes_labels=inputs.get("input_boxes_labels"))
        if route == "text_native_resize":
            # Meta's reference processor resizes logits then thresholds; Transformers resizes sigmoid probabilities.
            scores = outputs.pred_logits.sigmoid() * outputs.presence_logits.sigmoid()
            keep = scores[0] > threshold
            if keep.any():
                logits = outputs.pred_masks[0, keep].float().unsqueeze(0)
                masks = functional.interpolate(logits, (image.height, image.width), mode="bilinear", align_corners=False)[0] > 0
                mask = masks.any(dim=0).cpu().numpy().astype(np.uint8) * 255
            else:
                mask = np.zeros((image.height, image.width), dtype=np.uint8)
            return MaskResult(mask, tuple(float(value) for value in scores[0, keep].cpu()))
        result = processor.post_process_instance_segmentation(outputs, threshold=threshold, mask_threshold=0.5,
                                                               target_sizes=inputs["original_sizes"].cpu().tolist())[0]
        mask = (result["masks"].any(dim=0).cpu().numpy().astype(np.uint8) * 255
                if len(result["masks"]) else np.zeros((image.height, image.width), dtype=np.uint8))
        return MaskResult(mask, tuple(float(value) for value in result["scores"].cpu()))


def sync(model_name: ModelName) -> None:
    if model_name != "current":
        torch.cuda.synchronize()


def run(root: Path, output: Path, model_name: ModelName, split: str) -> None:
    cases = load_cases(root / "manifest.json")
    cohort = [case for case in cases if case.split == split]
    if not cohort:
        raise ValueError("Empty split")
    for case in cases:
        if sha256((ROOT / case.source).read_bytes()).hexdigest() != case.source_sha256 or sha256((ROOT / case.annotation).read_bytes()).hexdigest() != case.annotation_sha256:
            raise ValueError(f"Frozen source changed: {case.name}")
    selected: Selection | None = None
    selected_hash: str | None = None
    if split == "holdout":
        selection_path = root / "threshold-selection.json"
        selected = Selection.model_validate_json(selection_path.read_text())
        selected_hash = sha256(selection_path.read_bytes()).hexdigest()
        if selected.manifest_sha256 != sha256((root / "manifest.json").read_bytes()).hexdigest():
            raise ValueError("Threshold selection belongs to a different cohort")
    isolate(output)
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    start = perf_counter()
    predictor = CurrentPredictor() if model_name == "current" else SamPointPredictor(output) if model_name == "sam3-point" else SamTextPredictor(output)
    sync(model_name)
    load_seconds = perf_counter() - start
    records: list[Record] = []
    tuned_threshold = next((choice.threshold for choice in selected.choices if choice.model == model_name), 0.5) if selected else None
    default_threshold = 0.3 if model_name == "current" else 0.5
    context = torch.autocast("cuda", dtype=torch.bfloat16) if model_name != "current" else nullcontext()
    with torch.inference_mode(), context:
        warm_case = cohort[0]
        image = Image.open(ROOT / warm_case.source).convert("RGB")
        start = perf_counter()
        if isinstance(predictor, SamPointPredictor):
            predictor.point(image, warm_case, "point", None)
        else:
            predictor.text(image, warm_case, "person", default_threshold)
        sync(model_name)
        warm_seconds = perf_counter() - start
        if model_name != "current":
            torch.cuda.reset_peak_memory_stats()
        for case in cohort:
            print(json.dumps({"stage": "case", "model": model_name, "case": case.name, "split": split}), flush=True)
            image = Image.open(ROOT / case.source).convert("RGB")
            annotation = np.array(Image.open(ROOT / case.annotation))
            person_target, all_target = annotation == case.target_id, annotation > 0
            empty_target = np.zeros_like(all_target)
            if isinstance(predictor, (CurrentPredictor, SamPointPredictor)) and split == "holdout":
                predictor.reset()
                previous: torch.Tensor | None = None
                for route in ("point", "point_refined", "point_feedback", "box"):
                    sync(model_name)
                    start = perf_counter()
                    result, logits = predictor.point(image, case, route, previous)
                    sync(model_name)
                    seconds = perf_counter() - start
                    start = perf_counter()
                    repeat, _ = predictor.point(image, case, route, previous)
                    sync(model_name)
                    repeat_seconds = perf_counter() - start
                    records.append(measure(output, case, route, None, "largest_person", person_target, result, seconds, repeat, repeat_seconds))
                    if route == "point":
                        previous = logits
            if isinstance(predictor, (CurrentPredictor, SamTextPredictor)):
                if split == "calibration":
                    text_routes = [("text", "person", value) for value in THRESHOLDS] + [("absent", "zebra", value) for value in THRESHOLDS]
                else:
                    text_routes = [("text_default", "person", default_threshold), ("text_tuned", "person", tuned_threshold),
                                   ("absent_default", "zebra", default_threshold), ("absent_tuned", "zebra", tuned_threshold),
                                   ("absent_hard_default", "astronaut", default_threshold), ("absent_hard_tuned", "astronaut", tuned_threshold)]
                    if isinstance(predictor, SamTextPredictor):
                        text_routes += [(route, "person", tuned_threshold) for route in ("text_positive_box", "text_negative_box", "text_native_resize")]
                for route, text, threshold in text_routes:
                    predictor.reset()
                    sync(model_name)
                    start = perf_counter()
                    result = predictor.text(image, case, text, threshold, route) if isinstance(predictor, SamTextPredictor) else predictor.text(image, case, text, threshold)
                    sync(model_name)
                    seconds = perf_counter() - start
                    repeat, repeat_seconds = None, None
                    if split == "holdout":
                        start = perf_counter()
                        repeat = predictor.text(image, case, text, threshold, route) if isinstance(predictor, SamTextPredictor) else predictor.text(image, case, text, threshold)
                        sync(model_name)
                        repeat_seconds = perf_counter() - start
                    target = all_target if text == "person" else empty_target
                    records.append(measure(output, case, route, threshold, "all_people" if text == "person" else "empty", target, result, seconds, repeat, repeat_seconds))
            memory = psutil.Process().memory_info()
            import transformers
            from benchmark_selection_baseline import cached_revision
            payload = Results(status="complete" if case == cohort[-1] else "running", model=model_name, split=split,
                              measured_at_utc=datetime.now(timezone.utc).isoformat(), manifest_sha256=sha256((root / "manifest.json").read_bytes()).hexdigest(),
                              checkpoint_revision=cached_revision("facebook/sam2.1-hiera-tiny") if model_name == "current" else REVISION,
                              torch_version=torch.__version__, transformers_version=transformers.__version__, device="cpu" if model_name == "current" else "cuda",
                              precision="float32" if model_name == "current" else "bfloat16", load_seconds=load_seconds, warmup_seconds=warm_seconds,
                              peak_ram_bytes=getattr(memory, "peak_wset", memory.rss), cuda_allocated_bytes=torch.cuda.max_memory_allocated() if model_name != "current" else 0,
                              cuda_reserved_bytes=torch.cuda.max_memory_reserved() if model_name != "current" else 0, threshold_selection_sha256=selected_hash, records=records)
            (output / "results.json").write_text(payload.model_dump_json(indent=2), encoding="utf-8")
    print(json.dumps({"status": "complete", "records": len(records), "model": model_name, "split": split}), flush=True)


def select_thresholds(root: Path) -> None:
    choices: list[ThresholdChoice] = []
    audits: list[ThresholdChoice] = []
    for model in ("current", "sam3-text"):
        data = Results.model_validate_json((root / f"{model}-calibration/results.json").read_text())
        if data.status != "complete" or data.split != "calibration":
            raise ValueError("Calibration incomplete")
        for threshold in THRESHOLDS:
            positives = [record for record in data.records if record.route == "text" and record.threshold == threshold]
            negatives = [record for record in data.records if record.route == "absent" and record.threshold == threshold]
            if len(positives) != 6 or len(negatives) != 6:
                raise ValueError("Incomplete threshold sweep")
            audits.append(ThresholdChoice(model=model, threshold=threshold, mean_calibration_iou=statistics.mean(record.iou for record in positives),
                                          absent_false_positives=sum(record.selected_pixels > 0 for record in negatives)))
        candidates = [choice for choice in audits if choice.model == model and choice.absent_false_positives == 0]
        if not candidates:
            raise ValueError("No threshold satisfies the declared absent-object constraint")
        choices.append(max(candidates, key=lambda choice: (choice.mean_calibration_iou, choice.threshold)))
    payload = Selection(manifest_sha256=sha256((root / "manifest.json").read_bytes()).hexdigest(),
                        rule="Highest mean positive IoU with zero nonempty absent selections; ties prefer higher threshold", choices=choices)
    destination = root / "threshold-selection.json"
    with destination.open("x", encoding="utf-8") as stream:
        stream.write(payload.model_dump_json(indent=2))
    (root / "threshold-audit.json").write_text(json.dumps([choice.model_dump() for choice in audits], indent=2), encoding="utf-8")
    print(payload.model_dump_json(indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    runner = commands.add_parser("run")
    runner.add_argument("--root", type=Path, required=True)
    runner.add_argument("--output", type=Path, required=True)
    runner.add_argument("--model", choices=("current", "sam3-point", "sam3-text"), required=True)
    runner.add_argument("--split", choices=("calibration", "holdout"), required=True)
    chooser = commands.add_parser("select")
    chooser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "select":
        select_thresholds(args.root.resolve())
    else:
        if args.model == "sam3-point" and args.split == "calibration":
            parser.error("The point routes have no fitted threshold; run them only on holdout")
        run(args.root.resolve(), args.output.resolve(), args.model, args.split)
