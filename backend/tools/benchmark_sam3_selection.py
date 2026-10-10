"""Compare official SAM 3 with a frozen current-version selection manifest."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import statistics
from time import perf_counter
from typing import Literal, Protocol

import numpy as np
from PIL import Image
import psutil
from pydantic import ConfigDict, TypeAdapter
import torch
import transformers
from transformers import Sam3Model, Sam3Processor, Sam3TrackerModel, Sam3TrackerProcessor
from transformers.models.sam3.modeling_sam3 import Sam3VisionEncoderOutput

from benchmark_selection_baseline import ROOT, Case, Source, Measurement, overlay

REVISION = "3c879f39826c281e95690f02c7821c4de09afae7"
CHECKPOINT = ROOT / "models/research-cache" / f"sam3-{REVISION}"


@dataclass(frozen=True)
class Prediction:
    mask: np.ndarray
    scores: tuple[float, ...]
    detections: int | None
    error: str | None


@dataclass(frozen=True)
class CandidateMeasurement(Measurement):
    scores: tuple[float, ...]
    cuda_allocated_peak_bytes: int
    cuda_reserved_peak_bytes: int


class Predictor(Protocol):
    def clear_cache(self) -> None: ...
    def predict(self, image: Image.Image, case: Case) -> Prediction: ...


class PointPredictor:
    def __init__(self, output: Path, device: str, dtype: torch.dtype) -> None:
        self.device, self.dtype = device, dtype
        self.processor = Sam3TrackerProcessor.from_pretrained(CHECKPOINT, local_files_only=True)
        self.model, info = Sam3TrackerModel.from_pretrained(
            CHECKPOINT, local_files_only=True, use_safetensors=True, trust_remote_code=False,
            dtype=dtype, attn_implementation="sdpa", output_loading_info=True,
        )
        verify_load(output, info)
        self.model.to(device).eval()
        self.embeddings: list[torch.Tensor] | None = None

    def clear_cache(self) -> None:
        self.embeddings = None

    def predict(self, image: Image.Image, case: Case) -> Prediction:
        inputs = self.processor(images=image, input_points=[[[list(case.point)]]], input_labels=[[[1]]], return_tensors="pt").to(self.device)
        if self.embeddings is None:
            self.embeddings = self.model.get_image_embeddings(inputs["pixel_values"].to(self.dtype))
        outputs = self.model(image_embeddings=self.embeddings, input_points=inputs["input_points"],
                             input_labels=inputs["input_labels"], multimask_output=True)
        masks = self.processor.post_process_masks(outputs.pred_masks.cpu().float(), inputs["original_sizes"].cpu())[0]
        best = int(outputs.iou_scores[0, 0].argmax())
        return Prediction(masks[0, best].numpy().astype(np.uint8) * 255,
                          tuple(float(value) for value in outputs.iou_scores[0, 0].cpu()), None, None)


class TextPredictor:
    def __init__(self, output: Path, device: str, dtype: torch.dtype) -> None:
        self.device, self.dtype = device, dtype
        self.processor = Sam3Processor.from_pretrained(CHECKPOINT, local_files_only=True)
        self.model, info = Sam3Model.from_pretrained(
            CHECKPOINT, local_files_only=True, use_safetensors=True, trust_remote_code=False,
            dtype=dtype, attn_implementation="sdpa", output_loading_info=True,
        )
        verify_load(output, info)
        self.model.to(device).eval()
        self.embeddings: Sam3VisionEncoderOutput | None = None

    def clear_cache(self) -> None:
        self.embeddings = None

    def predict(self, image: Image.Image, case: Case) -> Prediction:
        inputs = self.processor(images=image, text=case.text, return_tensors="pt").to(self.device)
        if self.embeddings is None:
            self.embeddings = self.model.get_vision_features(inputs["pixel_values"].to(self.dtype))
        # Cache only the image; encode the prompt on every call, like production.
        outputs = self.model(vision_embeds=self.embeddings, input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
        result = self.processor.post_process_instance_segmentation(
            outputs, threshold=0.5, mask_threshold=0.5, target_sizes=inputs["original_sizes"].cpu().tolist(),
        )[0]
        count = len(result["masks"])
        mask = (result["masks"].any(dim=0).cpu().numpy().astype(np.uint8) * 255
                if count else np.zeros((image.height, image.width), dtype=np.uint8))
        return Prediction(mask, tuple(float(value) for value in result["scores"].cpu()), count,
                          None if count else "No instance at official 0.5 confidence threshold")


def verify_load(output: Path, info: dict[str, object]) -> None:
    # Library metadata may have optional fields. Never measure a partly random model.
    (output / "loading-info.json").write_text(json.dumps(info, indent=2, default=list), encoding="utf-8")
    if info.get("missing_keys") or info.get("mismatched_keys") or info.get("error_msgs"):
        raise RuntimeError("Incomplete checkpoint load; see loading-info.json")


def synchronize(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()


def run(baseline: Path, output: Path, route: Literal["point", "text"], device: str, precision: str, repeats: int) -> None:
    sources = TypeAdapter(list[Source], config=ConfigDict(extra="forbid")).validate_json((baseline / "manifest.json").read_text())
    for source in sources:
        if sha256((ROOT / source.case.source).read_bytes()).hexdigest() != source.sha256:
            raise ValueError(f"Frozen source changed: {source.case.name}")
    output.mkdir(parents=True, exist_ok=False)
    (output / "manifest.json").write_bytes((baseline / "manifest.json").read_bytes())
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    dtype = torch.bfloat16 if precision == "bfloat16" else torch.float32
    start_ram = psutil.virtual_memory().available
    start_rss = psutil.Process().memory_info().rss
    gpu_free = torch.cuda.mem_get_info()[0] if device == "cuda" else None
    start = perf_counter()
    predictor: Predictor = PointPredictor(output, device, dtype) if route == "point" else TextPredictor(output, device, dtype)
    synchronize(device)
    load_seconds = perf_counter() - start
    print(json.dumps({"stage": "loaded", "route": route, "device": device, "seconds": load_seconds}), flush=True)
    with Image.open(ROOT / sources[0].case.source) as uploaded:
        warm_image = uploaded.convert("RGB")
    with torch.inference_mode(), (torch.autocast("cuda", dtype=dtype) if device == "cuda" and precision == "bfloat16" else nullcontext()):
        start = perf_counter()
        predictor.predict(warm_image, sources[0].case)
        synchronize(device)
        warm_seconds = perf_counter() - start
        print(json.dumps({"stage": "warmup", "seconds": warm_seconds}), flush=True)
        records: list[CandidateMeasurement] = []
        for source in sources:
            predictor.clear_cache()
            if device == "cuda":
                torch.cuda.reset_peak_memory_stats()
            with Image.open(ROOT / source.case.source) as uploaded:
                image = uploaded.convert("RGB")
            predictions: list[Prediction] = []
            times: list[float] = []
            print(json.dumps({"stage": "case", "case": source.case.name}), flush=True)
            for _ in range(repeats + 1):
                synchronize(device)
                start = perf_counter()
                prediction = predictor.predict(image, source.case)
                synchronize(device)
                times.append(perf_counter() - start)
                predictions.append(prediction)
            first = predictions[0]
            Image.fromarray(first.mask).save(output / f"{source.case.name}-{route}-mask.png")
            overlay(image, first.mask).save(output / f"{source.case.name}-{route}-overlay.png")
            records.append(CandidateMeasurement(
                source.case.name, route, times[0], tuple(times[1:]), int((first.mask > 0).sum()),
                sha256(first.mask.tobytes()).hexdigest(), all(np.array_equal(first.mask, item.mask) for item in predictions[1:]),
                first.detections, first.error, first.scores,
                torch.cuda.max_memory_allocated() if device == "cuda" else 0,
                torch.cuda.max_memory_reserved() if device == "cuda" else 0,
            ))
            memory = psutil.Process().memory_info()
            payload = {
                "status": "complete" if len(records) == len(sources) else "running",
                "model": "facebook/sam3", "revision": REVISION, "route": route, "device": device, "precision": precision,
                "measured_at_utc": datetime.now(timezone.utc).isoformat(),
                "torch": torch.__version__, "transformers": transformers.__version__, "cpu_threads": torch.get_num_threads(),
                "attention": "sdpa", "tf32_allowed": device == "cuda", "load_seconds": load_seconds, "warmup_seconds": warm_seconds,
                "start_rss_bytes": start_rss, "rss_bytes": memory.rss, "process_peak_working_set_bytes": getattr(memory, "peak_wset", None),
                "start_available_ram_bytes": start_ram, "end_available_ram_bytes": psutil.virtual_memory().available,
                "start_gpu_free_bytes": gpu_free, "image_cache": "one image embedding; no text embedding cache",
                "score_threshold": 0.5 if route == "text" else None, "mask_threshold": 0.5 if route == "text" else "tracker default",
                "records": [asdict(item) for item in records],
            }
            (output / "results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
            print(json.dumps(asdict(records[-1])), flush=True)
    print(json.dumps({"uncached_median_seconds": statistics.median(item.uncached_seconds for item in records),
                      "cached_median_seconds": statistics.median(value for item in records for value in item.cached_seconds)}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--route", choices=("point", "text"), required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--precision", choices=("float32", "bfloat16"), required=True)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.repeats < 1 or (args.device == "cpu" and args.precision != "float32"):
        parser.error("Use positive repeats and CPU FP32 for the direct comparison")
    run(args.baseline.resolve(), args.output.resolve(), args.route, args.device, args.precision, args.repeats)
