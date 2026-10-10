"""Paired before/after masks on the real Klein 9B provider, without app documents.

Run with API models unloaded. Requires the retained 9 October fixture assets.
Every declared seed is retained; baseline reproduces the previous workspace.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from hashlib import sha256
import html
from importlib.metadata import version
from io import BytesIO
import json
from pathlib import Path
import shutil
import sys
import time
from typing import Literal

import numpy as np
from PIL import Image, ImageOps
from pydantic import BaseModel, TypeAdapter

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from app.compositing import InpaintWorkspace, prepare_inpaint_workspace, prepare_outpaint_workspace
from app.jobs import JobControl, JobManager
from app.models.base import ModelReference
from app.models.flux2_klein import MATCH_SURROUNDINGS, _inpaint_prompt
from app.models.flux2_klein_nvfp4 import BASE_REVISION_9B, QUANT_REVISION_9B, Flux2Klein9bNvfp4Provider
from app.runtime import torch_module
from app.schemas import InpaintForm, OutpaintForm
from app.services import prepare_outpaint_images
from benchmark_inpaint_quality import assert_api_unloaded
from benchmark_model_comparison import Case

Variant = Literal["before", "after"]


@dataclass(frozen=True)
class EditCase:
    name: str
    source: Path
    mask: Path
    request: InpaintForm
    references: tuple[ModelReference, ...] = ()


class Measurement(BaseModel):
    case: str
    variant: Variant
    seed: int
    seconds: float
    workspace_seconds: float
    peak_allocated_gib: float
    peak_reserved_gib: float
    protected_changes: int
    width: int
    height: int
    prompt: str
    reference_names: list[str]


def prepare_cases(output: Path) -> list[EditCase]:
    fixtures = ROOT / "data/model-comparison-2026-10-09"
    existing = TypeAdapter(list[Case]).validate_json((fixtures / "cases.json").read_text(encoding="utf-8"))
    assets = output / "assets"
    assets.mkdir()
    result: list[EditCase] = []
    for case in existing:
        if case.name not in ("shirt", "kitchen", "anna"):
            continue
        if case.source is None or case.mask is None:
            raise ValueError(f"Missing masked fixture: {case.name}")
        for name in (case.source, case.mask, *case.references):
            shutil.copyfile(fixtures / "assets" / name, assets / name)
        prompt = case.klein_prompt.removesuffix(MATCH_SURROUNDINGS).strip()
        operation: Literal["replace", "remove"] = "replace"
        if case.name == "kitchen":
            prompt = "Continuous white marble countertop with subtle gray veins and natural daylight."
            operation = "remove"
        request = InpaintForm(prompt=prompt, edit_operation=operation, blend_feather=0,
                              context_padding=case.context, seed=7)
        references = tuple(ModelReference("Anna", assets / name) for name in case.references)
        result.append(EditCase(case.name, assets / case.source, assets / case.mask, request, references))
    if {case.name for case in result} != {"shirt", "kitchen", "anna"}:
        raise ValueError("The retained cohort must contain shirt, kitchen and Anna fixtures.")
    for name, fixture, margins, size in (
        ("garden-right", "garden", (0, 0, 640, 0), None),
        ("kitchen-four-sides", "kitchen", (256, 256, 256, 256), (1536, 1536)),
    ):
        with Image.open(ROOT / f"benchmarks/fixtures/v1/{fixture}.png") as image:
            picture = image.convert("RGB")
        if size is not None:
            picture = picture.resize(size, Image.Resampling.LANCZOS)
        buffer = BytesIO()
        picture.save(buffer, "PNG")
        left, top, right, bottom = margins
        request = OutpaintForm(prompt="the existing garden and path" if fixture == "garden" else "the kitchen and marble countertop",
                              left=left, top=top, right=right, bottom=bottom, seed=7)
        canvas, mask = prepare_outpaint_images(buffer.getvalue(), request)
        source_path, mask_path = assets / f"{name}-canvas.png", assets / f"{name}-selection.png"
        source_path.write_bytes(canvas)
        mask_path.write_bytes(mask)
        result.append(EditCase(name, source_path, mask_path, request))
    return result


def workspace_for(case: EditCase, work: Path, variant: Variant) -> InpaintWorkspace:
    if isinstance(case.request, OutpaintForm):
        if variant == "after":
            return prepare_outpaint_workspace(case.source, case.mask, work, 1024, .5, 128)
        return prepare_inpaint_workspace(case.source, case.mask, work, 1024, .5, 128,
                                         mask_growth=16, composite_feather=0, match_edges=True)
    return prepare_inpaint_workspace(
        case.source, case.mask, work, 1024, 0, case.request.context_padding,
        composite_feather=case.request.blend_feather,
        generation_feather=case.request.blend_feather if variant == "before" else None,
        center_selection=True,
    )


def gallery(output: Path, cases: list[EditCase], measurements: list[Measurement]) -> None:
    cards: list[str] = []
    for case in cases:
        rows = []
        for measurement in measurements:
            if measurement.case != case.name:
                continue
            folder = f"{case.name}-{measurement.variant}-{measurement.seed}"
            rows.append(f'<figure><a href="{folder}/composite.png"><img src="{folder}/composite.png"></a>'
                        f'<figcaption>{measurement.variant} · seed {measurement.seed} · {measurement.seconds:.2f}s · '
                        f'{measurement.protected_changes} protected changes · '
                        f'<a href="{folder}/output.png">raw</a> · <a href="{folder}/mask.png">generation mask</a>'
                        '</figcaption></figure>')
        cards.append(f'<h2>{html.escape(case.name)}</h2><section>{"".join(rows)}</section>')
        before = [item for item in measurements if item.case == case.name and item.variant == "before"]
        if before:
            tiles = []
            for item in [entry for entry in measurements if entry.case == case.name]:
                with Image.open(output / f"{case.name}-{item.variant}-{item.seed}/composite.png") as image:
                    tiles.append(ImageOps.contain(image.convert("RGB"), (512, 512)))
            sheet = Image.new("RGB", (1024, 512 * len(before)), (24, 24, 24))
            for index, tile in enumerate(tiles):
                sheet.paste(tile, ((index % 2) * 512, (index // 2) * 512))
            sheet.save(output / f"{case.name}-comparison.jpg", quality=93)
    (output / "index.html").write_text(
        '<!doctype html><html lang="en"><meta charset="utf-8"><title>Mask quality comparison</title>'
        '<style>body{background:#181818;color:#eee;font:16px system-ui;margin:24px}section{display:grid;'
        'grid-template-columns:1fr 1fr;gap:20px}figure{margin:0}img{max-width:100%;max-height:700px}'
        'a{color:#8cd0ff}figcaption{padding:8px}</style><h1>Before / after · Klein 9B app provider</h1>'
        '<p>All seeds retained. Open PNGs for 100% review. The first call includes model loading; '
        'timings include inference and final compositing, but workspace preparation is recorded separately.</p>'
        + "".join(cards) + '</html>', encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 42, 20261009])
    args = parser.parse_args()
    assert_api_unloaded()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    cases = prepare_cases(output)
    protocol = {
        "model": "flux2-klein-9b-nvfp4", "base_revision": BASE_REVISION_9B,
        "quant_revision": QUANT_REVISION_9B, "encoder": "app RAM-streamed FP8",
        "seeds": args.seeds, "steps": 4, "guidance": 1, "strength": 1,
        "recipes": {
            "before": {"inpaint": "soft generation and composite alpha", "outpaint_generation_native_px": 16,
                       "outpaint_blend_native_px": 16, "drift_reference": "outside composite footprint"},
            "after": {"inpaint": "opaque generation footprint; original composite alpha",
                      "outpaint_generation_model_px": 32, "outpaint_blend_native_px": 16,
                      "drift_reference": "outside generation footprint"},
        },
        "packages": {name: version(name) for name in ("torch", "diffusers", "transformers", "comfy-kitchen")},
        "inputs": {path.name: sha256(path.read_bytes()).hexdigest() for path in (output / "assets").iterdir()},
        "cases": [{"name": case.name, "source": case.source.name, "mask": case.mask.name,
                   "request": case.request.model_dump(),
                   "references": [{"name": reference.name, "file": reference.path.name} for reference in case.references]}
                  for case in cases],
    }
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    torch = torch_module()
    provider, manager = Flux2Klein9bNvfp4Provider(), JobManager()
    measurements: list[Measurement] = []
    try:
        for case in cases:
            for seed in args.seeds:
                for variant in ("before", "after"):
                    work = output / f"{case.name}-{variant}-{seed}"
                    work.mkdir()
                    started = time.perf_counter()
                    workspace = workspace_for(case, work, variant)
                    workspace_seconds = time.perf_counter() - started
                    request = case.request.model_copy(update={"seed": seed})

                    def run(control: JobControl) -> None:
                        torch.cuda.reset_peak_memory_stats()
                        started = time.perf_counter()
                        provider.inpaint(request, workspace.model_input_path, workspace.model_mask_path,
                                         workspace.model_output_path, control, references=case.references)
                        workspace.composite_to(work / "composite.png")
                        seconds = time.perf_counter() - started
                        with Image.open(work / "composite.png") as image:
                            result = np.asarray(image.convert(workspace.source.mode))
                        protected = np.asarray(workspace.composite_mask) == 0
                        changes = int(np.count_nonzero(np.any(result != np.asarray(workspace.source), axis=2) & protected))
                        measurements.append(Measurement(
                            case=case.name, variant=variant, seed=seed, seconds=round(seconds, 3),
                            workspace_seconds=round(workspace_seconds, 3),
                            peak_allocated_gib=round(torch.cuda.max_memory_allocated() / 1024**3, 3),
                            peak_reserved_gib=round(torch.cuda.max_memory_reserved() / 1024**3, 3),
                            protected_changes=changes, width=workspace.source.width, height=workspace.source.height,
                            prompt=_inpaint_prompt(request, [reference.name for reference in case.references]),
                            reference_names=[reference.name for reference in case.references],
                        ))
                        if changes:
                            raise RuntimeError(f"Protected pixels changed: {changes}")

                    job = manager.submit("outpaint" if isinstance(request, OutpaintForm) else "inpaint", run)
                    assert job.future is not None
                    job.future.result()
                    status = manager.status(job.id)
                    (output / "results.json").write_text(json.dumps([item.model_dump() for item in measurements], indent=2), encoding="utf-8")
                    gallery(output, cases, measurements)
                    if status is None or status.status != "completed":
                        raise RuntimeError(status.error if status else "Missing job status")
                    print(measurements[-1].model_dump_json(), flush=True)
    finally:
        manager.shutdown()
        provider.unload()


if __name__ == "__main__":
    main()
