"""Measure the real app provider's cold, repeated and changed-prompt edits.

Run from backend/ after offline GPU cohorts have exited. Does not start an API
or modify documents. Uses the actual JobManager, provider and compositor.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
from app.compositing import prepare_inpaint_workspace
from app.jobs import JobControl, JobManager
from app.models.base import ModelReference
from app.models.flux2_klein_nvfp4 import Flux2KleinNvfp4Provider
from app.runtime import torch_module
from app.schemas import InpaintForm
from benchmark_inpaint_quality import Recipe, assert_api_unloaded
from benchmark_park_prompts import PARALLEL


class AppMeasurement(BaseModel):
    label: str
    seconds: float
    peak_allocated_gib: float
    peak_reserved_gib: float
    protected_changes: int | None = None
    error: str | None = None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert_api_unloaded()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    assets = args.assets.resolve()
    recipe = Recipe(name="app-reference", source="park.png", mask="box16.png", references=("anna512.png",), prompt=PARALLEL)
    torch = torch_module()
    provider = Flux2KleinNvfp4Provider()
    manager = JobManager()
    measurements: list[AppMeasurement] = []
    try:
        for index, seed in enumerate((7, 42, 20261008, 1234)):
            label = ("cold-first-edit", "cached-prompt-1", "cached-prompt-2", "changed-prompt")[index]
            prompt = PARALLEL
            if index == 3:
                prompt += " She looks to her left."
            request = InpaintForm(prompt=prompt, steps=4, guidance=1, strength=1, seed=seed, runtime_mode="balanced", context_padding=32, blend_feather=0)
            torch.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
            changed: int | None = None
            def render(control: JobControl) -> None:
                nonlocal changed
                work = output / label
                work.mkdir()
                workspace = prepare_inpaint_workspace(assets/recipe.source, assets/recipe.mask, work, 1024, 0, 32, composite_feather=0, generation_feather=0, center_selection=True)
                provider.inpaint(request, workspace.model_input_path, workspace.model_mask_path, workspace.model_output_path, control,
                    references=[ModelReference("Anna", assets/"anna512.png")])
                workspace.composite_to(work/"result.png")
                with Image.open(work/"result.png") as image:
                    delta = np.any(np.asarray(image.convert("RGBA")) != np.asarray(workspace.source.convert("RGBA")), axis=2)
                changed = int(np.count_nonzero(delta & (np.asarray(workspace.composite_mask)==0)))
            job = manager.submit("inpaint", render)
            if not manager.wait_for_completion(job.id, timeout=90):
                manager.cancel(job.id)
                measurements.append(AppMeasurement(label=label, seconds=round(time.perf_counter()-started, 3),
                    peak_allocated_gib=round(torch.cuda.max_memory_allocated()/1024**3, 3),
                    peak_reserved_gib=round(torch.cuda.max_memory_reserved()/1024**3, 3),
                    error="App provider exceeded 90 seconds; own job cancelled"))
                (output/"results.json").write_text(json.dumps([m.model_dump() for m in measurements], indent=2), encoding="utf-8")
                raise TimeoutError("App provider exceeded 90 seconds")
            status = manager.status(job.id)
            torch.cuda.synchronize()
            measurement = AppMeasurement(label=label, seconds=round(time.perf_counter()-started, 3),
                peak_allocated_gib=round(torch.cuda.max_memory_allocated()/1024**3, 3), peak_reserved_gib=round(torch.cuda.max_memory_reserved()/1024**3, 3),
                protected_changes=changed, error=status.error if status else "Missing status")
            measurements.append(measurement)
            (output/"results.json").write_text(json.dumps([m.model_dump() for m in measurements], indent=2), encoding="utf-8")
            print(measurement.model_dump_json(), flush=True)
            if measurement.error or changed != 0:
                raise RuntimeError("App benchmark failed; see results.json")
    finally:
        manager.shutdown()
        provider.unload()


if __name__ == "__main__":
    main()
