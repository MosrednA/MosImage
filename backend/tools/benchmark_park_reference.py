"""Paired Anna/park ablations through the existing API's single GPU worker.

Preserves the editor document. Every render, failure and declared seed is saved;
visual quality is reviewed separately from exact protected-pixel checks.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import shutil
import sys
import time
from urllib.request import Request, urlopen
from uuid import uuid4

import numpy as np
from PIL import Image
from pydantic import BaseModel

BACKEND = Path(__file__).resolve().parents[1]
if Path.cwd().resolve() != BACKEND:
    raise RuntimeError("Run park benchmark tools from backend/; no local configuration was read.")
sys.path.insert(0, str(BACKEND))
from app.compositing import _context_box, _model_size, _selection_alpha
from app.models.flux2_klein import _inpaint_prompt
from app.schemas import GenerationRecord, InpaintForm, JobResponse, JobStatus

API = "http://127.0.0.1:8000"
ROOT = Path(__file__).resolve().parents[2]
SEEDS = (7, 42, 20261007)
SOURCE = ROOT / "data/images/inputs/input-f6ba56b3318d4e28b2674de876c92f26.png"
MASK = ROOT / "data/images/masks/mask-556b8e0f258947e4ad2502ae52fbde6b.png"
REFERENCE = ROOT / "data/images/references/reference-daffa729928443a89e616dfe7dde53a6.png"
REFERENCE_ID = 2
SCENE = (
    "An adult woman seated on an ordinary adult-sized wooden garden bench beside the gravel path. "
    "Place the bench on the ground at the selected location, following the path's perspective. "
    "Use plausible adult proportions relative to the nearby flowers and fence, with natural contact shadows."
)


class Variant(BaseModel):
    name: str
    model: str = "flux2-klein-4b-nvfp4"
    context: int = 320
    steps: int = 4
    strength: float = 1.0
    scene: str = ""


VARIANTS = (
    Variant(name="context32", context=32),
    Variant(name="context192", context=192),
    Variant(name="context320"),
    Variant(name="context512", context=512),
    Variant(name="scene320", scene=SCENE),
    Variant(name="strength075", strength=0.75),
    Variant(name="steps8", steps=8),
    Variant(name="bf16-context320", model="flux2-klein-4b"),
)


class Trial(BaseModel):
    variant: str
    seed: int
    job_id: str | None = None
    generation: GenerationRecord | None = None
    elapsed_seconds: float | None = None
    job_seconds: float | None = None
    protected_pixels: int | None = None
    changed_protected_pixels: int | None = None
    error: str | None = None


class Protocol(BaseModel):
    seeds: tuple[int, ...] = SEEDS
    variants: tuple[Variant, ...] = VARIANTS
    source_sha256: str
    mask_sha256: str
    reference_sha256: str
    source_size: tuple[int, int]
    mask_bounds: tuple[int, int, int, int]
    rendered_prompts: dict[str, str]
    results: list[Trial] = []


def read_api(path: str) -> bytes:
    with urlopen(API + path, timeout=30) as response:
        return response.read()


def form_for(variant: Variant, seed: int) -> InpaintForm:
    return InpaintForm(
        prompt="sitting on a bench", reference_ids=[REFERENCE_ID], model_name=variant.model,
        seed=seed, steps=variant.steps, strength=variant.strength,
        context_padding=variant.context, scene_context=variant.scene, blend_feather=0,
    )


def submit(form: InpaintForm, source: Path, mask: Path) -> JobResponse:
    boundary = "mosimage-park-" + uuid4().hex
    fields = form.model_dump(exclude_none=True, exclude={"reference_ids"})
    parts = [
        f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n{value}\r\n'.encode()
        for name, value in fields.items()
    ]
    for reference_id in form.reference_ids:
        parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="reference_ids"\r\n\r\n{reference_id}\r\n'.encode())
    for name, path in (("image", source), ("mask", mask)):
        parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"; filename="{path.name}"\r\nContent-Type: image/png\r\n\r\n'.encode() + path.read_bytes() + b"\r\n")
    parts.append(f"--{boundary}--\r\n".encode())
    request = Request(API + "/api/inpaint", data=b"".join(parts), headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    with urlopen(request, timeout=30) as response:
        return JobResponse.model_validate_json(response.read())


def wait_for_job(job_id: str) -> JobStatus:
    started = time.monotonic()
    reported = started
    while time.monotonic() - started < 300:
        status = JobStatus.model_validate_json(read_api("/api/jobs/" + job_id))
        if status.status in {"completed", "failed", "cancelled"}:
            return status
        if time.monotonic() - reported > 25:
            print(f"  {status.phase}, step {status.step}/{status.steps}, {time.monotonic()-started:.0f}s", flush=True)
            reported = time.monotonic()
        time.sleep(0.25)
    with urlopen(Request(API + f"/api/jobs/{job_id}/cancel", data=b"", method="POST"), timeout=30):
        pass
    raise TimeoutError("Own job cancelled after 300 seconds; benchmark stopped before submitting more work.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--variants", required=True, nargs="+", choices=[variant.name for variant in VARIANTS])
    parser.add_argument("--retry-unsubmitted", action="store_true", help="Retry a confirmed connection failure without a job ID; retains the failed attempt.")
    args = parser.parse_args()
    output = args.output.resolve()
    manifest = output / "results.json"
    with Image.open(SOURCE) as image:
        source = np.asarray(image.convert("RGBA"))
        size = image.size
    with Image.open(MASK) as image:
        alpha = _selection_alpha(image)
        mask = np.asarray(alpha)
        box = alpha.getbbox()
    if box is None or mask.shape != source.shape[:2]:
        raise ValueError("Source and mask must match and the selection must be nonempty.")
    hashes = {f"{name}_sha256": sha256(path.read_bytes()).hexdigest() for name, path in (("source", SOURCE), ("mask", MASK), ("reference", REFERENCE))}
    if manifest.exists():
        protocol = Protocol.model_validate_json(manifest.read_bytes())
        for key, value in hashes.items():
            if getattr(protocol, key) != value:
                raise ValueError(f"Changed fixture: {key}")
    else:
        output.mkdir(parents=True, exist_ok=False)
        for name, path in (("source", SOURCE), ("mask", MASK), ("reference", REFERENCE)):
            shutil.copyfile(path, output / f"{name}.png")
        protocol = Protocol(**hashes, source_size=size, mask_bounds=box, rendered_prompts={
            variant.name: _inpaint_prompt(form_for(variant, SEEDS[0]), ["Anna"]) for variant in VARIANTS
        })
    def save() -> None:
        manifest.write_text(protocol.model_dump_json(indent=2), encoding="utf-8")
    save()
    for variant in protocol.variants:
        if args.variants and variant.name not in args.variants:
            continue
        crop = _context_box(box, None, 0, variant.context)
        native = crop[2] - crop[0], crop[3] - crop[1]
        print(f"VARIANT {variant.name}: {variant.model}, crop {native}, model {_model_size(native, 1024)}", flush=True)
        for seed in protocol.seeds:
            previous = [result for result in protocol.results if result.variant == variant.name and result.seed == seed]
            if previous and not (args.retry_unsubmitted and all(result.job_id is None for result in previous)):
                continue
            trial = Trial(variant=variant.name, seed=seed)
            protocol.results.append(trial)
            save()
            started = time.monotonic()
            try:
                job = submit(form_for(variant, seed), output / "source.png", output / "mask.png")
                trial.job_id = job.id
                save()
                print(f"  {variant.name} seed {seed}: {job.id}", flush=True)
                status = wait_for_job(job.id)
                trial.elapsed_seconds = round(time.monotonic() - started, 3)
                trial.job_seconds = status.elapsed_seconds
                if status.status != "completed" or status.generation_id is None:
                    trial.error = status.error or status.status
                else:
                    trial.generation = GenerationRecord.model_validate_json(read_api(f"/api/history/{status.generation_id}"))
                    path = output / f"{variant.name}-{seed}.png"
                    path.write_bytes(read_api(trial.generation.output_image_url))
                    with Image.open(path) as image:
                        edited = np.asarray(image.convert("RGBA"))
                    if edited.shape != source.shape:
                        trial.error = "Output dimensions differ from source."
                    else:
                        trial.protected_pixels = int(np.count_nonzero(mask == 0))
                        trial.changed_protected_pixels = int(np.count_nonzero(np.any(source != edited, axis=2) & (mask == 0)))
                        if trial.changed_protected_pixels:
                            trial.error = "Protected pixels changed."
                print(f"  generation {trial.generation.id if trial.generation else None}, {trial.elapsed_seconds}s, protected changes {trial.changed_protected_pixels}, error {trial.error}", flush=True)
            except Exception as error:
                trial.error = f"{type(error).__name__}: {error}"
                save()
                raise
            save()
            if trial.error:
                raise RuntimeError(trial.error)
    print(f"Saved {len(protocol.results)} outcomes in {manifest}", flush=True)


if __name__ == "__main__":
    main()
