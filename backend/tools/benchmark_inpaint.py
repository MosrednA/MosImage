"""Compare two context/blend settings on eight predeclared, fresh seeds.

Uses the running API's single worker. Writes every result, including failures;
visual subject/scene quality must be reviewed separately from pixel protection.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from urllib.request import Request, urlopen
from uuid import uuid4

import numpy as np
from PIL import Image
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.schemas import GenerationRecord, InpaintForm, JobResponse, JobStatus

API = "http://127.0.0.1:8000"
SEEDS = (7, 42, 1234, 20261007, 87654321, 314159265, 2718281828, 4287654321)
VARIANTS = (("context320-blend16", 320, 16), ("context192-blend8", 192, 8))


class Result(BaseModel):
    variant: str
    seed: int
    job_id: str
    generation: GenerationRecord | None = None
    error: str | None = None
    elapsed_seconds: float | None = None
    protected_pixels: int | None = None
    changed_protected_pixels: int | None = None


def submit(form: InpaintForm, source: Path, mask: Path) -> JobResponse:
    boundary = "mosimage-benchmark-" + uuid4().hex
    parts = [
        f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n{value}\r\n'.encode()
        for name, value in form.model_dump(exclude_none=True).items()
    ]
    for name, path in (("image", source), ("mask", mask)):
        parts.append(
            f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"; filename="{path.name}"\r\nContent-Type: image/png\r\n\r\n'.encode()
            + path.read_bytes() + b"\r\n"
        )
    parts.append(f"--{boundary}--\r\n".encode())
    request = Request(API + "/api/inpaint", data=b"".join(parts), headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    with urlopen(request, timeout=30) as response:
        return JobResponse.model_validate_json(response.read())


def wait_for_job(job_id: str) -> JobStatus:
    deadline = time.monotonic() + 300
    while time.monotonic() < deadline:
        with urlopen(API + "/api/jobs/" + job_id, timeout=30) as response:
            status = JobStatus.model_validate_json(response.read())
        if status.status in {"completed", "failed", "cancelled"}:
            return status
        time.sleep(1)
    raise TimeoutError(f"Job {job_id} still pending; stop the benchmark before submitting more work.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--mask", type=Path, required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source_path, mask_path = args.source.resolve(), args.mask.resolve()
    with Image.open(source_path) as image:
        source = np.array(image.convert("RGBA"))
    with Image.open(mask_path) as image:
        mask = np.array(image.getchannel("A") if "A" in image.getbands() else image.convert("L"))
    if mask.shape != source.shape[:2] or not np.any(mask):
        parser.error("The mask must be nonempty and match the source dimensions.")
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = args.output / "results.json"
    results: list[Result] = []

    def save() -> None:
        manifest.write_text(json.dumps({
            "source": str(source_path), "mask": str(mask_path), "prompt": args.prompt,
            "model": "flux2-klein-4b", "steps": 4, "guidance": 1, "strength": 1,
            "seeds": SEEDS, "note": "Both variants use native soft masks; this is a paired control-setting comparison, not a binary-versus-soft ablation.",
            "results": [result.model_dump(mode="json") for result in results],
        }, indent=2), encoding="utf-8")

    save()  # Declare the full seed set before any renders.
    for seed in SEEDS:
        for label, context, blend in VARIANTS:
            form = InpaintForm(prompt=args.prompt, seed=seed, context_padding=context, blend_feather=blend)
            started = time.monotonic()
            job = submit(form, source_path, mask_path)
            result = Result(variant=label, seed=seed, job_id=job.id)
            results.append(result)
            save()
            print(f"{len(results)}/16 {label}, seed {seed}: submitted {job.id}", flush=True)
            status = wait_for_job(job.id)
            result.elapsed_seconds = round(time.monotonic() - started, 3)
            if status.status != "completed" or status.generation_id is None:
                result.error = status.error or status.status
            else:
                with urlopen(f"{API}/api/history/{status.generation_id}", timeout=30) as response:
                    result.generation = GenerationRecord.model_validate_json(response.read())
                with urlopen(API + result.generation.output_image_url, timeout=30) as response:
                    with Image.open(response) as image:
                        edited = np.array(image.convert("RGBA"))
                if edited.shape != source.shape:
                    result.error = "Output dimensions do not match the source."
                else:
                    result.protected_pixels = int(np.count_nonzero(mask == 0))
                    result.changed_protected_pixels = int(np.count_nonzero(np.any(source != edited, axis=2) & (mask == 0)))
                    if result.changed_protected_pixels:
                        result.error = "Protected pixels changed."
            save()
            print(f"  generation {result.generation.id if result.generation else None}, protected changes {result.changed_protected_pixels}, error {result.error}", flush=True)
    print(f"All outcomes saved to {manifest}. Review every image for subject presence and scene integration.", flush=True)
    if any(result.error is not None for result in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
