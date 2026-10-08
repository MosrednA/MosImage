"""Compare the curved selection with a box using identical crop, prompt and seeds."""
from __future__ import annotations

import argparse
from hashlib import sha256
from importlib.metadata import version
import json
from pathlib import Path

from PIL import Image
from benchmark_park_base import BaseProtocol, BaseVariant, run_cohort
from benchmark_park_guides import GUIDED
from benchmark_park_prompts import PARALLEL
from app.compositing import _selection_alpha
from app.models.flux2_klein_nvfp4 import QUANT_REVISION


def prepare_box_mask(root: Path) -> dict[str, str]:
    with Image.open(root / "mask.png") as image:
        original = _selection_alpha(image)
    bounds = original.getbbox()
    if bounds is None:
        raise ValueError("Original selection is empty.")
    left, top, right, bottom = bounds
    core = Image.new("L", (right-left, bottom-top), 255)
    pixels = core.load()
    for y in range(core.height):
        for x in range(core.width):
            distance = min(x+1, y+1, core.width-x, core.height-y)
            t = min(distance/16, 1)
            pixels[x, y] = max(1, round(255*t*t*(3-2*t)))
    mask = Image.new("L", original.size, 0)
    mask.paste(core, (left, top))
    name = "mask-audit/box-mask.png"
    path = root / name
    path.parent.mkdir(exist_ok=True)
    if path.exists():
        with Image.open(path) as existing:
            if existing.convert("L").tobytes() != mask.tobytes():
                raise ValueError("Existing diagnostic mask differs; do not overwrite it.")
    else:
        mask.save(path)
    return {name: sha256(path.read_bytes()).hexdigest()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.input.resolve()
    assets = json.loads((root / "guides/manifest.json").read_text(encoding="utf-8"))["assets"]
    assets.update(prepare_box_mask(root))
    mask = "mask-audit/box-mask.png"
    protocol = BaseProtocol(
        model="Klein Distilled NVFP4 paired mask-shape comparison", revision=QUANT_REVISION,
        prompt=PARALLEL, prompts={"plain": PARALLEL, "guided": GUIDED}, asset_hashes=assets,
        variants=(
            BaseVariant(name="box-baseline", steps=8, mask_file=mask, reference_files=("guides/anna.png",)),
            BaseVariant(name="box-layout", steps=8, mask_file=mask, prompt_key="guided", reference_files=("guides/anna.png", "guides/layout.png")),
        ), per_render_seconds_limit=15,
        packages={name: version(name) for name in ("torch", "diffusers", "transformers", "accelerate", "safetensors", "comfy-kitchen")}
    )
    run_cohort(root, args.output, protocol)


if __name__ == "__main__":
    main()
