"""Prepare explicit pose/layout benchmark references without modifying the document."""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import shutil

import numpy as np
from PIL import Image, ImageOps

from benchmark_park_reference import Protocol
from app.compositing import prepare_inpaint_workspace

CANVAS = (384, 512)
TARGET_CENTER_X = 1056
TARGET_GROUND_Y = 610
TARGET_MAX_SIZE = (380, 420)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--cutout", type=Path, required=True)
    args = parser.parse_args()
    root = args.input.resolve()
    frozen = Protocol.model_validate_json((root / "results.json").read_bytes())
    for name in ("source", "mask", "reference"):
        if sha256((root / f"{name}.png").read_bytes()).hexdigest() != getattr(frozen, f"{name}_sha256"):
            raise ValueError(f"Frozen {name} file changed.")
    output = root / "guides"
    output.mkdir(exist_ok=False)
    shutil.copy2(args.cutout, output / "dummy-original.png")
    shutil.copy2(root / "dummy-generation.json", output / "generation.json")
    with Image.open(args.cutout) as image:
        if "A" not in image.getbands():
            raise ValueError("The generated dummy must have transparency.")
        dummy = image.convert("RGBA")
    opaque_box = dummy.getchannel("A").point(lambda value: 255 if value >= 16 else 0).getbbox()
    if opaque_box is None:
        raise ValueError("The dummy is empty.")
    dummy = ImageOps.contain(dummy.crop(opaque_box), TARGET_MAX_SIZE, Image.Resampling.LANCZOS)
    left = TARGET_CENTER_X - dummy.width // 2
    top = TARGET_GROUND_Y - dummy.height
    target = (left, top, left + dummy.width, top + dummy.height)
    with Image.open(root / "source.png") as image:
        source = image.convert("RGBA")
    with Image.open(root / "mask.png") as image:
        mask = image.getchannel("A")
    placed = Image.new("RGBA", source.size, (0,0,0,0))
    placed.paste(dummy, (left,top))
    overlaps = np.count_nonzero((np.asarray(placed.getchannel("A")) > 32) & (np.asarray(mask) == 0))
    if overlaps:
        raise ValueError(f"The guide crosses {overlaps} protected pixels; adjust placement before running.")
    preview = Image.alpha_composite(source, placed)
    preview = Image.composite(preview, source, mask)
    preview.save(output / "placement-preview.png")
    workspace = prepare_inpaint_workspace(root / "source.png", root / "mask.png", output,
        1024, 0, 32, composite_feather=0, generation_feather=0, center_selection=True)
    with Image.open(workspace.model_input_path) as image:
        model_size = image.size
        model_source = image.convert("RGBA")
    placed_model = placed.crop(workspace.model_box).resize(model_size, Image.Resampling.LANCZOS)
    neutral = Image.new("RGBA", model_size, (128,128,128,255))
    neutral = Image.alpha_composite(neutral, placed_model).convert("RGB")
    ImageOps.pad(neutral, CANVAS, Image.Resampling.LANCZOS, color=(128,128,128)).save(output / "pose.png")
    # Add the dummy after context preparation: mirror background scenery only,
    # so a bench near the image edge is not duplicated into the padded region.
    model_layout = Image.alpha_composite(model_source, placed_model).convert("RGB")
    ImageOps.pad(model_layout, CANVAS, Image.Resampling.LANCZOS, color=(128,128,128)).save(output / "layout.png")
    with Image.open(root / "reference.png") as image:
        ImageOps.pad(image.convert("RGB"), CANVAS, Image.Resampling.LANCZOS, color=(128,128,128)).save(output / "anna.png")
    names = ("anna.png", "pose.png", "layout.png", "placement-preview.png", "dummy-original.png", "generation.json")
    manifest = {
        "purpose": "Identity and structural references for a local controlled render; not an accepted edit.",
        "canvas": CANVAS, "context": 32, "model_size": model_size, "model_box": workspace.model_box,
        "target_max_size": TARGET_MAX_SIZE, "target_center_x": TARGET_CENTER_X, "target_ground_y": TARGET_GROUND_Y,
        "actual_cutout_box": target, "guide_protected_overlap_pixels": int(overlaps),
        "source_unchanged": True,
        "scale_status": "Explicit manually chosen test placement; not calibrated real-world size.",
        "generation": json.loads((output / "generation.json").read_text(encoding="utf-8")),
        "assets": {f"guides/{name}": sha256((output / name).read_bytes()).hexdigest() for name in names}
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({key: manifest[key] for key in ("canvas", "model_size", "model_box", "actual_cutout_box", "guide_protected_overlap_pixels")}, indent=2))
    print(f"Prepared identity, isolated pose and scene-layout guides: {output}")


if __name__ == "__main__":
    main()
