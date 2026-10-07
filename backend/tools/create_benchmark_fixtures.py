"""Rebuild deterministic masks and the four-image v1 inpainting benchmark."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
from typing import Literal

from PIL import Image, ImageDraw, ImageFont
from pydantic import BaseModel, ConfigDict

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.compositing import prepare_inpaint_workspace

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "benchmarks" / "fixtures" / "v1"


class Fixture(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    image: str
    mask: str
    width: int
    height: int
    image_sha256: str
    mask_sha256: str
    operation: Literal["replace", "remove"]
    prompt: str
    criteria: list[str]
    mask_bbox: tuple[int, int, int, int]
    selected_pixels: int
    context_box: tuple[int, int, int, int]
    model_size: tuple[int, int]


class Suite(BaseModel):
    model_config = ConfigDict(extra="forbid")
    version: Literal[1] = 1
    seeds: tuple[int, ...] = (7, 42, 20261007)
    steps: int = 4
    guidance: float = 1.0
    strength: float = 1.0
    context_padding: int = 320
    blend_feather: int = 16
    max_dimension: int = 1024
    fixtures: list[Fixture]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def create_mask(name: str, size: tuple[int, int]) -> Image.Image:
    if size != (1254, 1254):
        raise ValueError("v1 mask geometry requires the original 1254 × 1254 assets.")
    mask = Image.new("L", size)
    draw = ImageDraw.Draw(mask)
    if name == "garden":
        draw.rounded_rectangle((240, 725, 540, 1030), radius=36, fill=255)
    elif name == "kitchen":
        draw.polygon([(601, 515), (955, 515), (1065, 585), (1065, 830),
                      (1215, 918), (1224, 1050), (987, 1070), (613, 981),
                      (596, 855)], fill=255)
    elif name == "portrait":
        draw.ellipse((742, 655, 850, 769), fill=255)
    elif name == "architecture":
        draw.polygon([(302, 198), (542, 249), (543, 701), (295, 716)], fill=255)
    else:
        raise ValueError(f"Unknown fixture: {name}")
    return mask


def main() -> None:
    descriptions: tuple[tuple[str, Literal["replace", "remove"], str, list[str]], ...] = (
        ("garden", "replace", "A small glowing orange mushroom growing naturally among the purple flowers.",
         ["Complete mushroom replaces the terracotta sphere", "Stem is grounded and occluded by flowers", "Glow, texture and focus match the garden"]),
        ("kitchen", "remove", "Continuous white marble countertop with subtle gray veins and natural daylight.",
         ["Mug, handle, reflection and contact shadow are removed", "Marble veins continue naturally", "No seam, smear or yellow color residue"]),
        ("portrait", "replace", "A small embroidered white daisy stitched onto the blue denim shirt.",
         ["Daisy replaces the coral brooch", "Embroidery follows the denim folds and lighting", "Face, hair and remaining shirt are protected"]),
        ("architecture", "replace", "A minimalist blue botanical illustration printed on cream paper inside the existing poster frame.",
         ["Blue botanical illustration replaces the orange leaf", "Print follows the existing perspective and paper lighting", "Frame and bricks are protected"]),
    )
    fixtures: list[Fixture] = []
    previews: list[Image.Image] = []
    font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 22)
    for name, operation, prompt, criteria in descriptions:
        source_path = FIXTURES / f"{name}.png"
        with Image.open(source_path) as source_file:
            source = source_file.convert("RGB")
        mask = create_mask(name, source.size)
        mask_path = FIXTURES / f"{name}.mask.png"
        rgba = Image.new("RGBA", source.size, (255, 255, 255, 0))
        rgba.putalpha(mask)
        rgba.save(mask_path)
        mask.save(FIXTURES / f"{name}.mask-binary.png")
        work = FIXTURES / "prepared" / name
        work.mkdir(parents=True, exist_ok=True)
        workspace = prepare_inpaint_workspace(source_path, mask_path, work, 1024, 0, 320,
                                              composite_feather=16, generation_feather=16)
        overlay = source.convert("RGBA")
        tint = Image.new("RGBA", source.size, (82, 167, 255, 0))
        tint.putalpha(mask.point(lambda value: round(value * 0.34)))
        overlay = Image.alpha_composite(overlay, tint)
        ImageDraw.Draw(overlay).rectangle(workspace.crop_box, outline=(255, 208, 91), width=5)
        overlay.save(FIXTURES / f"{name}.preview.png")
        tile = Image.new("RGB", (640, 700), (25, 30, 37))
        tile.paste(overlay.convert("RGB").resize((620, 620), Image.Resampling.LANCZOS), (10, 45))
        ImageDraw.Draw(tile).text((16, 12), f"{name.title()} · {operation}", font=font, fill=(220, 231, 245))
        ImageDraw.Draw(tile).text((16, 672), "Blue: selection   Gold: protected context crop", font=font, fill=(170, 189, 208))
        previews.append(tile)
        with Image.open(workspace.model_input_path) as model_input:
            model_size = model_input.size
        bbox = mask.getbbox()
        assert bbox is not None
        fixtures.append(Fixture(name=name, image=source_path.name, mask=mask_path.name,
            width=source.width, height=source.height, image_sha256=digest(source_path),
            mask_sha256=digest(mask_path), operation=operation, prompt=prompt, criteria=criteria,
            mask_bbox=bbox, selected_pixels=sum(mask.histogram()[1:]),
            context_box=workspace.crop_box, model_size=model_size))
    suite = Suite(fixtures=fixtures)
    (FIXTURES / "manifest.json").write_text(suite.model_dump_json(indent=2), encoding="utf-8")
    sheet = Image.new("RGB", (1280, 1400))
    for index, tile in enumerate(previews):
        sheet.paste(tile, ((index % 2) * 640, (index // 2) * 700))
    sheet.save(FIXTURES / "overview.jpg", quality=94)
    print(suite.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
