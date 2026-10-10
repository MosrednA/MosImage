"""Review every retained SeedVR2 output beside identical source/SwinIR crops."""
from __future__ import annotations

from html import escape
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "data/seedvr2-3b-2026-10-10"
BOXES: dict[str, tuple[int, int, int, int]] = {
    "face": (320, 75, 560, 315), "portrait": (150, 110, 350, 310),
    "astronaut": (160, 40, 320, 200), "denim": (150, 150, 350, 350),
    "garden": (360, 470, 560, 670), "text": (60, 430, 260, 630),
}


def rgb(path: Path) -> Image.Image:
    with Image.open(path) as source:
        return source.convert("RGB")


def review(name: str, operation: str, folder: Path, generated: Path, suffix: str) -> str:
    box = BOXES[name]
    factor = 2 if operation == "native" else 1
    scaled_box = tuple(value * factor for value in box)
    width, height = (box[2] - box[0]) * factor, (box[3] - box[1]) * factor
    reference = rgb(folder / "assets" / f"{name}-reference.png")
    if factor == 2:
        reference = reference.resize((reference.width * 2, reference.height * 2), Image.Resampling.NEAREST)
    swin = ROOT / "data/upscale-fidelity-2026-10-10/swinir-light2" / f"{name}-{operation}.png"
    seed_label = generated.stem.rsplit("-", 2)[-2]
    entries = [("Source pixels (nearest 2x)" if factor == 2 else "Untouched source", reference),
               ("SwinIR Light 2x", rgb(swin)), (f"SeedVR2 3B {seed_label}", rgb(generated))]
    sheet = Image.new("RGB", (width * len(entries), height + 36), "#1b1e21")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.truetype("C:/Windows/Fonts/segoeui.ttf", 13)
    for index, (label, image) in enumerate(entries):
        draw.text((index * width + 8, 9), label, fill="white", font=font)
        sheet.paste(image.crop(scaled_box), (index * width, 36))
    filename = f"{name}-{operation}-{suffix}-review.png"
    sheet.save(OUTPUT / filename)
    return filename


def main() -> None:
    sections: list[str] = []
    count = 0
    for folder in (OUTPUT / "baseline", OUTPUT / "face-seeds", OUTPUT / "color-none"):
        if not folder.is_dir():
            continue
        sections.append(f"<h2>{escape(folder.name)}</h2>")
        for name in BOXES:
            for operation in ("native", "reconstruct"):
                for generated in sorted(folder.glob(f"{name}-{operation}-seed*-r*.png")):
                    suffix = generated.stem.removeprefix(f"{name}-{operation}-")
                    output_name = review(name, operation, folder, generated, f"{folder.name}-{suffix}")
                    relative = generated.relative_to(OUTPUT).as_posix()
                    baseline = (folder / f"{name}-{operation}-lanczos.png").relative_to(OUTPUT).as_posix()
                    count += 1
                    sections.append(f'<h3>{escape(name)} / {escape(operation)} / {escape(suffix)}</h3><p><a href="{relative}">SeedVR2 full PNG</a> · <a href="{baseline}">Lanczos full PNG</a></p><img src="{output_name}" alt="Matched source, SwinIR Light and SeedVR2 crops">')
    html = """<!doctype html><html lang="en"><meta charset="utf-8"><title>SeedVR2 3B review</title>
<style>body{font:16px system-ui;background:#171a1d;color:#eee;margin:28px}a{color:#9fcfff}img{display:block;max-width:100%;height:auto}h2{margin-top:44px}h3{margin-top:28px}p{max-width:1100px;line-height:1.5}</style>
<h1>SeedVR2 3B / natural texture</h1><p>Native: original input enlarged 2x, without high-resolution ground truth. Source nearest-neighbour exposes existing pixels. Reconstruction: the same source reduced 2x before recovery. Each image uses identical source-coordinate crops and display dimensions. SeedVR2 is generative: extra texture is inferred, and faces/letters/objects can change. Every generated result is retained. Open full PNGs to review actual pixels.</p>""" + f"<p>{count} retained outputs available.</p>" + "\n".join(sections) + "</html>"
    (OUTPUT / "index.html").write_text(html, encoding="utf-8")
    print(OUTPUT / "index.html", count, "outputs")


if __name__ == "__main__":
    main()
