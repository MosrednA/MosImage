"""Create matched-size review crops and links to every retained SR output."""
from __future__ import annotations

from html import escape
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "data/upscale-fidelity-2026-10-10"
MODELS = (("swinir-light2", "SwinIR Light 2x"), ("swinir-classical2", "SwinIR M 2x"),
          ("swin2sr-classical2", "Swin2SR 2x"))
BOXES: dict[str, tuple[int, int, int, int]] = {
    "face": (340, 90, 540, 290), "portrait": (150, 110, 350, 310),
    "astronaut": (160, 40, 320, 200), "denim": (150, 150, 350, 350),
    "garden": (360, 470, 560, 670), "text": (60, 430, 260, 630),
}


def rgb(path: Path) -> Image.Image:
    with Image.open(path) as source:
        return source.convert("RGB")


def sheet(name: str, operation: str, entries: list[tuple[str, Image.Image]], box: tuple[int, int, int, int]) -> None:
    factor = 2 if operation == "native" else 1
    crop_box = tuple(value * factor for value in box)
    width, height = (box[2] - box[0]) * factor, (box[3] - box[1]) * factor
    result = Image.new("RGB", (width * len(entries), height + 36), "#1b1e21")
    draw = ImageDraw.Draw(result)
    font = ImageFont.truetype("C:/Windows/Fonts/segoeui.ttf", 13)
    for index, (label, image) in enumerate(entries):
        draw.text((index * width + 8, 9), label, font=font, fill="white")
        result.paste(image.crop(crop_box), (index * width, 36))
    result.save(OUTPUT / f"{name}-{operation}-review.png")


def main() -> None:
    sections: list[str] = []
    for operation in ("native", "reconstruct"):
        sections.append(f"<h2>{escape(operation)}</h2>")
        for name, box in BOXES.items():
            base = OUTPUT / MODELS[0][0]
            reference = rgb(base / "assets" / f"{name}-reference.png")
            source = reference.resize((reference.width * 2, reference.height * 2), Image.Resampling.NEAREST) if operation == "native" else reference
            label = "Source pixels (nearest 2x)" if operation == "native" else "Untouched source"
            entries = [(label, source), ("Lanczos 2x", rgb(base / f"{name}-{operation}-lanczos.png"))]
            entries.extend((title, rgb(OUTPUT / model / f"{name}-{operation}.png")) for model, title in MODELS)
            sheet(name, operation, entries, box)
            if operation == "native" and name in ("face", "portrait", "astronaut"):
                sheet(f"{name}-light", operation, entries[:3], box)
            links = " · ".join(f'<a href="{model}/{name}-{operation}.png">{escape(title)} full PNG</a>' for model, title in MODELS)
            sections.append(f'<h3>{escape(name)}</h3><p>{links}</p><img src="{name}-{operation}-review.png" alt="{escape(name)} matched crops">')
    old = ROOT / "data/upscale-benchmark-2026-10-09/x2plus/face-2x.png"
    if old.is_file():
        base = OUTPUT / "swinir-light2"
        entries = [("Untouched source", rgb(base / "assets/face-reference.png")),
                   ("Rejected x2plus reconstruction", rgb(old)),
                   ("SwinIR Light reconstruction", rgb(base / "face-reconstruct.png"))]
        sheet("face-rejected", "reconstruct", entries, (320, 75, 560, 315))
        sections.insert(0, '<h2>Same reconstruction task as the rejected face</h2><img src="face-rejected-reconstruct-review.png" alt="Source, rejected x2plus and SwinIR Light">')
    html = """<!doctype html><html lang="en"><meta charset="utf-8"><title>SR fidelity review</title>
<style>body{font:16px system-ui;background:#171a1d;color:#eee;margin:28px}a{color:#9fcfff}img{display:block;max-width:100%;height:auto}h2{margin-top:44px}h3{margin-top:28px}p{max-width:1100px;line-height:1.5}</style>
<h1>Natural texture / 2x fidelity review</h1><p>Every candidate and all six cases are retained. Native: original input enlarged 2x, without a high-resolution ground truth. The first column uses nearest-neighbour to expose existing source pixels; Lanczos is the smooth geometric baseline. Reconstruction: the same source is reduced 2x before recovery; its untouched source is the reference. No denoise, face repair, added grain or diffusion. Warm FP32, 256px tiles with 64px context. Open full PNGs to judge at actual pixels.</p>""" + "\n".join(sections) + "</html>"
    (OUTPUT / "index.html").write_text(html, encoding="utf-8")
    print(OUTPUT / "index.html")


if __name__ == "__main__":
    main()
