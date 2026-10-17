"""Build the complete retained FLUX 3 / Klein comparison gallery and summary."""
from __future__ import annotations

import html
import json
from pathlib import Path
import statistics

from PIL import Image, ImageDraw, ImageFont, ImageOps

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "data/flux3-klein-2026-10-15"


def metric(value: float | None, unit: str) -> str:
    return f"{value:.2f} {unit}" if value is not None else "Pending"


def main() -> None:
    cases = json.loads((OUTPUT / "cases.json").read_text())
    runs = {p.stem.removeprefix("results-"): json.loads(p.read_text()) for p in sorted(OUTPUT.glob("results-*.json"))}
    summaries = {}
    for name, run in runs.items():
        good = [r for r in run["results"] if r["error"] is None]
        summaries[name] = {"completed": len(good), "failed": len(run["results"])-len(good),
                           "median_seconds": statistics.median(r["seconds"] for r in good) if good else None,
                           "cached_median_seconds": statistics.median(r["seconds"] for r in good if r["seed"] == 42) if any(r["seed"] == 42 for r in good) else None,
                           "max_allocated_gib": max((r["peak_allocated_gib"] for r in good), default=None),
                           "max_reserved_gib": max((r["peak_reserved_gib"] for r in good), default=None),
                           "max_rss_gib": max((r["peak_rss_gib"] for r in good), default=None),
                           "max_card_mib": max((r["peak_card_mib"] for r in good), default=None),
                           "protected_changes": sum(r["protected_changes"] or 0 for r in good)}
    (OUTPUT / "summary.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    details = OUTPUT / "details"
    details.mkdir(exist_ok=True)
    font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 20)
    sections = []
    artifacts = []
    for case in cases:
        name = case["name"]
        box = None
        source = None
        if case["source"]:
            source = Image.open(OUTPUT / "assets" / case["source"]).convert("RGB")
        if case["mask"]:
            mask = Image.open(OUTPUT / "assets" / case["mask"]).convert("RGBA")
            if mask.getextrema()[3] == (255, 255):
                mask = mask.convert("L")
            else:
                mask = mask.getchannel("A")
            bounds = mask.getbbox()
            if bounds:
                l, t, r, b = bounds
                margin = max(30, round(max(r-l, b-t)*.12))
                box = (max(0, l-margin), max(0, t-margin), min(mask.width, r+margin), min(mask.height, b+margin))
        if source:
            display_source = source.crop(box) if box else source
            display_source.save(details / f"{name}-source.jpg", quality=94)
        cards, panels = [], []
        for seed in (7, 42):
            for arm in ("klein9b", "flux3"):
                path = OUTPUT / "renders" / f"{arm}--{name}--{seed}.png"
                result = next((r for r in runs.get(arm, {}).get("results", []) if r["case"] == name and r["seed"] == seed), None)
                caption = f"{arm} · seed {seed}"
                if result and result["seconds"] is not None:
                    caption += f" · {result['seconds']:.2f}s · {result['peak_allocated_gib']:.2f} GiB"
                if path.exists():
                    with Image.open(path) as image:
                        image.verify()
                    with Image.open(path) as image:
                        display = image.convert("RGB")
                        if box:
                            display = display.crop(box)
                        display.save(details / f"{arm}-{name}-{seed}.jpg", quality=95)
                        panels.append((caption, display.copy()))
                        artifacts.append({"path": str(path.relative_to(OUTPUT)), "size": image.size})
                    relative = f"details/{arm}-{name}-{seed}.jpg"
                    cards.append(f'<figure><a href="renders/{path.name}"><img loading="lazy" src="{relative}" alt="{html.escape(caption)}"></a><figcaption>{html.escape(caption)}</figcaption></figure>')
                else:
                    error = result["error"] if result else "Pending"
                    cards.append(f'<figure><div class="failure">{html.escape(error or "No file")}</div><figcaption>{html.escape(caption)}</figcaption></figure>')
        if panels:
            contact = Image.new("RGB", (1400, 1040), "#171a20")
            draw = ImageDraw.Draw(contact)
            for index, (caption, panel) in enumerate(panels):
                x, y = (index%2)*700, (index//2)*520
                panel = ImageOps.contain(panel, (680, 475), Image.Resampling.LANCZOS)
                contact.paste(panel, (x+(700-panel.width)//2, y+37+(475-panel.height)//2))
                draw.text((x+12, y+9), caption, fill="white", font=font)
            contact.save(OUTPUT / f"review-{name}.jpg", quality=94)
        inputs = []
        if source:
            inputs.append(f'<figure><img src="details/{name}-source.jpg" alt="Source"><figcaption>Source / selected neighbourhood</figcaption></figure>')
        for reference in case["references"]:
            inputs.append(f'<figure><img src="assets/{reference}" alt="Reference"><figcaption>Reference</figcaption></figure>')
        controls = []
        control_panels = []
        for seed in (7, 42):
            path = OUTPUT / "renders" / f"klein9b-current--{name}--{seed}.png"
            if path.exists():
                with Image.open(path) as image:
                    image.verify()
                with Image.open(path) as image:
                    display = image.convert("RGB")
                    if box:
                        display = display.crop(box)
                    display.save(details / f"klein9b-current-{name}-{seed}.jpg", quality=95)
                    control_panels.append((f"Current app prompt, seed {seed}", display.copy()))
                    artifacts.append({"path": str(path.relative_to(OUTPUT)), "size": image.size})
                controls.append(f'<figure><a href="renders/{path.name}"><img src="details/klein9b-current-{name}-{seed}.jpg" alt="Current app prompt control"></a><figcaption>Current app prompt, seed {seed}</figcaption></figure>')
        if control_panels:
            contact = Image.new("RGB", (1400, 520), "#171a20")
            draw = ImageDraw.Draw(contact)
            for index, (caption, panel) in enumerate(control_panels):
                panel = ImageOps.contain(panel, (680, 475), Image.Resampling.LANCZOS)
                contact.paste(panel, (index*700+(700-panel.width)//2, 37+(475-panel.height)//2))
                draw.text((index*700+12, 9), caption, fill="white", font=font)
            contact.save(OUTPUT / f"review-{name}-current.jpg", quality=94)
        sections.append(f'<section id="{name}"><h2>{html.escape(case["title"])}</h2><div class="inputs">{"".join(inputs)}</div><p>{html.escape(" · ".join(case["criteria"]))}</p><div class="grid">{"".join(cards)}</div><details><summary>Current app prompt control</summary><div class="grid">{"".join(controls)}</div></details><details><summary>Frozen requests and prompts</summary><p>Original app prompt (control):</p><pre>{html.escape(case["klein_prompt"])}</pre><p>Identical full-scene caption used by both primary arms:</p><pre>{html.escape(json.loads((OUTPUT/"flux3-prompts.json").read_text())[name])}</pre></details></section>')
    page = '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><link rel="icon" href="data:,"><title>FLUX 3 vs Klein 9B — MosImage</title><style>
body{font:16px system-ui;background:#101318;color:#eceff5;margin:0 auto;padding:24px;max-width:1500px}h1{font-size:32px}a{color:#8bc5ff}nav{display:flex;gap:16px;flex-wrap:wrap;position:sticky;top:0;background:#101318ed;padding:14px 0;z-index:2}section{border-top:1px solid #3a424e;margin-top:30px;padding-top:16px}.grid{display:grid;grid-template-columns:1fr 1fr;gap:16px}.inputs{display:flex;gap:15px}.inputs figure{max-width:230px}.inputs img{height:180px}figure{margin:0;background:#1c222b;padding:10px;border-radius:8px}img{width:100%;max-height:650px;object-fit:contain}figcaption{padding:8px 0;color:#bac9da}pre{white-space:pre-wrap;background:#1c222b;padding:15px}.failure{min-height:120px;color:#ffa398;padding:20px}p{line-height:1.5}table{border-collapse:collapse;margin:20px 0;width:100%}td,th{text-align:left;padding:12px;border-bottom:1px solid #3a424e}details{margin:16px 0}summary{cursor:pointer;padding:10px 0}@media(max-width:700px){.grid{grid-template-columns:1fr}}
</style><h1>FLUX 3 Image Klein vs current Klein 9B</h1><p>RTX 5070 Ti 16 GB · 15 October 2026 · two frozen seeds per case. All retained outputs, including failures. Click a render for the full final image. Cropped panels show the selected neighbourhood. FLUX 3 has no native hard mask: both arms use MosImage compositing to protect pixels.</p><p>The primary arms receive identical full-scene captions. Current application-style Klein prompts are a separate control in each section. FLUX 3 main timings use cached embeddings; prompt encoding and complete request lifecycle are reported separately.</p>'''
    nav = '<nav>'+''.join(f'<a href="#{c["name"]}">{c["name"]}</a>' for c in cases)+'</nav>'
    metrics = '<h2>Measured performance</h2><table><tr><th>Arm</th><th>Images</th><th>Cached median</th><th>Peak CUDA allocated</th><th>Peak render RSS</th></tr>'
    for name, label in (("klein9b", "Klein 9B, matched captions"), ("flux3", "FLUX 3, matched captions"), ("klein9b-current", "Klein 9B, current app prompts")):
        if name in summaries:
            s = summaries[name]
            metrics += f'<tr><td>{label}</td><td>{s["completed"]}</td><td>{metric(s["cached_median_seconds"], "s")}</td><td>{metric(s["max_allocated_gib"], "GiB")}</td><td>{metric(s["max_rss_gib"], "GiB")}</td></tr>'
    metrics += '</table><p>CUDA allocation excludes some driver/desktop memory. FLUX 3 main renders use separately encoded text; Klein retains its CPU-streamed encoder. Compare the complete lifecycle below for new-prompt cost.</p>'
    cycle = runs.get("flux3-cycle")
    if cycle:
        metrics += '<h2>Complete FLUX 3 requests</h2><table><tr><th>Case</th><th>New prompt, including reloads</th><th>Cached prompt</th><th>Peak CUDA allocated</th></tr>'
        for case in ("duo-coaster", "garden", "typography"):
            rows = [r for r in cycle["results"] if r["case"] == case and not r["error"]]
            if len(rows) == 2:
                new = next(r for r in rows if r["seed"] == 7)
                cached = next(r for r in rows if r["seed"] == 42)
                metrics += f'<tr><td>{case}</td><td>{new["seconds"]:.3f} s</td><td>{cached["seconds"]:.3f} s</td><td>{max(r["peak_allocated_gib"] for r in rows):.3f} GiB</td></tr>'
        metrics += '</table><p>Six complete requests; all outputs are pixel-identical to the primary cohort. Encoder and renderer load sequentially. The new-prompt reload cost is included; downloads and initial warm-up are excluded.</p>'
        for row in cycle["results"]:
            path = OUTPUT / "renders" / f'flux3-cycle--{row["case"]}--{row["seed"]}.png'
            if path.exists():
                with Image.open(path) as image:
                    image.verify()
                with Image.open(path) as image:
                    artifacts.append({"path": str(path.relative_to(OUTPUT)), "size": image.size})
    failures = sorted(OUTPUT.glob("failure-*.json")) + sorted((OUTPUT / "attempts").glob("*.json"))
    failure_links = ''.join(f'<li><a href="{p.relative_to(OUTPUT).as_posix()}">{html.escape(p.name)}</a></li>' for p in failures)
    diagnostics = '<details><summary>Detailed measurements</summary><pre>'+html.escape(json.dumps(summaries, indent=2))+'</pre></details>'
    diagnostics += '<details><summary>Retained setup/input attempts</summary><ul>'+failure_links+'</ul></details>'
    page = page.replace('</style>', '.table-wrap{overflow-x:auto}.inputs{flex-wrap:wrap}pre,a{overflow-wrap:anywhere}.grid figure{display:flex;flex-direction:column}.grid figcaption{order:-1}section{scroll-margin-top:70px}@media(max-width:700px){section{scroll-margin-top:180px}}</style>')
    metrics = metrics.replace('<table>', '<div class="table-wrap"><table>').replace('</table>', '</table></div>')
    (OUTPUT / "index.html").write_text(page + nav + metrics + diagnostics + ''.join(sections)+'</html>', encoding="utf-8")
    (OUTPUT / "artifact-validation.json").write_text(json.dumps({"verified_images": len(artifacts), "artifacts": artifacts}, indent=2), encoding="utf-8")
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
