"""Retain every result and build a local review gallery for the Qwen Turbo trial."""
from __future__ import annotations

import html
import json
from pathlib import Path
import statistics

from PIL import Image, ImageDraw
from pydantic import BaseModel, ConfigDict
from benchmark_qwen_turbo import OUTPUT, Measurement, load_cases


class RunProtocol(BaseModel):
    model_config = ConfigDict(extra='ignore')
    variant: str
    encoding: str
    kv_cache: bool
    load_seconds: float | None = None
    load_error: str | None = None
    transformer_storage_gib: float | None = None


class RunData(BaseModel):
    protocol: RunProtocol
    results: list[Measurement]


def image_tile(path: Path, caption: str, box: tuple[int,int,int,int] | None) -> Image.Image:
    canvas = Image.new('RGB',(384,422),(27,31,36))
    with Image.open(path) as image:
        if box:
            image = image.crop(box)
        image = image.convert('RGBA')
        background = Image.new('RGBA',image.size,(196,196,196,255))
        d = ImageDraw.Draw(background)
        for y in range(0,image.height,24):
            for x in range(0,image.width,24):
                if (x//24+y//24)%2:
                    d.rectangle((x,y,x+23,y+23),fill=(228,228,228,255))
        image = Image.alpha_composite(background,image).convert('RGB')
        image.thumbnail((384,384),Image.Resampling.LANCZOS)
        canvas.paste(image,((384-image.width)//2,(384-image.height)//2))
    ImageDraw.Draw(canvas).text((8,391),caption,fill='white')
    return canvas


def main() -> None:
    cases = load_cases(OUTPUT)
    runs = {p.stem.removeprefix('results-'):RunData.model_validate_json(p.read_text(encoding='utf-8'))
            for p in sorted(OUTPUT.glob('results-*.json'))}
    labels = {'klein9b':'Klein 9B NVFP4 · 4 steps',
              'nvfp4':'Qwen NVFP4 · BF16 encoder', 'gguf':'Qwen GGUF Q4_K_M · BF16 encoder',
              'nvfp4-encnvfp4':'Qwen NVFP4 · native NVFP4 encoder',
              'nvfp4-encnvfp4-kv':'Qwen NVFP4 · native encoder · KV cache',
              'nvfp4-kv':'Qwen NVFP4 · BF16 encoder · KV cache',
              'nvfp4-cond512-kv':'Qwen NVFP4 · BF16 encoder · 512 conditioning',
              'nvfp4-encnvfp4-cond512-kv':'Qwen NVFP4 · native encoder · 512 conditioning',
              'gguf-cond512-kv':'Qwen GGUF Q4_K_M · BF16 encoder · 512 conditioning',
              'cycle-cond512':'Qwen NVFP4 · fresh encode + streaming · full VAE'}
    primary = [key for key in ('klein9b','nvfp4-encnvfp4-cond512-kv','gguf-cond512-kv') if key in runs]
    secondary = [key for key in runs if key not in primary]
    order = primary+secondary
    summary = {}
    table = []
    for key in order:
        run = runs[key]
        good = [r for r in run.results if r.seconds is not None and not r.error]
        times = [r.seconds for r in good]
        value = {'successes':len(good),'failures':sum(r.error is not None for r in run.results),
                 'median_render_seconds':round(statistics.median(times),3) if times else None,
                 'minimum_render_seconds':min(times) if times else None,'maximum_render_seconds':max(times) if times else None,
                 'peak_allocated_gib':max((r.peak_allocated_gib or 0 for r in good),default=0),
                 'peak_reserved_gib':max((r.peak_reserved_gib or 0 for r in good),default=0),
                 'load_seconds':run.protocol.load_seconds,'load_error':run.protocol.load_error,
                 'protected_pixel_failures':sum((r.protected_changes or 0)>0 for r in good)}
        summary[key] = value
        timing = f'{value["median_render_seconds"]} s ({value["minimum_render_seconds"]}–{value["maximum_render_seconds"]})' if times else 'No completed renders'
        table.append(f'<tr><td>{html.escape(labels.get(key,key))}</td><td>{len(good)}/{len(run.results)}</td><td>{timing}</td><td>{value["peak_allocated_gib"]:.2f} GiB</td></tr>')
    details = OUTPUT/'details'
    details.mkdir(exist_ok=True)
    sections = []
    for case in cases:
        references = ([case.source] if case.source else [])+list(case.references)
        inputs = ''.join(f'<a href="assets/{name}" target="_blank"><img src="assets/{name}" alt="Input {i+1}"></a>' for i,name in enumerate(references))
        bounds = None
        if case.mask:
            with Image.open(OUTPUT/'assets'/case.mask) as mask:
                selected = mask.getchannel('A') if mask.mode=='RGBA' else mask.convert('L')
                box = selected.getbbox()
                if box:
                    l,t,r,b = box
                    bounds = (max(0,l-48),max(0,t-48),min(mask.width,r+48),min(mask.height,b+48))
        cards,tiles = [],[]
        for seed in (7,42):
            for key in order:
                label = labels.get(key,key)
                result = next((r for r in runs[key].results if r.case==case.name and r.seed==seed),None)
                path = OUTPUT/'renders'/f'{key}--{case.name}--{seed}.png'
                title = f'{label} · seed {seed}'
                if result is None or result.error or not path.exists():
                    problem = result.error[:250] if result and result.error else runs[key].protocol.load_error or 'Not rendered'
                    css = 'fail' if (result and result.error) or runs[key].protocol.load_error else 'unrun'
                    cards.append(f'<article class="{css}" data-variant="{key}"><h3>{html.escape(title)}</h3><p>{html.escape(problem[:250])}</p></article>')
                    tile = Image.new('RGB',(384,422),(47,25,29))
                    ImageDraw.Draw(tile).text((8,8),f'{key} seed {seed}\nFAILED / NOT RENDERED',fill='white')
                    tiles.append(tile)
                    continue
                detail = details/f'{key}--{case.name}--{seed}.png'
                with Image.open(path) as image:
                    (image.crop(bounds) if bounds else image).save(detail)
                protected = f' · protected changes {result.protected_changes}' if result.protected_changes is not None else ''
                metadata = f'{result.seconds:.2f} s · {result.peak_allocated_gib:.2f} GiB{protected} · alpha {result.alpha_min}–{result.alpha_max}'
                relative = f'renders/{path.name}'
                raw = f'renders/{path.stem}-raw.png'
                cards.append(f'<article data-variant="{key}"><h3>{html.escape(title)}</h3><a href="{relative}" target="_blank"><img loading="lazy" class="result" src="details/{detail.name}" alt="{html.escape(title)}"></a><p>{metadata}</p><a href="{relative}" target="_blank">Full image</a> · <a href="{raw}" target="_blank">Raw model output</a> · <button onclick="pick(this)" data-key="{key}-{case.name}-{seed}">Mark for review</button></article>')
                tiles.append(image_tile(path,f'{key} seed {seed}',bounds))
        if tiles:
            sheet = Image.new('RGB',(384*len(order),422*2),(27,31,36))
            for i,tile in enumerate(tiles):
                sheet.paste(tile,((i%len(order))*384,(i//len(order))*422))
            sheet.save(OUTPUT/f'{case.name}-comparison.jpg',quality=94)
        sections.append(f'<section id="{case.name}"><h2>{html.escape(case.title)}</h2><div class="inputs">{inputs}</div><p>{html.escape(" · ".join(case.criteria))}</p><details><summary>Frozen prompt and selection</summary><pre>{html.escape(case.qwen_prompt)}</pre>{f"<a href=assets/{case.mask}>Selection mask</a>" if case.mask else ""}</details><div class="grid" style="--cols:{len(order)}">{"".join(cards)}</div></section>')
    nav = ''.join(f'<a href="#{c.name}">{html.escape(c.title)}</a>' for c in cases)
    filters = ''.join(f'<label><input type="checkbox" data-filter="{key}" {"checked" if key in primary else ""} onchange="filterVariants()"> {html.escape(labels.get(key,key))}</label><br>' for key in order)
    page = '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><link rel="icon" href="data:,"><title>Qwen Turbo · MosImage review</title>
<style>body{background:#10151b;color:#e8edf2;font:15px system-ui;margin:0;padding:28px}h1{font-size:32px}h2{margin-top:0}h3{font-size:13px;color:#a9d5f2}a{color:#9cd6ff}nav{display:flex;flex-wrap:wrap;gap:14px;padding:16px 0;position:sticky;top:0;background:#10151bf2;z-index:2}section{padding:24px 0;border-top:1px solid #354354}.grid{display:grid;grid-template-columns:repeat(var(--cols),minmax(220px,1fr));gap:14px;overflow-x:auto;margin-top:20px}article{background:#1c2631;border-radius:10px;padding:12px;min-width:200px}article p{font-size:12px;min-height:36px}.result{width:100%;height:340px;object-fit:contain;background:repeating-conic-gradient(#d2d2d2 0% 25%,#eee 0% 50%) 50%/24px 24px}.inputs{display:flex;gap:12px}.inputs img{height:180px;max-width:300px;object-fit:contain}table{border-collapse:collapse}td,th{border-bottom:1px solid #354354;padding:10px;text-align:left}button{margin-top:10px;background:#30495c;color:#eee;border:1px solid #63839c;border-radius:6px;padding:7px;cursor:pointer}button.selected{background:#815c2e}pre{white-space:pre-wrap}.fail{background:#40282d}.note{max-width:1000px;line-height:1.7;color:#b4c4d5}</style>
<h1>Qwen 2.1 Turbo · model review</h1><p class="note">RTX 5070 Ti 16 GB · two fixed seeds · official 8-step sigma schedule · every declared output and failure retained. Render time includes image conditioning, sampling, VAE decode, PNG save and masked compositing. Prompt encoding and model load are measured separately. Klein uses four distilled steps. Inpaint crops use the frozen comparison recipe; these are benchmark outputs. No Qwen model has been registered in the app.</p>
<p class="note"><strong>Review decision, 10 October 2026:</strong> the user prefers FLUX overall. Klein 9B remains the Quality model; Qwen is not selected for integration. This gallery retains the complete comparison and experiments for reference.</p>
<p class="note">Click a result for full resolution; “Raw model output” shows what the model changed before protected compositing. A checkerboard indicates actual alpha transparency. KV-cache rows are a separate memory experiment. The streamed lifecycle column includes a new encode and both CPU/GPU transfers. 512 conditioning keeps the output dimensions fixed but reduces source/reference detail. Your marks stay in this browser; send the case, model and seed you approve.</p><table><thead><tr><th>Variant</th><th>Complete</th><th>Median time (range)</th><th>Peak allocated CUDA</th></tr></thead><tbody>'''+''.join(table)+'</tbody></table><details><summary>Visible variants and memory experiments</summary>'+filters+'</details><nav>'+nav+'</nav>'+''.join(sections)+'''<script>function filterVariants(){let selected=[...document.querySelectorAll('input[data-filter]:checked')].map(x=>x.dataset.filter);document.querySelectorAll('article[data-variant]').forEach(a=>a.hidden=!selected.includes(a.dataset.variant));document.querySelectorAll('.grid').forEach(g=>g.style.setProperty('--cols',Math.max(1,selected.length)))}function pick(b){let k='mosimage-qwen-review:'+b.dataset.key;let v=localStorage.getItem(k)!=='yes';localStorage.setItem(k,v?'yes':'no');b.classList.toggle('selected',v);b.textContent=v?'Marked for review':'Mark for review'}document.querySelectorAll('button[data-key]').forEach(b=>{if(localStorage.getItem('mosimage-qwen-review:'+b.dataset.key)==='yes'){b.classList.add('selected');b.textContent='Marked for review'}});filterVariants()</script></html>'''
    (OUTPUT/'index.html').write_text(page,encoding='utf-8')
    (OUTPUT/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    main()
