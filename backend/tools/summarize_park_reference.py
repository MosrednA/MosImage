"""Build a static review of every Anna/park outcome, including recorded failures."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import median

from PIL import Image, ImageDraw, ImageOps
from pydantic import BaseModel

from benchmark_park_base import BaseProtocol
from benchmark_park_reference import Protocol, SEEDS


class ReviewRow(BaseModel):
    variant: str
    seed: int
    output: str | None
    seconds: float | None
    protected_changes: int | None
    error: str | None
    originally_protected_changes: int | None = None


HTML = """<!doctype html><html lang="nl"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<link rel="icon" href="data:,"><title>Anna in de tuin · MosImage test</title><style>
*{box-sizing:border-box}body{margin:0;background:#151a20;color:#e0e6ef;font:14px/1.5 'Segoe UI',sans-serif}header{padding:18px 24px;border-bottom:1px solid #38414d}h1{font-size:21px;margin:0 0 5px}p{margin:5px 0;color:#abb7c8}a{color:#b2ceff}.controls{display:flex;gap:16px;flex-wrap:wrap;padding:16px 24px;align-items:center}select,button{font:inherit;color:inherit;background:#252e3a;border:1px solid #485770;padding:7px;border-radius:6px}label{display:flex;gap:8px;align-items:center}main{display:grid;grid-template-columns:1fr 1fr;gap:16px;padding:0 24px}figure{margin:0;background:#0c1015;border:1px solid #354152;border-radius:8px;overflow:hidden}figcaption{padding:12px;font-size:15px;font-weight:600}figure img{display:block;width:100%;max-height:72vh;object-fit:contain}.meta{padding:12px;color:#b1becf;min-height:56px}.failure{color:#ffb9b9}details{margin:20px 24px}table{border-collapse:collapse;width:100%;font-size:13px}td,th{text-align:left;border-bottom:1px solid #354152;padding:8px}footer{padding:18px 24px;color:#9cabbf}.legend{display:flex;gap:12px;align-items:center}.legend img{width:68px;height:90px;object-fit:contain}@media(max-width:800px){main{grid-template-columns:1fr;padding:0 12px}.controls{padding:12px}header{padding:12px}figure img{max-height:65vh}details{margin:16px 12px}}
</style><header><h1>Anna in de tuin — dezelfde seeds, verschillende instellingen</h1><p>Alle resultaten zijn bewaard. Meer context of stappen is geen garantie voor passende schaal.</p><p>__ACCEPTANCE__</p></header>
<div class="controls"><label>Links <select id="left"></select></label><label>Rechts <select id="right"></select></label><label>Seed <select id="seed"></select></label><label><input type="checkbox" id="original"> Origineel</label><label><input type="checkbox" id="detail"> Detail</label></div>
<main><figure><figcaption id="left-title"></figcaption><img id="left-image" alt="Linker vergelijking"><div class="meta" id="left-meta"></div></figure><figure><figcaption id="right-title"></figcaption><img id="right-image" alt="Rechter vergelijking"><div class="meta" id="right-meta"></div></figure></main>
__MASKPREVIEW____GUIDEPREVIEW__<details><summary>Alle renders en mislukkingen</summary><table><thead><tr><th>Instelling</th><th>Seed</th><th>Seconden</th><th>Buiten testmasker veranderd</th><th>Oorspronkelijk beschermd veranderd</th><th>Status</th></tr></thead><tbody id="all"></tbody></table></details>
<footer><div class="legend"><img src="reference.png" alt="Anna referentie"><span>Anna-referentie · Alle foto's: 1254 × 1254.<br>De bronfoto blijft gelijk. Box-proeven gebruiken een zichtbaar ruimer testmasker binnen dezelfde rechthoek; de oorspronkelijke selectie blijft bewaard. De eerste render na laden kan extra tijd kosten.<br>Split- en Base-tijden sluiten laden, tekstencoding en warming uit; API-tijden bevatten die kosten waar van toepassing.</span></div><p><a href="results.json">Instellingen en API-resultaten</a> · __BASELINK____PROMPTLINK____CONTACTLINK____GUIDELINK____MASKLINK__<a href="review.json">Visuele beoordeling</a></p></footer>
<script>const rows=__ROWS__;let reviews=[];fetch('review.json').then(r=>r.json()).then(r=>{reviews=r.results;update()});const variants=[...new Set(rows.map(r=>r.variant))];
const el=id=>document.getElementById(id);for(const id of ['left','right'])for(const variant of variants){const option=document.createElement('option');option.value=variant;option.textContent=variant;el(id).append(option)}
for(const seed of __SEEDS__){const option=document.createElement('option');option.value=seed;option.textContent=seed;el('seed').append(option)}el('left').value=variants.includes('box-baseline')?'guide-baseline':variants.includes('guide-baseline')?'guide-baseline':variants.includes('prompt-baseline')?'prompt-baseline':'context32';el('right').value=variants.includes('box-baseline')?'box-baseline':variants.includes('guide-layout')?'guide-layout':variants.includes('prompt-parallel')?'prompt-parallel':variants.includes('prompt-contact')?'prompt-contact':variants.includes('split-small8')?'split-small8':'context320';
function show(side){const variant=el(side).value,seed=Number(el('seed').value);const candidates=rows.filter(r=>r.variant===variant&&r.seed===seed);const row=candidates.find(r=>r.output)||candidates.at(-1);el(side+'-title').textContent=variant+' · seed '+seed;const pic=el(side+'-image');if(!row?.output&&!el('original').checked){pic.removeAttribute('src');el(side+'-meta').textContent=row?.error||'Nog niet uitgevoerd';el(side+'-meta').className='meta failure';return}const filename=el('original').checked?'source.png':row.output;pic.src=el('detail').checked?filename.replace('.png','.detail.jpg'):filename;el(side+'-meta').className='meta';el(side+'-meta').textContent=el('original').checked?'Onbewerkte parkfoto':`${row.seconds?.toFixed(1)} s · buiten testmasker veranderd: ${row.protected_changes}; oorspronkelijk beschermd: ${row.originally_protected_changes??row.protected_changes}`;const review=reviews.find(r=>r.variant===variant&&r.seed===seed);if(review&&!el('original').checked)el(side+'-meta').textContent+=' · '+review.verdict+' · '+review.note}
function update(){show('left');show('right')}for(const id of ['left','right','seed','original','detail'])el(id).addEventListener('change',update);for(const row of rows){const tr=document.createElement('tr');for(const value of [row.variant,row.seed,row.seconds?.toFixed(1)??'—',row.protected_changes??'—',row.error||'Gegenereerd']){const td=document.createElement('td');td.textContent=value;tr.append(td)}el('all').append(tr)}update();</script></html>"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    args = parser.parse_args()
    root = args.input.resolve()
    protocol = Protocol.model_validate_json((root / "results.json").read_bytes())
    rows = [ReviewRow(variant=trial.variant, seed=trial.seed,
        output=f"{trial.variant}-{trial.seed}.png" if trial.generation else None,
        seconds=trial.elapsed_seconds, protected_changes=trial.changed_protected_pixels, error=trial.error)
        for trial in protocol.results]
    declared = [(variant.name, seed) for variant in protocol.variants for seed in protocol.seeds]
    for phase in ("base", "prompts", "prompts-contact", "guide-tests", "mask-tests"):
        manifest = root / phase / "results.json"
        if not manifest.is_file():
            continue
        cohort = BaseProtocol.model_validate_json(manifest.read_bytes())
        rows.extend(ReviewRow(variant=trial.variant, seed=trial.seed,
            output=f"{phase}/{trial.output}" if trial.output else None, seconds=trial.elapsed_seconds,
            protected_changes=trial.changed_protected_pixels, originally_protected_changes=trial.changed_originally_protected_pixels if trial.changed_originally_protected_pixels is not None else trial.changed_protected_pixels, error=trial.error) for trial in cohort.results)
        declared.extend((variant.name, seed) for variant in cohort.variants for seed in cohort.seeds)
    recorded = {(row.variant, row.seed) for row in rows}
    rows.extend(ReviewRow(variant=variant, seed=seed, output=None, seconds=None,
        protected_changes=None, error="Niet uitgevoerd: batch gestopt na geheugenfout of onderbreking.")
        for variant, seed in declared if (variant, seed) not in recorded)
    for name in {row.output for row in rows if row.output} | {"source.png"}:
        path = root / name
        with Image.open(path) as image:
            image.convert("RGB").crop((690, 30, 1254, 800)).save(path.with_suffix(".detail.jpg"), quality=94)
    for variant in dict.fromkeys(row.variant for row in rows):
        complete = [row for row in rows if row.variant == variant and row.output]
        if len(complete) != 3:
            continue
        sheet = Image.new("RGB", (1440, 800), (20, 25, 32))
        draw = ImageDraw.Draw(sheet)
        draw.text((15, 10), variant, fill="white")
        for index, row in enumerate(complete):
            with Image.open(root / row.output) as image:
                full = ImageOps.contain(image.convert("RGB"), (460, 350))
                sheet.paste(full, (index * 480 + (480 - full.width)//2, 40))
                detail = ImageOps.contain(image.convert("RGB").crop((690, 30, 1254, 800)), (460, 365))
                sheet.paste(detail, (index * 480 + (480-detail.width)//2, 405))
            draw.text((index*480+12, 780), f"seed {row.seed} - {row.seconds:.1f}s", fill="white")
        sheet.save(root / f"review-{variant}.jpg", quality=93)
        print(variant, "median all completed:", round(median(row.seconds for row in complete), 3))
    encoded = json.dumps([row.model_dump() for row in rows], ensure_ascii=False).replace("</", "<\\/")
    html = HTML.replace("__ROWS__", encoded).replace("__SEEDS__", json.dumps(SEEDS))
    html = html.replace("__BASELINK__", '<a href="base/results.json">Split- en Base-instellingen</a> · ' if (root / "base/results.json").is_file() else "")
    html = html.replace("__PROMPTLINK__", '<a href="prompts/results.json">Volledige promptproeven</a> · ' if (root / "prompts/results.json").is_file() else "")
    html = html.replace("__CONTACTLINK__", '<a href="prompts-contact/results.json">Compacte contactprompts</a> · ' if (root / "prompts-contact/results.json").is_file() else "")
    html = html.replace("__GUIDELINK__", '<a href="guide-tests/results.json">Visuele gidsproeven</a> · <a href="guides/manifest.json">Gidsen en generatieprompt</a> · ' if (root / "guide-tests/results.json").is_file() else "")
    html = html.replace("__GUIDEPREVIEW__", '<details><summary>Structurele testgidsen (dit zijn invoerbeelden)</summary><div class="controls"><figure><figcaption>Pose op neutrale achtergrond</figcaption><img src="guides/pose.png" alt="Neutrale zittende posegids"></figure><figure><figcaption>Dezelfde dummy op dezelfde plek in de tuin</figcaption><img src="guides/layout.png" alt="Tuin met geplaatste dummy"></figure></div><p><a href="guides/placement-preview.png">Volledige plaatsingspreview</a> · De gekozen schaal is een testhypothese; geen gemeten fysieke maat.</p></details>' if (root / "guides/manifest.json").is_file() else "")
    html = html.replace("__MASKLINK__", '<a href="mask-tests/results.json">Maskervormproeven</a> · ' if (root / "mask-tests/results.json").is_file() else "")
    html = html.replace("__MASKPREVIEW__", '<details><summary>Maskercontrole: blauwe zone mag veranderen, cyaan is de echte grens</summary><div class="controls"><figure><figcaption>Oorspronkelijke vorm snijdt door het gezicht (baseline seed 7)</figcaption><img src="mask-audit/guide-baseline-7-overlay.png" alt="Maskerrand dwars door het gezicht"></figure><figure><figcaption>Rechthoekige testvorm binnen dezelfde afmetingen</figcaption><img src="mask-audit/box-preview.png" alt="Ruimere rechthoekige testselectie"></figure></div><p>Deze box geeft 46.678 extra pixels binnen de rechthoek toestemming om te veranderen. Het originele document/masker is niet aangepast. Controleer de aparte kolommen voor testmasker en oorspronkelijke bescherming.</p></details>' if (root / "mask-audit/box-preview.png").is_file() else "")
    html = html.replace("__ACCEPTANCE__", "De eerdere 39 beelden zijn door de gebruiker afgewezen. Grootte, zithouding en grondcontact zijn de prioriteiten; prompt- en gidsproeven staan apart ter beoordeling.")
    (root / "index.html").write_text(html, encoding="utf-8")
    print(f"Review contains {len(rows)} render and skipped-attempt rows: {root / 'index.html'}")


if __name__ == "__main__":
    main()
