"""Local, pinned LoRA/outpaint/detail comparisons; run from backend with API unloaded.

Each stage runs in its own process so only one model is resident. All declared
seeds and full-resolution raw outputs survive, including unsuccessful results.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from hashlib import sha256
import html
import json
from pathlib import Path
import sys
import time
from typing import TYPE_CHECKING
from importlib.metadata import version

import numpy as np
import psutil
from PIL import Image, ImageDraw, ImageOps
from pydantic import BaseModel, TypeAdapter

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
from app.compositing import InpaintWorkspace, prepare_inpaint_workspace, prepare_outpaint_workspace
from app.jobs import JobControl, JobManager
from app.models.flux2_klein import Flux2KleinProvider, PipelineTask, _fit_area
from app.models.flux2_klein_nvfp4 import (BASE_REPO, BASE_REVISION, BASE_REVISION_9B,
                                        QUANT_REVISION, QUANT_REVISION_9B,
                                        Flux2Klein9bNvfp4Provider, Flux2KleinNvfp4Provider)
from app.models.klein_lora import KleinLora
from app.models.nvfp4_runtime import kernel_calls
from app.runtime import torch_module
from app.schemas import EditRequest, InpaintForm, OutpaintForm
from app.services import prepare_outpaint_images
from benchmark_inpaint_quality import assert_api_unloaded
from benchmark_model_comparison import Case as RetainedCase

if TYPE_CHECKING:
    from diffusers import Flux2KleinInpaintPipeline, Flux2KleinPipeline

CONSISTENCY = ROOT / "models/research-cache/Flux2-Klein-9B-Consistency/Flux2-Klein-9B-consistency-V2.safetensors"
FAL = ROOT / "models/research-cache/flux-2-klein-4B-outpaint-lora/flux-outpaint-lora.safetensors"
BASE4 = ROOT / "models/research-cache/FLUX.2-klein-base-4B/transformer"
SEEDS = (7, 42, 20261009)
REVISIONS = {"consistency": "8df0c7338cf68cfcd89ca7e461fe679905634607",
             "fal": "b11770ac6a3cf9325dcf81742c12b1c4e257880f",
             "base4": "a3b4f4849157f664bdbc776fd7453c2783562f4d",
             "distilled4_components": BASE_REVISION, "distilled9": BASE_REVISION_9B,
             "nvfp4_9": QUANT_REVISION_9B, "nvfp4_4": QUANT_REVISION}


def file_hash(path: Path) -> str:
    digest = sha256()
    with path.open('rb') as file:
        for block in iter(lambda: file.read(8*1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class Case:
    name: str
    source: Path
    request: EditRequest | InpaintForm
    mask: Path | None = None
    detail_box: tuple[int, int, int, int] | None = None


class Measurement(BaseModel):
    case: str
    variant: str
    seed: int
    seconds: float
    peak_allocated_gib: float
    peak_reserved_gib: float
    protected_changes: int
    unchanged_region_mae: float | None = None
    native_kernel_calls: int
    pipeline_seconds: float | None = None
    rss_after_gib: float | None = None
    available_ram_gib: float | None = None
    error: str | None = None


def cases_for(stage: str, assets: Path) -> list[Case]:
    cases: list[Case] = []
    fixtures = ROOT / "data/model-comparison-2026-10-09"
    old = {c.name:c for c in TypeAdapter(list[RetainedCase]).validate_json((fixtures / "cases.json").read_text())}
    if stage == "consistency":
        for name in ("dj", "sword-edit", "shirt"):
            c = old[name]
            if c.source is None:
                raise ValueError(f'Missing source fixture: {name}')
            source = assets / c.source
            source.write_bytes((fixtures / "assets" / c.source).read_bytes())
            mask = None
            if c.mask:
                mask = assets / c.mask
                mask.write_bytes((fixtures / "assets" / c.mask).read_bytes())
                request = InpaintForm(prompt=c.klein_prompt, context_padding=c.context, blend_feather=0)
            else:
                request = EditRequest(prompt=c.klein_prompt)
            cases.append(Case(name, source, request, mask))
    if stage in ("consistency", "outpaint"):
        for name in (("garden",) if stage == "consistency" else ("garden", "kitchen")):
            source = (ROOT / f"benchmarks/fixtures/v1/{name}.png").read_bytes()
            # 20% per expanded side, within the published 5–25% recipe.
            margin = 256
            request = OutpaintForm(prompt="the existing garden and path" if name == "garden" else "the kitchen and countertop",
                                   left=margin if name == "kitchen" else 0, right=margin,
                                   top=margin if name == "kitchen" else 0, bottom=margin if name == "kitchen" else 0)
            canvas, mask = prepare_outpaint_images(source, request)
            path, mask_path = assets / f"{name}-canvas.png", assets / f"{name}-selection.png"
            path.write_bytes(canvas)
            mask_path.write_bytes(mask)
            cases.append(Case(name, path, request, mask_path))
    if stage == "detail":
        for name, box, prompt in (
            ("architecture", (390, 340, 860, 850), "Restore fine natural brick and stone texture and clean architectural edges. Keep the exact building, poster artwork, windows, geometry, lighting and colors."),
            ("kitchen", (450, 630, 1050, 1030), "Restore subtle marble veins and natural material texture on the countertop. Keep the exact objects, layout, lighting and colors."),
            ("portrait", (450, 800, 900, 1200), "Restore fine natural woven fabric texture on the blue shirt. Keep the exact folds, clothing shape, person, lighting and colors."),
        ):
            source = assets / f"{name}.png"
            source.write_bytes((ROOT / f"benchmarks/fixtures/v1/{name}.png").read_bytes())
            mask = Image.new("L", (1254, 1254))
            ImageDraw.Draw(mask).rectangle((box[0], box[1], box[2]-1, box[3]-1), fill=255)
            mask_path = assets / f"{name}-selection.png"
            mask.save(mask_path)
            cases.append(Case(name, source, EditRequest(prompt=prompt), mask_path, box))
    return cases


class Base4Provider(Flux2KleinProvider):
    """Research-only undistilled transformer with identical cached 4B components."""
    display_name = "Klein Base 4B research"
    adapter: KleinLora

    def _create_pipeline(self, task: PipelineTask, control: JobControl) -> Flux2KleinPipeline:
        from diffusers import Flux2Transformer2DModel, Flux2KleinPipeline
        if task != 'text-to-image':
            raise ValueError('The published green-border recipe uses image-conditioned editing.')
        control.raise_if_cancelled()
        transformer = Flux2Transformer2DModel.from_pretrained(BASE4, torch_dtype=self._load_torch().bfloat16)
        self.adapter = KleinLora.attach(transformer, FAL, control.raise_if_cancelled)
        return Flux2KleinPipeline.from_pretrained(BASE_REPO, revision=BASE_REVISION,
            transformer=transformer, torch_dtype=self._load_torch().bfloat16, is_distilled=False,
            local_files_only=True)


class Fast4Provider(Flux2KleinNvfp4Provider):
    """Explicit ablation: the FAL weights on the app's distilled 4B NVFP4 model."""
    adapter: KleinLora

    def _create_pipeline(self, task: PipelineTask, control: JobControl) -> Flux2KleinPipeline | Flux2KleinInpaintPipeline:
        pipe=super()._create_pipeline(task,control)
        self.adapter=KleinLora.attach(pipe.transformer,FAL,control.raise_if_cancelled)
        return pipe


def workspace(case: Case, work: Path) -> InpaintWorkspace:
    if case.mask is None:
        raise ValueError("A selected case needs a mask")
    if isinstance(case.request, OutpaintForm):
        return prepare_outpaint_workspace(case.source, case.mask, work, 1024, .5, 128)
    padding = case.request.context_padding if isinstance(case.request, InpaintForm) else 96
    return prepare_inpaint_workspace(case.source, case.mask, work, 1024, 0, padding,
                                     composite_feather=0 if case.detail_box is None else 16,
                                     center_selection=True)


def gallery(output: Path, cases: list[Case], results: list[Measurement]) -> None:
    sections = []
    for case in cases:
        cards = []
        tiles = []
        variants = ('whole','crop') if case.detail_box else ('current9','fast4-green','fast4-fal-1.1','base4-green','base4-fal-1.1') if any(r.variant.startswith(('base4','fast4')) or r.variant=='current9' for r in results) else ('baseline','consistency-0.6','consistency-1.0')
        columns = len(variants)
        for seed in SEEDS:
            for variant in variants:
                r = next((item for item in results if item.case==case.name and item.seed==seed and item.variant==variant),None)
                stopped = variant.startswith('base4') and (output/'stopped-base4.json').exists()
                caption = f'{variant} · seed {seed} · {"stopped (latency)" if stopped else "pending"}'
                tile = Image.new('RGB',(512,512),(25,25,25))
                if r is None:
                    cards.append(f'<figure><p>{html.escape(caption)}</p></figure>')
                    ImageDraw.Draw(tile).text((8,8),caption,fill='white')
                    tiles.append(tile)
                    continue
                folder = f"{r.case}-{r.variant}-{r.seed}"
                caption = f"{r.variant} · seed {r.seed} · {r.seconds:.2f}s · {r.error or 'OK'}"
                links = ' '.join(f'<a href="{folder}/{name}">{label}</a>' for name,label in (
                    ('raw-model-output.png','raw'),('output.png','crop'),('whole-output.png','whole edit'),
                    ('green-input.png','green input'),('mask.png','generation mask')) if (output/folder/name).exists())
                cards.append(f'<figure><a href="{folder}/composite.png"><img src="{folder}/composite.png"></a><figcaption>{html.escape(caption)} · {r.protected_changes} protected changes · {links}</figcaption></figure>')
                path = output / folder / "composite.png"
                with Image.open(path) as image:
                    if case.detail_box:
                        image = image.crop(case.detail_box)
                    tile = ImageOps.pad(image.convert("RGB"), (512, 512), color=(25,25,25))
                    ImageDraw.Draw(tile).text((8,8), caption, fill="white", stroke_width=1, stroke_fill="black")
                    tiles.append(tile)
        if tiles:
            sheet = Image.new("RGB", (columns*512, ((len(tiles)+columns-1)//columns)*512), (25,25,25))
            for i, tile in enumerate(tiles):
                sheet.paste(tile, ((i%columns)*512, (i//columns)*512))
            sheet.save(output / f"{case.name}-comparison.jpg", quality=95)
        selection = f' · <a href="assets/{case.mask.name}">selection</a>' if case.mask else ''
        original = f'<figure class="original"><a href="assets/{case.source.name}"><img src="assets/{case.source.name}"></a><figcaption>Original input{selection}</figcaption></figure>'
        sections.append(f'<h2>{html.escape(case.name)}</h2>{original}<section style="grid-template-columns:repeat({columns},1fr)">{"".join(cards)}</section>')
    (output / "index.html").write_text('<!doctype html><html lang="en"><meta charset="utf-8"><title>Quality candidates</title><style>body{background:#181818;color:#eee;font:16px system-ui;margin:24px}section{display:grid;grid-template-columns:repeat(3,1fr);gap:16px}figure{margin:0}img{max-width:100%}</style><h1>All declared seeds and outputs</h1>'+''.join(sections), encoding="utf-8")


def render_detail(provider: Flux2KleinProvider, case: Case, w: InpaintWorkspace,
                  work: Path, variant: str, seed: int, control: JobControl) -> InpaintWorkspace:
    assert isinstance(case.request, EditRequest)
    source = case.source if variant == 'whole' else w.model_input_path
    dest = work/'whole-output.png' if variant == 'whole' else w.model_output_path
    provider.edit(case.request.model_copy(update={'seed':seed}),source,dest,control)
    if variant == 'whole':
        with Image.open(dest) as generated:
            generated.crop(w.crop_box).save(w.model_output_path)
        w = InpaintWorkspace(w.source,w.composite_mask,w.crop_box,w.model_input_path,
                            w.model_mask_path,w.model_output_path)
    w.composite_to(work/'composite.png')
    return w


def render_green(provider: Base4Provider | Fast4Provider, case: Case, w: InpaintWorkspace,
                 work: Path, seed: int, control: JobControl, steps: int, guidance: float) -> InpaintWorkspace:
    assert case.mask is not None
    pipe = provider._pipes['text-to-image']
    torch = provider._load_torch()
    with Image.open(case.source) as image, Image.open(case.mask) as mask:
        canvas = image.convert('RGB')
        canvas.paste((0,255,0),mask=mask.convert('L'))
    canvas = _fit_area(canvas)
    canvas.save(work/'green-input.png')
    embeds = provider._encode_prompt(pipe,'Fill the green spaces according to the image',control)
    result = pipe(image=canvas,prompt_embeds=embeds,height=canvas.height,width=canvas.width,
                  num_inference_steps=steps,guidance_scale=guidance,
                  generator=torch.Generator('cuda').manual_seed(seed),
                  callback_on_step_end=control.diffusion_callback(steps)).images[0]
    result.save(work/'raw-model-output.png')
    result.resize(w.source.size,Image.Resampling.LANCZOS).crop(w.crop_box).save(w.model_output_path)
    w = InpaintWorkspace(w.source,w.composite_mask,w.crop_box,w.model_input_path,w.model_mask_path,
                         w.model_output_path,seam_mask=w.seam_mask,seam_band=w.seam_band,
                         color_reference_mask=w.color_reference_mask)
    w.composite_to(work/'composite.png')
    return w


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("consistency", "outpaint", "detail"))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--outpaint-arm", choices=("current9", "base4", "fast4"), default="current9")
    args = parser.parse_args()
    assert_api_unloaded()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    assets = output / "assets"
    assets.mkdir(exist_ok=True)
    cases = cases_for(args.stage, assets)
    variants = ("baseline", "consistency-0.6", "consistency-1.0") if args.stage == "consistency" else ("whole", "crop") if args.stage == "detail" else ("current9",) if args.outpaint_arm == "current9" else (f"{args.outpaint_arm}-green", f"{args.outpaint_arm}-fal-1.1")
    protocol_path = output / f"protocol-{args.outpaint_arm if args.stage == 'outpaint' else args.stage}.json"
    checkpoints = (CONSISTENCY,) if args.stage=='consistency' else (FAL,BASE4/'diffusion_pytorch_model.safetensors') if args.stage=='outpaint' and args.outpaint_arm=='base4' else (FAL,) if args.stage=='outpaint' and args.outpaint_arm=='fast4' else ()
    protocol={"seeds": SEEDS, "variants": variants, "revisions": REVISIONS,
        "runner_sha256":file_hash(Path(__file__)),
        "packages": {name:version(name) for name in ('torch','diffusers','transformers','comfy-kitchen')},
        "files": {str(p.relative_to(ROOT)): file_hash(p) for p in checkpoints},
        "inputs": {p.name:sha256(p.read_bytes()).hexdigest() for p in assets.iterdir()},
        "cases": [{"name":c.name,"request":c.request.model_dump(),"detail_box":c.detail_box} for c in cases],
        "sampling": "4 steps/guidance 1 for current9 and distilled Fast4; undistilled Base4 50 steps/guidance 4; FAL green RGB(0,255,0), exact published prompt, scale 1.1",
        "detail": "Native 1254x1254 sources; whole vs 96px context crop, both approximately 1MP model edits; same final 16px inward feather and protected pixels"}
    # Never overwrite an earlier protocol while silently mixing changed runs.
    serialized=json.dumps(protocol,indent=2)
    if protocol_path.exists() and json.loads(protocol_path.read_text())!=json.loads(serialized):
        raise ValueError('The existing protocol differs. Use a fresh output folder.')
    protocol_path.write_text(serialized,encoding='utf-8')
    torch = torch_module()
    provider = Base4Provider() if args.stage == "outpaint" and args.outpaint_arm == "base4" else Fast4Provider() if args.stage=='outpaint' and args.outpaint_arm=='fast4' else Flux2Klein9bNvfp4Provider()
    manager = JobManager()
    results = [Measurement.model_validate(r) for r in json.loads((output/'results.json').read_text())] if (output/'results.json').exists() else []
    adapter: KleinLora | None = None
    try:
        for case in cases:
            for seed in SEEDS:
                for variant in variants:
                    work = output / f"{case.name}-{variant}-{seed}"
                    work.mkdir(exist_ok=True)
                    if any(r.case==case.name and r.seed==seed and r.variant==variant for r in results):
                        continue
                    w = workspace(case, work) if case.mask else None
                    def run(control: JobControl) -> None:
                        nonlocal adapter, w
                        load_started=time.perf_counter()
                        pipe = provider._load_pipe(provider._placement_for("balanced"), "text-to-image", control)
                        pipeline_seconds=time.perf_counter()-load_started
                        pipe.set_progress_bar_config(disable=True)
                        if adapter:
                            adapter.set_scale(0 if variant=="baseline" else .6 if variant.endswith("0.6") else 1)
                        if isinstance(provider, (Base4Provider,Fast4Provider)):
                            provider.adapter.set_scale(1.1 if variant.endswith("1.1") else 0)
                        before = kernel_calls(pipe.transformer)
                        torch.cuda.reset_peak_memory_stats()
                        started = time.perf_counter()
                        if args.stage == "detail":
                            assert w is not None
                            w=render_detail(provider,case,w,work,variant,seed,control)
                        elif isinstance(provider, (Base4Provider,Fast4Provider)):
                            assert w is not None
                            w=render_green(provider,case,w,work,seed,control,
                                           50 if isinstance(provider,Base4Provider) else 4,
                                           4 if isinstance(provider,Base4Provider) else 1)
                        elif w:
                            provider.inpaint(case.request.model_copy(update={"seed":seed}),w.model_input_path,w.model_mask_path,w.model_output_path,control)
                            w.composite_to(work/'composite.png')
                        else:
                            assert isinstance(case.request,EditRequest)
                            provider.edit(case.request.model_copy(update={"seed":seed}),case.source,work/'composite.png',control)
                        seconds=time.perf_counter()-started
                        with Image.open(work/'composite.png') as image, Image.open(case.source) as source:
                            mode=w.source.mode if w else 'RGB'
                            result=np.asarray(image.convert(mode)); original=np.asarray(source.convert(mode))
                        changes=0; mae=None
                        if w:
                            protected=np.asarray(w.composite_mask)==0
                            changes=int(np.count_nonzero(np.any(result!=original,axis=2)&protected))
                        if case.name=='sword-edit':
                            # The change is above this line; face/hoodie/street remain a diagnostic region.
                            mae=float(np.abs(result[800:].astype(float)-original[800:].astype(float)).mean())
                        measurement=Measurement(case=case.name,variant=variant,seed=seed,seconds=round(seconds,3),
                            peak_allocated_gib=round(torch.cuda.max_memory_allocated()/1024**3,3),
                            peak_reserved_gib=round(torch.cuda.max_memory_reserved()/1024**3,3),
                            protected_changes=changes,unchanged_region_mae=mae,native_kernel_calls=kernel_calls(pipe.transformer)-before)
                        measurement.pipeline_seconds=round(pipeline_seconds,3)
                        measurement.rss_after_gib=round(psutil.Process().memory_info().rss/1024**3,3)
                        measurement.available_ram_gib=round(psutil.virtual_memory().available/1024**3,3)
                        if args.stage=='consistency' and adapter is None:
                            assert isinstance(case.request,EditRequest)
                            adapter=KleinLora.attach(pipe.transformer,CONSISTENCY,control.raise_if_cancelled)
                            zero_path=work/'zero-scale-proof.png'
                            provider.edit(case.request.model_copy(update={"seed":seed}),case.source,zero_path,control)
                            with Image.open(zero_path) as zero:
                                zero_equal=bool(np.array_equal(np.asarray(zero.convert('RGB')),result))
                            if not zero_equal: raise RuntimeError('Loaded scale zero changed the baseline')
                            derived=provider._load_pipe('cuda','inpaint',control)
                            if derived.transformer is not pipe.transformer: raise RuntimeError('Task wrappers stopped sharing weights')
                            adapter.detach()
                            detach_path=work/'detached-proof.png'
                            provider.edit(case.request.model_copy(update={"seed":seed}),case.source,detach_path,control)
                            with Image.open(detach_path) as detached:
                                detach_equal=bool(np.array_equal(np.asarray(detached.convert('RGB')),result))
                            if not detach_equal: raise RuntimeError('Detached LoRA changed the baseline')
                            adapter=KleinLora.attach(pipe.transformer,CONSISTENCY,control.raise_if_cancelled)
                            (output/'adapter.json').write_text(json.dumps({"targets":len(adapter.targets),"bytes":adapter.storage_bytes,
                                "zero_scale_pixel_exact":zero_equal,"detached_pixel_exact":detach_equal,
                                "shared_task_transformer":True}),encoding="utf-8")
                        results.append(measurement)
                        if changes: raise RuntimeError('Protected pixels changed')
                    job=manager.submit('outpaint' if isinstance(case.request,OutpaintForm) else 'inpaint' if isinstance(case.request,InpaintForm) else 'edit',run)
                    assert job.future is not None
                    job.future.result()
                    status=manager.status(job.id)
                    if status is None or status.status!='completed':
                        (work/'failure.json').write_text(json.dumps({'case':case.name,'variant':variant,'seed':seed,
                            'error':status.error if status else 'Missing job status'}),encoding='utf-8')
                        raise RuntimeError(status.error if status else 'Missing job')
                    (output/'results.json').write_text(json.dumps([r.model_dump() for r in results],indent=2),encoding='utf-8')
                    gallery(output,cases,results)
                    print(results[-1].model_dump_json(),flush=True)
    finally:
        if adapter: adapter.detach()
        manager.shutdown()
        provider.unload()


if __name__ == '__main__':
    main()
