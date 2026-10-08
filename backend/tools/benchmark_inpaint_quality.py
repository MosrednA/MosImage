"""Reproducible local mask/reference ablations. Run from backend/ with API unloaded.

No app documents are changed. Every declared seed, input, mask and raw output is
retained. Split text/renderer timing is explicitly not interactive app latency.
"""
from __future__ import annotations

import argparse
import gc
from hashlib import sha256
from importlib.metadata import version
import json
from pathlib import Path
import sys
import time
from urllib.error import URLError
from urllib.request import urlopen

import numpy as np
from PIL import Image, ImageDraw, ImageOps
import psutil
from pydantic import BaseModel, ConfigDict

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
from app.compositing import _inner_feather, _selection_alpha, prepare_inpaint_workspace
from app.models.flux2_klein import _inpaint_prompt
from app.models.flux2_klein_nvfp4 import BASE_REPO, BASE_REVISION, QUANT_REVISION
from app.runtime import torch_module
from app.schemas import InpaintForm
from benchmark_park_prompts import PARALLEL
from benchmark_park_guides import GUIDED

PARK = ROOT / "data/park-scale-2026-10-07"
FIXTURES = ROOT / "benchmarks/fixtures/v1"


class Recipe(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    name: str
    source: str
    mask: str
    references: tuple[str, ...] = ()
    prompt: str
    context: int = 32
    steps: int = 4
    max_dimension: int = 1024
    strength: float = 1.0
    center_selection: bool = True
    upscale_long_edge: int | None = None


class Result(BaseModel):
    recipe: str
    seed: int
    seconds: float | None = None
    peak_allocated_gib: float | None = None
    peak_reserved_gib: float | None = None
    rss_after_gib: float | None = None
    available_ram_after_gib: float | None = None
    protected_changes: int | None = None
    selected_changes: int | None = None
    kernel_calls: int | None = None
    error: str | None = None


def assert_api_unloaded() -> None:
    try:
        with urlopen("http://127.0.0.1:8000/api/models", timeout=3) as response:
            if any(m["loaded"] for m in json.load(response)):
                raise RuntimeError("Unload the API model before this isolated benchmark")
    except URLError as error:
        if not isinstance(error.reason, ConnectionRefusedError):
            raise


def prepare(output: Path) -> list[Recipe]:
    assets = output / "assets"
    assets.mkdir()
    with Image.open(PARK / "source.png") as image:
        image.save(assets / "park.png")
    with Image.open(PARK / "mask.png") as image:
        original = _selection_alpha(image)
    original.save(assets / "curved.png")
    box = original.getbbox()
    if box is None:
        raise ValueError("Empty park mask")
    binary = Image.new("L", original.size)
    binary.paste(255, box)
    for feather in (0, 8, 16, 32):
        _inner_feather(binary, feather).save(assets / f"box{feather}.png")
    # A manually positioned, wider/shorter insertion zone for this scene only.
    # This supplies placement through the user's mask, without a pose image.
    target = Image.new("L", original.size)
    target.paste(255, (820, 210, min(1254, original.width), 680))
    _inner_feather(target, 16).save(assets / "target-mask.png")
    with Image.open(PARK / "reference.png") as image:
        full = image.convert("RGB")
        for size in (384, 512, 768):
            ref = full.copy()
            ref.thumbnail((size, size), Image.Resampling.LANCZOS)
            ref.save(assets / f"anna{size}.png")
        # Explicit hand-inspected crop of this fixture; never a generic face detector.
        face = full.crop((280, 15, 730, 530))
        for name, ref in (("identity", full), ("face", face)):
            ImageOps.pad(ref, (384, 512), color=(232, 232, 232)).save(assets / f"{name}.png")
    for name in ("pose", "layout"):
        with Image.open(PARK / f"guides/{name}.png") as image:
            image.save(assets / f"{name}.png")
    with Image.open(PARK / "guides/placement-preview.png") as image:
        image.save(assets / "placed.png")
    def park(name: str, **changes: object) -> Recipe:
        fields = dict(name=name, source="park.png", mask="box16.png", references=("anna512.png",), prompt=PARALLEL)
        fields.update(changes)
        return Recipe.model_validate(fields)
    recipes = [
        park("curved4", mask="curved.png"),
        park("box4"), park("box8", steps=8),
        park("hard4", mask="box0.png"), park("edge8", mask="box8.png"), park("edge32", mask="box32.png"),
        park("context96", context=96), park("context192", context=192),
        park("ref384", references=("anna384.png",)), park("ref768", references=("anna768.png",)),
        park("face-detail", references=("identity.png", "face.png"), prompt=PARALLEL + " Both references show the same woman. The close-up supplies facial detail; the full-body photograph supplies her clothes."),
        park("layout4", references=("identity.png", "layout.png"), prompt=GUIDED),
        park("upscale768", upscale_long_edge=768),
        park("upscale1024", upscale_long_edge=1024),
        park("target4", mask="target-mask.png"),
        park("target768", mask="target-mask.png", upscale_long_edge=768),
    ]
    placement_prompt = "Replace the grey mannequin sitting on the bench with Anna from the reference photograph. Preserve the existing seated pose, position, body size and wooden bench. Anna has the same face, curly auburn hair, mustard sweater, dark green trousers and white sneakers as the reference. Her hips rest on the seat and her shoes and the bench legs touch the gravel with natural contact shadows. Match the garden photograph's daylight."
    for strength in (0.6, 0.8, 1.0):
        recipes.append(park(f"placed{round(strength*100)}", source="placed.png", prompt=placement_prompt, strength=strength, steps=8))
    suite = json.loads((FIXTURES / "manifest.json").read_text())
    for fixture in suite["fixtures"]:
        name = fixture["name"]
        for key, target in (("image", f"{name}.png"), ("mask", f"{name}-mask.png")):
            path = FIXTURES / fixture[key]
            if sha256(path.read_bytes()).hexdigest() != fixture[f"{key}_sha256"]:
                raise ValueError(f"Changed fixture {path}")
            with Image.open(path) as image:
                image.save(assets / target)
        prompt = _inpaint_prompt(InpaintForm(prompt=fixture["prompt"], edit_operation=fixture["operation"]))
        for context in (96, 320):
            recipes.append(Recipe(name=f"{name}-c{context}", source=f"{name}.png", mask=f"{name}-mask.png", prompt=prompt, context=context, center_selection=False))
        recipes.append(Recipe(name=f"{name}-detail768", source=f"{name}.png", mask=f"{name}-mask.png", prompt=prompt, context=96, center_selection=False, upscale_long_edge=768))
    # A second reference task: transfer the actual mug's shape/material onto the garden sphere.
    with Image.open(FIXTURES / "kitchen.png") as image:
        ref = image.crop((600, 510, 1060, 930))
        ref.thumbnail((512, 512), Image.Resampling.LANCZOS)
        ref.save(assets / "mug-reference.png")
    for context in (96, 320):
        recipes.append(Recipe(name=f"mug-reference-c{context}", source="garden.png", mask="garden-mask.png", references=("mug-reference.png",),
            prompt="Replace the terracotta ball with the pale yellow ceramic mug from the reference photograph. Keep its cylindrical fluted body, rounded handle and pale yellow glaze. The complete mug stands upright nestled among the flowers, with a grounded base and natural daylight matching the garden.", context=context, center_selection=False))
    mug_mask = Image.new("L", original.size)
    mug_mask.paste(255, (210, 700, 655, 1075))
    _inner_feather(mug_mask, 16).save(assets / "mug-wide-mask.png")
    mug_prompt = "Replace the terracotta ball with the exact golden yellow ceramic coffee mug shown in the reference photograph. Keep its smooth glossy glaze, round rim and single rounded handle on the right. Show the entire mug including its handle and base. It stands upright among the garden flowers, with natural daylight and a soft contact shadow."
    for name, mask in (("mug-corrected", "garden-mask.png"), ("mug-wide", "mug-wide-mask.png")):
        recipes.append(Recipe(name=name, source="garden.png", mask=mask, references=("mug-reference.png",), prompt=mug_prompt, context=96, center_selection=False))
    # The historical kitchen mask missed the long reflection/shadow tail.
    kitchen_mask = Image.new("L", original.size)
    kitchen_mask.paste(255, (560, 490, 1254, 1254))
    _inner_feather(kitchen_mask, 24).save(assets / "kitchen-full-shadow.png")
    kitchen = next(r for r in recipes if r.name == "kitchen-c96")
    recipes.append(kitchen.model_copy(update={"name":"kitchen-full-shadow", "mask":"kitchen-full-shadow.png"}))
    return recipes


def gallery(output: Path, recipes: list[Recipe], results: list[Result]) -> None:
    import html
    sections = []
    for recipe in recipes:
        selected = [r for r in results if r.recipe == recipe.name]
        if not selected:
            continue
        with Image.open(output / "assets" / recipe.source) as source, Image.open(output / "assets" / recipe.mask) as mask:
            mask = _selection_alpha(mask)
            bounds = mask.getbbox()
            if bounds is None:
                raise ValueError("Empty mask")
            l, t, r, b = bounds
            bounds = (max(0, l-35), max(0, t-35), min(source.width, r+35), min(source.height, b+35))
            preview = source.convert("RGB")
            tint = Image.new("RGB", source.size, (30, 220, 235))
            preview = Image.composite(tint, preview, mask.point(lambda x: round(x*.35)))
            preview.crop(bounds).save(output / f"{recipe.name}-mask.jpg")
        cards = [f'<figure><img src="{recipe.name}-mask.jpg"><figcaption>Editable selection</figcaption></figure>']
        tiles = []
        for result in selected:
            stem = f"{recipe.name}-{result.seed}"
            if not result.error:
                with Image.open(output / f"{stem}.png") as image:
                    crop = image.crop(bounds).convert("RGB")
                    crop.save(output / f"{stem}-detail.jpg", quality=94)
                    tile = Image.new("RGB", (384, 540), (24, 29, 35))
                    tile.paste(ImageOps.contain(crop, (384, 505)), (0, 35))
                    ImageDraw.Draw(tile).text((8, 8), f"{recipe.name} / {result.seed} / {result.seconds:.2f}s", fill="white")
                    tiles.append(tile)
                cards.append(f'<figure><a href="{stem}.png"><img src="{stem}-detail.jpg"></a><figcaption>Seed {result.seed}: {result.seconds:.2f}s / {result.peak_allocated_gib:.2f} GiB</figcaption></figure>')
            else:
                cards.append(f'<p>{html.escape(result.error)}</p>')
        if tiles:
            sheet = Image.new("RGB", (384*len(tiles), 540), (24, 29, 35))
            for index, tile in enumerate(tiles):
                sheet.paste(tile, (384*index, 0))
            sheet.save(output / f"{recipe.name}-sheet.jpg", quality=94)
        refs = ''.join(f'<a href="assets/{ref}">{html.escape(ref)}</a> ' for ref in recipe.references)
        sections.append(f'<section><h2>{html.escape(recipe.name)}</h2><p>{recipe.steps} steps · context {recipe.context} · references: {refs or "none"}</p><div class="row">{"".join(cards)}</div><details><summary>Exact prompt</summary>{html.escape(recipe.prompt)}</details></section>')
    (output / "index.html").write_text('<!doctype html><meta charset="utf-8"><title>MosImage inpaint quality tests</title><style>body{background:#14191f;color:#e5ebed;font:16px system-ui;margin:36px}h1{font-size:32px}.row{display:flex;gap:12px;overflow:auto}figure{margin:0;min-width:240px;flex:1}img{max-width:100%;max-height:560px}section{margin:40px 0}a{color:#8ae3ca}figcaption{padding:8px}details{margin-top:16px}</style><h1>Inpaint quality experiments · 8 October 2026</h1><p>All declared outputs, including failures. Click an image for the full result. Timings: warmed split renderer; text encoding and loading excluded.</p>'+''.join(sections), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--variants", nargs="+")
    parser.add_argument("--seeds", nargs="+", type=int, default=[7, 42, 20261008])
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    assert_api_unloaded()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    recipes = prepare(output)
    if args.variants:
        missing = set(args.variants) - {r.name for r in recipes}
        if missing:
            raise ValueError(f"Unknown recipes: {missing}")
        recipes = [r for r in recipes if r.name in args.variants]
    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError("Duplicate seeds")
    protocol = dict(base_repo=BASE_REPO, base_revision=BASE_REVISION, quant_revision=QUANT_REVISION,
        placement="split: encoder released before renderer", seeds=args.seeds,
        recipes=[r.model_dump() for r in recipes],
        assets={p.name:sha256(p.read_bytes()).hexdigest() for p in (output/"assets").iterdir()},
        packages={p:version(p) for p in ("torch", "diffusers", "transformers", "comfy-kitchen")})
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    if args.prepare_only:
        return
    if psutil.virtual_memory().available < 4*1024**3:
        raise MemoryError("Insufficient RAM headroom")
    torch = torch_module()
    torch.cuda.set_per_process_memory_fraction(.65)
    from diffusers import Flux2KleinInpaintPipeline
    from transformers import Qwen3ForCausalLM, Qwen2TokenizerFast
    from nvfp4_transformer import load_transformer, kernel_calls
    started = time.perf_counter()
    print("Loading and encoding prompts separately", flush=True)
    encoder = Qwen3ForCausalLM.from_pretrained(BASE_REPO, revision=BASE_REVISION, subfolder="text_encoder", torch_dtype=torch.bfloat16, local_files_only=True, device_map={"":"cuda"})
    tokenizer = Qwen2TokenizerFast.from_pretrained(BASE_REPO, revision=BASE_REVISION, subfolder="tokenizer", local_files_only=True)
    text_pipe = Flux2KleinInpaintPipeline(scheduler=None, vae=None, text_encoder=encoder, tokenizer=tokenizer, transformer=None, is_distilled=True)
    embeddings = {}
    with torch.inference_mode():
        for recipe in recipes:
            if recipe.prompt not in embeddings:
                embeddings[recipe.prompt] = text_pipe.encode_prompt(recipe.prompt, device=torch.device("cuda"))[0].cpu()
    protocol["encode_and_load_seconds"] = time.perf_counter()-started
    del text_pipe, encoder, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    started = time.perf_counter()
    transformer = load_transformer()
    pipe = Flux2KleinInpaintPipeline.from_pretrained(BASE_REPO, revision=BASE_REVISION, transformer=transformer, text_encoder=None, tokenizer=None, torch_dtype=torch.bfloat16, local_files_only=True)
    del transformer
    pipe.to("cuda")
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()
    pipe.mask_processor.register_to_config(do_binarize=False)
    pipe.set_progress_bar_config(disable=True)
    protocol["renderer_load_seconds"] = time.perf_counter()-started
    protocol["gpu"] = torch.cuda.get_device_name()
    results: list[Result] = []
    def save() -> None:
        (output/"results.json").write_text(json.dumps(dict(protocol=protocol, results=[r.model_dump() for r in results]), indent=2), encoding="utf-8")
    save()
    for recipe in recipes:
        work = output / (recipe.name+"-work")
        work.mkdir()
        workspace = prepare_inpaint_workspace(output/"assets"/recipe.source, output/"assets"/recipe.mask, work, recipe.max_dimension, 0, recipe.context, composite_feather=0, generation_feather=0, center_selection=recipe.center_selection)
        with Image.open(workspace.model_input_path) as image:
            source = image.convert("RGB")
        with Image.open(workspace.model_mask_path) as image:
            mask = image.convert("L")
        if recipe.upscale_long_edge:
            scale = recipe.upscale_long_edge/max(source.size)
            size = tuple(max(32, round(side*scale/32)*32) for side in source.size)
            source = source.resize(size, Image.Resampling.LANCZOS)
            mask = mask.resize(size, Image.Resampling.LANCZOS)
            source.save(workspace.model_input_path)
            mask.save(workspace.model_mask_path)
        references = []
        for name in recipe.references:
            with Image.open(output/"assets"/name) as image:
                references.append(image.convert("RGB"))
        if len(references)>1 and any(r.size!=references[0].size for r in references):
            raise ValueError("Multi-reference dimensions differ")
        kwargs = dict(image=source, mask_image=mask, prompt_embeds=embeddings[recipe.prompt].to("cuda"), height=source.height, width=source.width, guidance_scale=1.0, strength=recipe.strength)
        if references:
            kwargs["image_reference"] = references if len(references)>1 else references[0]
        if not results:
            started = time.perf_counter()
            with torch.inference_mode():
                pipe(**kwargs, num_inference_steps=1, generator=torch.Generator("cuda").manual_seed(99999))
            protocol["warmup_seconds"] = time.perf_counter()-started
        original = np.asarray(workspace.source.convert("RGBA"))
        alpha = np.asarray(workspace.composite_mask)
        for seed in args.seeds:
            result = Result(recipe=recipe.name, seed=seed)
            results.append(result)
            save()
            started = time.perf_counter()
            calls = kernel_calls(pipe.transformer)
            torch.cuda.reset_peak_memory_stats()
            def progress(_pipe, _step, _timestep, values):
                if time.perf_counter()-started > 35:
                    raise TimeoutError("35-second render budget exceeded")
                if psutil.virtual_memory().available < 1.5*1024**3:
                    raise MemoryError("RAM headroom below 1.5 GiB")
                return values
            try:
                with torch.inference_mode():
                    generated = pipe(**kwargs, num_inference_steps=recipe.steps, generator=torch.Generator("cuda").manual_seed(seed), callback_on_step_end=progress).images[0]
                stem = f"{recipe.name}-{seed}"
                generated.save(workspace.model_output_path)
                generated.save(output / f"{stem}-raw.png")
                workspace.composite_to(output/f"{stem}.png")
                torch.cuda.synchronize()
                result.seconds = round(time.perf_counter()-started, 3)
                result.peak_allocated_gib = round(torch.cuda.max_memory_allocated()/1024**3, 3)
                result.peak_reserved_gib = round(torch.cuda.max_memory_reserved()/1024**3, 3)
                result.rss_after_gib = round(psutil.Process().memory_info().rss/1024**3, 3)
                result.available_ram_after_gib = round(psutil.virtual_memory().available/1024**3, 3)
                result.kernel_calls = kernel_calls(pipe.transformer)-calls
                with Image.open(output/f"{stem}.png") as image:
                    changed = np.any(np.asarray(image.convert("RGBA"))!=original, axis=2)
                result.protected_changes = int(np.count_nonzero(changed & (alpha==0)))
                result.selected_changes = int(np.count_nonzero(changed & (alpha>0)))
                if result.protected_changes or not result.selected_changes:
                    raise ValueError("Pixel preservation/change check failed")
                print(f"DONE {stem}: {result.seconds}s / {result.peak_allocated_gib} GiB", flush=True)
            except Exception as error:
                result.error = f"{type(error).__name__}: {error}"
                save()
                gallery(output, recipes, results)
                raise
            save()
        gallery(output, recipes, results)
    print(f"Review: {output/'index.html'}", flush=True)


if __name__ == "__main__":
    main()
