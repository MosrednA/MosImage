"""Memory-bounded Anna/park follow-up; run from backend with the API unloaded.

Text is encoded first and its model released before one renderer is loaded.
This diagnostic uses cached checkpoints and leaves the application unchanged.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import gc
from importlib.metadata import version
import json
from pathlib import Path
import time
from typing import Literal
from urllib.error import URLError

import numpy as np
from PIL import Image
import psutil
from pydantic import BaseModel

from benchmark_park_reference import ROOT, SEEDS, Protocol, read_api
from app.compositing import prepare_inpaint_workspace
from app.models.flux2_klein import _inpaint_prompt, _reference_images
from app.models.flux2_klein_nvfp4 import BASE_REPO, BASE_REVISION, QUANT_REPO, QUANT_REVISION, QUANT_FILENAME
from app.runtime import torch_module
from app.schemas import InpaintForm, ModelInfo

REPO = "black-forest-labs/FLUX.2-klein-base-4B"
REVISION = "a3b4f4849157f664bdbc776fd7453c2783562f4d"
TRANSFORMER = ROOT / "models/research-cache/models--black-forest-labs--FLUX.2-klein-base-4B/snapshots" / REVISION / "transformer"
SCENE = "Adult woman, seated naturally on a wooden garden bench beside the path, feet touching the ground."


class BaseVariant(BaseModel):
    name: str
    model: Literal["nvfp4", "base"] = "nvfp4"
    prompt_key: str = "plain"
    steps: int = 4
    guidance: float = 1.0
    context: int = 32
    reference_max: int = 512
    reference_files: tuple[str, ...] = ("reference.png",)
    scene: str = ""
    mask_file: str = "mask.png"


class BaseTrial(BaseModel):
    variant: str
    seed: int
    elapsed_seconds: float | None = None
    peak_allocated_gib: float | None = None
    peak_reserved_gib: float | None = None
    changed_protected_pixels: int | None = None
    changed_originally_protected_pixels: int | None = None
    protected_pixel_count: int | None = None
    output: str | None = None
    reference_sizes: list[tuple[int, int]] = []
    model_input_size: tuple[int, int] | None = None
    error: str | None = None


class BaseProtocol(BaseModel):
    model: str = "Klein Distilled NVFP4 / Klein Base BF16"
    revision: str = REVISION
    component_repo: str = BASE_REPO
    component_revision: str = BASE_REVISION
    quant_repo: str = QUANT_REPO
    quant_revision: str = QUANT_REVISION
    quant_filename: str = QUANT_FILENAME
    seeds: tuple[int, ...] = SEEDS
    variants: tuple[BaseVariant, ...] = (
        BaseVariant(name="split-full4", reference_max=1024),
        BaseVariant(name="split-full8", steps=8, reference_max=1024),
        BaseVariant(name="split-small4"),
        BaseVariant(name="split-small8", steps=8),
        BaseVariant(name="split-scene4", scene=SCENE, prompt_key="scene"),
        BaseVariant(name="base-small28", model="base", steps=28, guidance=4),
        BaseVariant(name="base-small50", model="base", steps=50, guidance=4),
    )
    prompt: str
    prompts: dict[str, str]
    cuda_memory_fraction: float = 0.65
    per_render_seconds_limit: float = 45.0
    placement: str = "CUDA text encoder released before CUDA renderer"
    load_seconds_by_model: dict[str, float] = {}
    warmup_seconds_by_model: dict[str, float] = {}
    prompt_encoding_seconds: float | None = None
    packages: dict[str, str]
    asset_hashes: dict[str, str] = {}
    results: list[BaseTrial] = []


def benchmark_references(input_dir: Path, variant: BaseVariant) -> list[Image.Image]:
    """Read declared assets without the app's multi-reference resampling.

    Multi-reference benchmark assets must already share one canvas; otherwise
    the pipeline crops later references to the first reference's dimensions.
    Single-reference preparation remains identical to earlier cohorts.
    """
    images: list[Image.Image] = []
    for name in variant.reference_files:
        path = (input_dir / name).resolve()
        if not path.is_relative_to(input_dir):
            raise ValueError("References must stay inside the frozen input directory.")
        image = _reference_images([path])[0]
        image.thumbnail((variant.reference_max, variant.reference_max), Image.Resampling.LANCZOS)
        images.append(image)
    if not images:
        raise ValueError("Declare at least one reference.")
    if any(image.size != images[0].size for image in images):
        raise ValueError("Prepare every reference on the same canvas before benchmarking.")
    return images


def run_cohort(input_dir: Path, output: Path, protocol: BaseProtocol) -> None:
    """Render a declared cohort sequentially, preserving every outcome."""
    if len({variant.name for variant in protocol.variants}) != len(protocol.variants):
        raise ValueError("Declare each variant only once.")
    if len(set(protocol.seeds)) != len(protocol.seeds):
        raise ValueError("Declare each seed only once.")
    for variant in protocol.variants:
        if variant.prompt_key not in protocol.prompts:
            raise ValueError(f"Missing prompt for {variant.name}: {variant.prompt_key}")
        if variant.model == "base" and "negative" not in protocol.prompts:
            raise ValueError("Base rendering requires an explicit negative prompt.")
    try:
        loaded = [model.name for model in (ModelInfo.model_validate(item) for item in json.loads(read_api("/api/models"))) if model.loaded]
    except URLError as error:
        if not isinstance(error.reason, ConnectionRefusedError) and getattr(error.reason, "winerror", None) != 10061:
            raise
        loaded = []
    if loaded:
        raise RuntimeError(f"Unload API models first: {loaded}")
    if psutil.virtual_memory().available < 4 * 1024**3:
        raise RuntimeError("Less than 4 GiB RAM available; no model loaded.")
    input_dir = input_dir.resolve()
    frozen = Protocol.model_validate_json((input_dir / "results.json").read_bytes())
    for name in ("source", "mask", "reference"):
        if sha256((input_dir / f"{name}.png").read_bytes()).hexdigest() != getattr(frozen, f"{name}_sha256"):
            raise ValueError(f"Frozen {name} file changed.")
    for name, expected_hash in protocol.asset_hashes.items():
        path = (input_dir / name).resolve()
        if not path.is_relative_to(input_dir):
            raise ValueError("Auxiliary assets must stay inside the frozen input directory.")
        if sha256(path.read_bytes()).hexdigest() != expected_hash:
            raise ValueError(f"Frozen guide asset changed: {name}")
    for variant in protocol.variants:
        if len(variant.reference_files) > 1 and any(name not in protocol.asset_hashes for name in variant.reference_files):
            raise ValueError("Declare hashes for every multi-reference benchmark asset.")
        mask_path = (input_dir / variant.mask_file).resolve()
        if not mask_path.is_relative_to(input_dir):
            raise ValueError("Masks must stay inside the frozen input directory.")
        if variant.mask_file != "mask.png" and variant.mask_file not in protocol.asset_hashes:
            raise ValueError("Declare a hash for each alternative mask.")
        with Image.open(mask_path) as mask_image, Image.open(input_dir / "source.png") as source_image:
            if mask_image.size != source_image.size:
                raise ValueError("Mask dimensions must match the source.")
        benchmark_references(input_dir, variant)
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    manifest = output / "results.json"
    def save() -> None:
        manifest.write_text(protocol.model_dump_json(indent=2), encoding="utf-8")
    save()
    torch = torch_module()
    torch.cuda.set_per_process_memory_fraction(protocol.cuda_memory_fraction)
    from diffusers import Flux2KleinInpaintPipeline, Flux2Transformer2DModel
    from transformers import Qwen3ForCausalLM, Qwen2TokenizerFast
    from nvfp4_transformer import load_transformer
    print("Encoding text alone; CUDA allocator budget 65%.", flush=True)
    started = time.perf_counter()
    encoder = Qwen3ForCausalLM.from_pretrained(BASE_REPO, revision=BASE_REVISION, subfolder="text_encoder",
        torch_dtype=torch.bfloat16, local_files_only=True, token=False, device_map={"": "cuda"}, low_cpu_mem_usage=True)
    tokenizer = Qwen2TokenizerFast.from_pretrained(BASE_REPO, revision=BASE_REVISION, subfolder="tokenizer", local_files_only=True, token=False)
    text_pipe = Flux2KleinInpaintPipeline(scheduler=None, vae=None, text_encoder=encoder, tokenizer=tokenizer, transformer=None, is_distilled=True)
    embeds = {}
    with torch.inference_mode():
        for name, text in protocol.prompts.items():
            embeds[name] = text_pipe.encode_prompt(text, device=torch.device("cuda"))[0].cpu()
    protocol.prompt_encoding_seconds = round(time.perf_counter() - started, 3)
    del text_pipe, encoder, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    save()
    print(f"Text encoded in {protocol.prompt_encoding_seconds}s; encoder released.", flush=True)
    with Image.open(input_dir / "source.png") as image:
        source = np.asarray(image.convert("RGBA"))
    from app.compositing import _selection_alpha
    with Image.open(input_dir / "mask.png") as image:
        original_alpha = np.asarray(_selection_alpha(image))
    pipe = None
    active = None
    for variant in protocol.variants:
        if active != variant.model:
            pipe = None
            gc.collect()
            torch.cuda.empty_cache()
            started = time.perf_counter()
            transformer = load_transformer() if variant.model == "nvfp4" else Flux2Transformer2DModel.from_pretrained(
                TRANSFORMER, torch_dtype=torch.bfloat16, local_files_only=True, use_safetensors=True,
                low_cpu_mem_usage=True, device_map={"": "cuda"})
            pipe = Flux2KleinInpaintPipeline.from_pretrained(BASE_REPO, revision=BASE_REVISION, transformer=transformer,
                text_encoder=None, tokenizer=None, is_distilled=variant.model == "nvfp4", torch_dtype=torch.bfloat16,
                local_files_only=True, token=False, use_safetensors=True, low_cpu_mem_usage=True)
            del transformer
            pipe.to("cuda")
            pipe.vae.enable_slicing()
            pipe.vae.enable_tiling()
            pipe.mask_processor.register_to_config(do_binarize=False)
            pipe.set_progress_bar_config(disable=True)
            active = variant.model
            protocol.load_seconds_by_model[active] = round(time.perf_counter() - started, 3)
            save()
            print(f"Loaded {active} in {protocol.load_seconds_by_model[active]}s; available RAM {psutil.virtual_memory().available/1024**3:.1f} GiB.", flush=True)
        workspace = prepare_inpaint_workspace(input_dir / "source.png", input_dir / variant.mask_file, output,
            1024, 0, variant.context, composite_feather=0, generation_feather=0, center_selection=True)
        with Image.open(workspace.model_input_path) as image:
            init_image = image.convert("RGB")
        with Image.open(workspace.model_mask_path) as image:
            mask_image = image.convert("L")
        alpha = np.asarray(workspace.composite_mask)
        references = benchmark_references(input_dir, variant)
        kwargs = dict(image=init_image, mask_image=mask_image,
            image_reference=references if len(references) > 1 else references[0],
            prompt_embeds=embeds[variant.prompt_key].to("cuda"),
            negative_prompt_embeds=embeds["negative"].to("cuda") if variant.model == "base" else None,
            height=init_image.height, width=init_image.width, guidance_scale=variant.guidance, strength=1)
        if active not in protocol.warmup_seconds_by_model:
            started = time.perf_counter()
            with torch.inference_mode():
                pipe(**kwargs, num_inference_steps=1, generator=torch.Generator("cuda").manual_seed(99999))
            protocol.warmup_seconds_by_model[active] = round(time.perf_counter() - started, 3)
            save()
        for seed in protocol.seeds:
            trial = BaseTrial(variant=variant.name, seed=seed,
                reference_sizes=[image.size for image in references], model_input_size=init_image.size,
                protected_pixel_count=int(np.count_nonzero(alpha == 0)))
            protocol.results.append(trial)
            save()
            started = time.perf_counter()
            torch.cuda.reset_peak_memory_stats()
            print(f"START {variant.name} seed {seed}", flush=True)
            try:
                def progress(_pipe, step: int, _timestep, values):
                    elapsed = time.perf_counter() - started
                    if elapsed > protocol.per_render_seconds_limit:
                        raise TimeoutError(f"Render exceeded {protocol.per_render_seconds_limit:g}-second diagnostic budget; model stopped.")
                    if psutil.virtual_memory().available < 1.5 * 1024**3:
                        raise MemoryError("RAM headroom below 1.5 GiB; model stopped.")
                    if step % 10 == 0:
                        print(f"  step {step+1}/{variant.steps}, {elapsed:.1f}s", flush=True)
                    return values
                with torch.inference_mode():
                    generated = pipe(**kwargs, num_inference_steps=variant.steps,
                        generator=torch.Generator("cuda").manual_seed(seed), callback_on_step_end=progress).images[0]
                generated.save(workspace.model_output_path)
                path = output / f"{variant.name}-{seed}.png"
                workspace.composite_to(path)
                torch.cuda.synchronize()
                trial.elapsed_seconds = round(time.perf_counter() - started, 3)
                trial.peak_allocated_gib = round(torch.cuda.max_memory_allocated()/1024**3, 3)
                trial.peak_reserved_gib = round(torch.cuda.max_memory_reserved()/1024**3, 3)
                trial.output = path.name
                with Image.open(path) as image:
                    pixels = np.asarray(image.convert("RGBA"))
                trial.changed_protected_pixels = int(np.count_nonzero(np.any(pixels != source, axis=2) & (alpha == 0)))
                trial.changed_originally_protected_pixels = int(np.count_nonzero(np.any(pixels != source, axis=2) & (original_alpha == 0)))
                if trial.changed_protected_pixels:
                    trial.error = "Protected pixels changed."
                print(f"DONE {variant.name}: {trial.elapsed_seconds}s, peak {trial.peak_allocated_gib} GiB, protected changes {trial.changed_protected_pixels}", flush=True)
            except Exception as error:
                trial.elapsed_seconds = round(time.perf_counter() - started, 3)
                trial.error = f"{type(error).__name__}: {error}"
                save()
                raise
            save()
    print(f"All outcomes saved: {manifest}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    form = InpaintForm(prompt="sitting on a bench", blend_feather=0, context_padding=32)
    prompt = _inpaint_prompt(form, ["Anna"])
    protocol = BaseProtocol(prompt=prompt, prompts={
        "plain": prompt, "scene": _inpaint_prompt(form.model_copy(update={"scene_context": SCENE}), ["Anna"]), "negative": ""
    }, packages={name: version(name) for name in ("torch", "diffusers", "transformers", "accelerate", "safetensors", "comfy-kitchen")})
    run_cohort(args.input, args.output, protocol)


if __name__ == "__main__":
    main()

