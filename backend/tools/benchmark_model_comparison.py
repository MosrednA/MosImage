"""Side-by-side model and text-encoder comparison. Run from backend/ with the API model unloaded.

    python tools/benchmark_model_comparison.py --output ../data/model-comparison-2026-10-09 --prepare-only
    python tools/benchmark_model_comparison.py --output ../data/model-comparison-2026-10-09 --arms all
    python tools/benchmark_model_comparison.py --output ../data/model-comparison-2026-10-09 --gallery

Creative cases come first: two people composed from references into a new scene,
whole-image edits and held-object swaps. The original retouch cases follow as a
control. Every arm renders the same cases, crops, masks and seeds, in its own process so
memory is fully released between arms. Prompts are encoded first and the encoder
is released before the renderer loads. Every output, including failures, is kept.
"""
from __future__ import annotations

import argparse
import gc
from hashlib import sha256
from importlib.metadata import version
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time
from typing import Literal

import numpy as np
from PIL import Image, ImageDraw
import psutil
from pydantic import BaseModel, ConfigDict

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from app.compositing import _inner_feather, _selection_alpha, prepare_inpaint_workspace
from app.models.flux2_klein import SHARED_COMPONENTS, _inpaint_prompt, _reference_images
from app.models.flux2_klein_nvfp4 import BASE_REPO as K4_REPO, BASE_REVISION as K4_REVISION
from app.models.flux2_klein_nvfp4 import QUANT_FILENAME as K4_QUANT_FILE, QUANT_REPO as K4_QUANT_REPO, QUANT_REVISION as K4_QUANT_REVISION
from app.schemas import InpaintForm
from benchmark_inpaint_quality import assert_api_unloaded
from benchmark_park_prompts import PARALLEL

FIXTURES = ROOT / "benchmarks/fixtures/v1"
CREATIVE = ROOT / "benchmarks/fixtures/v2"
PARK = ROOT / "data/park-scale-2026-10-07"
SEEDS = (7, 42, 20261009)
# Renders past this are stopped and recorded; the interactive target is 30 s.
RENDER_LIMIT_SECONDS = 90

K9_REPO, K9_REVISION = "black-forest-labs/FLUX.2-klein-9B", "92196c8e11f7b6cf2b7493e037d8c5345c559216"
K9_QUANT_REPO, K9_QUANT_REVISION = "black-forest-labs/FLUX.2-klein-9b-nvfp4", "e882f64f6aa086fcf8915a7763550e05af10ef13"
K9_QUANT_FILE = "flux-2-klein-9b-nvfp4.safetensors"
QWEN_REPO, QWEN_REVISION = "unsloth/Qwen-Image-2.1", "20f5a734a394f0fce241bf2f9dc1b3d97ac18b2a"
TURBO_REPO, TURBO_REVISION = "Viggle/Qwen-Image-2.1-viggle-turbo", "009a44a895ef85f7e643c80fdca9543795248867"
TURBO_FILE = "Qwen-Image-2.1-viggle-turbo-v0.3-6step-fp8_e4m3fn.safetensors"
# Raw sigma nodes from the turbo model card; the pipeline applies its own shift.
TURBO_SIGMAS = [1.0, 0.9375, 0.875, 0.75, 0.5, 0.25]


class Encoder(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    repo: str
    revision: str
    subfolder: str | None = None
    label: str


class Arm(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    name: str
    label: str
    family: str  # "klein4b", "klein9b" or "qwen21-turbo"
    encoder: Encoder
    steps: int


K4_STOCK = Encoder(repo=K4_REPO, revision=K4_REVISION, subfolder="text_encoder", label="Qwen3-4B (standaard)")
K9_STOCK = Encoder(repo=K9_REPO, revision=K9_REVISION, subfolder="text_encoder", label="Qwen3-8B (standaard)")
ARMS = {arm.name: arm for arm in (
    Arm(name="klein4b", label="Klein 4B NVFP4 (huidige app)", family="klein4b", encoder=K4_STOCK, steps=4),
    Arm(name="klein4b-instruct2507", label="Klein 4B NVFP4", family="klein4b", steps=4,
        encoder=Encoder(repo="unsloth/Qwen3-4B-Instruct-2507", revision="992063681dc2f7de4ee976110199552935cad284", label="Qwen3-4B-Instruct-2507")),
    Arm(name="klein4b-abliterated", label="Klein 4B NVFP4", family="klein4b", steps=4,
        encoder=Encoder(repo="BennyDaBall/Qwen3-4b-Z-Image-Turbo-AbliteratedV1", revision="ce497d288a7ddfd5d0f337c7139349d5d0236bfa", label="Qwen3-4B Z-Image abliterated")),
    Arm(name="klein9b", label="Klein 9B NVFP4", family="klein9b", encoder=K9_STOCK, steps=4),
    Arm(name="klein9b-abliterated", label="Klein 9B NVFP4", family="klein9b", steps=4,
        encoder=Encoder(repo="huihui-ai/Huihui-Qwen3-8B-abliterated-v2", revision="7d89db76029281fd8f3e6698a8e30738608105a9", label="Huihui Qwen3-8B abliterated v2")),
    Arm(name="qwen21-turbo", label="Qwen-Image 2.1 Viggle turbo v0.3 (FP8)", family="qwen21-turbo", steps=6,
        encoder=Encoder(repo=QWEN_REPO, revision=QWEN_REVISION, subfolder="text_encoder", label="Qwen3-VL-8B (standaard)")),
)}


class Case(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    name: str
    title: str
    category: str
    # inpaint: masked crop, composited back. edit: the whole source is image 1,
    # no mask. compose: a new scene from the references alone.
    mode: Literal["inpaint", "edit", "compose"] = "inpaint"
    source: str | None = None
    mask: str | None = None
    references: tuple[str, ...] = ()
    size: tuple[int, int] = (1216, 832)  # compose output, width x height
    # Klein receives the app's mask-aware inpaint prompt; Qwen edits the crop
    # without a mask, so it needs an instruction naming what to change.
    klein_prompt: str
    qwen_prompt: str
    context: int = 96
    center_selection: bool = False
    criteria: tuple[str, ...] = ()


class Result(BaseModel):
    arm: str
    case: str
    seed: int
    seconds: float | None = None
    peak_allocated_gib: float | None = None
    protected_changes: int | None = None
    selected_changes: int | None = None
    error: str | None = None


KEEP = " Keep everything else in the image exactly unchanged."


def _copy_fixture(assets: Path, fixture: dict, name: str) -> None:
    for key, target in (("image", f"{name}.png"), ("mask", f"{name}-mask.png")):
        path = FIXTURES / fixture[key]
        if sha256(path.read_bytes()).hexdigest() != fixture[f"{key}_sha256"]:
            raise ValueError(f"Changed fixture {path}")
        with Image.open(path) as image:
            image.save(assets / target)


def prepare(output: Path) -> list[Case]:
    assets = output / "assets"
    assets.mkdir(parents=True)
    suite = {f["name"]: f for f in json.loads((FIXTURES / "manifest.json").read_text())["fixtures"]}
    for name in suite:
        _copy_fixture(assets, suite[name], name)

    def klein(name: str, references: tuple[str, ...] = ()) -> str:
        fixture = suite[name]
        return _inpaint_prompt(InpaintForm(prompt=fixture["prompt"], edit_operation=fixture["operation"]), references)

    # The historical kitchen mask missed the long reflection/shadow tail (8 October study).
    with Image.open(assets / "kitchen-mask.png") as image:
        size = _selection_alpha(image).size
    kitchen = Image.new("L", size)
    kitchen.paste(255, (560, 490, 1254, 1254))
    _inner_feather(kitchen, 24).save(assets / "kitchen-full-mask.png")

    # Appearance change: the whole denim shirt, selected with the app's own SAM clicks.
    from app.selection import ObjectSelector
    with Image.open(assets / "portrait.png") as image:
        portrait = image.convert("RGB")
    positive = [(343, 818), (929, 640), (869, 919), (600, 1150), (1000, 1100), (700, 560), (255, 1085), (960, 1125)]
    negative = [(586, 250), (560, 1235), (640, 470)]
    shirt = ObjectSelector()._predict(portrait, "portrait-shirt", positive + negative,
                                      [1] * len(positive) + [0] * len(negative), multimask=False)
    # SAM leaves buttons and the brooch out; a new garment must cover them, so
    # fill every unselected area the image border cannot reach.
    outside = Image.new("L", (portrait.width + 2, portrait.height + 2), 255)
    outside.paste(Image.fromarray(np.where(shirt > 0, 0, 255).astype("uint8")), (1, 1))
    ImageDraw.floodfill(outside, (0, 0), 128)
    shirt = np.where((shirt > 0) | (np.asarray(outside)[1:-1, 1:-1] == 255), 255, 0).astype("uint8")
    if not 0.2 < (shirt > 0).mean() < 0.45:
        raise ValueError("Unexpected shirt selection size")
    _inner_feather(Image.fromarray(shirt, mode="L"), 12).save(assets / "shirt-mask.png")

    with Image.open(PARK / "source.png") as image:
        image.save(assets / "park.png")
    with Image.open(PARK / "mask.png") as image:
        box = _selection_alpha(image).getbbox()
        park_size = image.size
    if box is None:
        raise ValueError("Empty park mask")
    binary = Image.new("L", park_size)
    binary.paste(255, box)
    _inner_feather(binary, 16).save(assets / "park-mask.png")
    with Image.open(PARK / "reference.png") as image:
        anna = image.convert("RGB")
    anna.thumbnail((1024, 1024), Image.Resampling.LANCZOS)  # what the app's _reference_image keeps
    anna.save(assets / "anna.png")

    with Image.open(assets / "kitchen.png") as image:
        mug = image.crop((600, 510, 1060, 930))
    mug.thumbnail((512, 512), Image.Resampling.LANCZOS)
    mug.save(assets / "mug.png")
    mug_mask = Image.new("L", size)
    mug_mask.paste(255, (210, 700, 655, 1075))
    _inner_feather(mug_mask, 16).save(assets / "mug-mask.png")

    creative = json.loads((CREATIVE / "manifest.json").read_text())["sources"]
    for name, spec in creative.items():
        path = CREATIVE / spec["file"]
        if sha256(path.read_bytes()).hexdigest() != spec["sha256"]:
            raise ValueError(f"Changed fixture {path}")
        with Image.open(path) as image:
            image.save(assets / f"{name}.png")
    # The phone, the hand holding it and room above for a blade.
    sword = Image.new("L", (creative["phone"]["width"], creative["phone"]["height"]))
    sword.paste(255, (470, 60, 800, 880))
    _inner_feather(sword, 16).save(assets / "sword-mask.png")

    identity = " Keep each person's face, hair, accessories and clothing identical to their own reference image."
    coaster = ("Photograph of the woman from image 1 and the man from image 2 riding side by side in the front car of a bright red roller coaster "
               "at the top of a steep drop, both laughing and screaming with their arms raised and wind in their hair. "
               "Amusement park and blue summer sky behind them, motion blur on the track, sunny daylight." + identity)
    skihut = ("Photograph of the woman from image 1 and the man from image 2 sitting together at a wooden table inside a cosy alpine ski hut, "
              "toasting with mugs of hot chocolate and laughing. Ski jackets hang on the wall, snowy mountains show through the window, "
              "warm lamplight and a crackling fireplace." + identity)
    dj = ("Turn the woman in image 1 into a club DJ: she stands behind a DJ booth with turntables and a mixer, large over-ear headphones around her neck "
          "and one hand on a turntable, in a dark nightclub with colourful stage lights, haze and a dancing crowd behind her. "
          "Keep her face, curly hair and identity exactly the same.")
    sword_edit = ("Replace the smartphone in the young man's raised right hand in image 1 with a medieval steel longsword: his fingers grip its "
                  "leather-wrapped hilt below a cross guard and the polished blade points straight up." + KEEP)
    duo_criteria = ("Both faces are recognisable from their references", "Both people are complete, interacting naturally, with plausible scale",
                    "The scene is clearly the requested place, with coherent light")
    mug_prompt = ("Replace the terracotta ball with the exact golden yellow ceramic coffee mug shown in the reference photograph. "
                  "Keep its smooth glossy glaze, round rim and single rounded handle on the right. Show the entire mug including its handle and base. "
                  "It stands upright among the garden flowers, with natural daylight and a soft contact shadow.")
    shirt_prompt = "A chunky knitted burgundy wool sweater with a crew neck and visible cable-knit texture, worn naturally with realistic folds."
    cases = [
        Case(name="duo-coaster", title="Duo in de achtbaan", category="Creatief: twee personen", mode="compose",
             references=("anna.png", "man.png"), klein_prompt=coaster, qwen_prompt=coaster, criteria=duo_criteria),
        Case(name="duo-skihut", title="Duo in de skihut", category="Creatief: twee personen", mode="compose",
             references=("anna.png", "man.png"), klein_prompt=skihut, qwen_prompt=skihut, criteria=duo_criteria),
        Case(name="dj", title="Portret wordt DJ", category="Creatief: hele beeld", mode="edit", source="portrait.png",
             klein_prompt=dj, qwen_prompt=dj,
             criteria=("Same face and curly hair as the original", "Convincing DJ booth, headphones and club lighting",
                       "Hands and equipment are anatomically and physically plausible")),
        Case(name="sword", title="Telefoon wordt zwaard (selectie)", category="Creatief: object in hand", source="phone.png", mask="sword-mask.png",
             klein_prompt=_inpaint_prompt(InpaintForm(prompt="A medieval steel longsword held upright in his raised right hand, gripped by its "
                                                             "leather-wrapped hilt below a cross guard, the polished blade pointing straight up")),
             qwen_prompt="Replace the smartphone in his raised right hand with a medieval steel longsword, gripped by its leather-wrapped hilt below a cross guard, "
                         "the polished blade pointing straight up." + KEEP,
             criteria=("No phone remains", "Fingers wrap the hilt believably", "Blade is straight, complete and lit like the street")),
        Case(name="sword-edit", title="Telefoon wordt zwaard (zonder selectie)", category="Creatief: object in hand", mode="edit", source="phone.png",
             klein_prompt=sword_edit, qwen_prompt=sword_edit,
             criteria=("No phone remains", "Fingers wrap the hilt believably", "Face, hoodie and street stay unchanged")),
        Case(name="garden", title="Tuin: bol wordt paddenstoel", category="Vervangen", source="garden.png", mask="garden-mask.png",
             klein_prompt=klein("garden"), criteria=tuple(suite["garden"]["criteria"]),
             qwen_prompt="Replace the terracotta ball among the flowers with a small glowing orange mushroom growing naturally among the purple flowers." + KEEP),
        Case(name="kitchen", title="Keuken: mok weghalen", category="Weghalen", source="kitchen.png", mask="kitchen-full-mask.png",
             klein_prompt=klein("kitchen"), criteria=tuple(suite["kitchen"]["criteria"]),
             qwen_prompt="Remove the yellow coffee mug together with its reflection and shadow, revealing continuous white marble countertop with subtle gray veins." + KEEP),
        Case(name="portrait", title="Portret: broche wordt madeliefje", category="Klein detail", source="portrait.png", mask="portrait-mask.png",
             klein_prompt=klein("portrait"), criteria=tuple(suite["portrait"]["criteria"]),
             qwen_prompt="Replace the round coral brooch on the shirt with a small embroidered white daisy stitched onto the blue denim." + KEEP),
        Case(name="architecture", title="Gevel: andere poster", category="Vervangen", source="architecture.png", mask="architecture-mask.png",
             klein_prompt=klein("architecture"), criteria=tuple(suite["architecture"]["criteria"]),
             qwen_prompt="Replace the leaf illustration inside the poster frame with a minimalist blue botanical illustration printed on cream paper. Keep the frame." + KEEP),
        Case(name="shirt", title="Portret: ander shirt", category="Uiterlijk", source="portrait.png", mask="shirt-mask.png",
             klein_prompt=_inpaint_prompt(InpaintForm(prompt=shirt_prompt)),
             criteria=("Sweater replaces the whole denim shirt", "Knit texture, folds and light are believable", "Face, hair, hands and trousers are protected"),
             qwen_prompt="Change her denim shirt into a chunky knitted burgundy wool sweater with a crew neck and cable-knit texture. Keep her face, hair, hands and trousers." + KEEP),
        Case(name="anna", title="Park: Anna op een bankje", category="Persoon uit referentie", source="park.png", mask="park-mask.png",
             references=("anna.png",), klein_prompt=PARALLEL, context=32, center_selection=True,
             criteria=("Anna's face, hair and clothes match the reference", "Complete seated body on a bench, both shoes on the gravel", "Scale, light and contact shadows fit the park"),
             qwen_prompt="Add the woman from image 2 to the right side of image 1: she sits upright on a wooden garden bench beside the gravel path, facing the camera, "
                         "both feet on the gravel. Keep her face, curly auburn hair, mustard sweater, dark green trousers and white sneakers exactly as in image 2. "
                         "Natural adult proportions, the garden's perspective and daylight, soft contact shadows." + KEEP),
        Case(name="mug", title="Tuin: mok uit referentie", category="Object uit referentie", source="garden.png", mask="mug-mask.png",
             references=("mug.png",), klein_prompt=mug_prompt,
             criteria=("The mug matches the reference in shape, colour and handle", "Complete, upright and grounded among the flowers", "Light and focus fit the garden"),
             qwen_prompt="Replace the terracotta ball in image 1 with the golden yellow ceramic coffee mug from image 2. Keep its glossy glaze, round rim and rounded handle. "
                         "The complete mug stands upright among the flowers with natural daylight and a soft contact shadow." + KEEP),
    ]
    (output / "cases.json").write_text(json.dumps([c.model_dump() for c in cases], indent=2), encoding="utf-8")
    (output / "assets.json").write_text(json.dumps({p.name: sha256(p.read_bytes()).hexdigest() for p in sorted(assets.iterdir())}, indent=2), encoding="utf-8")
    return cases


def load_cases(output: Path) -> list[Case]:
    recorded = json.loads((output / "assets.json").read_text())
    for name, digest in recorded.items():
        if sha256((output / "assets" / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Changed comparison asset {name}")
    return [Case.model_validate(c) for c in json.loads((output / "cases.json").read_text())]


class Workspace(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    workspace: object = None  # only for inpaint cases
    source: Image.Image | None = None
    mask: Image.Image | None = None
    references: list[Image.Image]
    size: tuple[int, int]

    def images(self) -> list[Image.Image]:
        return ([self.source] if self.source else []) + self.references


def _fit_area(image: Image.Image, area: int = 1024 * 1024) -> Image.Image:
    width = round((area * image.width / image.height) ** 0.5 / 16) * 16
    return image.resize((width, round(area / width / 16) * 16), Image.Resampling.LANCZOS)


def workspaces(output: Path, arm: Arm, cases: list[Case]) -> dict[str, Workspace]:
    prepared = {}
    for case in cases:
        # The app's reference handling: several references share one padded size.
        references = _reference_images([output / "assets" / name for name in case.references])
        if case.mode == "compose":
            prepared[case.name] = Workspace(references=references, size=case.size)
            continue
        if case.mode == "edit":
            with Image.open(output / "assets" / case.source) as image:
                source = _fit_area(image.convert("RGB"))
            prepared[case.name] = Workspace(source=source, references=references, size=source.size)
            continue
        work = output / "work" / arm.name / case.name
        work.mkdir(parents=True, exist_ok=True)
        workspace = prepare_inpaint_workspace(output / "assets" / case.source, output / "assets" / case.mask, work, 1024, 0, case.context,
                                              composite_feather=0, generation_feather=0, center_selection=case.center_selection)
        with Image.open(workspace.model_input_path) as image:
            source = image.convert("RGB")
        with Image.open(workspace.model_mask_path) as image:
            mask = image.convert("L")
        prepared[case.name] = Workspace(workspace=workspace, source=source, mask=mask, references=references, size=source.size)
    return prepared


def _gpu_budget_gib(torch) -> float:
    """What this process may use: free VRAM minus headroom for the desktop's own use."""
    free, _ = torch.cuda.mem_get_info()
    return free / 1024**3 - 1.5


def _encoder_kwargs(torch, encoder: Encoder, headroom_gib: float = 1) -> dict:
    # Large encoders spill their last layers to the CPU instead of exhausting VRAM.
    kwargs = dict(revision=encoder.revision, torch_dtype=torch.bfloat16, device_map="auto",
                  max_memory={0: f"{max(2.0, _gpu_budget_gib(torch) - headroom_gib):.1f}GiB", "cpu": "24GiB"}, local_files_only=True)
    if encoder.subfolder:
        kwargs["subfolder"] = encoder.subfolder
    return kwargs


def _release(torch) -> None:
    gc.collect()
    torch.cuda.empty_cache()


class KleinRenderer:
    def __init__(self, arm: Arm, cases: list[Case], prepared: dict[str, Workspace], protocol: dict) -> None:
        import torch
        from diffusers import Flux2KleinInpaintPipeline, Flux2KleinPipeline
        from huggingface_hub import hf_hub_download
        from transformers import Qwen2TokenizerFast, Qwen3ForCausalLM
        from app.models.nvfp4_runtime import kernel_api, load_transformer

        base, revision = (K4_REPO, K4_REVISION) if arm.family == "klein4b" else (K9_REPO, K9_REVISION)
        quant = (K4_QUANT_REPO, K4_QUANT_FILE, K4_QUANT_REVISION) if arm.family == "klein4b" else (K9_QUANT_REPO, K9_QUANT_FILE, K9_QUANT_REVISION)
        started = time.perf_counter()
        encoder = Qwen3ForCausalLM.from_pretrained(arm.encoder.repo, **_encoder_kwargs(torch, arm.encoder))
        # The stock tokenizer of the matching Klein: the Qwen3 family shares its vocabulary.
        tokenizer = Qwen2TokenizerFast.from_pretrained(base, revision=revision, subfolder="tokenizer", local_files_only=True)
        text_pipe = Flux2KleinInpaintPipeline(scheduler=None, vae=None, text_encoder=encoder, tokenizer=tokenizer, transformer=None, is_distilled=True)
        self.embeddings = {}
        with torch.inference_mode():
            for case in cases:
                self.embeddings[case.name] = text_pipe.encode_prompt(case.klein_prompt, device=torch.device("cuda"))[0].cpu()
        protocol["encode_seconds"] = round(time.perf_counter() - started, 3)
        del text_pipe, encoder, tokenizer
        _release(torch)

        started = time.perf_counter()
        config = hf_hub_download(base, "transformer/config.json", revision=revision, local_files_only=True)
        checkpoint = hf_hub_download(quant[0], quant[1], revision=quant[2], local_files_only=True)
        transformer = load_transformer(Path(config), Path(checkpoint), kernel_api(), lambda: None)
        pipe = Flux2KleinInpaintPipeline.from_pretrained(base, revision=revision, transformer=transformer, text_encoder=None, tokenizer=None,
                                                         torch_dtype=torch.bfloat16, local_files_only=True)
        del transformer
        pipe.to("cuda")
        pipe.vae.enable_tiling()
        pipe.mask_processor.register_to_config(do_binarize=False)
        pipe.set_progress_bar_config(disable=True)
        # Edit and compose use the multi-reference pipeline on the same modules,
        # constructed directly like the app does so the NVFP4 scales stay intact.
        self.edit_pipe = Flux2KleinPipeline(**{name: getattr(pipe, name) for name in SHARED_COMPONENTS}, is_distilled=True)
        self.edit_pipe.set_progress_bar_config(disable=True)
        protocol["renderer_load_seconds"] = round(time.perf_counter() - started, 3)
        self.pipe, self.prepared, self.steps = pipe, prepared, arm.steps

    def render(self, case: Case, seed: int, callback, steps: int | None = None) -> Image.Image:
        import torch
        item = self.prepared[case.name]
        width, height = item.size
        kwargs = dict(prompt_embeds=self.embeddings[case.name].to("cuda"), height=height, width=width, guidance_scale=1.0)
        if case.mode == "inpaint":
            pipe = self.pipe
            kwargs.update(image=item.source, mask_image=item.mask, strength=1.0)
            if item.references:
                kwargs["image_reference"] = item.references[0] if len(item.references) == 1 else item.references
        else:
            pipe = self.edit_pipe
            kwargs["image"] = item.images()
        with torch.inference_mode():
            return pipe(**kwargs, num_inference_steps=steps or self.steps, generator=torch.Generator("cuda").manual_seed(seed),
                        callback_on_step_end=callback).images[0]


def _fp8_linear_class():
    import torch
    from torch import nn

    class ScaledFp8Linear(nn.Module):
        """Per-tensor scaled FP8 weights, dequantized to BF16 for each call."""

        def __init__(self, weight, scale, bias) -> None:
            super().__init__()
            if weight.dtype != torch.float8_e4m3fn or weight.ndim != 2 or scale.numel() != 1:
                raise ValueError("Expected 2-D float8_e4m3fn weights with one scale")
            self.out_features, self.in_features = weight.shape
            self.register_buffer("weight", weight.contiguous(), persistent=False)
            self.register_buffer("scale", scale.float().reshape(()), persistent=False)
            self.register_buffer("bias", bias, persistent=False)

        def forward(self, x):
            weight = (self.weight.to(torch.float32) * self.scale).to(x.dtype)
            return nn.functional.linear(x, weight, None if self.bias is None else self.bias.to(x.dtype))

    return ScaledFp8Linear


def load_turbo_transformer(config_path: Path, checkpoint: Path):
    import torch
    from accelerate import init_empty_weights
    from diffusers import QwenImage21Transformer2DModel
    from safetensors import safe_open
    from torch import nn

    ScaledFp8Linear = _fp8_linear_class()
    config = json.loads(config_path.read_text())
    with init_empty_weights():
        model = QwenImage21Transformer2DModel.from_config(config).to(dtype=torch.bfloat16)
    expected = set(model.state_dict())
    state, quantized, consumed_bias = {}, 0, set()
    with safe_open(str(checkpoint), framework="pt", device="cpu") as saved:
        keys = set(saved.keys())
        for key in sorted(keys):
            if key.endswith(".weight_scale"):
                continue
            if key.endswith(".comfy_quant"):
                described = json.loads(bytes(saved.get_tensor(key).tolist()))
                if described.get("format") != "float8_e4m3fn":
                    raise ValueError(f"Unsupported quantization {described} at {key}")
                continue
            tensor = saved.get_tensor(key)
            if tensor.dtype != torch.float8_e4m3fn:
                state[key] = tensor
                continue
            source = key.removesuffix(".weight")
            scale = saved.get_tensor(source + ".weight_scale")
            if source.endswith("img_mlp.gate_up"):
                # ComfyUI fuses the SwiGLU gate and up projections, gate rows first.
                gate, up = tensor.chunk(2, dim=0)
                targets = [(source.replace("gate_up", "gate_layer"), gate), (source.replace("gate_up", "proj"), up)]
            else:
                targets = [(source, tensor)]
            for module, weight in targets:
                previous = model.get_submodule(module)
                if not isinstance(previous, nn.Linear) or tuple(weight.shape) != (previous.out_features, previous.in_features):
                    raise ValueError(f"Unexpected FP8 target {module}")
                if previous.bias is not None and source + ".bias" not in keys:
                    raise ValueError(f"Missing bias for {module}")
                bias = saved.get_tensor(source + ".bias") if previous.bias is not None else None
                parent, leaf = module.rsplit(".", 1)
                setattr(model.get_submodule(parent), leaf, ScaledFp8Linear(weight, scale, bias))
                expected -= {module + ".weight", module + ".bias"}
                quantized += 1
            consumed_bias.add(source + ".bias")
    state = {k: v for k, v in state.items() if k not in consumed_bias}
    unexpected = set(state) - expected
    if unexpected:
        raise ValueError(f"Unexpected checkpoint keys: {sorted(unexpected)[:8]}")
    model.load_state_dict(state, strict=False, assign=True)
    missing = [name for name, parameter in (*model.named_parameters(), *model.named_buffers()) if parameter.is_meta]
    if missing:
        raise ValueError(f"Checkpoint left uninitialized tensors: {missing[:8]}")
    return model.eval(), quantized


def qwen_condition_images(pipe_cls_module, image_processor, images: list[Image.Image], resolution: int) -> list[Image.Image]:
    """The pipeline's own resize, so cached embeddings see exactly what the renderer would."""
    resized = []
    for image in images:
        image = image.convert("RGBA")
        width, height, _ = pipe_cls_module.calculate_dimensions(resolution * resolution, image.width / image.height)
        resized.append(image_processor.resize(image, width=width, height=height))
    return resized


class QwenRenderer:
    RESOLUTION = 1024

    def __init__(self, arm: Arm, cases: list[Case], prepared: dict[str, Workspace], protocol: dict) -> None:
        import torch
        from diffusers import FlowMatchEulerDiscreteScheduler, QwenImage21Pipeline
        from diffusers.pipelines.qwenimage21 import pipeline_qwenimage21 as module
        from huggingface_hub import hf_hub_download
        from transformers import AutoImageProcessor, AutoVideoProcessor, Qwen2TokenizerFast
        from transformers import Qwen3VLForConditionalGeneration, Qwen3VLProcessor

        started = time.perf_counter()
        # Assembled from its parts: the folder has no config.json, which the
        # processor loader otherwise looks up offline and fails on.
        parts = dict(revision=QWEN_REVISION, subfolder="processor", local_files_only=True)
        tokenizer = Qwen2TokenizerFast.from_pretrained(QWEN_REPO, **parts)
        processor = Qwen3VLProcessor(image_processor=AutoImageProcessor.from_pretrained(QWEN_REPO, **parts), tokenizer=tokenizer,
                                     video_processor=AutoVideoProcessor.from_pretrained(QWEN_REPO, **parts), chat_template=tokenizer.chat_template)
        # Vision-tower activations for up to three megapixel images need several GiB.
        encoder = Qwen3VLForConditionalGeneration.from_pretrained(arm.encoder.repo, **_encoder_kwargs(torch, arm.encoder, headroom_gib=5))
        text_pipe = QwenImage21Pipeline(scheduler=None, vae=None, text_encoder=encoder, processor=processor, transformer=None)
        self.embeddings = {}
        with torch.inference_mode():
            for case in cases:
                item = prepared[case.name]
                images = qwen_condition_images(module, text_pipe.image_processor, item.images(), self.RESOLUTION)
                embeds, mask, pads = text_pipe.encode_prompt(prompt=case.qwen_prompt, image=images, device=torch.device("cuda"))
                self.embeddings[case.name] = tuple(None if t is None else t.cpu() for t in (embeds, mask, pads))
        protocol["encode_seconds"] = round(time.perf_counter() - started, 3)
        del text_pipe, encoder
        _release(torch)

        started = time.perf_counter()
        config = hf_hub_download(TURBO_REPO, "transformer/config.json", revision=TURBO_REVISION, local_files_only=True)
        checkpoint = hf_hub_download(TURBO_REPO, TURBO_FILE, revision=TURBO_REVISION, local_files_only=True)
        transformer, protocol["fp8_linears"] = load_turbo_transformer(Path(config), Path(checkpoint))
        pipe = QwenImage21Pipeline.from_pretrained(QWEN_REPO, revision=QWEN_REVISION, transformer=transformer, text_encoder=None,
                                                   processor=processor, torch_dtype=torch.bfloat16, local_files_only=True)
        pipe.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(TURBO_REPO, revision=TURBO_REVISION, subfolder="scheduler", local_files_only=True)
        del transformer
        pipe.to("cuda")
        pipe.vae.enable_tiling()
        pipe.set_progress_bar_config(disable=True)
        protocol["renderer_load_seconds"] = round(time.perf_counter() - started, 3)
        self.pipe, self.prepared, self.steps = pipe, prepared, arm.steps

    def render(self, case: Case, seed: int, callback, steps: int | None = None) -> Image.Image:
        import torch
        item = self.prepared[case.name]
        embeds, mask, pads = (None if t is None else t.to("cuda") for t in self.embeddings[case.name])
        # The pipeline re-encodes whenever it receives condition images; hand it the cached result instead.
        self.pipe.encode_prompt = lambda **_: (embeds, mask, pads)
        sigmas = TURBO_SIGMAS if steps is None else TURBO_SIGMAS[-steps:]
        with torch.inference_mode():
            return self.pipe(image=item.images(), prompt=case.qwen_prompt, height=item.size[1], width=item.size[0],
                             output_resolution=self.RESOLUTION, num_inference_steps=len(sigmas), sigmas=sigmas, true_cfg_scale=1.0,
                             # The condition-token KV cache (~2 GiB per megapixel image) does not fit beside
                             # the 7 GiB FP8 weights on 16 GB; recomputing those tokens each step does.
                             use_kv_cache=False,
                             generator=torch.Generator("cuda").manual_seed(seed), callback_on_step_end=callback).images[0]


def run_arm(output: Path, arm: Arm, seeds: tuple[int, ...], only: set[str] | None) -> None:
    import torch

    cases = [c for c in load_cases(output) if not only or c.name in only]
    if psutil.virtual_memory().available < 8 * 1024**3:
        raise MemoryError("Less than 8 GiB of free RAM")
    budget = _gpu_budget_gib(torch)
    if budget < 8:
        raise MemoryError(f"Only {budget:.1f} GiB of VRAM is free; close GPU-heavy applications first")
    torch.cuda.set_per_process_memory_fraction(budget * 1024**3 / torch.cuda.get_device_properties(0).total_memory)
    prepared = workspaces(output, arm, cases)
    protocol = dict(arm=arm.model_dump(), seeds=list(seeds), gpu=torch.cuda.get_device_name(), vram_budget_gib=round(budget, 2),
                    packages={p: version(p) for p in ("torch", "diffusers", "transformers", "comfy-kitchen")})
    path = output / f"results-{arm.name}.json"
    results: list[Result] = []

    def save() -> None:
        path.write_text(json.dumps(dict(protocol=protocol, results=[r.model_dump() for r in results]), indent=2), encoding="utf-8")

    try:
        renderer = (QwenRenderer if arm.family == "qwen21-turbo" else KleinRenderer)(arm, cases, prepared, protocol)
    except Exception as error:
        protocol["error"] = f"{type(error).__name__}: {error}"
        save()
        raise
    started = time.perf_counter()
    renderer.render(cases[0], 99999, None, steps=1)
    protocol["warmup_seconds"] = round(time.perf_counter() - started, 3)
    save()
    for case in cases:
        item = prepared[case.name]
        if item.workspace:
            original = np.asarray(item.workspace.source.convert("RGBA"))
            alpha = np.asarray(item.workspace.composite_mask)
        for seed in seeds:
            result = Result(arm=arm.name, case=case.name, seed=seed)
            results.append(result)
            stem = output / "renders" / f"{arm.name}--{case.name}--{seed}"
            stem.parent.mkdir(exist_ok=True)
            started = time.perf_counter()

            def budget(_pipe, _step, _timestep, values):
                if time.perf_counter() - started > RENDER_LIMIT_SECONDS:
                    raise TimeoutError(f"Render exceeded {RENDER_LIMIT_SECONDS} s")
                if psutil.virtual_memory().available < 2 * 1024**3:
                    raise MemoryError("Free RAM below 2 GiB")
                return values

            torch.cuda.reset_peak_memory_stats()
            try:
                generated = renderer.render(case, seed, budget)
                if item.workspace:
                    generated.save(item.workspace.model_output_path)
                    generated.save(stem.with_name(stem.name + "-raw.png"))
                    item.workspace.composite_to(stem.with_suffix(".png"))
                else:
                    generated.save(stem.with_suffix(".png"))
                torch.cuda.synchronize()
                result.seconds = round(time.perf_counter() - started, 3)
                result.peak_allocated_gib = round(torch.cuda.max_memory_allocated() / 1024**3, 3)
                if item.workspace:
                    with Image.open(stem.with_suffix(".png")) as image:
                        changed = np.any(np.asarray(image.convert("RGBA")) != original, axis=2)
                    result.protected_changes = int(np.count_nonzero(changed & (alpha == 0)))
                    result.selected_changes = int(np.count_nonzero(changed & (alpha > 0)))
                print(f"DONE {arm.name} {case.name} {seed}: {result.seconds}s / {result.peak_allocated_gib} GiB", flush=True)
            except Exception as error:  # Keep going: a failed case is a result too.
                result.error = f"{type(error).__name__}: {error}"
                print(f"FAIL {arm.name} {case.name} {seed}: {result.error}", flush=True)
                _release(torch)
            save()


def gallery(output: Path) -> None:
    import html

    cases = load_cases(output)
    runs = {}
    for path in sorted(output.glob("results-*.json")):
        data = json.loads(path.read_text())
        runs[data["protocol"]["arm"]["name"]] = data
    arms = [ARMS[name] for name in ARMS if name in runs]
    details = output / "details"
    details.mkdir(exist_ok=True)
    sections, nav = [], []
    for case in cases:
        # Inpaint cases show the selection's neighbourhood; edits and new scenes the whole frame.
        bounds, intro, original = None, "", None
        if case.mode == "inpaint":
            with Image.open(output / "assets" / case.source) as source, Image.open(output / "assets" / case.mask) as mask_file:
                mask = _selection_alpha(mask_file)
                left, top, right, bottom = mask.getbbox()
                margin = round(0.12 * max(right - left, bottom - top)) + 24
                bounds = (max(0, left - margin), max(0, top - margin), min(source.width, right + margin), min(source.height, bottom + margin))
                rgb = source.convert("RGB")
                rgb.crop(bounds).save(details / f"{case.name}--original.jpg", quality=92)
                tint = Image.new("RGB", rgb.size, (30, 200, 235))
                Image.composite(tint, rgb, mask.point(lambda x: round(x * 0.4))).crop(bounds).save(details / f"{case.name}--mask.jpg", quality=92)
            original = f"details/{case.name}--original.jpg"
            intro = f'<figure><img src="details/{case.name}--mask.jpg" alt="Selectie"><figcaption>Selectie</figcaption></figure>'
        elif case.mode == "edit":
            with Image.open(output / "assets" / case.source) as source:
                rgb = source.convert("RGB")
            rgb.thumbnail((900, 900), Image.Resampling.LANCZOS)
            rgb.save(details / f"{case.name}--original.jpg", quality=92)
            original = f"details/{case.name}--original.jpg"
            intro = f'<figure><img src="{original}" alt="Origineel"><figcaption>Origineel</figcaption></figure>'
        refs ="".join(f'<figure class="ref"><img src="assets/{html.escape(r)}" alt="Referentie"><figcaption>Referentie</figcaption></figure>' for r in case.references)
        rows = []
        for arm in arms:
            data = runs[arm.name]
            results = [r for r in data["results"] if r["case"] == case.name]
            if not results and "error" in data["protocol"]:
                rows.append(f'<div class="arm"><div class="label"><b>{html.escape(arm.label)}</b><span>{html.escape(arm.encoder.label)}</span></div><p class="error">{html.escape(data["protocol"]["error"])}</p></div>')
                continue
            times = [r["seconds"] for r in results if r["seconds"]]
            cells = []
            for r in results:
                key = f'{arm.name}--{case.name}--{r["seed"]}'
                if r["error"]:
                    cells.append(f'<div class="cell failed"><p>{html.escape(r["error"])}</p><small>seed {r["seed"]}</small></div>')
                    continue
                with Image.open(output / "renders" / f"{key}.png") as image:
                    detail = image.convert("RGB").crop(bounds) if bounds else image.convert("RGB")
                detail.thumbnail((900, 900), Image.Resampling.LANCZOS)
                detail.save(details / f"{key}.jpg", quality=92)
                hover = f' data-original="{original}"' if original else ""
                cells.append(f'<div class="cell"><a href="renders/{key}.png" target="_blank"><img src="details/{key}.jpg"{hover} alt="{html.escape(key)}"></a>'
                             f'<div class="meta"><small>seed {r["seed"]} · {r["seconds"]:.1f} s</small><button class="star" data-key="{key}" aria-label="Favoriet">☆</button></div></div>')
            median = f"{statistics.median(times):.1f} s mediaan" if times else ""
            rows.append(f'<div class="arm"><div class="label"><b>{html.escape(arm.label)}</b><span>{html.escape(arm.encoder.label)}</span><small>{median}</small></div><div class="cells">{"".join(cells)}</div></div>')
        criteria = "".join(f"<li>{html.escape(c)}</li>" for c in case.criteria)
        nav.append(f'<a href="#{case.name}">{html.escape(case.title)}</a>')
        sections.append(f'<section id="{case.name}"><header><span class="tag">{html.escape(case.category)}</span><h2>{html.escape(case.title)}</h2></header>'
                        f'<div class="intro">{intro}{refs}<ul>{criteria}</ul></div>'
                        f'<details><summary>Exacte prompts</summary><p><b>Klein:</b> {html.escape(case.klein_prompt)}</p><p><b>Qwen:</b> {html.escape(case.qwen_prompt)}</p></details>'
                        f'{"".join(rows)}</section>')
    summary = []
    for arm in arms:
        protocol = runs[arm.name]["protocol"]
        times = [r["seconds"] for r in runs[arm.name]["results"] if r["seconds"]]
        peaks = [r["peak_allocated_gib"] for r in runs[arm.name]["results"] if r["peak_allocated_gib"]]
        failed = sum(1 for r in runs[arm.name]["results"] if r["error"])
        summary.append(f"<tr><td>{html.escape(arm.label)}</td><td>{html.escape(arm.encoder.label)}</td><td>{arm.steps}</td>"
                       f"<td>{statistics.median(times):.1f} s</td><td>{max(times):.1f} s</td><td>{max(peaks):.1f} GiB</td><td>{failed}</td>"
                       f"<td>{protocol.get('encode_seconds', 0):.0f} s / {protocol.get('renderer_load_seconds', 0):.0f} s</td></tr>" if times else
                       f"<tr><td>{html.escape(arm.label)}</td><td>{html.escape(arm.encoder.label)}</td><td colspan=6>{html.escape(protocol.get('error', 'geen resultaten'))}</td></tr>")
    style = """
:root{--bg:#f6f5f2;--fg:#1d2124;--muted:#5f676d;--card:#fff;--line:#dedbd4;--accent:#0f766e;--bad:#b42318}
@media (prefers-color-scheme:dark){:root{--bg:#121518;--fg:#e6eaec;--muted:#98a2a8;--card:#1b2024;--line:#2c3338;--accent:#5eead4;--bad:#f97066}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,sans-serif}
main{max-width:1500px;margin:0 auto;padding:24px 16px 80px}h1{font-size:28px;margin:0 0 4px}p.lead{color:var(--muted);margin:0 0 20px;max-width:900px}
nav{position:sticky;top:0;z-index:2;background:var(--bg);display:flex;gap:6px;flex-wrap:wrap;padding:10px 0;border-bottom:1px solid var(--line)}
nav a,button.copy{font:inherit;font-size:13px;color:var(--fg);text-decoration:none;border:1px solid var(--line);border-radius:99px;padding:4px 12px;background:var(--card);cursor:pointer}
table{border-collapse:collapse;margin:16px 0 8px;width:100%;font-size:14px;background:var(--card)}td,th{border:1px solid var(--line);padding:6px 10px;text-align:left}
section{margin:44px 0}header{display:flex;gap:12px;align-items:baseline}h2{margin:0;font-size:22px}.tag{font-size:12px;text-transform:uppercase;letter-spacing:.06em;color:var(--accent)}
.intro{display:flex;gap:12px;align-items:flex-start;margin:12px 0}.intro figure{margin:0;width:200px}.intro img{width:100%;border-radius:6px}.intro ul{margin:0;color:var(--muted)}
figcaption,small{color:var(--muted);font-size:12px}details{margin:8px 0 12px;color:var(--muted)}
.arm{display:grid;grid-template-columns:220px 1fr;gap:12px;padding:12px 0;border-top:1px solid var(--line)}.label{display:flex;flex-direction:column;gap:2px}.label span{font-size:13px}
.cells{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:10px}.cell img{width:100%;border-radius:6px;display:block;cursor:zoom-in}
.meta{display:flex;justify-content:space-between;align-items:center}.star{background:none;border:0;font-size:22px;color:var(--accent);cursor:pointer;padding:0 4px}
.cell.failed,.error{color:var(--bad);font-size:13px;background:var(--card);border-radius:6px;padding:10px}
@media (max-width:760px){.arm{grid-template-columns:1fr}.cells{grid-template-columns:1fr}.intro{flex-wrap:wrap}table{display:block;overflow-x:auto}}
"""
    script = """
const KEY='mosimage-model-comparison-stars';let stars={};try{stars=JSON.parse(localStorage.getItem(KEY)||'{}')}catch(e){}
function paint(){document.querySelectorAll('.star').forEach(b=>{b.textContent=stars[b.dataset.key]?'★':'☆'})}
document.querySelectorAll('.star').forEach(b=>b.addEventListener('click',()=>{stars[b.dataset.key]=!stars[b.dataset.key];if(!stars[b.dataset.key])delete stars[b.dataset.key];try{localStorage.setItem(KEY,JSON.stringify(stars))}catch(e){}paint()}));
document.querySelectorAll('.cell img[data-original]').forEach(img=>{const own=img.src;const show=()=>img.src=img.dataset.original,hide=()=>img.src=own;img.addEventListener('mouseenter',()=>{img._t=setTimeout(show,700)});img.addEventListener('mouseleave',()=>{clearTimeout(img._t);hide()})});
document.querySelector('.copy').addEventListener('click',async e=>{const text=Object.keys(stars).sort().join('\\n')||'(geen favorieten)';try{await navigator.clipboard.writeText(text);e.target.textContent='Gekopieerd'}catch(err){prompt('Kopieer je favorieten',text)}});
paint();
"""
    head = ("<tr><th>Model</th><th>Text-encoder</th><th>Steps</th><th>Mediaan</th><th>Langzaamste</th><th>Piek VRAM</th><th>Mislukt</th><th>Encode / laden</th></tr>")
    page = (f'<!doctype html><html lang="nl"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Modelvergelijking MosImage</title>'
            f'<style>{style}</style><main><h1>Modelvergelijking</h1><p class="lead">Zelfde uitsnede, masker en seeds voor elk model. Beweeg even over een beeld om het origineel te zien, '
            f'klik voor het volledige resultaat, en geef favorieten een ster. Tijden: één render na opwarmen, zonder prompt-encoding en laden.</p>'
            f'<table>{head}{"".join(summary)}</table><nav>{"".join(nav)}<button class="copy">Kopieer favorieten</button></nav>{"".join(sections)}</main><script>{script}</script></html>')
    (output / "index.html").write_text(page, encoding="utf-8")
    print(f"Review: {output / 'index.html'}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--arms", nargs="+", help=f"'all' or any of: {', '.join(ARMS)}")
    parser.add_argument("--arm", choices=ARMS, help=argparse.SUPPRESS)
    parser.add_argument("--cases", nargs="+")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--gallery", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    if args.prepare_only:
        output.mkdir(parents=True, exist_ok=False)
        prepare(output)
        return
    if args.arm:
        run_arm(output, ARMS[args.arm], tuple(args.seeds), set(args.cases) if args.cases else None)
        return
    if args.arms:
        assert_api_unloaded()
        if not (output / "cases.json").exists():
            output.mkdir(parents=True, exist_ok=True)
            prepare(output)
        names = list(ARMS) if args.arms == ["all"] else args.arms
        for name in names:
            if name not in ARMS:
                raise ValueError(f"Unknown arm {name}")
            command = [sys.executable, __file__, "--output", str(output), "--arm", name, "--seeds", *map(str, args.seeds)]
            if args.cases:
                command += ["--cases", *args.cases]
            print(f"=== {name}", flush=True)
            if subprocess.run(command, check=False).returncode:
                print(f"=== {name} failed; continuing", flush=True)
            gallery(output)
        return
    if args.gallery:
        gallery(output)
        return
    parser.error("Choose --prepare-only, --arms or --gallery")


if __name__ == "__main__":
    main()
