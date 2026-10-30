"""Bria Fibo 2.0 edit-turbo against current Klein 9B on the FLUX 3 benchmark fixtures.

Isolated research tool, not a MosImage provider. Fibo needs diffusers 0.42, which is
installed with --no-deps under models/research-cache/fibo2/pylib and put first on
sys.path for the Fibo stages only; the Klein stage runs the app's own environment.
"""
from __future__ import annotations

import argparse
import gc
from hashlib import sha256
from importlib.metadata import version
import json
from pathlib import Path
import shutil
import sys
import time
from typing import Literal

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "data/flux3-klein-2026-10-15"
OUTPUT = ROOT / "data/fibo2-2026-10-30"
RESEARCH = ROOT / "models/research-cache/fibo2"
WEIGHTS = RESEARCH / "weights"
REVISION = "9a24ee6810a782dd6aad4639fadcf48db0957ef4"
SKIPPED = {"typography": "Edit-turbo only edits; text-to-image needs Fibo 2.0 dev"}

if "--stage" in sys.argv and sys.argv[sys.argv.index("--stage") + 1] in ("encode", "fibo"):
    sys.path.insert(0, str(RESEARCH / "pylib"))
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT / "backend/tools"))

import numpy as np
from PIL import Image, ImageDraw
import psutil

import benchmark_flux3_klein as base
from benchmark_flux3_klein import GIB, SEEDS, Measurement, MemorySample, align_references, finish_image
from benchmark_model_comparison import ARMS, Case, Workspace, workspaces

Quant = Literal["nvfp4", "fp8"]
CAP_GIB = 12.5


def save_json(name: str, value: object) -> None:
    (OUTPUT / name).write_text(json.dumps(value, indent=2), encoding="utf-8")


def setup_gpu(required_vram_gib: float, required_ram_gib: float) -> None:
    """Budget from CUDA's own free figure: nvidia-smi also counts desktop memory WDDM moves aside."""
    import torch
    base.assert_api_unloaded()
    free, total = torch.cuda.mem_get_info()
    budget = min(CAP_GIB * GIB, free - 1.25 * GIB)
    available_ram = psutil.virtual_memory().available
    save_json("preflight.json", {"gpu": torch.cuda.get_device_name(), "cuda_free_gib": free / GIB, "allocation_budget_gib": budget / GIB,
                                "available_ram_gib": available_ram / GIB, "required_vram_gib": required_vram_gib, "required_ram_gib": required_ram_gib})
    if budget < required_vram_gib * GIB or available_ram < required_ram_gib * GIB:
        raise RuntimeError(f"Insufficient free GPU/RAM: GPU budget {budget/GIB:.2f} GiB, available RAM {available_ram/GIB:.2f} GiB")
    torch.cuda.set_per_process_memory_fraction(budget / total)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    print(f"GPU budget {budget/GIB:.2f} GiB; available RAM {available_ram/GIB:.2f} GiB", flush=True)


# Every shared helper (preflight, results, case loading, the Klein runner) now uses this folder and budget.
base.OUTPUT = OUTPUT
base.setup_gpu = setup_gpu


def prepare() -> None:
    """Copy the frozen FLUX 3 fixtures; Fibo receives the app-style instructions Klein's current arm uses."""
    OUTPUT.mkdir(parents=True, exist_ok=True)
    shutil.copytree(SOURCE / "assets", OUTPUT / "assets", dirs_exist_ok=True)
    cases = [case for case in json.loads((SOURCE / "cases.json").read_text()) if case["name"] not in SKIPPED]
    save_json("cases.json", cases)
    # The shared Klein runner reads this file even for its current-prompt arm.
    shutil.copyfile(SOURCE / "flux3-prompts.json", OUTPUT / "flux3-prompts.json")
    hashes = {p.name: sha256(p.read_bytes()).hexdigest() for p in sorted((OUTPUT / "assets").glob("*.png"))}
    if hashes != json.loads((SOURCE / "assets.json").read_text()):
        raise ValueError("Fixture assets differ from the FLUX 3 benchmark")
    save_json("assets.json", hashes)
    save_json("provenance.json", {"model": "briaai/Fibo-2.0-edit-turbo", "model_revision": REVISION, "seeds": SEEDS,
                                "skipped": SKIPPED, "prompt_protocol": "Both arms receive the application-style Klein prompts as plain edit instructions; no JSON captions or Gemini prompt rewriting.",
                                "packages": {name: version(name) for name in ("torch", "transformers", "comfy-kitchen", "safetensors")}})
    print(f"Prepared {len(cases)} cases", flush=True)


def prepared_cases(only: list[str] | None) -> tuple[list[Case], dict[str, Workspace]]:
    """The Klein arm's exact conditions: workspace crops, reference fit and 32-pixel grid."""
    cases = base.selected_cases(only)
    prepared = workspaces(OUTPUT, ARMS["klein9b"], cases)
    align_references(prepared)
    for item in prepared.values():
        item.size = tuple(n // 32 * 32 for n in item.size)
        if item.source and item.source.size != item.size:
            item.source = item.source.resize(item.size, Image.Resampling.LANCZOS)
            if item.mask:
                item.mask = item.mask.resize(item.size, Image.Resampling.LANCZOS)
    return cases, prepared


def fibo_inputs(case: Case, item: Workspace) -> tuple[list[Image.Image], Image.Image | None]:
    """Fibo greys out a masked region before encoding, and only accepts a mask with a single image.

    With references, the source is greyed out the same way and passed beside them without a mask.
    """
    from diffusers.pipelines.bria_fibo2.image_processor import BriaFibo2ImageProcessor
    if case.mode != "inpaint":
        return item.images(), None
    if not item.references:
        return [item.source], item.mask
    return [BriaFibo2ImageProcessor.grey_out(item.source, item.mask), *item.references], None


def encode(only: list[str] | None) -> None:
    import torch
    from diffusers import BriaFibo2Pipeline
    from diffusers.pipelines.bria_fibo2.image_processor import BriaFibo2ImageProcessor
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    setup_gpu(9.5, 4)
    cases, prepared = prepared_cases(only)
    folder = OUTPUT / "embeddings"
    folder.mkdir(exist_ok=True)
    started = time.perf_counter()
    with MemorySample() as memory:
        encoder = Qwen3VLForConditionalGeneration.from_pretrained(str(WEIGHTS / "text_encoder"), dtype=torch.bfloat16,
                                                                  device_map="cuda").eval()
        processor = AutoProcessor.from_pretrained(str(WEIGHTS / "processor"))
        torch.cuda.synchronize()
    protocol = {"load_seconds": round(time.perf_counter() - started, 3), "load_peak_rss_gib": round(memory.peak_rss / GIB, 3), "cases": []}
    images_processor = BriaFibo2ImageProcessor(vae_scale_factor=16)
    for case in cases:
        item = prepared[case.name]
        images, mask = fibo_inputs(case, item)
        result = Measurement(arm="fibo2-encoder", case=case.name, seed=0)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        with MemorySample() as memory, torch.inference_mode():
            _, encoder_images = images_processor.preprocess(images, height=item.size[1], width=item.size[0], mask=mask)
            embeds, embeds_mask = BriaFibo2Pipeline._get_qwen_prompt_embeds(encoder, processor, case.klein_prompt,
                                                                           image=encoder_images, device=torch.device("cuda"))
            torch.cuda.synchronize()
            result.seconds = round(time.perf_counter() - started, 3)
        memory.record(result)
        torch.save((embeds.cpu(), embeds_mask.cpu()), folder / f"{case.name}.pt")
        protocol["cases"].append(result.model_dump() | {"tokens": int(embeds_mask.sum())})
        save_json("encoding.json", protocol)
        print(f"ENCODE {case.name}: {result.seconds}s, {int(embeds_mask.sum())} tokens, {result.peak_allocated_gib} GiB", flush=True)


def quantized_linear_classes():
    import torch
    from torch import nn
    from app.models.nvfp4_runtime import Nvfp4Linear, kernel_api
    kernel = kernel_api()

    class DynamicNvfp4Linear(Nvfp4Linear):
        """Round-to-nearest NVFP4 weights; the activation scale is measured per call, as no calibration exists."""

        @classmethod
        def from_weight(cls, weight: torch.Tensor) -> DynamicNvfp4Linear:
            # The FFN's hidden width is a multiple of 64, not 128: zero rows pad it and are sliced off again.
            rows = weight.shape[0]
            weight = nn.functional.pad(weight, (0, 0, 0, -rows % 128))
            scale = (weight.abs().amax().float() / (448 * 6)).clamp_min(1e-12).reshape(1)
            packed, blocks = kernel.quantize_nvfp4(weight.contiguous(), scale, pad_16x=False)
            layer = cls(packed, blocks, scale, torch.ones(1, device=weight.device), kernel)
            layer.rows = rows
            return layer

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            self.input_scale = (x.abs().amax().float() / (448 * 6)).clamp_min(1e-12).reshape(1)
            return super().forward(x.to(torch.bfloat16))[..., :self.rows]

    class RowFp8Linear(nn.Module):
        """Per-output-row FP8 weights, dequantized to BF16 for each call."""

        def __init__(self, weight: torch.Tensor) -> None:
            super().__init__()
            self.out_features, self.in_features = weight.shape
            scale = (weight.abs().amax(dim=1, keepdim=True).float() / 448).clamp_min(1e-12)
            self.register_buffer("weight", (weight.float() / scale).to(torch.float8_e4m3fn), persistent=False)
            self.register_buffer("scale", scale, persistent=False)
            self.calls = 0

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            self.calls += 1
            return nn.functional.linear(x, (self.weight.float() * self.scale).to(x.dtype))

    return {"nvfp4": DynamicNvfp4Linear.from_weight, "fp8": RowFp8Linear}


def load_fibo_transformer(quant: Quant):
    """Quantizes the main blocks' bias-free linears on the GPU while streaming the BF16 shards."""
    import torch
    from accelerate import init_empty_weights
    from diffusers import BriaFibo2Transformer2DModel
    from safetensors import safe_open
    from torch import nn
    folder = WEIGHTS / "transformer"
    with init_empty_weights():
        model = BriaFibo2Transformer2DModel.from_config(BriaFibo2Transformer2DModel.load_config(str(folder))).to(torch.bfloat16)
    targets = {name for name, module in model.named_modules()
               if name.startswith("layers.") and isinstance(module, nn.Linear) and module.bias is None
               and module.in_features % 64 == 0}
    build = quantized_linear_classes()[quant]
    state: dict[str, torch.Tensor] = {}
    for shard in sorted(folder.glob("*.safetensors")):
        with safe_open(str(shard), framework="pt") as saved:
            for key in saved.keys():
                tensor = saved.get_tensor(key).to("cuda", torch.bfloat16)
                name = key.removesuffix(".weight")
                if name in targets:
                    parent, leaf = name.rsplit(".", 1)
                    replacement = build(tensor)
                    if not hasattr(model, "quant_error"):
                        # One layer against its BF16 weights on random activations: a layout or scale bug shows here.
                        x = torch.randn((64, tensor.shape[1]), device="cuda", dtype=torch.bfloat16)
                        exact = nn.functional.linear(x, tensor).float()
                        model.quant_error = ((replacement(x).float() - exact).norm() / exact.norm()).item()
                        replacement.calls = 0
                    setattr(model.get_submodule(parent), leaf, replacement)
                    targets.discard(name)
                else:
                    state[key] = tensor
                del tensor
    if targets:
        raise ValueError(f"Missing quantized weights: {sorted(targets)[:3]}")
    model.load_state_dict(state, strict=True, assign=True)
    if any(t.is_meta for t in (*model.parameters(), *model.buffers())):
        raise ValueError("Incomplete Fibo checkpoint")
    return model.eval().requires_grad_(False)


def quant_calls(model) -> int:
    return sum(getattr(module, "calls", 0) for module in model.modules() if hasattr(module, "calls") and module is not model)


def run_fibo(quant: Quant, only: list[str] | None, seeds: list[int], resume: bool) -> None:
    import torch
    from diffusers import AutoencoderKLFlux2, BriaFibo2Pipeline, FlowMatchEulerDiscreteScheduler
    from app.models.nvfp4_runtime import transformer_storage_bytes
    tag = f"fibo2-{quant}"
    setup_gpu(9, 4)
    cases, prepared = prepared_cases(only)
    result_path = OUTPUT / f"results-{tag}.json"
    retained: list[Measurement] = []
    if result_path.exists():
        if not resume:
            raise FileExistsError("Existing run requires --resume")
        (OUTPUT / "attempts").mkdir(exist_ok=True)
        shutil.copyfile(result_path, OUTPUT / "attempts" / f"{tag}-{time.time_ns()}.json")
        retained = [Measurement.model_validate(r) for r in json.loads(result_path.read_text())["results"] if r["error"] is None]
    started = time.perf_counter()
    with MemorySample() as memory:
        model = load_fibo_transformer(quant)
        vae = AutoencoderKLFlux2.from_pretrained(str(WEIGHTS / "vae"), torch_dtype=torch.bfloat16).to("cuda")
        scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(str(WEIGHTS / "scheduler"))
        pipe = BriaFibo2Pipeline(transformer=model, scheduler=scheduler, vae=vae, text_encoder=None, processor=None)
        pipe.set_progress_bar_config(disable=True)
        warm = torch.load(OUTPUT / "embeddings" / f"{cases[0].name}.pt", weights_only=True)
        with torch.inference_mode():
            images, mask = fibo_inputs(cases[0], prepared[cases[0].name])
            pipe(prompt_embeds=warm[0].cuda(), prompt_embeds_mask=warm[1].cuda(), image=images, mask=mask,
                 height=256, width=256, num_inference_steps=1, guidance_scale=1.0)
        torch.cuda.synchronize()
    protocol = {"arm": tag, "steps": 4, "guidance_scale": 1.0, "seeds": seeds, "gpu": torch.cuda.get_device_name(),
                "quantization": {"nvfp4": "Round-to-nearest NVFP4 of every bias-free main-block linear, native Comfy Kitchen GEMM, per-call activation scale",
                                 "fp8": "Per-row FP8 weights of the same linears, dequantized to BF16 per call"}[quant],
                "timing": "Cached prompt embeddings (encoded separately with the image); includes VAE encode/decode, sampling and both PNG writes",
                "load_and_warmup_seconds": round(time.perf_counter() - started, 3), "load_peak_rss_gib": round(memory.peak_rss / GIB, 3),
                "resident_transformer_gib": round(transformer_storage_bytes(model) / GIB, 3), "diffusers": version("diffusers"),
                "first_layer_relative_error": round(model.quant_error, 5)}
    print(f"Loaded {tag}: {protocol['resident_transformer_gib']} GiB transformer in {protocol['load_and_warmup_seconds']}s", flush=True)
    results = retained
    completed = {(r.case, r.seed) for r in retained}
    failures = 0
    (OUTPUT / "renders").mkdir(exist_ok=True)
    for case in cases:
        item = prepared[case.name]
        embeds, embeds_mask = (t.cuda() for t in torch.load(OUTPUT / "embeddings" / f"{case.name}.pt", weights_only=True))
        images, mask = fibo_inputs(case, item)
        for seed in seeds:
            if (case.name, seed) in completed:
                continue
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            result = Measurement(arm=tag, case=case.name, seed=seed)
            results.append(result)
            before = quant_calls(model)
            started = time.perf_counter()
            try:
                with MemorySample() as memory, torch.inference_mode():
                    generated = pipe(prompt_embeds=embeds, prompt_embeds_mask=embeds_mask, image=images, mask=mask,
                                     height=item.size[1], width=item.size[0], num_inference_steps=4, guidance_scale=1.0,
                                     generator=torch.Generator("cuda").manual_seed(seed)).images[0]
                    torch.cuda.synchronize()
                    finish_image(item, generated, OUTPUT / "renders" / f"{tag}--{case.name}--{seed}", result)
                    result.seconds = round(time.perf_counter() - started, 3)
                memory.record(result)
                result.native_calls = quant_calls(model) - before
                failures = 0
                print(f"DONE {tag} {case.name} {seed}: {result.seconds}s / {result.peak_allocated_gib} GiB", flush=True)
            except Exception as error:
                result.error = f"{type(error).__name__}: {error}"
                failures += 1
                memory.record(result)
                print(f"FAIL {tag} {case.name} {seed}: {result.error}", flush=True)
                gc.collect()
                torch.cuda.empty_cache()
            save_json(f"results-{tag}.json", {"protocol": protocol, "results": [r.model_dump() for r in results]})
            if failures >= 2:
                raise RuntimeError("Stopped after two consecutive failures; evidence retained")


def sheets() -> None:
    """One contact sheet per case: input, then every arm's seed 7 and 42 below each other."""
    from PIL import ImageFont
    arms = [tag for tag in ("klein9b-current", "fibo2-nvfp4", "fibo2-fp8") if (OUTPUT / f"results-{tag}.json").exists()]
    folder = OUTPUT / "sheets"
    folder.mkdir(exist_ok=True)
    font = ImageFont.load_default(size=22)
    for case in base.selected_cases(None):
        tiles = [("input", OUTPUT / "assets" / (case.source or case.references[0]))]
        tiles += [(f"{arm} {seed}", OUTPUT / "renders" / f"{arm}--{case.name}--{seed}.png") for arm in arms for seed in SEEDS]
        height = 420
        thumbs = []
        for label, path in tiles:
            if not path.exists():
                continue
            with Image.open(path) as image:
                thumb = image.convert("RGB")
            thumb = thumb.resize((round(thumb.width * height / thumb.height), height), Image.Resampling.LANCZOS)
            ImageDraw.Draw(thumb).text((8, 6), label, fill="white", font=font, stroke_width=2, stroke_fill="black")
            thumbs.append(thumb)
        sheet = Image.new("RGB", (sum(t.width for t in thumbs) + 8 * len(thumbs), height), "white")
        x = 0
        for thumb in thumbs:
            sheet.paste(thumb, (x, 0))
            x += thumb.width + 8
        sheet.save(folder / f"{case.name}.jpg", quality=90)
    print(f"Sheets for {len(arms)} arms in {folder}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("prepare", "encode", "fibo", "klein9b", "sheets"), required=True)
    parser.add_argument("--quant", choices=("nvfp4", "fp8"), default="nvfp4")
    parser.add_argument("--only", nargs="*")
    parser.add_argument("--seeds", nargs="*", type=int, default=list(SEEDS))
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.stage == "prepare":
        prepare()
    elif args.stage == "encode":
        encode(args.only)
    elif args.stage == "fibo":
        run_fibo(args.quant, args.only, args.seeds, args.resume)
    elif args.stage == "klein9b":
        base.run("klein9b", args.only, args.seeds, "klein9b-current", "current", args.resume)
    else:
        sheets()


if __name__ == "__main__":
    main()
