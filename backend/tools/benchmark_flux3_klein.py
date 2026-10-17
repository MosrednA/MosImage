"""Isolated FLUX 3 vs the current Klein 9B provider. No app registration.

Run sequential stages from the repository root with backend/.venv/Scripts/python.exe.
The Apache-2.0 BFL reference source and gated weights live under models/research-cache.
All cases, prompts, inputs, failures and measurements are retained under data/.
"""
from __future__ import annotations

import argparse
import gc
from hashlib import sha256
from importlib.metadata import version
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time
import traceback
from typing import TYPE_CHECKING, Literal
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "data/flux3-klein-2026-10-15"
RESEARCH = ROOT / "models/research-cache/flux3-klein"
WEIGHTS = RESEARCH / "weights"
REVISION = "8454017f9ebaada30c998fccd9f67a80c7580f68"
REFERENCE_REVISION = "45e57c9fb06f9751cdced29a65c5d2171003be05"
OUTPUT.mkdir(parents=True, exist_ok=True)
os.environ["HF_XET_CACHE"] = str(RESEARCH / "xet-cache")
os.environ["MOSIMAGE_DATA_DIR"] = str(OUTPUT / "app-data")
os.environ["MOSIMAGE_DB_PATH"] = str(OUTPUT / "app-data/app.db")
os.environ["MOSIMAGE_IMAGE_DIR"] = str(OUTPUT / "app-data/images")
sys.path[:0] = [str(ROOT / "backend"), str(RESEARCH / "reference/src")]
os.chdir(OUTPUT)

import numpy as np
from PIL import Image, ImageDraw
import psutil
from pydantic import BaseModel, ConfigDict

if TYPE_CHECKING:
    from diffusers import Flux2KleinInpaintPipeline, Flux2KleinPipeline

from app.config import settings
from app.jobs import Job, JobControl, JobManager
from app.models.flux2_klein import SHARED_COMPONENTS, PipelineTask
from app.models.flux2_klein_nvfp4 import Flux2Klein9bNvfp4Provider, Flux2KleinNvfp4Provider
os.chdir(ROOT / "backend")  # Historical fixture helpers validate their working directory.
from benchmark_inpaint_quality import assert_api_unloaded
from benchmark_model_comparison import ARMS, Case, Workspace, load_cases, workspaces
os.chdir(OUTPUT)

SEEDS = (7, 42)
GIB = 1024**3


class BenchmarkKleinProvider(Flux2Klein9bNvfp4Provider):
    """Same app components/inference; move the renderer first to avoid a transient RAM double residency."""

    def _create_pipeline(self, task: PipelineTask, control: JobControl) -> Flux2KleinInpaintPipeline | Flux2KleinPipeline:
        from diffusers import Flux2KleinInpaintPipeline, Flux2KleinPipeline
        from huggingface_hub import snapshot_download
        from app.models.klein_text_encoder import load_text_encoder
        pipe = Flux2KleinNvfp4Provider._create_pipeline(self, task, control)
        pipe.to("cuda")
        root = snapshot_download(self.base_repo, revision=self.base_revision, allow_patterns=["text_encoder/*"],
                                 token=settings.huggingface_token, local_files_only=True)
        components = {name: getattr(pipe, name) for name in SHARED_COMPONENTS}
        components["text_encoder"] = load_text_encoder(Path(root), control.raise_if_cancelled)
        pipeline_class = Flux2KleinInpaintPipeline if task == "inpaint" else Flux2KleinPipeline
        return pipeline_class(**components, is_distilled=pipe.config.get("is_distilled", False))


class Measurement(BaseModel):
    model_config = ConfigDict(extra="forbid")
    arm: str
    case: str
    seed: int
    seconds: float | None = None
    encoding_seconds: float | None = None
    peak_allocated_gib: float | None = None
    peak_reserved_gib: float | None = None
    peak_rss_gib: float | None = None
    min_available_ram_gib: float | None = None
    peak_card_mib: int = 0
    native_calls: int = 0
    protected_changes: int | None = None
    selected_changes: int | None = None
    error: str | None = None


def save_json(name: str, value: object) -> None:
    (OUTPUT / name).write_text(json.dumps(value, indent=2), encoding="utf-8")


def download() -> None:
    """Fetch immutable source and only the distilled NVFP4 package; no dependency installation."""
    from huggingface_hub import get_token, snapshot_download
    names = ("__init__.py", "autoencoder.py", "bbox.py", "checkpoint.py", "dit.py", "fp8.py", "nvfp4.py",
             "pipeline.py", "sampling.py", "text_encoder.py")
    paths = ["LICENSE", "README.md", "pyproject.toml", *("src/flux3/" + name for name in names)]
    hashes = {}
    for path in paths:
        url = f"https://raw.githubusercontent.com/black-forest-labs/flux-3-image-klein/{REFERENCE_REVISION}/{path}"
        with urllib.request.urlopen(url, timeout=30) as response:
            data = response.read()
        target = RESEARCH / "reference" / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        hashes[path] = sha256(data).hexdigest()
    auth = type(settings)(_env_file=ROOT / "backend/.env")
    snapshot_download("black-forest-labs/flux-3-image-klein", revision=REVISION, local_dir=WEIGHTS,
                      token=auth.huggingface_token or get_token(), max_workers=3,
                      allow_patterns=["README.md", "LICENSE.md", "NOTICE.md", "config.json", "manifest.json",
                                      "ae.safetensors", "text_encoder/*", "variants/nvfp4/*"])
    save_json("download-provenance.json", {"model_revision": REVISION, "reference_revision": REFERENCE_REVISION,
                                          "source_sha256": hashes})


class MemorySample:
    def __init__(self) -> None:
        self.peak_rss = 0
        self.min_available = psutil.virtual_memory().available
        self.peak_card_mib = 0
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.sample, daemon=True)

    def sample(self) -> None:
        process = psutil.Process()
        while not self.stop.is_set():
            self.peak_rss = max(self.peak_rss, process.memory_info().rss)
            self.min_available = min(self.min_available, psutil.virtual_memory().available)
            try:
                used = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                                               text=True, timeout=3)
                self.peak_card_mib = max(self.peak_card_mib, int(used.strip().splitlines()[0]))
            except (OSError, ValueError, subprocess.SubprocessError):
                pass
            self.stop.wait(0.3)

    def __enter__(self) -> MemorySample:
        self.thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.stop.set()
        self.thread.join()

    def record(self, result: Measurement) -> None:
        import torch
        result.peak_rss_gib = round(self.peak_rss / GIB, 3)
        result.min_available_ram_gib = round(self.min_available / GIB, 3)
        result.peak_card_mib = self.peak_card_mib
        result.peak_allocated_gib = round(torch.cuda.max_memory_allocated() / GIB, 3)
        result.peak_reserved_gib = round(torch.cuda.max_memory_reserved() / GIB, 3)


def setup_gpu(required_vram_gib: float, required_ram_gib: float) -> None:
    import torch
    assert_api_unloaded()
    free, total = torch.cuda.mem_get_info()
    observed = int(subprocess.check_output(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                                          text=True).strip().splitlines()[0]) * 1024**2
    budget = min(free, observed) - int(1.25 * GIB)
    available_ram = psutil.virtual_memory().available
    save_json("preflight.json", {"gpu": torch.cuda.get_device_name(), "physical_free_gib": observed/GIB,
                                "cuda_free_gib": free/GIB, "allocation_budget_gib": budget/GIB,
                                "available_ram_gib": available_ram/GIB,
                                "required_vram_gib": required_vram_gib, "required_ram_gib": required_ram_gib})
    if budget < required_vram_gib * GIB or available_ram < required_ram_gib * GIB:
        raise RuntimeError(f"Insufficient free GPU/RAM: GPU budget {budget/GIB:.2f} GiB, available RAM {available_ram/GIB:.2f} GiB")
    torch.cuda.set_per_process_memory_fraction(budget / total)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    print(f"GPU budget {budget/GIB:.2f} GiB; available RAM {psutil.virtual_memory().available/GIB:.2f} GiB", flush=True)


def prepare() -> None:
    original = ROOT / "data/model-comparison-2026-10-09"
    cases = load_cases(original)
    keep = {"duo-coaster", "dj", "sword", "sword-edit", "garden", "kitchen", "portrait", "architecture", "anna", "mug"}
    cases = [case for case in cases if case.name in keep]
    shutil.copytree(original / "assets", OUTPUT / "assets", dirs_exist_ok=True)
    typography = next(case for case in load_cases(ROOT / "data/qwen-turbo-official-2026-10-10") if case.name == "typography")
    cases.append(typography)
    with Image.open(OUTPUT / "assets/garden.png") as image:
        center = image.convert("RGB").resize((768, 768), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (1024, 768), (127, 127, 127))
    canvas.paste(center, (128, 0))
    canvas.save(OUTPUT / "assets/expand.png")
    mask = Image.new("L", canvas.size, 255)
    mask.paste(0, (128, 0, 896, 768))
    mask.save(OUTPUT / "assets/expand-mask.png")
    expansion = "Extend the garden photograph into both grey side borders. Continue the purple flowers, green foliage and softly blurred garden in matching daylight, perspective and depth of field. Keep the central photograph unchanged. No grey border, no frame, no duplicated terracotta ball."
    cases.append(Case(name="expand", title="Garden side expansion", category="Expansion", source="expand.png", mask="expand-mask.png",
                      klein_prompt=expansion, qwen_prompt=expansion, criteria=("Both borders filled", "Continuous flowers and light", "No seam or duplicated subject")))
    save_json("cases.json", [case.model_dump() for case in cases])
    save_json("assets.json", {p.name: sha256(p.read_bytes()).hexdigest() for p in sorted((OUTPUT / "assets").glob("*.png"))})
    # Freeze recommended caption-style prompts before seeing any FLUX 3 outputs.
    captions = {
        "duo-coaster": cases[0].qwen_prompt,
        "dj": "A photographic portrait of the short curly brown-haired woman from the reference, now standing behind a professional DJ booth with turntables and a mixer in a dark nightclub. Large headphones rest around her neck and one hand operates a turntable. Colourful stage lighting, haze and a dancing crowd fill the background. Her face, curly hair and identity match the reference exactly.",
        "sword": "A young man outdoors on the same street as the reference, wearing the same hoodie, holding a medieval steel longsword upright in his raised right hand. His fingers wrap the leather hilt below a cross guard; the complete straight polished blade points upward into the space above his hand. The phone has been entirely replaced. His face, clothing, pose, street, lighting and camera framing match the reference.",
        "sword-edit": "A young man outdoors on the same street as the reference, wearing the same hoodie, holding a medieval steel longsword upright in his raised right hand. His fingers wrap the leather hilt below a cross guard; the complete straight polished blade points upward. There is no phone. His face, clothing, pose, street, lighting and framing match the reference.",
        "garden": "A close-up photograph of a small glowing orange mushroom growing naturally among purple flowers and green garden foliage, in the location of the terracotta ball in the reference. The terracotta ball is absent. The complete mushroom is small relative to the surrounding flowers, grounded with a soft contact shadow and subtle warm local glow. Preserve the reference's flower arrangement, soft daylight, shallow depth of field and framing.",
        "kitchen": "A photograph of the same clean white marble kitchen countertop as the reference, now empty in the area previously occupied by the yellow mug. Continuous subtle grey marble veins run naturally through that area. There is no mug, handle, yellow reflection or mug shadow. The kitchen, framing, white balance and soft daylight remain identical to the reference.",
        "portrait": "A photographic portrait matching the reference woman, face, curly hair, blue denim shirt, fabric folds, lighting and framing. A small embroidered white daisy with a yellow centre is stitched onto the blue denim exactly where the coral brooch was. The coral brooch is absent. The daisy follows the fabric perspective and looks like real embroidery.",
        "architecture": "A photograph of the same brick facade and poster frame as the reference. Inside the existing frame, a minimalist blue botanical illustration is printed on cream paper, replacing the orange leaf illustration. The existing frame, perspective, brickwork, illumination and camera framing remain identical.",
        "anna": "A realistic garden photograph matching reference image 1. The woman from reference image 2 sits upright on a wooden garden bench on the right beside the gravel path, facing the camera. Her complete body fits in the garden, with both white sneakers resting on the gravel, parallel legs, natural adult proportions and plausible scale. Her face, curly auburn hair, mustard sweater and dark green trousers match image 2. The garden perspective, daylight and soft contact shadows are coherent.",
        "mug": "A close-up photograph of the complete golden yellow ceramic coffee mug from reference image 2, standing upright among the purple flowers of image 1 where the terracotta ball used to be. There is no terracotta ball. The small mug has the reference's glossy glaze, round rim and single rounded handle on the right, with its entire base and handle visible. Its scale, soft contact shadow, natural daylight and depth of field match the surrounding garden.",
        "typography": typography.klein_prompt,
        "expand": "A wide garden photograph with purple flowers, green foliage and the central terracotta ball, in soft natural daylight and shallow depth of field. The garden continues naturally to the full left and right edges, filling both grey borders of the reference with coherent flowers, greenery and blurred background. The central original garden stays identical. There are no grey bars, frames, seams or duplicated terracotta balls.",
    }
    save_json("flux3-prompts.json", captions)
    save_json("provenance.json", {"model_revision": REVISION, "reference_revision": REFERENCE_REVISION,
                                "seeds": SEEDS, "prompt_upsampler": False,
                                "prompt_protocol": "The primary arms use identical pre-frozen full-scene captions. A separate Klein current-style control retains the original application prompts. Native mask conditioning differs by model; final compositing is shared.",
                                "packages": {name: version(name) for name in ("torch", "diffusers", "transformers", "comfy-kitchen", "safetensors")}})
    print(f"Prepared {len(cases)} cases, two seeds each", flush=True)


def selected_cases(only: list[str] | None) -> list[Case]:
    return [case for case in load_cases(OUTPUT) if not only or case.name in only]


def align_references(prepared: dict[str, Workspace]) -> None:
    # BFL's VAE patchifies pairs of latent pixels. Shared 16px-grid inputs avoid a
    # shape error on the historical 683px-wide Anna reference in either arm.
    from PIL import ImageOps
    for item in prepared.values():
        item.references = [ImageOps.fit(im, (im.width//16*16, im.height//16*16), Image.Resampling.LANCZOS)
                           if im.width%16 or im.height%16 else im for im in item.references]


def encode(only: list[str] | None) -> None:
    import torch
    from flux3.text_encoder import TextEncoder
    setup_gpu(9, 12)
    prompts = json.loads((OUTPUT / "flux3-prompts.json").read_text())
    cases = selected_cases(only)
    folder = OUTPUT / "embeddings"
    folder.mkdir(exist_ok=True)
    started = time.perf_counter()
    with MemorySample() as memory:
        encoder = TextEncoder(str(WEIGHTS / "text_encoder"))
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
    protocol = {"load_seconds": round(elapsed, 3), "load_peak_rss_gib": memory.peak_rss/GIB, "cases": []}
    for case in cases:
        result = Measurement(arm="flux3-encoder", case=case.name, seed=0)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        with MemorySample() as memory:
            tokens, ids = encoder.encode(prompts[case.name])
            torch.cuda.synchronize()
            result.seconds = round(time.perf_counter() - started, 3)
        memory.record(result)
        torch.save((tokens.cpu(), ids.cpu()), folder / f"{case.name}.pt")
        protocol["cases"].append(result.model_dump())
        save_json("encoding.json", protocol)
        print(f"ENCODE {case.name}: {result.seconds}s / {result.peak_allocated_gib} GiB", flush=True)


def load_flux3_transformer():
    import torch
    from torch import nn
    from safetensors import safe_open
    from comfy_kitchen.float_utils import swap_nibbles, to_blocked
    from app.models.nvfp4_runtime import Nvfp4Linear, kernel_api
    from flux3.dit import DiT
    kernel = kernel_api()
    class Flux3Linear(Nvfp4Linear):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # BFL explicitly casts flow-map FP32 residuals before NVFP4 activation quantization.
            return super().forward(x.to(torch.bfloat16))
    with torch.device("meta"):
        model = DiT(flow_map=True).to(dtype=torch.bfloat16)
    consumed: set[str] = set()
    count = 0
    with safe_open(str(WEIGHTS / "variants/nvfp4/model.safetensors"), framework="pt") as saved:
        for key in saved.keys():
            if not key.endswith(".weight_scale_2"):
                continue
            name = key.removesuffix(".weight_scale_2")
            previous = model.get_submodule(name)
            if not isinstance(previous, nn.Linear) or previous.bias is not None:
                raise ValueError(f"Unexpected quantized layer {name}")
            names = [name + suffix for suffix in (".weight", ".weight_scale", ".weight_scale_2", ".input_scale")]
            packed, blocks, scale, input_scale = [saved.get_tensor(n).to("cuda") for n in names]
            # BFL exports even-low nibble, row-major scales; Kitchen uses even-high and cuBLAS tiles.
            packed = swap_nibbles(packed)
            blocks = to_blocked(blocks.view(torch.uint8), flatten=False).view(torch.float8_e4m3fn)
            replacement = Flux3Linear(packed, blocks, scale, input_scale, kernel)
            if (replacement.in_features, replacement.out_features) != (previous.in_features, previous.out_features):
                raise ValueError(f"Dimensions differ at {name}")
            parent, leaf = name.rsplit(".", 1)
            setattr(model.get_submodule(parent), leaf, replacement)
            consumed.update(names)
            count += 1
        state = {key: saved.get_tensor(key).to(device="cuda", dtype=torch.bfloat16) for key in saved.keys() if key not in consumed}
    model.load_state_dict(state, strict=True, assign=True)
    if count != 198 or any(t.is_meta for t in (*model.parameters(), *model.buffers())):
        raise ValueError("Incomplete FLUX 3 checkpoint")
    return model.eval().requires_grad_(False)


def probe() -> None:
    import torch
    import torch.nn.functional as F
    from safetensors import safe_open
    from comfy_kitchen.float_utils import swap_nibbles, to_blocked
    from app.models.nvfp4_runtime import Nvfp4Linear, kernel_api
    setup_gpu(1, 1)
    kernel = kernel_api()
    name = "single_blocks.0.q_proj"
    with safe_open(str(WEIGHTS / "variants/nvfp4/model.safetensors"), framework="pt") as saved:
        packed = saved.get_tensor(name + ".weight")[:128].to("cuda")
        scales = saved.get_tensor(name + ".weight_scale")[:128].to("cuda")
        scale = saved.get_tensor(name + ".weight_scale_2").to("cuda")
        act_scale = saved.get_tensor(name + ".input_scale").to("cuda")
    lut = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6], device="cuda")
    unpacked = torch.stack([packed & 15, packed >> 4], dim=-1).reshape(128, -1).long()
    exact = (lut[unpacked] * scales.float().repeat_interleave(16, dim=1) * scale).to(torch.bfloat16)
    converted = swap_nibbles(packed)
    tiled = to_blocked(scales.view(torch.uint8), flatten=False).view(torch.float8_e4m3fn)
    dequantized = kernel.dequantize_nvfp4(converted, scale, tiled)
    torch.testing.assert_close(dequantized, exact, rtol=0, atol=0)
    linear = Nvfp4Linear(converted, tiled, scale, act_scale, kernel)
    x = torch.randn((32, exact.shape[1]), device="cuda", dtype=torch.bfloat16,
                    generator=torch.Generator("cuda").manual_seed(123))
    qx, bx = kernel.quantize_nvfp4(x, act_scale, pad_16x=True)
    reconstructed_x = kernel.dequantize_nvfp4(qx, act_scale, bx)[:32]
    reference = F.linear(reconstructed_x, exact)
    output = linear(x)
    error = ((output.float()-reference.float()).norm()/reference.float().norm()).item()
    if error > .015:
        raise ValueError(f"Native NVFP4 GEMM disagrees with quantized reference: {error}")
    save_json("kernel-proof.json", {"weight_dequantization_exact": True, "gemm_relative_error": error,
                                  "native_calls": linear.calls, "layer": name,
                                  "conversion": "Nibble swap and cuBLAS scale tiling only; no weight requantization"})
    print(f"Native kernel proof passed: exact weights, GEMM relative error {error:.6f}", flush=True)


def finish_image(item: Workspace, generated: Image.Image, stem: Path, result: Measurement) -> None:
    generated.save(stem.with_name(stem.name + "-raw.png"))
    if item.workspace is None:
        generated.save(stem.with_suffix(".png"))
        return
    generated.save(item.workspace.model_output_path)
    item.workspace.composite_to(stem.with_suffix(".png"))
    original = np.asarray(item.workspace.source.convert("RGBA"))
    with Image.open(stem.with_suffix(".png")) as final:
        changed = np.any(np.asarray(final.convert("RGBA")) != original, axis=2)
    mask = np.asarray(item.workspace.composite_mask)
    result.protected_changes = int(np.count_nonzero(changed & (mask == 0)))
    result.selected_changes = int(np.count_nonzero(changed & (mask > 0)))


def cycle() -> None:
    """Actual requests with sequential encoder/renderer loading, bounded in both GPU and host memory."""
    import torch
    from flux3.autoencoder import AutoEncoder
    from flux3.checkpoint import load_checkpoint
    from flux3.pipeline import Pipeline
    from flux3.text_encoder import TextEncoder
    from app.models.nvfp4_runtime import kernel_calls
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    setup_gpu(9, 3)
    def load_encoder() -> TextEncoder:
        value = TextEncoder.__new__(TextEncoder)
        value.model = Qwen3VLForConditionalGeneration.from_pretrained(str(WEIGHTS / "text_encoder"),
                                      dtype=torch.bfloat16, device_map="cuda").eval()
        value.processor = AutoProcessor.from_pretrained(str(WEIGHTS / "text_encoder"))
        value.processor.tokenizer.padding_side = "right"
        value.device = "cuda"
        return value
    cases = selected_cases(["duo-coaster", "garden", "typography"])
    prepared = workspaces(OUTPUT, ARMS["klein9b"], cases)
    align_references(prepared)
    prompts = json.loads((OUTPUT / "flux3-prompts.json").read_text())
    started = time.perf_counter()
    with MemorySample() as memory:
        encoder = load_encoder()
        warm_text = encoder.encode("a quiet garden")
        del encoder
        gc.collect()
        torch.cuda.empty_cache()
        model = load_flux3_transformer()
        ae = load_checkpoint(AutoEncoder(), str(WEIGHTS / "ae.safetensors"), "cuda", torch.float32)
        renderer = Pipeline.__new__(Pipeline)
        renderer.dits, renderer.device, renderer.offload, renderer.autoencoder = {"distilled": model}, "cuda", False, ae
        class CachedEncoder:
            values: tuple[torch.Tensor, torch.Tensor]
            def encode(self, prompt: str) -> tuple[torch.Tensor, torch.Tensor]:
                return self.values
        cached = CachedEncoder()
        cached.values = warm_text
        renderer.text_encoder = cached
        with torch.inference_mode():
            renderer.generate("distilled", "a quiet garden", [], 256, 256, 1, None, 99999)
        torch.cuda.synchronize()
        elapsed = time.perf_counter()-started
    protocol = {"arm": "flux3", "steps": 16, "timing": "Actual prompt-to-file; a new prompt releases the renderer, loads the encoder directly on CUDA, encodes, releases it and reloads the renderer. Second seed retains renderer and text cache. All component reloads included; download excluded.",
                "load_seconds": round(elapsed, 3), "load_peak_rss_gib": round(memory.peak_rss/GIB, 3), "prompt_upsampler": False}
    results: list[Measurement] = []
    for case in cases:
        for seed in (7, 42):
            result = Measurement(arm="flux3-cycle", case=case.name, seed=seed)
            results.append(result)
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
            before = kernel_calls(model)
            try:
                with MemorySample() as memory, torch.inference_mode():
                    encode_start = time.perf_counter()
                    if seed == 7:
                        renderer.dits.clear()
                        del model
                        gc.collect()
                        torch.cuda.empty_cache()
                        encoder = load_encoder()
                        cached.values = encoder.encode(prompts[case.name])
                        del encoder
                        gc.collect()
                        torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    result.encoding_seconds = round(time.perf_counter()-encode_start, 3)
                    if seed == 7:
                        model = load_flux3_transformer()
                        renderer.dits["distilled"] = model
                        before = 0
                    item = prepared[case.name]
                    size = tuple(n//32*32 for n in item.size)
                    images = [im.resize(size, Image.Resampling.LANCZOS) if index == 0 and item.source else im
                              for index, im in enumerate(item.images())]
                    generated = renderer.generate("distilled", prompts[case.name], images, *size, 16, None, seed)
                    stem = OUTPUT / "renders" / f"flux3-cycle--{case.name}--{seed}"
                    finish_image(item, generated, stem, result)
                    torch.cuda.synchronize()
                    result.seconds = round(time.perf_counter()-started, 3)
                memory.record(result)
                result.native_calls = kernel_calls(model)-before
                print(f"CYCLE {case.name} {seed}: {result.seconds}s, encode+transfer {result.encoding_seconds}s, peak {result.peak_allocated_gib} GiB, RAM {result.peak_rss_gib} GiB", flush=True)
            except Exception as error:
                result.error = f"{type(error).__name__}: {error}"
                memory.record(result)
                print(f"FAIL CYCLE {case.name}: {result.error}", flush=True)
                save_json("results-flux3-cycle.json", {"protocol": protocol, "results": [r.model_dump() for r in results]})
                raise
            save_json("results-flux3-cycle.json", {"protocol": protocol, "results": [r.model_dump() for r in results]})


def run(arm: Literal["flux3", "klein9b"], only: list[str] | None, seeds: list[int], tag: str, prompt_style: str, resume: bool) -> None:
    import torch
    from app.models.nvfp4_runtime import kernel_calls, transformer_storage_bytes
    setup_gpu(9, 3 if arm == "flux3" else 7)
    cases = selected_cases(only)
    captions = json.loads((OUTPUT / "flux3-prompts.json").read_text())
    if arm == "klein9b" and prompt_style == "caption":
        cases = [case.model_copy(update={"klein_prompt": captions[case.name]}) for case in cases]
    prepared = workspaces(OUTPUT, ARMS["klein9b"], cases)
    align_references(prepared)
    for item in prepared.values():
        item.size = tuple(n // 32 * 32 for n in item.size)
        if item.source and item.source.size != item.size:
            item.source = item.source.resize(item.size, Image.Resampling.LANCZOS)
            if item.mask:
                item.mask = item.mask.resize(item.size, Image.Resampling.LANCZOS)
    save_json(f"conditions-{tag}.json", {
        name: {"size": item.size,
               "source": {"size": item.source.size, "sha256": sha256(item.source.tobytes()).hexdigest()} if item.source else None,
               "mask": {"size": item.mask.size, "sha256": sha256(item.mask.tobytes()).hexdigest()} if item.mask else None,
               "references": [{"size": im.size, "sha256": sha256(im.tobytes()).hexdigest()} for im in item.references]}
        for name, item in prepared.items()})
    result_path = OUTPUT / f"results-{tag}.json"
    retained: list[Measurement] = []
    previous_protocol = None
    if result_path.exists():
        if not resume:
            raise FileExistsError("Existing run requires --resume or a fresh --tag")
        previous = json.loads(result_path.read_text())
        archive = OUTPUT / "attempts"
        archive.mkdir(exist_ok=True)
        shutil.copyfile(result_path, archive / f"{tag}-{time.time_ns()}.json")
        retained = [Measurement.model_validate(r) for r in previous["results"] if r["error"] is None]
        previous_protocol = previous["protocol"]
    protocol = {"arm": arm, "tag": tag, "gpu": torch.cuda.get_device_name(), "steps": 16 if arm == "flux3" else 4,
                "prompt_style": "caption" if arm == "flux3" else prompt_style,
                "seeds": seeds, "prompt_upsampler": False, "warmup": "256-square text generation, excluded",
                "timing": "Klein includes streamed prompt encoding; FLUX 3 cached-embedding renders, separate encoding and lifecycle records"}
    if arm == "klein9b":
        protocol["baseline_load_order"] = "Current app components and inference; renderer placed on CUDA before loading the unchanged CPU-streamed encoder, avoiding transient host double residency"
    if previous_protocol:
        protocol["retained_run_protocol"] = previous_protocol
    started = time.perf_counter()
    with MemorySample() as memory:
        if arm == "flux3":
            from flux3.autoencoder import AutoEncoder
            from flux3.checkpoint import load_checkpoint
            from flux3.pipeline import Pipeline
            model = load_flux3_transformer()
            ae = load_checkpoint(AutoEncoder(), str(WEIGHTS / "ae.safetensors"), "cuda", torch.float32)
            renderer = Pipeline.__new__(Pipeline)
            renderer.dits, renderer.device, renderer.offload, renderer.autoencoder = {"distilled": model}, "cuda", False, ae
            class CachedEncoder:
                values: tuple[torch.Tensor, torch.Tensor]
                def encode(self, prompt: str) -> tuple[torch.Tensor, torch.Tensor]:
                    return self.values
            cached = CachedEncoder()
            renderer.text_encoder = cached
            prompts = json.loads((OUTPUT / "flux3-prompts.json").read_text())
            cached.values = tuple(t.to("cuda") for t in torch.load(OUTPUT / "embeddings" / f"{cases[0].name}.pt", weights_only=True))
            with torch.inference_mode():
                renderer.generate("distilled", prompts[cases[0].name], [], 256, 256, 1, None, 99999)
        else:
            from diffusers import Flux2KleinPipeline
            manager = JobManager()
            manager._jobs["benchmark"] = Job(id="benchmark", kind="text-to-image")
            class BenchmarkControl(JobControl):
                def raise_if_cancelled(self) -> None:
                    super().raise_if_cancelled()
                    available = psutil.virtual_memory().available
                    if available < .5 * GIB:
                        raise MemoryError(f"Stopped provider loading with {available/GIB:.2f} GiB host RAM left")
            control = BenchmarkControl(manager, "benchmark")
            provider = BenchmarkKleinProvider()
            pipe = provider._load_pipe("cuda", "inpaint", control)
            edit_pipe = Flux2KleinPipeline(**{n: getattr(pipe, n) for n in SHARED_COMPONENTS}, is_distilled=True)
            for p in (pipe, edit_pipe):
                p.set_progress_bar_config(disable=True)
            model = pipe.transformer
            with torch.inference_mode():
                embeds = provider._encode_prompt(pipe, "a quiet garden", control)
                edit_pipe(prompt_embeds=embeds, height=256, width=256, num_inference_steps=1, guidance_scale=1).images[0]
        torch.cuda.synchronize()
    protocol.update(load_and_warmup_seconds=round(time.perf_counter()-started, 3),
                    load_peak_rss_gib=round(memory.peak_rss/GIB, 3), resident_transformer_gib=round(transformer_storage_bytes(model)/GIB, 3))
    results: list[Measurement] = retained
    completed = {(r.case, r.seed) for r in retained}
    failures = 0
    def guard(_module: torch.nn.Module, _args: tuple[object, ...]) -> None:
        if time.perf_counter() - started > 120:
            raise TimeoutError("Render exceeded 120 seconds")
        if psutil.virtual_memory().available < 1.5 * GIB:
            raise MemoryError("Stopped before exhausting host RAM")
    model.register_forward_pre_hook(guard)
    for case in cases:
        item = prepared[case.name]
        for seed in seeds:
            if (case.name, seed) in completed:
                continue
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            result = Measurement(arm=arm, case=case.name, seed=seed)
            results.append(result)
            before = kernel_calls(model)
            stem = OUTPUT / "renders" / f"{tag}--{case.name}--{seed}"
            stem.parent.mkdir(exist_ok=True)
            started = time.perf_counter()
            try:
                with MemorySample() as memory, torch.inference_mode():
                    if arm == "flux3":
                        cached.values = tuple(t.to("cuda") for t in torch.load(OUTPUT / "embeddings" / f"{case.name}.pt", weights_only=True))
                        generated = renderer.generate("distilled", prompts[case.name], item.images(), *item.size, 16, None, seed)
                    else:
                        encoded = time.perf_counter()
                        embeds = provider._encode_prompt(pipe, case.klein_prompt, control)
                        torch.cuda.synchronize()
                        result.encoding_seconds = round(time.perf_counter()-encoded, 3)
                        kwargs = dict(prompt_embeds=embeds, height=item.size[1], width=item.size[0], guidance_scale=1.0,
                                      num_inference_steps=4, generator=torch.Generator("cuda").manual_seed(seed))
                        if case.mode == "inpaint":
                            kwargs.update(image=item.source, mask_image=item.mask, strength=1.0)
                            if item.references:
                                kwargs["image_reference"] = item.references[0] if len(item.references) == 1 else item.references
                            generated = pipe(**kwargs).images[0]
                        else:
                            if item.images():
                                kwargs["image"] = item.images()
                            generated = edit_pipe(**kwargs).images[0]
                    torch.cuda.synchronize()
                    finish_image(item, generated, stem, result)
                    result.seconds = round(time.perf_counter()-started, 3)
                memory.record(result)
                result.native_calls = kernel_calls(model)-before
                failures = 0
                print(f"DONE {tag} {case.name} {seed}: {result.seconds}s / {result.peak_allocated_gib} GiB / RSS {result.peak_rss_gib} GiB", flush=True)
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("download", "prepare", "probe", "encode", "flux3", "klein9b", "cycle"), required=True)
    parser.add_argument("--only", nargs="*")
    parser.add_argument("--seeds", nargs="*", type=int, default=list(SEEDS))
    parser.add_argument("--tag")
    parser.add_argument("--prompt-style", choices=("current", "caption"), default="caption")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.stage == "download":
        download()
    elif args.stage == "prepare":
        prepare()
    elif args.stage == "probe":
        probe()
    elif args.stage == "encode":
        encode(args.only)
    elif args.stage == "cycle":
        cycle()
    else:
        run(args.stage, args.only, args.seeds, args.tag or args.stage, args.prompt_style, args.resume)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        if "download" not in sys.argv:
            save_json(f"failure-{time.time_ns()}.json", {"arguments": sys.argv[1:], "error": f"{type(error).__name__}: {error}",
                                                       "traceback": traceback.format_exc()})
        raise
