"""Run the frozen four-image suite on BF16 or official native NVFP4 Klein.

Run precisions in separate processes, with no loaded API model on the GPU.
Results measure whole warmed pipeline calls, including text/vae and offload.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
from importlib.metadata import version
import json
from pathlib import Path
import sys
import threading
import time
from typing import Literal
from urllib.request import urlopen

import numpy as np
from PIL import Image
import psutil
from pydantic import BaseModel, ConfigDict

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.compositing import prepare_inpaint_workspace
from app.models.flux2_klein import _inpaint_prompt
from app.runtime import torch_module
from app.schemas import InpaintForm
from create_benchmark_fixtures import FIXTURES, Suite, digest
from nvfp4_transformer import (BASE_REPO, BASE_REVISION, QUANT_REPO, QUANT_REVISION,
                             kernel_calls, load_transformer, transformer_storage_bytes)


class Outcome(BaseModel):
    model_config = ConfigDict(extra="forbid")
    fixture: str
    seed: int
    prompt: str
    output: str | None = None
    elapsed_seconds: float | None = None
    cuda_peak_allocated_gib: float | None = None
    cuda_peak_reserved_gib: float | None = None
    process_peak_rss_gib: float | None = None
    nvfp4_kernel_calls: int = 0
    protected_pixels: int | None = None
    changed_protected_pixels: int | None = None
    changed_selected_pixels: int | None = None
    error: str | None = None


class Measurement(BaseModel):
    precision: Literal["bf16", "nvfp4"]
    placement: Literal["model-cpu-offload"] = "model-cpu-offload"
    base_repo: str = BASE_REPO
    base_revision: str = BASE_REVISION
    transformer_repo: str
    transformer_revision: str
    date_utc: str
    packages: dict[str, str]
    gpu: str
    compute_capability: tuple[int, int]
    manifest_sha256: str
    seeds: list[int]
    transformer_storage_gib: float | None = None
    load_seconds: float | None = None
    warmup_seconds: float | None = None
    component_end_allocated_gib: dict[str, float] = {}
    results: list[Outcome] = []


def assert_api_idle() -> None:
    # The user-facing API must not contend with the offline GPU benchmark.
    with urlopen("http://127.0.0.1:8000/api/models", timeout=5) as response:
        models = json.load(response)
    if not isinstance(models, list) or any(model.get("loaded") for model in models):
        raise RuntimeError("Unload all API pipelines before running this offline benchmark.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--precision", choices=("bf16", "nvfp4"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed-count", type=int, choices=(1, 2, 3), default=3)
    parser.add_argument("--fixture", choices=("garden", "kitchen", "portrait", "architecture"))
    parser.add_argument("--profile-components", action="store_true", help="Record allocated memory at component returns; run separately from the latency cohort.")
    args = parser.parse_args()
    assert_api_idle()
    suite = Suite.model_validate_json((FIXTURES / "manifest.json").read_text(encoding="utf-8"))
    for fixture in suite.fixtures:
        if digest(FIXTURES / fixture.image) != fixture.image_sha256 or digest(FIXTURES / fixture.mask) != fixture.mask_sha256:
            raise ValueError(f"Frozen asset hash mismatch: {fixture.name}")
    args.output.mkdir(parents=True, exist_ok=False)
    torch = torch_module()
    from diffusers import Flux2KleinInpaintPipeline

    quantized = args.precision == "nvfp4"
    measure = Measurement(precision=args.precision,
        transformer_repo=QUANT_REPO if quantized else BASE_REPO,
        transformer_revision=QUANT_REVISION if quantized else BASE_REVISION,
        date_utc=datetime.now(timezone.utc).isoformat(),
        packages={package: version(package) for package in ("torch", "diffusers", "transformers", "accelerate", "safetensors", "pillow")},
        gpu=torch.cuda.get_device_name(), compute_capability=torch.cuda.get_device_capability(),
        manifest_sha256=digest(FIXTURES / "manifest.json"), seeds=list(suite.seeds[:args.seed_count]))
    manifest = args.output / "results.json"

    def save() -> None:
        manifest.write_text(measure.model_dump_json(indent=2), encoding="utf-8")

    save()  # Freeze the manifest and seed list before viewing generated outputs.
    print(f"Loading {args.precision}; no other API pipeline is resident.", flush=True)
    started = time.perf_counter()
    if quantized:
        transformer = load_transformer()
        measure.packages["comfy-kitchen"] = version("comfy-kitchen")
        pipe = Flux2KleinInpaintPipeline.from_pretrained(BASE_REPO, revision=BASE_REVISION,
            transformer=transformer, torch_dtype=torch.bfloat16, local_files_only=True,
            token=False, use_safetensors=True, low_cpu_mem_usage=True)
        del transformer
    else:
        pipe = Flux2KleinInpaintPipeline.from_pretrained(BASE_REPO, revision=BASE_REVISION,
            torch_dtype=torch.bfloat16, local_files_only=True, token=False,
            use_safetensors=True, low_cpu_mem_usage=True)
    pipe.mask_processor.register_to_config(do_binarize=False)
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()
    measure.transformer_storage_gib = transformer_storage_bytes(pipe.transformer) / 1024**3
    # No dtype conversion after the packed FP8/FP4 checkpoint was loaded.
    pipe.enable_model_cpu_offload()
    torch.cuda.synchronize()
    measure.load_seconds = time.perf_counter() - started
    save()
    print(f"Loaded in {measure.load_seconds:.2f}s; transformer storage {measure.transformer_storage_gib:.3f} GiB.", flush=True)

    def render(fixture_name: str, seed: int, destination: Path) -> str:
        fixture = next(item for item in suite.fixtures if item.name == fixture_name)
        work = destination.parent / (destination.stem + "-work")
        work.mkdir(parents=True, exist_ok=True)
        workspace = prepare_inpaint_workspace(FIXTURES / fixture.image, FIXTURES / fixture.mask,
            work, suite.max_dimension, 0, suite.context_padding,
            composite_feather=suite.blend_feather, generation_feather=suite.blend_feather)
        request = InpaintForm(prompt=fixture.prompt, edit_operation=fixture.operation,
            seed=seed, steps=suite.steps, guidance=suite.guidance, strength=suite.strength,
            context_padding=suite.context_padding, blend_feather=suite.blend_feather)
        prompt = _inpaint_prompt(request)
        with Image.open(workspace.model_input_path) as source, Image.open(workspace.model_mask_path) as mask:
            generated = pipe(image=source.convert("RGB"), mask_image=mask.convert("L"),
                prompt=prompt, height=source.height, width=source.width,
                num_inference_steps=suite.steps, guidance_scale=suite.guidance, strength=suite.strength,
                generator=torch.Generator("cuda").manual_seed(seed)).images[0]
        generated.save(workspace.model_output_path)
        workspace.composite_to(destination)
        return prompt

    # This seed is outside the declared evaluation cohort.
    print("Warming the pipeline on garden, seed 99999 (excluded from results).", flush=True)
    started = time.perf_counter()
    with torch.inference_mode():
        render("garden", 99999, args.output / "warmup.png")
    torch.cuda.synchronize()
    measure.warmup_seconds = time.perf_counter() - started
    save()
    print(f"Warmup {measure.warmup_seconds:.2f}s.", flush=True)
    profile_handles = []
    if args.profile_components:
        # The outer accelerate hook has already placed the component before
        # its child forward runs. These returns therefore observe residency
        # without conflating it with the previous component being offloaded.
        def record_memory(name: str):
            def on_return(_module, _inputs, _output):
                allocated = torch.cuda.memory_allocated() / 1024**3
                measure.component_end_allocated_gib[name] = max(measure.component_end_allocated_gib.get(name, 0), allocated)
            return on_return
        for name, module in (("text_encoder", pipe.text_encoder), ("transformer", pipe.transformer),
                             ("vae_encoder", pipe.vae.encoder), ("vae_decoder", pipe.vae.decoder)):
            profile_handles.append(module.register_forward_hook(record_memory(name)))
    fixtures = [fixture for fixture in suite.fixtures if args.fixture is None or fixture.name == args.fixture]
    process = psutil.Process()
    for fixture in fixtures:
        for seed in measure.seeds:
            assert_api_idle()
            output = args.output / f"{fixture.name}-{seed}.png"
            outcome = Outcome(fixture=fixture.name, seed=seed, prompt=fixture.prompt)
            measure.results.append(outcome)
            save()
            stop = threading.Event()
            rss_peak = [process.memory_info().rss]

            def sample_memory() -> None:
                while not stop.wait(0.1):
                    rss_peak[0] = max(rss_peak[0], process.memory_info().rss)

            sampler = threading.Thread(target=sample_memory, daemon=True)
            sampler.start()
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            calls_before = kernel_calls(pipe.transformer)
            started = time.perf_counter()
            try:
                with torch.inference_mode():
                    outcome.prompt = render(fixture.name, seed, output)
                torch.cuda.synchronize()
                outcome.elapsed_seconds = time.perf_counter() - started
                outcome.output = output.name
                with Image.open(FIXTURES / fixture.image) as source_file, Image.open(output) as edited_file, Image.open(FIXTURES / fixture.mask) as mask_file:
                    source = np.asarray(source_file.convert("RGB"))
                    edited = np.asarray(edited_file.convert("RGB"))
                    mask = np.asarray(mask_file.getchannel("A"))
                if source.shape != edited.shape:
                    raise ValueError("Output dimensions changed.")
                changed = np.any(source != edited, axis=2)
                outcome.protected_pixels = int(np.count_nonzero(mask == 0))
                outcome.changed_protected_pixels = int(np.count_nonzero(changed & (mask == 0)))
                outcome.changed_selected_pixels = int(np.count_nonzero(changed & (mask != 0)))
                if outcome.changed_protected_pixels:
                    raise ValueError("The compositor changed protected pixels.")
                outcome.nvfp4_kernel_calls = kernel_calls(pipe.transformer) - calls_before
                if quantized and outcome.nvfp4_kernel_calls == 0:
                    raise ValueError("The claimed NVFP4 run did not execute any NVFP4 kernels.")
            except Exception as exc:
                outcome.error = f"{type(exc).__name__}: {exc}"
                raise
            finally:
                stop.set()
                sampler.join(timeout=2)
                outcome.process_peak_rss_gib = rss_peak[0] / 1024**3
                outcome.cuda_peak_allocated_gib = torch.cuda.max_memory_allocated() / 1024**3
                outcome.cuda_peak_reserved_gib = torch.cuda.max_memory_reserved() / 1024**3
                save()
            print(f"{len(measure.results)}/{len(fixtures) * len(measure.seeds)} {fixture.name} seed {seed}: "
                  f"{outcome.elapsed_seconds:.2f}s, allocated {outcome.cuda_peak_allocated_gib:.2f} GiB, "
                  f"NVFP4 calls {outcome.nvfp4_kernel_calls}, protected changes {outcome.changed_protected_pixels}", flush=True)
    for handle in profile_handles:
        handle.remove()
    del pipe
    gc.collect()
    torch.cuda.empty_cache()
    print(f"All outcomes saved to {manifest}.", flush=True)


if __name__ == "__main__":
    main()
