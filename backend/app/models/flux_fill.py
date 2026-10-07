from __future__ import annotations

import gc
import random
from pathlib import Path
from typing import Any

from PIL import Image

from app.jobs import JobCancelled, JobControl
from app.config import settings
from app.runtime import require_generation_device, selected_device, selected_dtype, torch_module
from app.schemas import InpaintForm, LoadModelRequest, ModelInfo, TextToImageRequest


class FluxFillProvider:
    """Optional specialist fill model, loaded only when explicitly selected."""

    name = "flux1-fill-dev"
    display_name = "FLUX.1 Fill Dev (specialist)"
    model_id = "black-forest-labs/FLUX.1-Fill-dev"

    def __init__(self) -> None:
        self._pipe: Any | None = None
        self._torch: Any | None = None

    def info(self) -> ModelInfo:
        return ModelInfo(
            name=self.name,
            display_name=self.display_name,
            model_id=self.model_id,
            capabilities=["inpaint", "outpaint"],
            loaded=self._pipe is not None,
            loaded_pipelines=["inpaint:cpu-offload"] if self._pipe is not None else [],
        )

    def generate(self, request: TextToImageRequest, output_path: Path, control: JobControl) -> tuple[int, int]:
        del request, output_path, control
        raise ValueError("FLUX.1 Fill Dev only supports filling a masked image")

    def inpaint(
        self,
        request: InpaintForm,
        input_path: Path,
        mask_path: Path,
        output_path: Path,
        control: JobControl,
    ) -> tuple[int, int]:
        try:
            control.raise_if_cancelled()
            require_generation_device()
            torch = self._load_torch()
            if self._pipe is None:
                control.set_phase("loading", self.display_name)
            pipe = self._load_pipe()
            seed = request.seed if request.seed is not None else random.randrange(0, 2**32)
            generator = torch.Generator(selected_device(torch)).manual_seed(seed)
            with Image.open(input_path) as source_file:
                source = source_file.convert("RGB")
            with Image.open(mask_path) as mask_file:
                mask = mask_file.convert("L")
            control.set_progress(0.15)
            control.set_phase("generating")
            image = pipe(
                prompt=". ".join(part for part in (request.prompt.strip(), request.scene_context.strip()) if part),
                image=source,
                mask_image=mask,
                height=source.height,
                width=source.width,
                strength=request.strength,
                num_inference_steps=request.steps,
                guidance_scale=request.guidance,
                generator=generator,
                callback_on_step_end=control.diffusion_callback(request.steps),
            ).images[0]
            control.set_progress(0.95)
            image.save(output_path)
            return seed, image.width
        except JobCancelled:
            self.unload()
            raise

    def preload(self, request: LoadModelRequest, control: JobControl) -> None:
        if request.task != "inpaint":
            raise ValueError("FLUX.1 Fill Dev can only be preloaded for inpainting")
        try:
            control.raise_if_cancelled()
            require_generation_device()
            control.set_phase("loading", self.display_name)
            self._load_pipe()
            control.set_progress(0.95)
        except JobCancelled:
            self.unload()
            raise

    def unload(self) -> None:
        self._pipe = None
        self._collect()

    def _load_pipe(self) -> Any:
        if self._pipe is None:
            from diffusers import FluxFillPipeline

            self._pipe = FluxFillPipeline.from_pretrained(
                self.model_id,
                torch_dtype=selected_dtype(),
                low_cpu_mem_usage=True,
                token=settings.huggingface_token,
                use_safetensors=True,
            )
            vae = getattr(self._pipe, "vae", None)
            if vae is not None and hasattr(vae, "enable_slicing"):
                vae.enable_slicing()
            if vae is not None and hasattr(vae, "enable_tiling"):
                vae.enable_tiling()
            # This 12B specialist is intentionally never made GPU-resident on
            # the 16 GB target. Offload preserves headroom for contextual crops.
            self._pipe.enable_model_cpu_offload()
        return self._pipe

    def _load_torch(self) -> Any:
        if self._torch is None:
            self._torch = torch_module()
        return self._torch

    def _collect(self) -> None:
        gc.collect()
        if self._torch is not None and self._torch.cuda.is_available():
            self._torch.cuda.empty_cache()
