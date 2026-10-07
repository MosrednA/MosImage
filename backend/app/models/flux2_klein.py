from __future__ import annotations

import gc
import random
from pathlib import Path
from typing import Any, Literal

from PIL import Image

from app.config import settings
from app.jobs import JobCancelled, JobControl
from app.runtime import require_generation_device, selected_device, selected_dtype, torch_module
from app.schemas import InpaintForm, LoadModelRequest, ModelInfo, RuntimeMode, TextToImageRequest


Placement = Literal["cuda", "offload"]
PipelineTask = Literal["text-to-image", "inpaint"]


def _inpaint_prompt(request: InpaintForm) -> str:
    if request.edit_operation == "remove":
        instruction = "Remove the selected subject completely and reconstruct the background naturally inside the masked region. " + request.prompt.strip()
    else:
        instruction = request.prompt.strip().rstrip(".") + "."
    instruction += " Match the surrounding lighting, color palette, perspective, texture and depth of field."
    if request.scene_context.strip():
        instruction += " Scene context: " + request.scene_context.strip()
    return instruction


class Flux2KleinProvider:
    """Generation and mask-conditioned editing with one resident pipeline."""

    name = "flux2-klein-4b"
    display_name = "FLUX.2 Klein 4B"
    model_id = "black-forest-labs/FLUX.2-klein-4B"

    def __init__(self) -> None:
        self._pipe: Any | None = None
        self._placement: Placement | None = None
        self._task: PipelineTask | None = None
        self._torch: Any | None = None

    def info(self) -> ModelInfo:
        loaded_pipelines = [f"{self._task}:{self._placement}"] if self._pipe is not None else []
        return ModelInfo(
            name=self.name,
            display_name=self.display_name,
            model_id=self.model_id,
            capabilities=["text-to-image", "inpaint"],
            loaded=self._pipe is not None,
            loaded_pipelines=loaded_pipelines,
        )

    def generate(self, request: TextToImageRequest, output_path: Path, control: JobControl) -> tuple[int, int]:
        try:
            control.raise_if_cancelled()
            require_generation_device()
            torch = self._load_torch()
            pipe = self._load_pipe(self._placement_for(request.runtime_mode), "text-to-image")
            control.set_progress(0.15)
            seed = request.seed if request.seed is not None else random.randrange(0, 2**32)
            generator = torch.Generator(selected_device(torch)).manual_seed(seed)
            image = pipe(
                prompt=request.prompt,
                height=request.height,
                width=request.width,
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
            pipe = self._load_pipe(self._placement_for(request.runtime_mode), "inpaint")
            seed = request.seed if request.seed is not None else random.randrange(0, 2**32)
            generator = torch.Generator(selected_device(torch)).manual_seed(seed)
            with Image.open(input_path) as source:
                init_image = source.convert("RGB")
            with Image.open(mask_path) as selection:
                mask_image = selection.convert("L")
            control.set_progress(0.15)
            image = pipe(
                image=init_image,
                mask_image=mask_image,
                prompt=_inpaint_prompt(request),
                height=init_image.height,
                width=init_image.width,
                num_inference_steps=request.steps,
                guidance_scale=request.guidance,
                strength=request.strength,
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
        try:
            control.raise_if_cancelled()
            require_generation_device()
            self._load_pipe(self._placement_for(request.runtime_mode), request.task)
            control.set_progress(0.95)
        except JobCancelled:
            self.unload()
            raise

    def unload(self) -> None:
        self._pipe = None
        self._placement = None
        self._task = None
        self._collect()

    def _load_pipe(self, placement: Placement, task: PipelineTask) -> Any:
        if self._pipe is not None and (self._placement != placement or self._task != task):
            self._pipe = None
            self._placement = None
            self._task = None
            self._collect()
        if self._pipe is None:
            from diffusers import Flux2KleinInpaintPipeline, Flux2KleinPipeline

            pipeline_class = Flux2KleinInpaintPipeline if task == "inpaint" else Flux2KleinPipeline
            self._pipe = pipeline_class.from_pretrained(
                self.model_id,
                torch_dtype=selected_dtype(),
                low_cpu_mem_usage=True,
                token=settings.huggingface_token,
                use_safetensors=True,
            )
            if task == "inpaint":
                # Keep the inward alpha ramp during latent sampling; the
                # default binary processor would discard the soft boundary.
                self._pipe.mask_processor.register_to_config(do_binarize=False)
            self._place_pipeline(self._pipe, placement)
            self._placement = placement
            self._task = task
        return self._pipe

    def _load_torch(self) -> Any:
        if self._torch is None:
            self._torch = torch_module()
        return self._torch

    def _place_pipeline(self, pipe: Any, placement: Placement) -> None:
        torch = self._load_torch()
        device = selected_device(torch)
        self._configure_memory(pipe)
        if device == "cuda":
            if placement == "offload" and hasattr(pipe, "enable_model_cpu_offload"):
                pipe.enable_model_cpu_offload()
            else:
                pipe.to("cuda")
        else:
            pipe.to("cpu")

    def _configure_memory(self, pipe: Any) -> None:
        vae = getattr(pipe, "vae", None)
        if vae is not None and hasattr(vae, "enable_slicing"):
            vae.enable_slicing()
        if vae is not None and hasattr(vae, "enable_tiling"):
            vae.enable_tiling()
        transformer = getattr(pipe, "transformer", None)
        if transformer is not None and hasattr(transformer, "set_attention_backend"):
            if settings.attention_backend in {"flash", "_flash_3"}:
                transformer.set_attention_backend(settings.attention_backend)

    def _placement_for(self, runtime_mode: RuntimeMode) -> Placement:
        if runtime_mode == "performance" or not settings.enable_cpu_offload:
            return "cuda"
        # The official 4B recipe recommends model offload. It keeps the 16 GB
        # card below its limit while the transformer still does its work on CUDA.
        return "offload"

    def _collect(self) -> None:
        gc.collect()
        if self._torch is not None and self._torch.cuda.is_available():
            self._torch.cuda.empty_cache()
