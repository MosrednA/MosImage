from __future__ import annotations

import gc
import random
from pathlib import Path
from typing import Any

from PIL import Image

from app.config import settings
from app.jobs import JobCancelled, JobControl
from app.runtime import require_generation_device, selected_device, selected_dtype, torch_module
from app.schemas import InpaintForm, LoadModelRequest, ModelInfo, RuntimeMode, TextToImageRequest


Placement = str


class ZImageTurboProvider:
    name = "z-image-turbo"
    display_name = "Z-Image Turbo"
    model_id = "Tongyi-MAI/Z-Image-Turbo"

    def __init__(self) -> None:
        self._text_pipe: Any | None = None
        self._inpaint_pipe: Any | None = None
        self._text_placement: Placement | None = None
        self._inpaint_placement: Placement | None = None
        self._torch: Any | None = None

    def info(self) -> ModelInfo:
        loaded_pipelines = []
        if self._text_pipe is not None:
            loaded_pipelines.append(f"text-to-image:{self._text_placement}")
        if self._inpaint_pipe is not None:
            loaded_pipelines.append(f"inpaint:{self._inpaint_placement}")
        return ModelInfo(
            name=self.name,
            display_name=self.display_name,
            model_id=self.model_id,
            capabilities=["text-to-image", "inpaint", "outpaint"],
            loaded=self._text_pipe is not None or self._inpaint_pipe is not None,
            loaded_pipelines=loaded_pipelines,
        )

    def generate(self, request: TextToImageRequest, output_path: Path, control: JobControl) -> tuple[int, int]:
        try:
            control.raise_if_cancelled()
            require_generation_device()
            torch = torch_module()
            placement = self._placement_for(request.runtime_mode, request.width, request.height)
            if self._text_pipe is None or self._text_placement != placement:
                control.set_phase("loading", self.display_name)
            pipe = self._load_text_pipe(placement)
            control.set_progress(0.15)
            control.set_phase("generating")
            seed = request.seed if request.seed is not None else random.randrange(0, 2**32)
            generator = torch.Generator(selected_device(torch)).manual_seed(seed)

            image = pipe(
                prompt=request.prompt,
                negative_prompt=request.negative_prompt,
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
            torch = torch_module()
            seed = request.seed if request.seed is not None else random.randrange(0, 2**32)
            generator = torch.Generator(selected_device(torch)).manual_seed(seed)

            with Image.open(input_path) as source_file, Image.open(mask_path) as mask_file:
                init_image = source_file.convert("RGB")
                mask_image = mask_file.convert("L").resize(init_image.size)
            placement = self._inpaint_placement_for(request.runtime_mode)
            if self._inpaint_pipe is None or self._inpaint_placement != placement:
                control.set_phase("loading", self.display_name)
            pipe = self._load_inpaint_pipe(placement)
            control.set_progress(0.15)
            control.set_phase("generating")
            image = pipe(
                prompt=". ".join(part for part in (request.prompt.strip(), request.scene_context.strip()) if part),
                negative_prompt=request.negative_prompt,
                image=init_image,
                mask_image=mask_image,
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

    def unload(self) -> None:
        self._text_pipe = None
        self._inpaint_pipe = None
        self._text_placement = None
        self._inpaint_placement = None
        self._collect()

    def preload(self, request: LoadModelRequest, control: JobControl) -> None:
        try:
            control.raise_if_cancelled()
            require_generation_device()
            placement = (
                self._inpaint_placement_for(request.runtime_mode)
                if request.task == "inpaint"
                else self._placement_for(request.runtime_mode, request.width, request.height)
            )
            control.set_phase("loading", self.display_name)
            if request.task == "inpaint":
                self._load_inpaint_pipe(placement)
            else:
                self._load_text_pipe(placement)
            control.set_progress(0.95)
        except JobCancelled:
            self.unload()
            raise

    def _load_text_pipe(self, placement: Placement) -> Any:
        if self._text_pipe is not None and self._text_placement != placement:
            self._text_pipe = None
            self._text_placement = None
            self._collect()
        if self._text_pipe is None:
            from diffusers import ZImagePipeline

            if self._inpaint_pipe is not None:
                self._text_pipe = ZImagePipeline.from_pipe(self._inpaint_pipe, dtype=selected_dtype())
                if settings.max_loaded_pipelines <= 1:
                    self._inpaint_pipe = None
                    self._inpaint_placement = None
                    self._collect()
            else:
                self._text_pipe = ZImagePipeline.from_pretrained(
                    self.model_id,
                    torch_dtype=selected_dtype(),
                    low_cpu_mem_usage=True,
                    token=settings.huggingface_token,
                    use_safetensors=True,
                )
            self._place_pipeline(self._text_pipe, placement)
            self._text_placement = placement
        return self._text_pipe

    def _load_inpaint_pipe(self, placement: Placement) -> Any:
        if self._inpaint_pipe is not None and self._inpaint_placement != placement:
            self._inpaint_pipe = None
            self._inpaint_placement = None
            self._collect()
        if self._inpaint_pipe is None:
            from diffusers import ZImageInpaintPipeline

            if self._text_pipe is not None:
                self._inpaint_pipe = ZImageInpaintPipeline.from_pipe(self._text_pipe, dtype=selected_dtype())
                if settings.max_loaded_pipelines <= 1:
                    self._text_pipe = None
                    self._text_placement = None
                    self._collect()
            else:
                self._inpaint_pipe = ZImageInpaintPipeline.from_pretrained(
                    self.model_id,
                    torch_dtype=selected_dtype(),
                    low_cpu_mem_usage=True,
                    token=settings.huggingface_token,
                    use_safetensors=True,
                )
            self._place_pipeline(self._inpaint_pipe, placement)
            self._inpaint_placement = placement
        return self._inpaint_pipe

    def _load_torch(self) -> Any:
        if self._torch is None:
            self._torch = torch_module()
        return self._torch

    def _place_pipeline(self, pipe: Any, placement: Placement) -> None:
        torch = self._load_torch()
        device = selected_device(torch)
        self._configure_attention(pipe)
        if device == "cuda":
            if placement == "offload" and hasattr(pipe, "enable_model_cpu_offload"):
                pipe.enable_model_cpu_offload()
            else:
                pipe.to("cuda")
        else:
            pipe.to("cpu")

    def _configure_attention(self, pipe: Any) -> None:
        vae = getattr(pipe, "vae", None)
        if vae is not None and hasattr(vae, "enable_slicing"):
            vae.enable_slicing()
        if vae is not None and hasattr(vae, "enable_tiling"):
            vae.enable_tiling()
        transformer = getattr(pipe, "transformer", None)
        if transformer is None or not hasattr(transformer, "set_attention_backend"):
            return
        if settings.attention_backend in {"flash", "_flash_3"}:
            transformer.set_attention_backend(settings.attention_backend)

    def _placement_for(self, runtime_mode: RuntimeMode, width: int, height: int) -> Placement:
        if runtime_mode == "safe":
            return "offload"
        if runtime_mode == "performance":
            return "cuda"
        pixels = width * height
        if settings.enable_cpu_offload and pixels > settings.gpu_resident_max_pixels:
            return "offload"
        return "cuda"

    def _inpaint_placement_for(self, runtime_mode: RuntimeMode) -> Placement:
        if runtime_mode == "performance" or not settings.enable_cpu_offload:
            return "cuda"
        # The inpaint pipeline needs additional source, mask, and VAE latents.
        # Module-level offload keeps the transformer resident while denoising,
        # but moves the text encoder and VAE out of the 16 GB budget between stages.
        return "offload"

    def _collect(self) -> None:
        gc.collect()
        if self._torch is not None and self._torch.cuda.is_available():
            self._torch.cuda.empty_cache()
