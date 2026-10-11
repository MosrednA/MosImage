"""FLUX.2 Klein on Apple silicon: the original BF16 weights on Metal (MPS).

NVFP4 needs Blackwell tensor cores, so a Mac runs the BF16 checkpoints the
NVFP4 builds were quantized from. CPU and GPU share one memory there, so every
component stays resident: no model offload and no streamed text encoder.

With ``apple_weights`` set to q8 or q4, the transformer and text encoder run in
MLX instead, quantized at load (``klein_mlx``); the pipeline around them, with
its masks, references and VAE, stays the same.
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import TYPE_CHECKING, Any

from PIL import Image

from app.compositing import FAST_PNG
from app.config import APPLE_VAE_TILE, settings
from app.jobs import JobControl
from app.models.flux2_klein import SHARED_COMPONENTS, Flux2KleinProvider, PipelineTask, _fit_area
from app.models.flux2_klein_nvfp4 import BASE_REPO, BASE_REPO_9B, BASE_REVISION, BASE_REVISION_9B
from app.runtime import require_generation_device, selected_device, selected_dtype
from app.schemas import OutpaintForm

if TYPE_CHECKING:
    from app.models.klein_lora import KleinLora

OUTPAINT_PROMPT = "Fill the green spaces according to the image"
# Shown after the model name, so the editor says which weights run.
WEIGHT_LABELS = {"q8": " (8-bit)", "q4": " (4-bit)"}


class Flux2KleinAppleProvider(Flux2KleinProvider):
    """Klein 4B, the Fast model on a Mac, with the specialized outpaint LoRA."""

    name = "flux2-klein-4b"
    base_display_name = "FLUX.2 Klein 4B"
    model_id = BASE_REPO
    revision = BASE_REVISION

    def __init__(self) -> None:
        super().__init__()
        self._outpaint_adapter: KleinLora | None = None

    @property
    def display_name(self) -> str:  # type: ignore[override]
        return self.base_display_name + WEIGHT_LABELS.get(settings.apple_weights, "")

    def unload(self) -> None:
        self._outpaint_adapter = None
        super().unload()

    def _configure_edit_adapters(self, pipe: Any, enabled: bool, control: JobControl, detail: float = 0.0) -> None:
        super()._configure_edit_adapters(pipe, enabled, control, detail)
        if self._outpaint_adapter is not None:
            self._outpaint_adapter.set_scale(0)

    def outpaint(self, request: OutpaintForm, input_path: Path, output_path: Path,
                 control: JobControl) -> tuple[int, int]:
        if self.name != settings.fast_model or not request.specialized_outpaint:
            raise ValueError("Specialized outpaint requires the Fast model.")
        control.raise_if_cancelled()
        require_generation_device()
        pipe = self._load_pipe(self._placement_for(request.runtime_mode), "text-to-image", control)
        self._configure_edit_adapters(pipe, False, control)
        if self._outpaint_adapter is None:
            # Without offload hooks the LoRA can join already resident weights.
            from app.models.klein_lora import FAL_OUTPAINT, KleinLora, resolve_checkpoint
            self._outpaint_adapter = KleinLora.attach(pipe.transformer,
                resolve_checkpoint(FAL_OUTPAINT, control), control.raise_if_cancelled)
        self._outpaint_adapter.set_scale(1.1)
        torch = self._load_torch()
        seed = request.seed if request.seed is not None else random.randrange(0, 2**32)
        with Image.open(input_path) as source:
            size = source.size
            canvas = _fit_area(source.convert("RGB"))
        embeds = self._encode_prompt(pipe, OUTPAINT_PROMPT, control)
        control.set_phase("generating")
        image = pipe(image=canvas, prompt_embeds=embeds, height=canvas.height, width=canvas.width,
            num_inference_steps=4, guidance_scale=1,
            generator=torch.Generator(selected_device(torch)).manual_seed(seed),
            callback_on_step_end=control.diffusion_callback(4)).images[0]
        control.raise_if_cancelled()
        image.resize(size, Image.Resampling.LANCZOS).save(output_path, **FAST_PNG)
        return seed, size[0]

    def _configure_memory(self, pipe: Any) -> None:
        super()._configure_memory(pipe)
        vae = getattr(pipe, "vae", None)
        if settings.apple_vae_tiling and isinstance(getattr(vae, "tile_sample_min_size", None), int):
            # Keep the VAE's pixel-to-latent scale (8) for the smaller tiles.
            scale = vae.tile_sample_min_size // vae.tile_latent_min_size
            vae.tile_sample_min_size, vae.tile_latent_min_size = APPLE_VAE_TILE, APPLE_VAE_TILE // scale

    def _create_pipeline(self, task: PipelineTask, control: JobControl) -> Any:
        from diffusers import Flux2KleinInpaintPipeline, Flux2KleinPipeline

        if settings.apple_weights != "bf16":
            return self._create_mlx_pipeline(task, control)
        control.raise_if_cancelled()
        pipeline_class = Flux2KleinInpaintPipeline if task == "inpaint" else Flux2KleinPipeline
        return pipeline_class.from_pretrained(
            self.model_id, revision=self.revision, torch_dtype=selected_dtype(), low_cpu_mem_usage=True,
            token=settings.huggingface_token, use_safetensors=True,
        )

    def _create_mlx_pipeline(self, task: PipelineTask, control: JobControl) -> Any:
        """The diffusers pipeline with its transformer and text encoder quantized in MLX."""
        from diffusers import Flux2KleinInpaintPipeline, Flux2KleinPipeline
        from huggingface_hub import snapshot_download
        from app.models.klein_mlx import load_text_encoder, load_transformer

        control.raise_if_cancelled()
        pipeline_class = Flux2KleinInpaintPipeline if task == "inpaint" else Flux2KleinPipeline
        pipe = pipeline_class.from_pretrained(
            self.model_id, revision=self.revision, transformer=None, text_encoder=None, torch_dtype=selected_dtype(),
            low_cpu_mem_usage=True, token=settings.huggingface_token, use_safetensors=True,
        )
        control.raise_if_cancelled()
        root = Path(snapshot_download(self.model_id, revision=self.revision, allow_patterns=["transformer/*", "text_encoder/*"],
                                      token=settings.huggingface_token))
        components = {name: getattr(pipe, name) for name in SHARED_COMPONENTS}
        components["transformer"] = load_transformer(root / "transformer", settings.apple_weights, control.raise_if_cancelled)
        components["text_encoder"] = load_text_encoder(root / "text_encoder", settings.apple_weights, control.raise_if_cancelled)
        return pipeline_class(**components, is_distilled=pipe.config.get("is_distilled", False))


class Flux2Klein9bAppleProvider(Flux2KleinAppleProvider):
    """Klein 9B, the Quality model on a Mac, with the Consistency and detail-enhancer LoRAs.

    Its Qwen3-8B encoder is loaded without the layers Klein never reads, which
    saves about 4 GB and a fifth of every prompt encode. The weights are under
    a non-commercial licence.
    """

    name = "flux2-klein-9b"
    base_display_name = "FLUX.2 Klein 9B"
    model_id = BASE_REPO_9B
    revision = BASE_REVISION_9B
    supports_edit_adapters = True

    def outpaint(self, request: OutpaintForm, input_path: Path, output_path: Path,
                 control: JobControl) -> tuple[int, int]:
        raise ValueError("Specialized outpaint requires the Fast model.")

    def _create_pipeline(self, task: PipelineTask, control: JobControl) -> Any:
        from diffusers import Flux2KleinInpaintPipeline, Flux2KleinPipeline
        from huggingface_hub import snapshot_download
        from app.models.klein_text_encoder import load_resident_text_encoder

        if settings.apple_weights != "bf16":
            return self._create_mlx_pipeline(task, control)
        control.raise_if_cancelled()
        pipeline_class = Flux2KleinInpaintPipeline if task == "inpaint" else Flux2KleinPipeline
        # Skip the full encoder; the trimmed one is attached below.
        pipe = pipeline_class.from_pretrained(
            self.model_id, revision=self.revision, text_encoder=None, torch_dtype=selected_dtype(),
            low_cpu_mem_usage=True, token=settings.huggingface_token, use_safetensors=True,
        )
        control.raise_if_cancelled()
        root = snapshot_download(self.model_id, revision=self.revision, allow_patterns=["text_encoder/*"],
                                 token=settings.huggingface_token)
        components = {name: getattr(pipe, name) for name in SHARED_COMPONENTS}
        components["text_encoder"] = load_resident_text_encoder(Path(root), selected_device(), control.raise_if_cancelled)
        return pipeline_class(**components, is_distilled=pipe.config.get("is_distilled", False))
