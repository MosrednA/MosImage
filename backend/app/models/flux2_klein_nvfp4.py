from __future__ import annotations

from pathlib import Path
import random
from typing import TYPE_CHECKING, Any

from PIL import Image

from app.compositing import FAST_PNG
from app.config import settings
from app.jobs import JobControl
from app.models.flux2_klein import SHARED_COMPONENTS, Flux2KleinProvider, Placement, PipelineTask, _fit_area
from app.runtime import require_generation_device, selected_device, selected_dtype
from app.schemas import OutpaintForm, RuntimeMode

if TYPE_CHECKING:
    from diffusers import Flux2KleinInpaintPipeline, Flux2KleinPipeline
    from app.models.klein_lora import KleinLora

BASE_REPO = "black-forest-labs/FLUX.2-klein-4B"
BASE_REVISION = "e7b7dc27f91deacad38e78976d1f2b499d76a294"
QUANT_REPO = "black-forest-labs/FLUX.2-klein-4b-nvfp4"
QUANT_REVISION = "1db2b2f776c24b76f1122e5f69ab1949fc620068"
QUANT_FILENAME = "flux-2-klein-4b-nvfp4.safetensors"
BASE_REPO_9B = "black-forest-labs/FLUX.2-klein-9B"
BASE_REVISION_9B = "92196c8e11f7b6cf2b7493e037d8c5345c559216"
QUANT_REPO_9B = "black-forest-labs/FLUX.2-klein-9b-nvfp4"
QUANT_REVISION_9B = "e882f64f6aa086fcf8915a7763550e05af10ef13"
QUANT_FILENAME_9B = "flux-2-klein-9b-nvfp4.safetensors"


class Flux2KleinNvfp4Provider(Flux2KleinProvider):
    """Official distilled NVFP4 transformer, with BF16 encoder and VAE."""

    name = "flux2-klein-4b-nvfp4"
    display_name = "FLUX.2 Klein 4B NVFP4"
    model_id = QUANT_REPO
    base_repo = BASE_REPO
    base_revision = BASE_REVISION
    quant_repo = QUANT_REPO
    quant_revision = QUANT_REVISION
    quant_filename = QUANT_FILENAME

    def __init__(self) -> None:
        super().__init__()
        self._outpaint_adapter: KleinLora | None = None
        self._want_outpaint_adapter = False

    def unload(self) -> None:
        self._outpaint_adapter = None
        self._want_outpaint_adapter = False
        super().unload()

    def _configure_edit_adapters(self, pipe: Any, enabled: bool, control: JobControl, detail: float = 0.0) -> None:
        super()._configure_edit_adapters(pipe, enabled, control, detail)
        if self._outpaint_adapter is not None:
            self._outpaint_adapter.set_scale(0)

    def outpaint(self, request: OutpaintForm, input_path: Path, output_path: Path,
                 control: JobControl) -> tuple[int, int]:
        if self.name != "flux2-klein-4b-nvfp4" or not request.specialized_outpaint:
            raise ValueError("Specialized outpaint requires Fast NVFP4.")
        control.raise_if_cancelled()
        require_generation_device()
        # Attach before Accelerate installs its offload hooks. This costs
        # one reload the first time it is enabled on an already loaded 4B.
        placement = self._placement_for(request.runtime_mode)
        if self._outpaint_adapter is None or (self._pipes and self._placement != placement):
            self.unload()
            self._want_outpaint_adapter = True
        pipe = self._load_pipe(placement, "text-to-image", control.subrange(0, 0.15))
        assert self._outpaint_adapter is not None
        self._outpaint_adapter.set_scale(1.1)
        torch = self._load_torch()
        seed = request.seed if request.seed is not None else random.randrange(0, 2**32)
        with Image.open(input_path) as source:
            size = source.size
            canvas = _fit_area(source.convert("RGB"))
        embeds = self._encode_prompt(pipe, "Fill the green spaces according to the image", control)
        control.set_phase("generating")
        image = pipe(image=canvas, prompt_embeds=embeds, height=canvas.height, width=canvas.width,
            num_inference_steps=4, guidance_scale=1,
            generator=torch.Generator("cuda").manual_seed(seed),
            callback_on_step_end=control.diffusion_callback(4)).images[0]
        control.raise_if_cancelled()
        image.resize(size, Image.Resampling.LANCZOS).save(output_path, **FAST_PNG)
        return seed, size[0]

    def _validate_runtime(self) -> None:
        torch = self._load_torch()
        if selected_device(torch) != "cuda" or torch.cuda.get_device_capability()[0] < 10:
            raise RuntimeError("NVFP4 requires a Blackwell CUDA GPU (RTX 50 series or newer). Choose FLUX.2 Klein 4B (BF16) on other GPUs.")
        cuda_version = torch.version.cuda
        if cuda_version is None or int(cuda_version.split(".")[0]) < 13:
            raise RuntimeError("NVFP4 requires CUDA 13 PyTorch. Install requirements-torch-cu130.txt and restart the backend.")
        if selected_dtype(torch) != torch.bfloat16:
            raise RuntimeError("NVFP4 requires MOSIMAGE_PREFERRED_DTYPE=bfloat16 for the encoder and VAE.")

    def _create_pipeline(self, task: PipelineTask, control: JobControl) -> Flux2KleinInpaintPipeline | Flux2KleinPipeline:
        self._validate_runtime()
        control.raise_if_cancelled()
        # Keep CUDA/Diffusers imports out of registry construction/API startup.
        from diffusers import Flux2KleinInpaintPipeline, Flux2KleinPipeline
        from huggingface_hub import hf_hub_download
        from app.models.nvfp4_runtime import kernel_api, load_transformer

        kernel = kernel_api()
        config = hf_hub_download(self.base_repo, "transformer/config.json", revision=self.base_revision,
                                 token=settings.huggingface_token)
        control.raise_if_cancelled()
        checkpoint = hf_hub_download(self.quant_repo, self.quant_filename, revision=self.quant_revision,
                                     token=settings.huggingface_token)
        control.raise_if_cancelled()
        transformer = load_transformer(Path(config), Path(checkpoint), kernel, control.raise_if_cancelled)
        pipeline_class = Flux2KleinInpaintPipeline if task == "inpaint" else Flux2KleinPipeline
        # No dtype cast after loading packed weights and FP8/FP32 scale buffers.
        pipe = pipeline_class.from_pretrained(
            self.base_repo, revision=self.base_revision, transformer=transformer,
            torch_dtype=self._load_torch().bfloat16, low_cpu_mem_usage=True,
            token=settings.huggingface_token, use_safetensors=True,
            **self._pipeline_overrides(control),
        )
        if self._want_outpaint_adapter:
            from app.models.klein_lora import FAL_OUTPAINT, KleinLora, resolve_checkpoint
            self._outpaint_adapter = KleinLora.attach(pipe.transformer,
                resolve_checkpoint(FAL_OUTPAINT, control), control.raise_if_cancelled)
        return pipe

    def _pipeline_overrides(self, control: JobControl) -> dict[str, Any]:
        return {}


class Flux2Klein9bNvfp4Provider(Flux2KleinNvfp4Provider):
    """The 9B transformer, with its 16 GB Qwen3-8B encoder streamed from RAM.

    The encoder is trimmed and stored as FP8 in pinned memory (about 6 GB) and
    visits the GPU for each new prompt only, so transformer and VAE stay
    resident on a 16 GB card. The weights are under a non-commercial licence.
    """

    name = "flux2-klein-9b-nvfp4"
    display_name = "FLUX.2 Klein 9B NVFP4"
    model_id = QUANT_REPO_9B
    supports_edit_adapters = True
    base_repo = BASE_REPO_9B
    base_revision = BASE_REVISION_9B
    quant_repo = QUANT_REPO_9B
    quant_revision = QUANT_REVISION_9B
    quant_filename = QUANT_FILENAME_9B

    def unload(self) -> None:
        super().unload()
        torch = self._torch
        # Pinned host blocks stay cached by PyTorch until released explicitly.
        if torch is not None and hasattr(torch._C, "_host_emptyCache"):
            torch._C._host_emptyCache()

    def _pipeline_overrides(self, control: JobControl) -> dict[str, Any]:
        # Skip the BF16 encoder; _create_pipeline attaches the streamed one.
        return {"text_encoder": None}

    def _create_pipeline(self, task: PipelineTask, control: JobControl) -> Flux2KleinInpaintPipeline | Flux2KleinPipeline:
        from diffusers import Flux2KleinInpaintPipeline, Flux2KleinPipeline
        from huggingface_hub import snapshot_download
        from app.models.klein_text_encoder import load_text_encoder

        pipe = super()._create_pipeline(task, control)
        control.set_progress(0.05)
        root = snapshot_download(self.base_repo, revision=self.base_revision, allow_patterns=["text_encoder/*"],
                                 token=settings.huggingface_token)
        control.raise_if_cancelled()
        components = {name: getattr(pipe, name) for name in SHARED_COMPONENTS}
        # The encoder is nearly all of this (about 16 of 17 s), so its progress is the load's.
        components["text_encoder"] = load_text_encoder(Path(root), control.raise_if_cancelled,
                                                       lambda share: control.set_progress(0.05 + 0.95 * share))
        pipeline_class = Flux2KleinInpaintPipeline if task == "inpaint" else Flux2KleinPipeline
        return pipeline_class(**components, is_distilled=pipe.config.get("is_distilled", False))

    def _placement_for(self, runtime_mode: RuntimeMode) -> Placement:
        # The streamed encoder is what keeps 9B inside 16 GB; model offload
        # would only move the transformer back and forth on every edit.
        return "cuda"
