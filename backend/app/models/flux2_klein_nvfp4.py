from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from app.config import settings
from app.jobs import JobControl
from app.models.flux2_klein import SHARED_COMPONENTS, Flux2KleinProvider, Placement, PipelineTask
from app.runtime import selected_device, selected_dtype
from app.schemas import RuntimeMode

if TYPE_CHECKING:
    from diffusers import Flux2KleinInpaintPipeline, Flux2KleinPipeline

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
        return pipeline_class.from_pretrained(
            self.base_repo, revision=self.base_revision, transformer=transformer,
            torch_dtype=self._load_torch().bfloat16, low_cpu_mem_usage=True,
            token=settings.huggingface_token, use_safetensors=True,
            **self._pipeline_overrides(control),
        )

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
        root = snapshot_download(self.base_repo, revision=self.base_revision, allow_patterns=["text_encoder/*"],
                                 token=settings.huggingface_token)
        control.raise_if_cancelled()
        components = {name: getattr(pipe, name) for name in SHARED_COMPONENTS}
        components["text_encoder"] = load_text_encoder(Path(root), control.raise_if_cancelled)
        pipeline_class = Flux2KleinInpaintPipeline if task == "inpaint" else Flux2KleinPipeline
        return pipeline_class(**components, is_distilled=pipe.config.get("is_distilled", False))

    def _placement_for(self, runtime_mode: RuntimeMode) -> Placement:
        # The streamed encoder is what keeps 9B inside 16 GB; model offload
        # would only move the transformer back and forth on every edit.
        return "cuda"
