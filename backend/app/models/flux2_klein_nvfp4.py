from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from app.config import settings
from app.jobs import JobControl
from app.models.flux2_klein import Flux2KleinProvider, PipelineTask
from app.runtime import selected_device, selected_dtype

if TYPE_CHECKING:
    from diffusers import Flux2KleinInpaintPipeline, Flux2KleinPipeline

BASE_REPO = "black-forest-labs/FLUX.2-klein-4B"
BASE_REVISION = "e7b7dc27f91deacad38e78976d1f2b499d76a294"
QUANT_REPO = "black-forest-labs/FLUX.2-klein-4b-nvfp4"
QUANT_REVISION = "1db2b2f776c24b76f1122e5f69ab1949fc620068"
QUANT_FILENAME = "flux-2-klein-4b-nvfp4.safetensors"
KLEIN_MODELS = frozenset({"flux2-klein-4b", "flux2-klein-4b-nvfp4"})


class Flux2KleinNvfp4Provider(Flux2KleinProvider):
    """Official distilled NVFP4 transformer, with BF16 encoder and VAE."""

    name = "flux2-klein-4b-nvfp4"
    display_name = "FLUX.2 Klein 4B NVFP4"
    model_id = QUANT_REPO

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
        config = hf_hub_download(BASE_REPO, "transformer/config.json", revision=BASE_REVISION,
                                 token=settings.huggingface_token)
        control.raise_if_cancelled()
        checkpoint = hf_hub_download(QUANT_REPO, QUANT_FILENAME, revision=QUANT_REVISION,
                                     token=settings.huggingface_token)
        control.raise_if_cancelled()
        transformer = load_transformer(Path(config), Path(checkpoint), kernel, control.raise_if_cancelled)
        pipeline_class = Flux2KleinInpaintPipeline if task == "inpaint" else Flux2KleinPipeline
        # No dtype cast after loading packed weights and FP8/FP32 scale buffers.
        return pipeline_class.from_pretrained(
            BASE_REPO, revision=BASE_REVISION, transformer=transformer,
            torch_dtype=self._load_torch().bfloat16, low_cpu_mem_usage=True,
            token=settings.huggingface_token, use_safetensors=True,
        )
