"""Cached-only benchmark entry point for the application's NVFP4 runtime."""
from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from diffusers import Flux2Transformer2DModel
from huggingface_hub import hf_hub_download
from app.models.flux2_klein_nvfp4 import BASE_REPO, BASE_REVISION, QUANT_REPO, QUANT_REVISION, QUANT_FILENAME
from app.models.nvfp4_runtime import Nvfp4Linear, kernel_api, kernel_calls, transformer_storage_bytes
from app.models.nvfp4_runtime import load_transformer as load_native_transformer


def load_transformer() -> Flux2Transformer2DModel:
    kernel = kernel_api()
    config_path = hf_hub_download(BASE_REPO, "transformer/config.json", revision=BASE_REVISION,
                                  local_files_only=True, token=False)
    checkpoint = hf_hub_download(QUANT_REPO, QUANT_FILENAME, revision=QUANT_REVISION,
                                  local_files_only=True, token=False)
    return load_native_transformer(Path(config_path), Path(checkpoint), kernel, lambda: None)
