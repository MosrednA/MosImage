from __future__ import annotations

import os
from functools import lru_cache
from typing import Any

from .config import settings


@lru_cache(maxsize=1)
def torch_module() -> Any:
    if settings.cuda_alloc_conf:
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", settings.cuda_alloc_conf)

    import torch

    configure_torch(torch)
    return torch


def configure_torch(torch: Any) -> None:
    if settings.enable_tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")
        if settings.cuda_memory_fraction is not None:
            torch.cuda.set_per_process_memory_fraction(settings.cuda_memory_fraction)


def selected_device(torch: Any | None = None) -> str:
    torch = torch or torch_module()
    if settings.preferred_device == "cuda" and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def selected_dtype(torch: Any | None = None) -> Any:
    torch = torch or torch_module()
    if selected_device(torch) == "cuda" and settings.preferred_dtype == "bfloat16":
        return torch.bfloat16
    if selected_device(torch) == "cuda" and settings.preferred_dtype == "float16":
        return torch.float16
    return torch.float32


def require_generation_device() -> None:
    torch = torch_module()
    if selected_device(torch) == "cuda" or settings.allow_cpu_generation:
        return
    raise RuntimeError(
        "CUDA is not available, so Z-Image generation was stopped before model load. "
        "Your RTX 5070 Ti should run this on GPU; install a CUDA-enabled PyTorch build "
        "for your NVIDIA driver, then restart the backend. Set "
        "MOSIMAGE_ALLOW_CPU_GENERATION=true only for debugging, because CPU generation "
        "will be impractically slow."
    )


def system_info() -> dict[str, Any]:
    try:
        torch = torch_module()
    except Exception as exc:
        return {
            "hardware_profile": settings.hardware_profile,
            "torch_available": False,
            "error": str(exc),
        }

    info: dict[str, Any] = {
        "hardware_profile": settings.hardware_profile,
        "torch_available": True,
        "torch_version": torch.__version__,
        "preferred_device": settings.preferred_device,
        "selected_device": selected_device(torch),
        "preferred_dtype": settings.preferred_dtype,
        "enable_tf32": settings.enable_tf32,
        "enable_cpu_offload": settings.enable_cpu_offload,
        "allow_cpu_generation": settings.allow_cpu_generation,
        "runtime_mode": settings.runtime_mode,
        "gpu_resident_max_pixels": settings.gpu_resident_max_pixels,
        "max_loaded_pipelines": settings.max_loaded_pipelines,
        "attention_backend": settings.attention_backend,
        "cuda_alloc_conf": settings.cuda_alloc_conf,
    }
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        info.update(
            {
                "cuda_available": True,
                "cuda_device": torch.cuda.get_device_name(0),
                "cuda_version": torch.version.cuda,
                "vram_gb": round(props.total_memory / (1024**3), 2),
                "cuda_memory_fraction": settings.cuda_memory_fraction,
            }
        )
    else:
        info["cuda_available"] = False
        info["setup_hint"] = (
            "CUDA is unavailable to PyTorch. Install a CUDA-enabled PyTorch wheel "
            "for your NVIDIA driver and restart the backend."
        )
    return info
