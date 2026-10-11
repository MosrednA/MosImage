from __future__ import annotations

import os
import platform
import subprocess
import sys
from functools import lru_cache
from typing import Any

from .config import physical_memory_gb, settings


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


def mps_available(torch: Any) -> bool:
    backend = getattr(torch.backends, "mps", None)
    return bool(backend is not None and backend.is_available())


def selected_device(torch: Any | None = None) -> str:
    torch = torch or torch_module()
    if settings.preferred_device == "cuda" and torch.cuda.is_available():
        return "cuda"
    if settings.preferred_device == "mps" and mps_available(torch):
        return "mps"
    return "cpu"


def selected_dtype(torch: Any | None = None) -> Any:
    torch = torch or torch_module()
    accelerated = selected_device(torch) in ("cuda", "mps")
    if accelerated and settings.preferred_dtype == "bfloat16":
        return torch.bfloat16
    if accelerated and settings.preferred_dtype == "float16":
        return torch.float16
    return torch.float32


def empty_cache(torch: Any) -> None:
    """Returns cached, freed GPU blocks to the system."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if mps_available(torch):
        torch.mps.empty_cache()
    # MLX keeps its own cache of freed buffers (the quantized Klein runtime on a Mac).
    mlx = sys.modules.get("mlx.core")
    if mlx is not None:
        mlx.clear_cache()


def release_cached_memory() -> None:
    """Apple silicon: hands the GPU caches back after a job.

    PyTorch (VAE) and MLX (quantized Klein) each keep freed buffers for reuse,
    and on a Mac both count against the one unified memory. Releasing them
    between jobs lowered the next job's peak by 3 to 5 GiB at no measurable
    cost. CUDA keeps its caching allocator as it is.
    """
    torch = sys.modules.get("torch")
    if settings.apple_silicon and torch is not None:
        empty_cache(torch)


def require_generation_device() -> None:
    torch = torch_module()
    if selected_device(torch) != "cpu" or settings.allow_cpu_generation:
        return
    if settings.apple_silicon:
        raise RuntimeError(
            "Metal (MPS) is not available to PyTorch, so generation was stopped before model load. "
            "MosImage needs an Apple silicon Mac with macOS 14 or newer and the arm64 PyTorch build; "
            "recreate backend/.venv with ./start.sh, then restart the backend. Set "
            "MOSIMAGE_ALLOW_CPU_GENERATION=true only for debugging, because CPU generation "
            "will be impractically slow."
        )
    raise RuntimeError(
        "CUDA is not available, so generation was stopped before model load. "
        "Your RTX 5070 Ti should run this on GPU; install a CUDA-enabled PyTorch build "
        "for your NVIDIA driver, then restart the backend. Set "
        "MOSIMAGE_ALLOW_CPU_GENERATION=true only for debugging, because CPU generation "
        "will be impractically slow."
    )


@lru_cache(maxsize=1)
def _apple_chip() -> tuple[str, int | None]:
    """The chip name and unified memory size, as About This Mac shows them."""
    try:
        name = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True,
                              timeout=2, check=True).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        name = ""
    memory = physical_memory_gb()
    return name or f"Apple silicon ({platform.machine()})", round(memory) if memory is not None else None


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
        "max_loaded_pipelines": settings.max_loaded_pipelines,
        "attention_backend": settings.attention_backend,
        "cuda_alloc_conf": settings.cuda_alloc_conf,
        "quality_model": settings.quality_model,
        "fast_model": settings.fast_model,
        # The editor starts on this model's preset, which suits the device.
        "warm_up_model": settings.warm_up_model,
    }
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        info.update(
            {
                "cuda_available": True,
                "cuda_device": torch.cuda.get_device_name(0),
                "device_name": torch.cuda.get_device_name(0),
                "cuda_version": torch.version.cuda,
                "vram_gb": round(props.total_memory / (1024**3), 2),
                "cuda_memory_fraction": settings.cuda_memory_fraction,
            }
        )
    else:
        info["cuda_available"] = False
    if settings.apple_silicon:
        chip, memory = _apple_chip()
        info.update({"mps_available": mps_available(torch), "device_name": chip, "unified_memory_gb": memory,
                     "apple_weights": settings.apple_weights, "apple_vae_tiling": settings.apple_vae_tiling})
        if not info["mps_available"]:
            info["setup_hint"] = (
                "Metal (MPS) is unavailable to PyTorch. MosImage needs an Apple silicon Mac with "
                "macOS 14 or newer; recreate backend/.venv with ./start.sh and restart the backend."
            )
    elif not info["cuda_available"]:
        info["setup_hint"] = (
            "CUDA is unavailable to PyTorch. Install a CUDA-enabled PyTorch wheel "
            "for your NVIDIA driver and restart the backend."
        )
    return info
