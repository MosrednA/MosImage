import os
import sys
from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Apple silicon runs the BF16 weights on Metal (MPS); everywhere else the
# defaults target the RTX 5070 Ti profile with native NVFP4 on CUDA.
APPLE_SILICON = sys.platform == "darwin"
if APPLE_SILICON:
    # PyTorch reads this once, when it is imported: an operator Metal lacks
    # then runs on the CPU instead of failing the edit.
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
# The Quality and Fast models of each profile, by registry name.
CUDA_MODELS = {"quality": "flux2-klein-9b-nvfp4", "fast": "flux2-klein-4b-nvfp4"}
APPLE_MODELS = {"quality": "flux2-klein-9b", "fast": "flux2-klein-4b"}
AppleWeights = Literal["bf16", "q8", "q4"]
# Unified memory (GB) from which a Mac runs Klein in BF16 by default, and from
# which in 8-bit; below that, in 4-bit.
APPLE_BF16_MEMORY_GB = 64
APPLE_Q8_MEMORY_GB = 48
# Quality's peak footprint (GiB) per weight format on an M5 Max, and the room
# macOS and the UI need beside it; without that room, warm up Fast instead.
# See benchmarks/MLX_KLEIN_2026-10-11.md.
APPLE_QUALITY_PEAK_GB: dict[str, float] = {"bf16": 43, "q8": 31, "q4": 24}
APPLE_HEADROOM_GB = 12
# Side (px) of the VAE tiles on a Mac without that room. Untiled, a 1024²
# decode takes 6.5 GiB of unified memory; in these tiles 1.3 GiB.
APPLE_VAE_TILE = 512


def physical_memory_gb() -> float | None:
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1024**3
    except (ValueError, OSError, AttributeError):
        return None


def default_apple_weights(memory_gb: float | None) -> AppleWeights:
    if memory_gb is None or memory_gb >= APPLE_BF16_MEMORY_GB:
        return "bf16"
    return "q8" if memory_gb >= APPLE_Q8_MEMORY_GB else "q4"


class Settings(BaseSettings):
    app_name: str = "MosImage"
    hardware_profile: str = "apple-silicon" if APPLE_SILICON else "rtx5070ti-16gb"
    project_root: Path = Path(__file__).resolve().parents[2]
    data_dir: Path = project_root / "data"
    db_path: Path = data_dir / "app.db"
    image_dir: Path = data_dir / "images"
    # Vite dev server, plus the Tauri desktop shell (Windows, then macOS/Linux).
    cors_origins: list[str] = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://tauri.localhost",
        "https://tauri.localhost",
        "tauri://localhost",
    ]
    max_loaded_pipelines: int = 1
    enable_cpu_offload: bool = True
    runtime_mode: str = "balanced"
    # cuda, mps (Apple silicon) or cpu.
    preferred_device: str = "mps" if APPLE_SILICON else "cuda"
    preferred_dtype: str = "bfloat16"
    # Apple silicon: how the Klein transformer and text encoder hold their weights.
    # bf16 runs them in PyTorch on MPS; q8 and q4 run them in MLX, quantized
    # group-wise to 8 or 4 bits. Masked edits, references and LoRAs work with all
    # three. Empty picks by unified memory (default_apple_weights).
    apple_weights: Literal["", "bf16", "q8", "q4"] = ""
    # Apple silicon: decode and encode the VAE in APPLE_VAE_TILE tiles. Saves
    # about 5 GiB at 1024², at the cost of faint colour shifts between tiles.
    # Unset turns it on where Quality's peak does not fit (as warm_up_model).
    apple_vae_tiling: bool | None = None
    enable_tf32: bool = True
    allow_cpu_generation: bool = False
    cuda_memory_fraction: float | None = None
    cuda_alloc_conf: str = "expandable_segments:True"
    attention_backend: str = "sdpa"
    supervised: bool = False
    # Set by the desktop app on macOS and Linux: exit when the process that
    # started the engine is gone, so a crashed app leaves no models in memory.
    exit_with_parent: bool = False
    # Also exit once this process is gone: start.sh passes its own PID, since
    # the engine's parent is the supervisor loop.
    watch_pid: int | None = None
    cancel_grace_seconds: float = 8.0
    # Inference area budget of a masked edit, as the side of a square.
    inpaint_max_dimension: int = 1024
    # Smaller crops are rendered at least this large and scaled back down.
    inpaint_minimum_dimension: int = 768
    inpaint_context_ratio: float = 0.5
    inpaint_minimum_context: int = 128
    # Uploads are read into memory, so cap them before they reach the job queue.
    max_upload_bytes: int = 64 * 1024 * 1024
    # Pillow refuses to decode images above twice this many pixels (decompression bombs).
    max_image_pixels: int = 64_000_000
    # Finished jobs stay queryable this long, so a slow poll still sees the result.
    job_retention_seconds: float = 600.0
    # Load and warm the Quality model and SAM while the UI opens.
    warm_up_on_start: bool = True
    # Empty means the Quality model of the device profile (Fast on a Mac without room for it).
    warm_up_model: str = ""
    # Set by the desktop app that started this engine, so only it can stop it.
    shutdown_token: str | None = Field(default=None, repr=False)
    huggingface_token: str | None = Field(
        default=None,
        validation_alias=AliasChoices("HF_TOKEN", "MOSIMAGE_HF_TOKEN"),
        repr=False,
    )

    model_config = SettingsConfigDict(env_prefix="MOSIMAGE_", env_file=".env")

    @model_validator(mode="after")
    def _device_defaults(self) -> "Settings":
        memory = physical_memory_gb() if self.apple_silicon else None
        if not self.apple_weights:
            self.apple_weights = default_apple_weights(memory)
        room = APPLE_QUALITY_PEAK_GB[self.apple_weights] + APPLE_HEADROOM_GB
        small = memory is not None and memory < room
        if not self.warm_up_model:
            self.warm_up_model = self.fast_model if small else self.quality_model
        if self.apple_vae_tiling is None:
            self.apple_vae_tiling = small
        return self

    @property
    def apple_silicon(self) -> bool:
        """The Metal profile: BF16 Klein models in unified memory."""
        return self.preferred_device == "mps"

    @property
    def quality_model(self) -> str:
        return (APPLE_MODELS if self.apple_silicon else CUDA_MODELS)["quality"]

    @property
    def fast_model(self) -> str:
        return (APPLE_MODELS if self.apple_silicon else CUDA_MODELS)["fast"]

    @property
    def input_dir(self) -> Path:
        return self.image_dir / "inputs"

    @property
    def mask_dir(self) -> Path:
        return self.image_dir / "masks"

    @property
    def output_dir(self) -> Path:
        return self.image_dir / "outputs"

    @property
    def document_dir(self) -> Path:
        return self.image_dir / "documents"

    @property
    def revision_dir(self) -> Path:
        return self.image_dir / "revisions"

    @property
    def reference_dir(self) -> Path:
        return self.image_dir / "references"

    @property
    def layer_dir(self) -> Path:
        return self.image_dir / "layers"


settings = Settings()
