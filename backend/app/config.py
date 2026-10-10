import sys
from pathlib import Path

from pydantic import AliasChoices, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Apple silicon runs the BF16 weights on Metal (MPS); everywhere else the
# defaults target the RTX 5070 Ti profile with native NVFP4 on CUDA.
APPLE_SILICON = sys.platform == "darwin"
# The Quality and Fast models of each profile, by registry name.
CUDA_MODELS = {"quality": "flux2-klein-9b-nvfp4", "fast": "flux2-klein-4b-nvfp4"}
APPLE_MODELS = {"quality": "flux2-klein-9b", "fast": "flux2-klein-4b"}


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
    enable_tf32: bool = True
    allow_cpu_generation: bool = False
    cuda_memory_fraction: float | None = None
    cuda_alloc_conf: str = "expandable_segments:True"
    attention_backend: str = "sdpa"
    supervised: bool = False
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
    # Empty means the Quality model of the device profile.
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
    def _default_warm_up_model(self) -> "Settings":
        if not self.warm_up_model:
            self.warm_up_model = self.quality_model
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
