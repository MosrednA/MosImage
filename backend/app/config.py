from pathlib import Path

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "MosImage"
    hardware_profile: str = "rtx5070ti-16gb"
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
    gpu_resident_max_pixels: int = 802816
    preferred_device: str = "cuda"
    preferred_dtype: str = "bfloat16"
    enable_tf32: bool = True
    allow_cpu_generation: bool = False
    cuda_memory_fraction: float | None = None
    cuda_alloc_conf: str = "expandable_segments:True"
    attention_backend: str = "sdpa"
    supervised: bool = False
    cancel_grace_seconds: float = 8.0
    inpaint_max_dimension: int = 1024
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
    warm_up_model: str = "flux2-klein-9b-nvfp4"
    # Set by the desktop app that started this engine, so only it can stop it.
    shutdown_token: str | None = Field(default=None, repr=False)
    huggingface_token: str | None = Field(
        default=None,
        validation_alias=AliasChoices("HF_TOKEN", "MOSIMAGE_HF_TOKEN"),
        repr=False,
    )

    model_config = SettingsConfigDict(env_prefix="MOSIMAGE_", env_file=".env")

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
