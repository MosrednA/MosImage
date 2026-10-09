from datetime import datetime
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


TaskKind = Literal["text-to-image", "edit", "inpaint", "outpaint", "select", "load-model"]
JobState = Literal["queued", "running", "cancelling", "cancelled", "completed", "failed"]
JobPhase = Literal["loading", "warming", "preparing", "generating", "finishing"]
RuntimeMode = Literal["safe", "balanced", "performance"]
MAX_REFERENCES_PER_EDIT = 3


class ModelInfo(BaseModel):
    name: str
    display_name: str
    model_id: str
    capabilities: list[TaskKind]
    loaded: bool
    loaded_pipelines: list[str] = []


class LoadModelRequest(BaseModel):
    model_name: str = "flux2-klein-9b-nvfp4"
    task: Literal["text-to-image", "inpaint"] = "text-to-image"
    runtime_mode: RuntimeMode = "balanced"
    width: int = Field(default=1024, ge=256, le=2048)
    height: int = Field(default=1024, ge=256, le=2048)
    warm_up: bool = False


class PromptRequest(BaseModel):
    prompt: str = Field(min_length=1)
    negative_prompt: str | None = None
    model_name: str = "flux2-klein-9b-nvfp4"
    seed: int | None = None
    runtime_mode: RuntimeMode = "balanced"
    steps: int = Field(default=4, ge=1, le=100)
    guidance: float = Field(default=1.0, ge=0.0, le=30.0)
    count: int = Field(default=1, ge=1, le=4)
    # People or objects to bring into the image, named in this order.
    reference_ids: list[Annotated[int, Field(ge=1)]] = Field(default_factory=list, max_length=MAX_REFERENCES_PER_EDIT)

    @field_validator("reference_ids")
    @classmethod
    def _unique_references(cls, value: list[int]) -> list[int]:
        return list(dict.fromkeys(value))


class TextToImageRequest(PromptRequest):
    width: int = Field(default=1024, ge=256, le=2048)
    height: int = Field(default=1024, ge=256, le=2048)


class EditRequest(PromptRequest):
    """Edits the whole picture by instruction; the result keeps its size."""


class CopyReferencesRequest(BaseModel):
    reference_ids: list[Annotated[int, Field(ge=1)]] = Field(min_length=1, max_length=MAX_REFERENCES_PER_EDIT)


class InpaintForm(BaseModel):
    prompt: str = Field(min_length=1)
    negative_prompt: str | None = None
    model_name: str = "flux2-klein-9b-nvfp4"
    seed: int | None = None
    runtime_mode: RuntimeMode = "balanced"
    steps: int = Field(default=4, ge=1, le=100)
    guidance: float = Field(default=1.0, ge=0.0, le=30.0)
    strength: float = Field(default=1.0, ge=0.0, le=1.0)
    count: int = Field(default=1, ge=1, le=4)
    edit_operation: Literal["replace", "remove"] = "replace"
    blend_feather: float = Field(default=16, ge=0, le=64, allow_inf_nan=False)
    context_padding: int = Field(default=320, ge=32, le=1024)
    scene_context: str = Field(default="", max_length=1500)
    # Order matters: the prompt names them as reference image 1, 2, ...
    reference_ids: list[Annotated[int, Field(ge=1)]] = Field(default_factory=list, max_length=MAX_REFERENCES_PER_EDIT)

    @field_validator("reference_ids")
    @classmethod
    def _unique_references(cls, value: list[int]) -> list[int]:
        return list(dict.fromkeys(value))


class OutpaintForm(InpaintForm):
    model_config = ConfigDict(extra="forbid")
    reference_ids: list[int] = Field(default_factory=list, max_length=0)
    model_name: str = "z-image-turbo"
    steps: int = Field(default=8, ge=1, le=100)
    guidance: float = Field(default=0.0, ge=0.0, le=30.0)
    left: int = Field(default=0, ge=0, le=2048)
    top: int = Field(default=0, ge=0, le=2048)
    right: int = Field(default=0, ge=0, le=2048)
    bottom: int = Field(default=0, ge=0, le=2048)

    @model_validator(mode="after")
    def requires_expansion(self) -> "OutpaintForm":
        if not (self.left or self.top or self.right or self.bottom):
            raise ValueError("Add space on at least one side of the image")
        return self


class SelectionPoint(BaseModel):
    model_config = ConfigDict(extra="forbid")
    x: float = Field(ge=0, allow_inf_nan=False)
    y: float = Field(ge=0, allow_inf_nan=False)
    label: Literal[0, 1]


class SelectionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    points: list[SelectionPoint] = Field(min_length=1, max_length=24)

    @model_validator(mode="after")
    def requires_positive_point(self) -> "SelectionRequest":
        if not any(point.label == 1 for point in self.points):
            raise ValueError("Click an object before excluding points")
        return self


class JobResponse(BaseModel):
    id: str
    kind: TaskKind
    status: JobState


class JobStatus(BaseModel):
    id: str
    kind: TaskKind
    status: JobState
    progress: float
    error: str | None = None
    generation_id: int | None = None
    generation_ids: list[int] = Field(default_factory=list)
    mask_image_url: str | None = None
    phase: JobPhase | None = None
    detail: str | None = None
    step: int | None = None
    steps: int | None = None
    item: int | None = None
    items: int | None = None
    elapsed_seconds: float | None = None
    phase_elapsed_seconds: float | None = None
    eta_seconds: float | None = None
    waiting_for: TaskKind | None = None


class GenerationRecord(BaseModel):
    id: int
    task: TaskKind
    prompt: str
    negative_prompt: str | None = None
    model_name: str
    runtime_mode: RuntimeMode = "balanced"
    seed: int
    steps: int
    guidance: float
    width: int | None = None
    height: int | None = None
    strength: float | None = None
    edit_operation: Literal["replace", "remove"] | None = None
    blend_feather: float | None = None
    context_padding: int | None = None
    scene_context: str | None = None
    reference_image_paths: list[str] = Field(default_factory=list)
    input_image_path: str | None = None
    mask_image_path: str | None = None
    output_image_path: str
    output_image_url: str
    created_at: datetime


class ReferenceRecord(BaseModel):
    id: int
    # None for loose references on the start screen.
    document_id: int | None
    name: str
    width: int
    height: int
    image_url: str
    subject_url: str | None = None
    created_at: datetime


class DocumentUpdate(BaseModel):
    name: str = Field(min_length=1, max_length=160)


class AlternativeProposalRequest(BaseModel):
    generation_ids: list[int] = Field(min_length=1, max_length=4)
    operation_type: Literal["generate", "inpaint", "outpaint"]
    base_revision_id: int | None = None
    # Expand adds this much canvas on the left and top; layers move along with the content.
    layer_offset_x: int = Field(default=0, ge=0)
    layer_offset_y: int = Field(default=0, ge=0)


class LayerState(BaseModel):
    """One image layer: centre position in canvas pixels, scale (negative flips) and rotation in degrees."""

    id: str = Field(pattern=r"^[A-Za-z0-9_-]{1,40}$")
    name: str = Field(min_length=1, max_length=80)
    image_path: str
    mask_path: str | None = None
    x: float
    y: float
    scale_x: float = 1.0
    scale_y: float = 1.0
    rotation: float = 0.0
    opacity: float = Field(default=1.0, ge=0, le=1)
    visible: bool = True

    @field_validator("scale_x", "scale_y")
    @classmethod
    def usable_scale(cls, value: float) -> float:
        if not 0.005 <= abs(value) <= 50:
            raise ValueError("Layer scale must stay between 0.5% and 5000%")
        return value


class LayerRecord(LayerState):
    width: int
    height: int
    image_url: str
    mask_url: str | None = None


class LayersUpdate(BaseModel):
    base_revision_id: int
    layers: list[LayerState] = Field(max_length=32)
    # Short description of the change, such as "Move layer".
    label: str = Field(default="Edit layers", max_length=60)


class LayerAsset(BaseModel):
    path: str
    url: str
    width: int
    height: int


class AlternativeRecord(BaseModel):
    id: int
    revision_id: int
    generation_id: int | None = None
    seed: int
    model_name: str
    output_image_path: str
    output_image_url: str
    metadata: dict[str, object]
    selected: bool


class RevisionRecord(BaseModel):
    id: int
    document_id: int
    parent_revision_id: int | None = None
    operation_type: str
    composite_image_path: str
    composite_image_url: str
    mask_image_path: str | None = None
    operation_parameters: dict[str, object]
    selected_alternative_id: int | None = None
    accepted: bool
    created_at: datetime
    alternatives: list[AlternativeRecord] = Field(default_factory=list)
    layers: list[LayerRecord] = Field(default_factory=list)


class DocumentRecord(BaseModel):
    id: int
    name: str
    canvas_width: int
    canvas_height: int
    color_profile: str
    current_revision_id: int | None = None
    current_image_url: str | None = None
    # Layers above the background in the current revision, bottom first.
    layers: list[LayerRecord] = Field(default_factory=list)
    can_undo: bool
    can_redo: bool
    created_at: datetime
    updated_at: datetime


class EngineStatus(BaseModel):
    warm_up: JobStatus | None = None
    models: list[ModelInfo]
