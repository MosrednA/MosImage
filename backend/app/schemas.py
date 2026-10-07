from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


TaskKind = Literal["text-to-image", "inpaint", "outpaint", "select", "load-model"]
JobState = Literal["queued", "running", "cancelling", "cancelled", "completed", "failed"]
RuntimeMode = Literal["safe", "balanced", "performance"]


class ModelInfo(BaseModel):
    name: str
    display_name: str
    model_id: str
    capabilities: list[TaskKind]
    loaded: bool
    loaded_pipelines: list[str] = []


class LoadModelRequest(BaseModel):
    model_name: str = "flux2-klein-4b"
    task: Literal["text-to-image", "inpaint"] = "text-to-image"
    runtime_mode: RuntimeMode = "balanced"
    width: int = Field(default=1024, ge=256, le=2048)
    height: int = Field(default=1024, ge=256, le=2048)


class TextToImageRequest(BaseModel):
    prompt: str = Field(min_length=1)
    negative_prompt: str | None = None
    model_name: str = "flux2-klein-4b"
    seed: int | None = None
    runtime_mode: RuntimeMode = "balanced"
    steps: int = Field(default=4, ge=1, le=100)
    guidance: float = Field(default=1.0, ge=0.0, le=30.0)
    width: int = Field(default=1024, ge=256, le=2048)
    height: int = Field(default=1024, ge=256, le=2048)
    count: int = Field(default=1, ge=1, le=4)


class InpaintForm(BaseModel):
    prompt: str = Field(min_length=1)
    negative_prompt: str | None = None
    model_name: str = "flux2-klein-4b"
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


class OutpaintForm(InpaintForm):
    model_config = ConfigDict(extra="forbid")
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
    input_image_path: str | None = None
    mask_image_path: str | None = None
    output_image_path: str
    output_image_url: str
    created_at: datetime


class DocumentUpdate(BaseModel):
    name: str = Field(min_length=1, max_length=160)


class AlternativeProposalRequest(BaseModel):
    generation_ids: list[int] = Field(min_length=1, max_length=4)
    operation_type: Literal["generate", "inpaint", "outpaint"]
    base_revision_id: int | None = None


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


class DocumentRecord(BaseModel):
    id: int
    name: str
    canvas_width: int
    canvas_height: int
    color_profile: str
    current_revision_id: int | None = None
    current_image_url: str | None = None
    can_undo: bool
    can_redo: bool
    created_at: datetime
    updated_at: datetime
