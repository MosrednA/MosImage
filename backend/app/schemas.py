from datetime import datetime
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .config import settings


TaskKind = Literal["text-to-image", "edit", "inpaint", "outpaint", "select", "load-model"]
JobState = Literal["queued", "running", "cancelling", "cancelled", "completed", "failed"]
JobPhase = Literal["loading", "warming", "preparing", "generating", "finishing"]
RuntimeMode = Literal["safe", "balanced", "performance"]
MAX_REFERENCES_PER_EDIT = 3
# Fix faces redraws up to this many faces at a time.
MAX_FACES = 8
FaceBox = tuple[Annotated[int, Field(ge=0)], Annotated[int, Field(ge=0)], Annotated[int, Field(ge=0)], Annotated[int, Field(ge=0)]]
# The transformer a model runs, and so the LoRAs that fit it.
LoraArchitecture = Literal["klein-4b", "klein-9b"]
# Own LoRAs one edit can stack on top of the built-in ones.
MAX_LORAS = 3


class LoraChoice(BaseModel):
    """A file from the LoRA folder, by its name without ``.safetensors``, and its strength."""

    model_config = ConfigDict(extra="forbid")
    # A file name only: no folders, drive letters, hidden files or control characters.
    name: str = Field(min_length=1, max_length=200, pattern=r'^[^./\\:*?"<>|\x00-\x1f][^/\\:*?"<>|\x00-\x1f]*$')
    scale: float = Field(default=1.0, gt=0, le=2, allow_inf_nan=False)


def _unique_loras(value: list[LoraChoice]) -> list[LoraChoice]:
    if len({choice.name for choice in value}) != len(value):
        raise ValueError("Each LoRA can be chosen once.")
    return value


class ModelInfo(BaseModel):
    name: str
    display_name: str
    model_id: str
    capabilities: list[TaskKind]
    loaded: bool
    loaded_pipelines: list[str] = []
    architecture: LoraArchitecture
    # The model this device runs for the Quality or Fast preset, if any.
    preset: Literal["quality", "fast"] | None = None


def _quality_model() -> str:
    return settings.quality_model


class LoadModelRequest(BaseModel):
    model_name: str = Field(default_factory=_quality_model)
    task: Literal["text-to-image", "inpaint"] = "text-to-image"
    runtime_mode: RuntimeMode = "balanced"
    width: int = Field(default=1024, ge=256, le=2048)
    height: int = Field(default=1024, ge=256, le=2048)
    warm_up: bool = False


class PromptRequest(BaseModel):
    prompt: str = Field(min_length=1)
    negative_prompt: str | None = None
    model_name: str = Field(default_factory=_quality_model)
    seed: int | None = None
    runtime_mode: RuntimeMode = "balanced"
    steps: int = Field(default=4, ge=1, le=100)
    guidance: float = Field(default=1.0, ge=0.0, le=30.0)
    count: int = Field(default=1, ge=1, le=4)
    # People or objects to bring into the image, named in this order.
    reference_ids: list[Annotated[int, Field(ge=1)]] = Field(default_factory=list, max_length=MAX_REFERENCES_PER_EDIT)
    loras: list[LoraChoice] = Field(default_factory=list, max_length=MAX_LORAS)

    @field_validator("reference_ids")
    @classmethod
    def _unique_references(cls, value: list[int]) -> list[int]:
        return list(dict.fromkeys(value))

    _unique_loras = field_validator("loras")(_unique_loras)


class TextToImageRequest(PromptRequest):
    width: int = Field(default=1024, ge=256, le=2048)
    height: int = Field(default=1024, ge=256, le=2048)


class EditRequest(PromptRequest):
    """Edits the whole picture by instruction; the result keeps its size.

    Sketch mode turns rough coloured strokes into what the prompt names, face
    mode restores one face crop for Fix faces, and enhance mode adds fine
    detail to a picture or one tile of it.
    """

    mode: Literal["edit", "sketch", "face", "enhance"] = "edit"
    consistency_lora: bool = False
    # Scale of the detail-enhancer LoRA; zero leaves it off.
    detail: float = Field(default=0, ge=0, le=1, allow_inf_nan=False)
    # Below one the picture is partly redrawn: noise is added to it and only
    # the later steps run, which keeps its expression, pose and lighting.
    strength: float = Field(default=1.0, gt=0, le=1, allow_inf_nan=False)

    @model_validator(mode="after")
    def compatible_options(self) -> "EditRequest":
        if self.consistency_lora and (self.model_name != settings.quality_model or self.mode != "edit"):
            raise ValueError("Consistency is available for Quality image edits only.")
        if self.detail and (self.model_name != settings.quality_model or self.mode not in ("face", "enhance")):
            raise ValueError("The detail enhancer is available for Quality Enhance and Fix faces only.")
        if self.strength < 1 and self.mode not in ("face", "enhance"):
            raise ValueError("Partial redraws are available for Enhance and Fix faces only.")
        return self


class FaceFixRequest(PromptRequest):
    """Redraws every face (or those in the selection) sharper; the prompt is optional.

    Faithful redraws only the later steps from the face itself, so expression
    and lighting stay; otherwise the face is drawn anew from its crop.
    """

    prompt: str = Field(default="", max_length=500)
    faithful: bool = True
    # The faces picked on the canvas as left, top, right and bottom pixels; they
    # replace detection and the selection. Without them the faces are found anew.
    faces: list[FaceBox] | None = Field(default=None, min_length=1, max_length=MAX_FACES)

    @field_validator("faces")
    @classmethod
    def faces_have_area(cls, faces: list[FaceBox] | None) -> list[FaceBox] | None:
        if faces and any(right <= left or bottom <= top for left, top, right, bottom in faces):
            raise ValueError("Every face needs a width and a height.")
        return faces


class FaceCandidate(BaseModel):
    box: FaceBox
    score: float
    # Inside a bigger person: a real face, which Fix faces picks by itself.
    suggested: bool


class FaceDetectionResponse(BaseModel):
    faces: list[FaceCandidate]


class EnhanceRequest(PromptRequest):
    """Adds fine detail to the picture or the selection: with the detail-enhancer LoRA on Quality."""

    prompt: str = Field(default="", max_length=500)
    # The LoRA scale: 0.4 adds believable texture, 0.7 and up starts to reinvent.
    # Fast has no such LoRA and redraws more of the picture instead.
    detail: float = Field(default=0.4, ge=0.1, le=1, allow_inf_nan=False)
    reference_ids: list[int] = Field(default_factory=list, max_length=0)


class InpaintForm(BaseModel):
    prompt: str = Field(min_length=1)
    negative_prompt: str | None = None
    model_name: str = Field(default_factory=_quality_model)
    seed: int | None = None
    runtime_mode: RuntimeMode = "balanced"
    steps: int = Field(default=4, ge=1, le=100)
    guidance: float = Field(default=1.0, ge=0.0, le=30.0)
    strength: float = Field(default=1.0, ge=0.0, le=1.0)
    count: int = Field(default=1, ge=1, le=4)
    edit_operation: Literal["replace", "remove", "sketch"] = "replace"
    blend_feather: float = Field(default=16, ge=0, le=64, allow_inf_nan=False)
    context_padding: int = Field(default=320, ge=32, le=1024)
    scene_context: str = Field(default="", max_length=1500)
    consistency_lora: bool = False
    detail_crop: bool = False
    # Order matters: the prompt names them as reference image 1, 2, ...
    reference_ids: list[Annotated[int, Field(ge=1)]] = Field(default_factory=list, max_length=MAX_REFERENCES_PER_EDIT)
    loras: list[LoraChoice] = Field(default_factory=list, max_length=MAX_LORAS)

    @field_validator("reference_ids")
    @classmethod
    def _unique_references(cls, value: list[int]) -> list[int]:
        return list(dict.fromkeys(value))

    _unique_loras = field_validator("loras")(_unique_loras)

    @model_validator(mode="after")
    def compatible_quality_options(self) -> "InpaintForm":
        if self.consistency_lora and (self.model_name != settings.quality_model or self.edit_operation != "replace"):
            raise ValueError("Consistency is available for Quality replacement and expansion only.")
        if self.detail_crop and (self.edit_operation != "replace" or self.strength != 1):
            raise ValueError("Fine detail requires Replace at full strength.")
        return self


class OutpaintForm(InpaintForm):
    model_config = ConfigDict(extra="forbid")
    reference_ids: list[int] = Field(default_factory=list, max_length=0)
    left: int = Field(default=0, ge=0, le=2048)
    top: int = Field(default=0, ge=0, le=2048)
    right: int = Field(default=0, ge=0, le=2048)
    bottom: int = Field(default=0, ge=0, le=2048)
    specialized_outpaint: bool = False

    @model_validator(mode="after")
    def requires_expansion(self) -> "OutpaintForm":
        if not (self.left or self.top or self.right or self.bottom):
            raise ValueError("Add space on at least one side of the image")
        if self.detail_crop:
            raise ValueError("Fine detail is available for selected replacements only.")
        if self.specialized_outpaint and (self.model_name != settings.fast_model or self.steps != 4 or self.guidance != 1 or self.strength != 1):
            raise ValueError("Specialized outpaint requires the Fast model at four steps and guidance one.")
        if self.specialized_outpaint and self.loras:
            raise ValueError("Specialized outpaint runs its own LoRA; turn your LoRAs off for it.")
        return self


class LoraFile(BaseModel):
    name: str
    size_bytes: int
    # The models it fits: empty when it fits none, with the reason in ``problem``.
    models: list[str]
    problem: str | None = None


class LoraLibrary(BaseModel):
    folder: str
    loras: list[LoraFile]


class SelectionPoint(BaseModel):
    model_config = ConfigDict(extra="forbid")
    x: float = Field(ge=0, allow_inf_nan=False)
    y: float = Field(ge=0, allow_inf_nan=False)
    label: Literal[0, 1]


class TextSelectionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: str = Field(min_length=1, max_length=120)


class CutoutRequest(BaseModel):
    """Keeps the main subject, or what the text names, with soft edges."""

    model_config = ConfigDict(extra="forbid")
    text: str = Field(default="", max_length=120)


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
    edit_operation: Literal["replace", "remove", "sketch", "faces", "enhance"] | None = None
    blend_feather: float | None = None
    context_padding: int | None = None
    scene_context: str | None = None
    consistency_lora: bool = False
    detail_crop: bool = False
    specialized_outpaint: bool = False
    loras: list[LoraChoice] = Field(default_factory=list)
    reference_image_paths: list[str] = Field(default_factory=list)
    input_image_path: str | None = None
    mask_image_path: str | None = None
    output_image_path: str
    output_image_url: str
    created_at: datetime


ReferenceRole = Literal["subject", "clothing", "pose", "style"]


class ReferenceRecord(BaseModel):
    id: int
    # None for loose references on the start screen.
    document_id: int | None
    name: str
    role: ReferenceRole = "subject"
    width: int
    height: int
    image_url: str
    subject_url: str | None = None
    created_at: datetime


class ReferenceUpdate(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=80)
    role: ReferenceRole | None = None


class DocumentUpdate(BaseModel):
    name: str = Field(min_length=1, max_length=160)


class CropRequest(BaseModel):
    base_revision_id: int
    # The new canvas in background pixels; it may reach past the picture.
    x: int = Field(ge=-4096, le=4096)
    y: int = Field(ge=-4096, le=4096)
    width: int = Field(ge=16, le=4096)
    height: int = Field(ge=16, le=4096)


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
    # A switched-off mask is kept but not applied, like Shift-clicking a mask in Photoshop.
    mask_enabled: bool = True
    # AI proposals kept in one layer, like Photoshop's generative variations; image_path is the one shown.
    variant_paths: list[str] = Field(default_factory=list, max_length=8)
    x: float
    y: float
    scale_x: float = 1.0
    scale_y: float = 1.0
    rotation: float = 0.0
    opacity: float = Field(default=1.0, ge=0, le=1)
    visible: bool = True
    # The layer an AI edit or Blend in replaced, kept hidden just below; merging this one lets it go.
    original_id: str | None = Field(default=None, pattern=r"^[A-Za-z0-9_-]{1,40}$")

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
    variant_urls: list[str] = Field(default_factory=list)


class LayersUpdate(BaseModel):
    base_revision_id: int
    layers: list[LayerState] = Field(max_length=40)
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
    # The layers that make up the picture in the current revision, bottom first.
    layers: list[LayerRecord] = Field(default_factory=list)
    can_undo: bool
    can_redo: bool
    created_at: datetime
    updated_at: datetime


class EngineStatus(BaseModel):
    warm_up: JobStatus | None = None
    models: list[ModelInfo]
