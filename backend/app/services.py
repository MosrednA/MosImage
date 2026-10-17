from __future__ import annotations

import json
from dataclasses import replace
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import uuid4
from typing import Literal

from PIL import Image

from . import references, storage
from .compositing import FAST_PNG, INPAINT_GENERATION_OVERLAP, expansion_mask, prepare_inpaint_workspace, prepare_outpaint_workspace
from .config import settings
from .jobs import JobCancelled, JobControl
from .models.base import ImageModelProvider, ModelReference
from .models.registry import registry
from .schemas import EditRequest, InpaintForm, OutpaintForm, TextToImageRequest


def run_text_to_image(request: TextToImageRequest, control: JobControl) -> list[int]:
    provider = registry.activate(request.model_name)
    output_paths: list[Path] = []
    generation_values: list[dict[str, object]] = []
    model_references = _model_references(request.reference_ids)
    # Only reference-aware providers receive the extra argument.
    extra = {"references": model_references} if model_references else {}
    try:
        for index in range(request.count):
            output_path = _new_image_path(settings.output_dir, "output")
            output_paths.append(output_path)
            iteration = request.model_copy(
                update={"seed": _alternative_seed(request.seed, index), "count": 1}
            )
            seed, _ = provider.generate(iteration, output_path, control.slice(index, request.count), **extra)
            control.raise_if_cancelled()
            generation_values.append(
                {
                    "task": "text-to-image",
                    "prompt": request.prompt,
                    "negative_prompt": request.negative_prompt,
                    "model_name": request.model_name,
                    "runtime_mode": request.runtime_mode,
                    "seed": seed,
                    "steps": request.steps,
                    "guidance": request.guidance,
                    "width": request.width,
                    "height": request.height,
                    "strength": None,
                    "input_image_path": None,
                    "mask_image_path": None,
                    "output_image_path": storage.to_relative(output_path),
                    "reference_image_paths": json.dumps([storage.to_relative(reference.path) for reference in model_references]) if model_references else None,
                }
            )
        control.raise_if_cancelled()
        return storage.create_generations(generation_values)
    except BaseException as error:
        release_after_failure(provider, error)
        for output_path in output_paths:
            output_path.unlink(missing_ok=True)
        raise


def run_image_edit(request: EditRequest, image_bytes: bytes, image_name: str | None, control: JobControl) -> list[int]:
    """Edits the whole picture by instruction; every proposal keeps the picture's size."""
    provider = registry.activate(request.model_name)
    input_path = _save_bytes(image_bytes, settings.input_dir, "input", image_name)
    output_paths: list[Path] = []
    generation_values: list[dict[str, object]] = []
    model_references = _model_references(request.reference_ids)
    extra = {"references": model_references} if model_references else {}
    try:
        with Image.open(input_path) as picture:
            width, height = picture.size
        for index in range(request.count):
            output_path = _new_image_path(settings.output_dir, "output")
            output_paths.append(output_path)
            iteration = request.model_copy(update={"seed": _alternative_seed(request.seed, index), "count": 1})
            seed, _ = provider.edit(iteration, input_path, output_path, control.slice(index, request.count), **extra)
            control.raise_if_cancelled()
            generation_values.append(
                {
                    "task": "edit",
                    "prompt": request.prompt,
                    "negative_prompt": request.negative_prompt,
                    "model_name": request.model_name,
                    "runtime_mode": request.runtime_mode,
                    "seed": seed,
                    "steps": request.steps,
                    "guidance": request.guidance,
                    "width": width,
                    "height": height,
                    "strength": None,
                    "input_image_path": storage.to_relative(input_path),
                    "consistency_lora": request.consistency_lora,
                    "mask_image_path": None,
                    "output_image_path": storage.to_relative(output_path),
                    "reference_image_paths": json.dumps([storage.to_relative(reference.path) for reference in model_references]) if model_references else None,
                }
            )
        control.raise_if_cancelled()
        return storage.create_generations(generation_values)
    except BaseException as error:
        release_after_failure(provider, error)
        for path in (input_path, *output_paths):
            path.unlink(missing_ok=True)
        raise


def _model_references(reference_ids: list[int]) -> list[ModelReference]:
    return [
        ModelReference(name=record["name"], path=references.model_image_path(record["id"]), role=record.get("role", "subject"))
        for record in (references.get_reference(reference_id) for reference_id in reference_ids)
        if record is not None
    ]


def run_inpaint(
    request: InpaintForm,
    image_bytes: bytes,
    image_name: str | None,
    mask_bytes: bytes,
    mask_name: str | None,
    control: JobControl,
) -> list[int]:
    return _run_masked_edit(request, image_bytes, image_name, mask_bytes, mask_name, control, "inpaint")


def prepare_outpaint_images(content: bytes, request: OutpaintForm) -> tuple[bytes, bytes]:
    with Image.open(BytesIO(content)) as uploaded:
        source = uploaded.convert("RGBA")
    size = (source.width + request.left + request.right, source.height + request.top + request.bottom)
    if max(size) > 4096 or size[0] * size[1] > 16_777_216:
        raise ValueError("The expanded canvas must fit within 4096 × 4096 pixels")
    canvas = Image.new("RGBA", size, (128, 128, 128, 0))
    canvas.paste(source, (request.left, request.top))
    mask = expansion_mask(size, (request.left, request.top, request.left + source.width, request.top + source.height))
    image_buffer, mask_buffer = BytesIO(), BytesIO()
    canvas.save(image_buffer, "PNG", **FAST_PNG)
    mask.save(mask_buffer, "PNG", **FAST_PNG)
    return image_buffer.getvalue(), mask_buffer.getvalue()


def run_outpaint(request: OutpaintForm, image_bytes: bytes, control: JobControl) -> list[int]:
    canvas, mask = prepare_outpaint_images(image_bytes, request)
    return _run_masked_edit(request, canvas, "canvas.png", mask, "mask.png", control, "outpaint")


def _run_masked_edit(
    request: InpaintForm,
    image_bytes: bytes,
    image_name: str | None,
    mask_bytes: bytes,
    mask_name: str | None,
    control: JobControl,
    task: Literal["inpaint", "outpaint"],
) -> list[int]:
    provider = registry.activate(request.model_name)
    input_path = _save_bytes(image_bytes, settings.input_dir, "input", image_name)
    mask_path = _save_bytes(mask_bytes, settings.mask_dir, "mask", mask_name)
    output_paths: list[Path] = []
    generation_values: list[dict[str, object]] = []
    model_references = _model_references(request.reference_ids) if task == "inpaint" else []

    try:
        with TemporaryDirectory(prefix="inpaint-", dir=settings.data_dir) as temporary_dir:
            if task == "outpaint":
                workspace = prepare_outpaint_workspace(
                    input_path, mask_path, Path(temporary_dir), settings.inpaint_max_dimension,
                    settings.inpaint_context_ratio, settings.inpaint_minimum_context,
                    minimum_dimension=settings.inpaint_minimum_dimension,
                )
            else:
                # Generate throughout the selected footprint, one token past
                # it; the soft mask belongs to final compositing, not to
                # retaining old content during denoising. Pixels beyond that
                # footprint stay exact.
                workspace = prepare_inpaint_workspace(
                    input_path, mask_path, Path(temporary_dir), settings.inpaint_max_dimension,
                    0, request.context_padding, composite_feather=request.blend_feather,
                    center_selection=True, generation_growth=INPAINT_GENERATION_OVERLAP,
                    minimum_dimension=settings.inpaint_minimum_dimension,
                )
            for index in range(request.count):
                output_path = _new_image_path(settings.output_dir, "output")
                output_paths.append(output_path)
                model_output_path = Path(temporary_dir) / f"output-{index}.png"
                iteration_workspace = replace(workspace, model_output_path=model_output_path)
                iteration = request.model_copy(
                    update={"seed": _alternative_seed(request.seed, index), "count": 1}
                )
                iteration_control = control.slice(index, request.count)
                # Only reference-aware providers receive the extra argument.
                extra = {"references": model_references} if model_references else {}
                if isinstance(iteration, OutpaintForm) and iteration.specialized_outpaint:
                    green_path = Path(temporary_dir) / "green-input.png"
                    green = workspace.source.convert("RGB")
                    with Image.open(mask_path) as border:
                        green.paste((0, 255, 0), mask=border.convert("L"))
                    green.save(green_path, **FAST_PNG)
                    full_output = Path(temporary_dir) / f"full-output-{index}.png"
                    seed, _model_width = provider.outpaint(iteration, green_path, full_output, iteration_control)
                    with Image.open(full_output) as result:
                        result.resize(workspace.source.size, Image.Resampling.LANCZOS).crop(workspace.crop_box).save(model_output_path, **FAST_PNG)
                    iteration_workspace = replace(iteration_workspace, model_box=None)
                elif iteration.detail_crop:
                    detail_request = EditRequest(prompt=iteration.prompt + (
                        " Scene context: " + iteration.scene_context if iteration.scene_context else ""),
                        negative_prompt=iteration.negative_prompt,
                        model_name=iteration.model_name, runtime_mode=iteration.runtime_mode,
                        seed=iteration.seed, steps=iteration.steps, guidance=iteration.guidance,
                        reference_ids=iteration.reference_ids, consistency_lora=iteration.consistency_lora)
                    seed, _model_width = provider.edit(detail_request, iteration_workspace.model_input_path,
                        model_output_path, iteration_control, **extra)
                else:
                    seed, _model_width = provider.inpaint(iteration, iteration_workspace.model_input_path,
                        iteration_workspace.model_mask_path, model_output_path, iteration_control, **extra)
                iteration_control.set_progress(0.97)
                iteration_control.set_phase("finishing")
                width, height = iteration_workspace.composite_to(output_path)
                control.raise_if_cancelled()
                generation_values.append(
                    {
                        "task": task,
                        "prompt": request.prompt,
                        "negative_prompt": request.negative_prompt,
                        "model_name": request.model_name,
                        "runtime_mode": request.runtime_mode,
                        "seed": seed,
                        "steps": request.steps,
                        "guidance": request.guidance,
                        "width": width,
                        "height": height,
                        "strength": request.strength,
                        "consistency_lora": request.consistency_lora,
                        "detail_crop": request.detail_crop,
                        "specialized_outpaint": isinstance(request, OutpaintForm) and request.specialized_outpaint,
                        "edit_operation": request.edit_operation if task == "inpaint" else None,
                        "blend_feather": request.blend_feather if task == "inpaint" else None,
                        "context_padding": request.context_padding if task == "inpaint" else None,
                        "scene_context": request.scene_context if task == "inpaint" else None,
                        "reference_image_paths": json.dumps([storage.to_relative(reference.path) for reference in model_references]) if model_references else None,
                        "input_image_path": storage.to_relative(input_path),
                        "mask_image_path": storage.to_relative(mask_path),
                        "output_image_path": storage.to_relative(output_path),
                    }
                )
        control.raise_if_cancelled()
        return storage.create_generations(generation_values)
    except BaseException as error:
        release_after_failure(provider, error)
        for path in (input_path, mask_path, *output_paths):
            path.unlink(missing_ok=True)
        raise


def release_after_failure(provider: ImageModelProvider, error: BaseException) -> None:
    """Unload after a failure, which may leave GPU memory in an unknown state.

    A cancelled run or a rejected input leaves the weights intact, and keeping
    them loaded spares the next edit a full model load.
    """
    if not isinstance(error, (JobCancelled, ValueError)):
        provider.unload()


def _alternative_seed(seed: int | None, index: int) -> int | None:
    if seed is None:
        return None
    return (seed + index) % (2**32)


def _save_bytes(content: bytes, directory: Path, prefix: str, filename: str | None) -> Path:
    suffix = Path(filename or "").suffix.lower()
    if suffix not in {".png", ".jpg", ".jpeg", ".webp"}:
        suffix = ".png"
    path = directory / f"{prefix}-{uuid4().hex}{suffix}"
    path.write_bytes(content)
    return path


def _new_image_path(directory: Path, prefix: str) -> Path:
    return directory / f"{prefix}-{uuid4().hex}.png"
