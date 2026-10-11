from __future__ import annotations

import gc
import math
from collections import OrderedDict
import random
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Sequence

from PIL import Image, ImageOps

from app.compositing import FAST_PNG
from app.config import settings
from app.jobs import JobCancelled, JobControl
from app.models.base import ModelReference
from app.runtime import empty_cache, require_generation_device, selected_device, selected_dtype, torch_module
from app.schemas import EditRequest, InpaintForm, LoadModelRequest, ModelInfo, OutpaintForm, RuntimeMode, TextToImageRequest

if TYPE_CHECKING:
    from app.models.klein_lora import KleinLora


# Resident on the GPU, in unified memory on Apple silicon, or offloaded to RAM between uses.
Placement = Literal["cuda", "mps", "cpu", "offload"]
PipelineTask = Literal["text-to-image", "inpaint"]

# Several references share a token budget: two at this size cost what one
# 1024 px reference does, and kept both identities in the local trial.
MULTI_REFERENCE_AREA = 512 * 704
REFERENCE_BACKGROUND = (232, 232, 232)
SHARED_COMPONENTS = ("scheduler", "vae", "text_encoder", "tokenizer", "transformer")
PROMPT_CACHE_SIZE = 8
GENERIC_REFERENCE_NAME = re.compile(r"^Reference \d+$")
# Klein's native resolution; whole-image edits render here and are scaled back.
EDIT_AREA = 1024 * 1024
# A face crop needs less: Fix faces scales the result back to the crop anyway.
FACE_AREA = 768 * 768
# The detail-enhancer LoRA's trigger, and its author's full prompt.
ENHANCE_TRIGGER = "Enhance this image to high resolution with rich fine details."
ENHANCE_PROMPT = (ENHANCE_TRIGGER + " Sharpen textures, add realistic micro-details, improve clarity and definition."
                  " Keep the same composition, subjects, colours and lighting.")
MATCH_SURROUNDINGS = " Match the surrounding lighting, color palette, perspective, texture and depth of field."


def _inpaint_prompt(request: InpaintForm, references: Sequence[str] = ()) -> str:
    if isinstance(request, OutpaintForm):
        # The masked border is neutral grey padding around the original picture.
        instruction = (f"Extend the picture outward into the masked border with {request.prompt.strip().rstrip('.')}."
                       " Continue the existing scene seamlessly: lines, horizon, objects and textures that reach the edge"
                       " carry on naturally, with no frame, border or visible seam.")
    elif request.edit_operation == "sketch":
        # In the local trial this followed a crude deer and top hat closely,
        # with the sketched crop also given as the reference image.
        instruction = (f"Turn the rough coloured sketch in the masked area into {request.prompt.strip().rstrip('.')}."
                       " Follow the sketch's shape, position, size, pose and colours, but render it as a realistic,"
                       " detailed part of the picture.")
    elif request.edit_operation == "remove":
        instruction = "Remove the selected subject completely and reconstruct the background naturally inside the masked region. " + request.prompt.strip()
    else:
        instruction = request.prompt.strip().rstrip(".") + "."
        if len(references) == 1:
            instruction += (" The subject is the person or object from the reference image: keep its face, hair, clothing,"
                            " colours and proportions identical to the reference, and show it complete within the selection.")
        elif references:
            named = [f"{name} is reference image {index}" for index, name in enumerate(references, 1) if not GENERIC_REFERENCE_NAME.match(name)]
            instruction += f" The subjects come from the {len(references)} reference images"
            instruction += (": " + ", ".join(named) + ".") if named else "."
            instruction += (" Keep each one's face, hair, clothing, colours and proportions identical to their reference,"
                            " and show them all complete within the selection.")
    instruction += MATCH_SURROUNDINGS
    if request.scene_context.strip():
        instruction += " Scene context: " + request.scene_context.strip()
    return instruction


def _generation_prompt(prompt: str, references: Sequence[str] = ()) -> str:
    """Names each reference by its position, as the pipeline orders them."""
    instruction = prompt.strip()
    if not references:
        return instruction
    instruction = instruction.rstrip(".") + "."
    if len(references) == 1:
        subject = "The subject" if GENERIC_REFERENCE_NAME.match(references[0]) else references[0]
        return instruction + (f" {subject} is the person or object from the reference image: keep its face, hair,"
                              " accessories, clothing, colours and proportions identical to the reference.")
    named = [f"{name} is reference image {index}" for index, name in enumerate(references, 1) if not GENERIC_REFERENCE_NAME.match(name)]
    instruction += f" The subjects come from the {len(references)} reference images"
    instruction += (": " + ", ".join(named) + ".") if named else "."
    # "Accessories" rather than a list of items: naming glasses moved one
    # person's glasses onto the other in the two-person trial.
    return instruction + " Keep each one's face, hair, accessories, clothing and proportions identical to their own reference image."


def _edit_prompt(prompt: str, references: Sequence[str] = ()) -> str:
    """The picture is image 1; references follow it as images 2 and up."""
    instruction = "Edit image 1: " + prompt.strip().rstrip(".") + "."
    if len(references) == 1:
        subject = "The subject" if GENERIC_REFERENCE_NAME.match(references[0]) else references[0]
        instruction += (f" {subject} is the person or object from image 2: keep its face, hair, accessories, clothing,"
                        " colours and proportions identical to image 2.")
    elif references:
        named = [f"{name} is image {index}" for index, name in enumerate(references, 2) if not GENERIC_REFERENCE_NAME.match(name)]
        instruction += f" Images 2 to {len(references) + 1} show the people or objects to bring in"
        instruction += (": " + ", ".join(named) + ".") if named else "."
        instruction += " Keep each one's face, hair, accessories, clothing and proportions identical to their own image."
    return instruction + " Keep everything the instruction does not change as it is in image 1, including each person's face and identity."


def _sketch_prompt(prompt: str) -> str:
    """Image 1 is the picture with rough strokes on it, or a sketch on a blank canvas."""
    return ("Image 1 contains a rough coloured sketch. Turn it into a finished, detailed image: "
            + prompt.strip().rstrip(".") + ". Follow the sketch's layout, shapes, sizes, poses and colours closely."
            " Keep every part of image 1 that is not sketched as it is.")


def _face_prompt(prompt: str, references: Sequence[str] = ()) -> str:
    """Image 1 is a close crop around one face; references follow as images 2 and up."""
    # Naming teeth made Klein open closed mouths into a grin, so the list leaves them out.
    instruction = ("Restore image 1, a close-up of a face, as a sharp and detailed photograph: clear eyes, natural skin texture,"
                   " crisp hair and eyebrows. Keep the same identity, age, expression, head pose, gaze, lighting,"
                   " colours, framing and background exactly.")
    if prompt.strip():
        instruction += " Also: " + prompt.strip().rstrip(".") + "."
    if len(references) == 1:
        name = "This person" if GENERIC_REFERENCE_NAME.match(references[0]) else references[0]
        instruction += f" {name} is the person in image 2: make the face match image 2."
    elif references:
        instruction += f" The person is one of the people in images 2 to {len(references) + 1}: make the face match their image."
    return instruction


def _enhance_prompt(prompt: str) -> str:
    """The detail-enhancer LoRA's own prompt; the user's words only add to it."""
    instruction = ENHANCE_PROMPT
    if prompt.strip():
        instruction += " Also: " + prompt.strip().rstrip(".") + "."
    return instruction


def _mode_prompt(request: EditRequest, references: Sequence[str]) -> str:
    if request.mode == "sketch":
        return _sketch_prompt(request.prompt)
    if request.mode == "enhance":
        return _enhance_prompt(request.prompt)
    if request.mode == "face":
        # The LoRA was trained on its trigger sentence, so it leads.
        return (ENHANCE_TRIGGER + " " if request.detail else "") + _face_prompt(request.prompt, references)
    return _edit_prompt(request.prompt, references)


def _fit_area(image: Image.Image, area: int = EDIT_AREA) -> Image.Image:
    """Scales to about ``area`` pixels with sides in multiples of 16, keeping the aspect ratio."""
    width = max(16, round(math.sqrt(area * image.width / image.height) / 16) * 16)
    height = max(16, round(area / width / 16) * 16)
    return image.resize((width, height), Image.Resampling.LANCZOS)


def _opaque(picture: Image.Image) -> Image.Image:
    """Flattens transparency onto neutral grey; a cut-out layer would otherwise sit on black."""
    if picture.mode not in ("RGBA", "LA", "PA") and "transparency" not in picture.info:
        return picture.convert("RGB")
    rgba = picture.convert("RGBA")
    opaque = Image.new("RGB", rgba.size, REFERENCE_BACKGROUND)
    opaque.paste(rgba, mask=rgba.getchannel("A"))
    return opaque


def _reference_image(path: Path) -> Image.Image:
    with Image.open(path) as reference:
        image = reference.convert("RGB")
    # The pipeline caps references near one megapixel anyway; shrinking once
    # here keeps large phone photos cheap to decode for every proposal.
    image.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
    return image


def _reference_images(paths: Sequence[Path]) -> list[Image.Image]:
    """One reference keeps its own size; several are padded to one shared size.

    The pipeline crops every reference to the first one's size, so differing
    aspect ratios would cut people off without the shared padding.
    """
    images = [_reference_image(path) for path in paths]
    if len(images) < 2:
        return images
    aspect = sum(image.width / image.height for image in images) / len(images)
    width = max(16, round(math.sqrt(MULTI_REFERENCE_AREA * aspect) / 16) * 16)
    height = max(16, round(MULTI_REFERENCE_AREA / width / 16) * 16)
    return [ImageOps.pad(image, (width, height), Image.Resampling.LANCZOS, color=REFERENCE_BACKGROUND) for image in images]


class Flux2KleinProvider:
    """Generation and mask-conditioned editing on one set of resident weights.

    The text and inpaint pipelines are thin wrappers around the same modules,
    so switching between them costs neither a reload nor extra memory.
    """

    name = "flux2-klein-4b"
    display_name = "FLUX.2 Klein 4B (BF16)"
    model_id = "black-forest-labs/FLUX.2-klein-4B"
    # The Consistency and detail-enhancer LoRAs were trained on the distilled 9B.
    supports_edit_adapters = False

    def __init__(self) -> None:
        self._pipes: dict[PipelineTask, Any] = {}
        self._placement: Placement | None = None
        self._torch: Any | None = None
        self._prompt_cache: OrderedDict[str, Any] = OrderedDict()
        self._consistency_adapter: KleinLora | None = None
        self._detail_adapter: KleinLora | None = None

    def info(self) -> ModelInfo:
        return ModelInfo(
            name=self.name,
            display_name=self.display_name,
            model_id=self.model_id,
            capabilities=["text-to-image", "edit", "inpaint", "outpaint"],
            loaded=bool(self._pipes),
            loaded_pipelines=[f"{task}:{self._placement}" for task in self._pipes],
        )

    def generate(
        self,
        request: TextToImageRequest,
        output_path: Path,
        control: JobControl,
        references: Sequence[ModelReference] = (),
    ) -> tuple[int, int]:
        control.raise_if_cancelled()
        require_generation_device()
        torch = self._load_torch()
        pipe = self._load_pipe(self._placement_for(request.runtime_mode), "text-to-image", control)
        self._configure_edit_adapters(pipe, False, control)
        reference_images = _reference_images([reference.path for reference in references])
        control.set_progress(0.15)
        seed = request.seed if request.seed is not None else random.randrange(0, 2**32)
        generator = torch.Generator(selected_device(torch)).manual_seed(seed)
        prompt_embeds = self._encode_prompt(pipe, _generation_prompt(request.prompt, [reference.name for reference in references]), control)
        control.set_phase("generating")
        image = pipe(
            # Each reference becomes its own block of image tokens.
            **({"image": reference_images} if reference_images else {}),
            prompt_embeds=prompt_embeds,
            height=request.height,
            width=request.width,
            num_inference_steps=request.steps,
            guidance_scale=request.guidance,
            generator=generator,
            callback_on_step_end=control.diffusion_callback(request.steps),
        ).images[0]
        control.set_progress(0.95)
        image.save(output_path, **FAST_PNG)
        return seed, image.width

    def edit(
        self,
        request: EditRequest,
        input_path: Path,
        output_path: Path,
        control: JobControl,
        references: Sequence[ModelReference] = (),
    ) -> tuple[int, int]:
        """Redraws the whole picture by instruction and saves it at the picture's own size."""
        control.raise_if_cancelled()
        require_generation_device()
        torch = self._load_torch()
        partial = request.strength < 1
        # A partial redraw needs the source latent, which only the inpaint pipeline noises.
        pipe = self._load_pipe(self._placement_for(request.runtime_mode), "inpaint" if partial else "text-to-image", control)
        self._configure_edit_adapters(pipe, request.consistency_lora, control, request.detail)
        with Image.open(input_path) as picture:
            source = _opaque(picture)
        fitted = _fit_area(source, FACE_AREA if request.mode == "face" else EDIT_AREA)
        if request.mode in ("sketch", "enhance"):
            references = ()
        reference_images = _reference_images([reference.path for reference in references])
        control.set_progress(0.15)
        seed = request.seed if request.seed is not None else random.randrange(0, 2**32)
        generator = torch.Generator(selected_device(torch)).manual_seed(seed)
        prompt_embeds = self._encode_prompt(pipe, _mode_prompt(request, [reference.name for reference in references]), control)
        control.set_phase("generating")
        if partial:
            # A full mask repaints everything. The pipeline still passes the
            # picture as the first reference, so references stay images 2 and up.
            reference: Any = reference_images if len(reference_images) > 1 else reference_images[0] if reference_images else None
            image = pipe(
                image=fitted,
                mask_image=Image.new("L", fitted.size, 255),
                **({"image_reference": reference} if reference is not None else {}),
                prompt_embeds=prompt_embeds,
                height=fitted.height,
                width=fitted.width,
                num_inference_steps=request.steps,
                guidance_scale=request.guidance,
                strength=request.strength,
                generator=generator,
                callback_on_step_end=control.diffusion_callback(request.steps),
            ).images[0]
        else:
            image = pipe(
                # The picture leads, so the output follows its layout; references follow it.
                image=[fitted, *reference_images],
                prompt_embeds=prompt_embeds,
                height=fitted.height,
                width=fitted.width,
                num_inference_steps=request.steps,
                guidance_scale=request.guidance,
                generator=generator,
                callback_on_step_end=control.diffusion_callback(request.steps),
            ).images[0]
        control.set_progress(0.95)
        if image.size != source.size:
            image = image.resize(source.size, Image.Resampling.LANCZOS)
        image.save(output_path, **FAST_PNG)
        return seed, image.width

    def inpaint(
        self,
        request: InpaintForm,
        input_path: Path,
        mask_path: Path,
        output_path: Path,
        control: JobControl,
        references: Sequence[ModelReference] = (),
    ) -> tuple[int, int]:
        control.raise_if_cancelled()
        require_generation_device()
        torch = self._load_torch()
        pipe = self._load_pipe(self._placement_for(request.runtime_mode), "inpaint", control)
        self._configure_edit_adapters(pipe, request.consistency_lora, control)
        if request.edit_operation in ("remove", "sketch"):
            references = ()
        reference_images = _reference_images([reference.path for reference in references])
        seed = request.seed if request.seed is not None else random.randrange(0, 2**32)
        generator = torch.Generator(selected_device(torch)).manual_seed(seed)
        with Image.open(input_path) as source:
            init_image = source.convert("RGB")
        with Image.open(mask_path) as selection:
            mask_image = selection.convert("L")
        control.set_progress(0.15)
        prompt_embeds = self._encode_prompt(pipe, _inpaint_prompt(request, [reference.name for reference in references]), control)
        control.set_phase("generating")
        # The sketched crop doubles as the reference, so no stroke survives the edit.
        reference: Any = init_image if request.edit_operation == "sketch" else (
            reference_images if len(reference_images) > 1 else reference_images[0] if reference_images else None)
        image = pipe(
            image=init_image,
            mask_image=mask_image,
            # A list of equally sized images becomes one reference token block each.
            **({"image_reference": reference} if reference is not None else {}),
            prompt_embeds=prompt_embeds,
            height=init_image.height,
            width=init_image.width,
            num_inference_steps=request.steps,
            guidance_scale=request.guidance,
            strength=request.strength,
            generator=generator,
            callback_on_step_end=control.diffusion_callback(request.steps),
        ).images[0]
        control.set_progress(0.95)
        image.save(output_path, **FAST_PNG)
        return seed, image.width

    def preload(self, request: LoadModelRequest, control: JobControl) -> None:
        try:
            control.raise_if_cancelled()
            require_generation_device()
            placement = self._placement_for(request.runtime_mode)
            pipe = self._load_pipe(placement, request.task, control)
            self._configure_edit_adapters(pipe, False, control)
            control.set_progress(0.6)
            if request.warm_up:
                self._warm_up(pipe, request.task, control)
                # The other task shares the weights, so having it ready is free.
                self._load_pipe(placement, "text-to-image" if request.task == "inpaint" else "inpaint", control)
            control.set_progress(0.95)
        except JobCancelled:
            self.unload()
            raise

    def unload(self) -> None:
        self._consistency_adapter = None
        self._detail_adapter = None
        self._pipes = {}
        self._placement = None
        self._prompt_cache.clear()
        self._collect()

    def _configure_edit_adapters(self, pipe: Any, enabled: bool, control: JobControl, detail: float = 0.0) -> None:
        """Consistency and the detail enhancer patch the same layers, so one is attached at a time.

        Each loads on first use and stays attached at scale zero until the other one is needed.
        """
        if (enabled or detail) and not self.supports_edit_adapters:
            raise ValueError("Consistency and Enhance require the distilled Quality 9B model.")
        if enabled and detail:
            raise ValueError("Consistency and the detail enhancer cannot be combined.")
        if (enabled and self._consistency_adapter is None) or (detail and self._detail_adapter is None):
            from app.models.klein_lora import CONSISTENCY_V2, DETAIL_ENHANCER, KleinLora, resolve_checkpoint
            path = resolve_checkpoint(CONSISTENCY_V2 if enabled else DETAIL_ENHANCER, control)
            other = self._detail_adapter if enabled else self._consistency_adapter
            if other is not None:
                other.detach()
                self._consistency_adapter = self._detail_adapter = None
                self._collect()
            adapter = KleinLora.attach(pipe.transformer, path, control.raise_if_cancelled)
            if enabled:
                self._consistency_adapter = adapter
            else:
                self._detail_adapter = adapter
        if self._consistency_adapter is not None:
            self._consistency_adapter.set_scale(0.6 if enabled else 0)
        if self._detail_adapter is not None:
            self._detail_adapter.set_scale(detail)

    def outpaint(self, request: OutpaintForm, input_path: Path, output_path: Path,
                 control: JobControl) -> tuple[int, int]:
        raise ValueError("Specialized outpaint requires the Fast NVFP4 model.")

    def _encode_prompt(self, pipe: Any, prompt: str, control: JobControl | None = None) -> Any:
        """Encode each prompt once.

        Under CPU offload every encode moves the 8 GB text encoder onto the GPU
        and back. Proposals and retries share their prompt, so they skip that.
        """
        embeds = self._prompt_cache.get(prompt)
        if embeds is None:
            if control is not None:
                control.set_phase("preparing")
            # This call runs outside Diffusers' no-grad pipeline __call__.
            # Caching its autograd graph retains encoder activations per prompt
            # and can push subsequent edits into shared GPU memory on Windows.
            with self._load_torch().inference_mode():
                embeds = pipe.encode_prompt(prompt=prompt, device=pipe._execution_device)[0]
            # The encoder's weights pass through the GPU (about 5 GiB for 9B).
            # PyTorch keeps those freed blocks reserved, and Windows does not
            # refuse the denoiser's next allocations but moves them to system
            # RAM, so an edit then runs 20 times slower. Hand them back now.
            self._collect()
            self._prompt_cache[prompt] = embeds
            while len(self._prompt_cache) > PROMPT_CACHE_SIZE:
                self._prompt_cache.popitem(last=False)
        else:
            self._prompt_cache.move_to_end(prompt)
        return embeds

    def _warm_up(self, pipe: Any, task: PipelineTask, control: JobControl) -> None:
        """One tiny step compiles the CUDA kernels the first real edit would wait for (about 3 s)."""
        control.set_phase("warming", self.display_name)
        torch = self._load_torch()
        generator = torch.Generator(selected_device(torch)).manual_seed(0)
        canvas = Image.new("RGB", (256, 256), REFERENCE_BACKGROUND)
        extra = {"image": canvas, "mask_image": Image.new("L", canvas.size, 255), "strength": 1.0} if task == "inpaint" else {}
        pipe(prompt="a quiet garden", height=256, width=256, num_inference_steps=1, guidance_scale=1.0, generator=generator, **extra)

    def _load_pipe(self, placement: Placement, task: PipelineTask, control: JobControl) -> Any:
        control.raise_if_cancelled()
        if self._pipes and self._placement != placement:
            self.unload()
        if task in self._pipes:
            return self._pipes[task]
        shared = next(iter(self._pipes.values()), None)
        if shared is None:
            control.set_phase("loading", self.display_name)
        pipe = None
        try:
            pipe = self._create_pipeline(task, control) if shared is None else self._derive_pipeline(shared, task)
            control.raise_if_cancelled()
            if task == "inpaint":
                # Keep mask resampling coverage; final user feathering is
                # applied by the compositor, independently of generation.
                pipe.mask_processor.register_to_config(do_binarize=False)
            self._place_pipeline(pipe, placement)
            control.raise_if_cancelled()
        except BaseException:
            pipe = None
            if shared is None:
                self.unload()
            raise
        # Publish only after loading, placement and cancellation checks.
        self._pipes[task] = pipe
        self._placement = placement
        return pipe

    def _derive_pipeline(self, shared: Any, task: PipelineTask) -> Any:
        from diffusers import Flux2KleinInpaintPipeline, Flux2KleinPipeline

        pipeline_class = Flux2KleinInpaintPipeline if task == "inpaint" else Flux2KleinPipeline
        components = {name: getattr(shared, name) for name in SHARED_COMPONENTS}
        # Direct construction: from_pipe would cast the packed NVFP4 scales.
        return pipeline_class(**components, is_distilled=shared.config.get("is_distilled", False))

    def _create_pipeline(self, task: PipelineTask, control: JobControl) -> Any:
        from diffusers import Flux2KleinInpaintPipeline, Flux2KleinPipeline

        control.raise_if_cancelled()
        pipeline_class = Flux2KleinInpaintPipeline if task == "inpaint" else Flux2KleinPipeline
        return pipeline_class.from_pretrained(
            self.model_id,
            torch_dtype=selected_dtype(),
            low_cpu_mem_usage=True,
            token=settings.huggingface_token,
            use_safetensors=True,
        )

    def _load_torch(self) -> Any:
        if self._torch is None:
            self._torch = torch_module()
        return self._torch

    def _place_pipeline(self, pipe: Any, placement: Placement) -> None:
        torch = self._load_torch()
        device = selected_device(torch)
        self._configure_memory(pipe)
        if device == "cuda":
            if placement == "offload" and hasattr(pipe, "enable_model_cpu_offload"):
                pipe.enable_model_cpu_offload()
            else:
                pipe.to("cuda")
        else:
            pipe.to(device)

    def _configure_memory(self, pipe: Any) -> None:
        vae = getattr(pipe, "vae", None)
        if vae is not None and hasattr(vae, "enable_slicing"):
            vae.enable_slicing()
        if vae is not None and hasattr(vae, "enable_tiling"):
            vae.enable_tiling()
        transformer = getattr(pipe, "transformer", None)
        if transformer is not None and hasattr(transformer, "set_attention_backend"):
            if settings.attention_backend in {"flash", "_flash_3"}:
                transformer.set_attention_backend(settings.attention_backend)

    def _placement_for(self, runtime_mode: RuntimeMode) -> Placement:
        if settings.apple_silicon:
            # CPU and GPU share one memory, so offloading would only copy weights around.
            return "mps"
        if runtime_mode == "performance" or not settings.enable_cpu_offload:
            return "cuda"
        # The official 4B recipe recommends model offload. It keeps the 16 GB
        # card below its limit while the transformer still does its work on CUDA.
        return "offload"

    def _collect(self) -> None:
        gc.collect()
        if self._torch is not None:
            empty_cache(self._torch)
