# Quality, speed and UX sweep, 10 October 2026

Status: Historical
Last verified: 2026-10-11
Current state: [Guide](../guide.md) and [roadmap](../roadmap.md).

**Recorded scope:** Implemented and measured on the default profile (RTX 5070 Ti 16 GB, Quality = Klein 9B NVFP4, four steps). Timings are end-to-end API times from upload to finished job, with the model already loaded. Visual review used fixed seeds and side-by-side crops; outputs stayed in a local scratch folder and are not part of the repository.

## What changed

| Area | Change | Why |
|---|---|---|
| Inpaint edges | Generation mask grows one 16 px token after resizing; the final blend keeps the visible selection | Klein blends the original back per token, sampled at the token centre, so tokens cut by the edge kept old colour (the blue rim on a recoloured shirt) |
| Inpaint edges | The tone offset in that hidden ring fades into the selection, with a 32-level RMS tolerance | A removed object's shadow outside the selection left a hard rectangle; the tolerance stops leftovers of the old object (or a different object) from being smeared in as a ghost |
| Detail | Grain matching: noise lost to the model is measured outside the selection (Immerkær operator, robust median) and added back as seeded monochrome noise, at most 5 levels | New content looked smoother than the photo around it, which reads as plastic |
| Detail | Model size uses an area budget (about 1024², long side at most 2048) and renders crops below 768 × 768 at that size | Klein draws flat detail at low resolution, and wide crops lost a third of their pixels under the old long-side cap |
| Selection edges | Auto Expand = longest side / 200 (4–24 px), auto Soften = longest side / 150 (6–32 px); double-click Expand for automatic | Expand 4 with a 31 px outward ramp removed the rim but smeared the old colour; Expand 6 with Soften 8 was crisp with no rim |
| Speed | Array mask morphology replaces repeated 3 × 3 Pillow filters (identical output, 7–130× faster); intermediate PNGs use compression level 1 | Mask preparation and file writes were measurable per edit |
| Speed | The model stays loaded after a cancel or a rejected request; only real failures unload it | Reloading 9B costs about 24 s |
| Speed | Job polling every 250 ms instead of 800 ms; a finished result shows before history and model status refresh | Up to about half a second less waiting after every job |
| UX | An opened picture starts in Edit with the Object tool; clicking an object switches to Replace | Typing "make it a snowy night" right after opening used to do nothing, because Replace needs a selection |

## Timings

| Case | Before | After | Note |
|---|---:|---:|---|
| Generate 1024² | 3.0 s | 3.0 s | |
| Generate 1024², two proposals | 5.6 s | 5.9 s | |
| Edit whole picture | 7.5 s | 7.5–8.2 s | Repeat runs vary by about half a second |
| Replace (640² picture) | 4.0 s | 4.4 s | Small crop now rendered at 768² |
| Remove a small object | 2.6–2.8 s | 4.0 s | Same: more detail, about a second more |
| Replace on a 2048² picture | 6.0 s | 8.5–8.9 s | Crop now renders at 1152 × 896 instead of 1024 × 784 (+28% pixels) |
| Expand left and right | 4.6–4.8 s | 4.8 s | |
| Expand four sides | 5.8 s | 5.6 s | |
| Expand left and right, 2K | 8.8 s | 10.0 s | Area budget |

The slower cases trade time for resolution. Lowering `MOSIMAGE_INPAINT_MINIMUM_DIMENSION` (0 disables the upscale) or `MOSIMAGE_INPAINT_MAX_DIMENSION` restores the old speed.

## Visual results

- **Shirt recolour:** the blue rim disappears with about 8 px more selection; the hidden token alone did not remove it because the fixture mask stops inside the shirt. The new automatic edges give the same result in the app.
- **Expansion:** seams on the garden and photo cases stay invisible. Grain now continues across the seam, where the new area used to be visibly smoother.
- **Remove with a tight selection:** the hard vertical edge beside a shoe's shadow becomes a gradual transition. The rectangle stays faintly visible because the model paints clean floor where the shadow used to be; selecting the shadow too still gives the best result.
- **Texture instruction rejected:** adding "keep the same fine detail, sharpness and grain" to the inpaint prompt changed little on the sweater and painted gritty texture on a defocused floor, making the selection outline more visible. It was reverted.

## ComfyUI community check

The question was whether ComfyUI users have tricks for Klein seams and the plastic look that MosImage lacks.

- **Inpaint conditioning:** Klein workflows pass the source as `ReferenceLatent` and mask the latent with `Set Latent Noise Mask` at full denoise. Diffusers' `Flux2KleinInpaintPipeline` already does this: the encoded source is always the first reference. Nothing to port.
- **Crop and stitch:** context crop, mask grow/blur and colour matching are already in place (see the [7 October note](2026-10-07-inpainting.md#comfyui-community-leads-checked-7-october-2026)). The "4K inpainting" trick renders the crop larger for detail, which the new minimum size and area budget do within the speed target. ([MyAIForce](https://myaiforce.com/flux-2-klein-inpaiting/), [CropAndStitch](https://github.com/lquesada/ComfyUI-Inpaint-CropAndStitch))
- **Plastic faces:** a popular 9B workflow fixes plastic faces with [Elusarca's Detail Enhancer for Klein 9B](https://civitai.com/models/2442399?modelVersionId=2746136) at 0.7 ([workflow](https://civitai.com/models/2543188/flux2-klein-9b-ultimate-6-in-1-workflow-face-swap-inpaint-auto-mask-nag-refine-upscale-8gb-vram)). It is an edit LoRA (trigger "Enhance this image", 632 MB BF16, FLUX.1 [dev] non-commercial licence) meant for a refinement pass, so it would roughly double an edit's time. Tested below.
- **CFG advice does not apply:** the realism guide's "lower CFG to 3.5" targets the undistilled model; distilled Klein runs at guidance 1 and degrades past about six steps. ([Civitai guide](https://civitai.com/articles/27526/flux2-klein-9b-realistic-enhanced-details-lora), [Diffusion Doodles](https://diffusiondoodles.substack.com/p/flux2-klein-shrinking-flux2-dev))
- **Negative prompts:** NAG (normalized attention guidance) gives distilled models negative prompts such as "plastic skin, over-smoothed". It needs a custom attention processor and costs extra attention per step; untested.
- **LanPaint:** still the main sampler lead for seams, at about three times the model calls with two thinking iterations. It is listed in the 7 October note and remains untested.
- **Latent compositing research:** [PELC/DecFormer](https://arxiv.org/abs/2512.05198) targets exactly the latent-mask seam, but needs a trained network and has no released code.

## Detail-enhancer LoRA A/B

Elusarca's Detail Enhancer v2 for Klein 9B ([Hugging Face](https://huggingface.co/reverentelusarca/detail-enhancer-flux-klein-9b), revision `82dd21d`, SHA-256 `520832b4…f9072`, identical to the Civitai file) is a rank-128 BF16 LoRA on the double and single blocks. `KleinLora` attaches it to the NVFP4 transformer unchanged. The test redrew whole pictures with the author's "Enhance this image…" prompt (four steps, seed 7) on three portraits, a kitchen and a garden, at scales 0, 0.4, 0.7 and 1.0. "Detail" below is the RMS of a Laplacian filter, so it counts noise as well as texture.

- **Cost:** attaching takes 0.9 s and 0.65 GB VRAM. An active pass takes 7.8–8.4 s instead of 7.1–7.5 s at 1 MP. Peak allocation was 11.6 GB.
- **Prompt alone over-processes:** without the LoRA, faces became up to 40% more saturated and orange, and detail rose 40–160%, mostly as invented noise: grain on a smooth mug and an orange-peel surface on a terracotta ball.
- **0.4 looks best:** colours stay close to the original (the lowest drift from the source in four of five pictures), and the picture gains believable texture (pores, freckles, knit, stone) with 18–71% more detail.
- **0.7 and above reinvent:** one portrait's face shape changed, and the marble got a new vein pattern. At 1.0, drift from the source is 1.5–1.8 times the 0.4 value.
- **Not for generation:** used during text-to-image at 0.5, it gave finer skin but a different person.
- **Resolution:** the result is limited to the 1 MP edit size, so a whole-picture enhance would lose resolution on a large photo. It suits generated pictures and crops (Fix faces, a selection).
- **Stability:** the first run coincided with an NVIDIA driver timeout and bugcheck 0xD1 on the test machine. The same bugcheck occurred on 7 October during plain Fast inpainting without any LoRA. A rerun that checked the driver event log after every pass completed all 20 passes without errors, at 150–170 W and below 65 °C.

The LoRA is under the FLUX.1 [dev] non-commercial licence, like the 9B weights.

## Fix faces changed expressions

A report: Fix faces sometimes turned a neutral face into a smiling one. A trial on four faces from two pictures (95–420 px, seeds 1 and 2) compared the old full redraw with partial redraws at strengths 0.5–0.75, with and without the detail LoRA.

- **Cause:** the built-in prompt asked for "natural teeth". Klein read that as "show teeth" and opened closed mouths into a grin, in every variant. Without the word, neutral faces stayed neutral, so the prompt now names skin, hair and eyebrows only.
- **Small faces also lost their light:** a full redraw of a 95 px face under blue light came back evenly lit. It has to invent everything the blurry crop lacks.
- **Faithful (new default):** noise added to the crop and only the later steps run (steps × 2 at strength 0.625, five model passes from noise level 0.93), with the detail LoRA at 0.4 on Quality. Expression, light and identity stayed on all four faces; small faces gained less sharpness than with a full redraw. About 4.5 s per face instead of 3 s.
- **Redraw** keeps the old full redraw as an option for faces too blurry to sharpen.
- On large faces both modes looked alike.

## Enhance

The open question from the A/B became an Enhance command: the detail LoRA on the whole picture or a selection, as a partial redraw at strength 0.625 with the original's broad tones restored under the new detail. Pictures beyond 1.1 MP render in up to nine overlapping tiles of at most 1024², so a large photo keeps its resolution. Subtle uses 0.4 and Strong 0.7.

The first real run spilled into system memory at 29 s per step instead of about 1 s. Each new prompt copies the streamed text encoder (about 5 GiB) onto the GPU. PyTorch keeps those freed blocks reserved, and Windows' sysmem fallback never makes an allocation fail, so the allocator never hands them back. With the LoRA's extra 0.65 GB, the denoiser no longer fit. Prompt encoding now empties the CUDA cache right after a new prompt. This affects every Quality job, not only Enhance; edits without the LoRA were simply small enough to fit.

The rerun went through the service code with PyTorch capped at 12.5 GiB, so it could not spill. It compared the partial redraw (eight steps at 0.625, five passes) with a full redraw (four steps), Subtle (0.4), seed 7. "Drift" is the mean difference of the blurred region from the source in 8-bit levels, and "detail" is the ratio of Laplacian RMS to the source.

| Case | Parts | Partial: time · drift · detail | Full: time · drift · detail |
|---|---:|---|---|
| Portrait 1254² | 2 | 20 s · 2.6 · 1.44× | 17 s · 5.4 · 1.53× |
| Kitchen 1254² | 2 | 20 s · 2.1 · 1.27× | 17 s · 4.5 · 1.81× |
| 2000 × 2400 selection on a 3000 × 4000 photo | 8 | 82 s · 3.2 · 1.36× | 67 s · 4.3 · 1.72× |

- **Memory:** 8.8 GB peak during denoising and 11.6 GB while the first prompt was encoded, with no driver events.
- **Partial redraw stays:** both modes sharpen denim, skin and a heatsink's coating. The full redraw doubles the drift, changes a pin's icon and the marble veins, and leaves double tick marks on a tape measure where tiles overlap, because neighbouring tiles invent different detail. Partial tiles agree well enough to blend without visible seams.
- **Time:** about ten seconds per part. The first job adds model loading (about 24 s) and the LoRA download on first use.

## Open

- Try NAG with a fixed anti-plastic negative prompt, measuring the extra time.
- A one-click "include shadow" control for Remove (grow the selection downward) would address the remaining rectangle.
