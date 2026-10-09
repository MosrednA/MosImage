# Regional inpainting and blending

**Status:** Core editing path implemented locally, 7 October 2026; visual reliability remains experimental. Current behavior for the English Focus editor; linked from [README](../../README.md), [Focus validation](2026-10-07-focus-implementation.md) and the [v2 roadmap](../roadmap.md).

## Root cause and correction

The previous Klein provider discarded `mask_path`, used ordinary image editing on the whole crop and clipped the result afterward. A generated mushroom could extend beyond the selection and be cut off, while recolored scenery remained inside the unused part of the mask. Exact protected pixels alone did not demonstrate correct masked generation.

The installed stable Diffusers 0.41.0 already provides `Flux2KleinInpaintPipeline`. MosImage now passes the real generation mask and edit strength to that pipeline. Native latent masking preserves surrounding context during sampling; the final native-resolution compositor independently enforces the user's pixel boundary. Task switches release the previous large pipeline, retaining BF16, TF32 and CPU offload on the RTX 5070 Ti profile. No additional dependencies or checkpoints were installed.

This native API is not the same as an inpaint-trained checkpoint. Inspection of the installed pipeline shows that the Klein transformer receives generated/source image latents and text; its mask blends source and generated latents after each scheduler step. The Fill pipeline instead passes masked image latents and the mask as additional transformer input. Klein's current route enforces a spatial editing boundary but does not give its general editing checkpoint the same explicit mask conditioning as a dedicated Fill model.

The [8 October follow-up](../../benchmarks/INPAINT_QUALITY_2026-10-08.md) compares mask shape, edge width, context, reference resolution, layout conditioning and object references. It supplements the historical experiments below; application defaults have not been changed by the benchmark.

## Context, selection and blend

These controls have different responsibilities:

| Control | What it changes | Default |
|---|---|---|
| Painted selection | Where generated pixels may be returned | User selection |
| Expand | Grows (or shrinks) the selection before the edge ramp, shown live on the canvas | +4 native px |
| Soften | Outward smoothstep from the selection to the untouched image, shown live on the canvas | Auto: longest side / 40, 8–64 px |
| Surrounding context | How much nearby scenery the model sees around the selection's bounding rectangle | 320 native px per side |
| Scene notes | Optional lighting, perspective, material or foreground cues | Empty |

The context window is centred on the selection; where it reaches past the image, the edge is mirrored for the model only, and only the part inside the image is composited back (see the follow-up below). It is resized to a maximum model-facing edge of 1024, with dimensions divisible by 32. The canvas Context toggle shows its protected rectangle at the current zoom. Inpainting uses the exact chosen padding; no hidden proportional increase changes the displayed rectangle. A full-frame second reference is not added automatically.

The Klein generation mask preserves selection alpha and uses an inward smoothstep from zero at the painted boundary to full opacity inside. Its native mask processor is configured with `do_binarize=False`; otherwise the vendor default discards the gradual transition. The pipeline resizes mask values into latent space and mixes source and generated latents at each denoising step. Original scenery therefore contributes progressively near the edge during generation, rather than only being pasted back afterward. Other providers continue to receive the binary selected footprint. The editor now sends its final soft mask with `blend_feather=0`, so Klein receives exactly the alpha shown on the canvas. The API's inward blend remains for other clients.

Final blending uses the same alpha at native resolution. Existing soft alpha caps any API-side inward transition instead of being multiplied twice. There is no hidden generation-mask growth: Expand and Soften are visible canvas controls. Outpainting retains its separate 16 px model overlap and exact protected original; inpaint controls do not silently change its behavior.

Select enough room for the complete object, shadow and glow. A mask describes editable space, not a guaranteed subject silhouette. Add context when nearby lighting, scale or perspective is missing; excessive context can reduce the subject's model-facing resolution. Start with a concise subject description, then add specific scene notes if needed. More feather cannot repair an already-clipped subject.

For removal, include the existing object's halo and shadow as well as its silhouette. The default +4 px Expand and outward Soften cover anti-aliased object edges; raise Expand if traces remain. The removal UI sends concise background content to Klein, while the provider adds the removal instruction once.

## Local evidence

The fixture uses the user's original 1254 × 1254 dusk scene and mask bounds `(136, 924, 372, 1186)`. Klein comparisons use 4 steps, guidance 1, strength 1 and seed 3936008243.

| Generation | Experiment | Observed result |
|---|---|---|
| 29 | Earlier instruction edit followed by clipping | Saturated red mushroom, clipped cap and conspicuous recolored area |
| 30 | Native mask with simple subject prompt | Small complete glowing mushroom; established native route |
| 31 | Native mask plus automatic full-scene reference and verbose positioning instructions | Copied the tree rather than creating the requested mushroom; rejected |
| 32 | Native mask with verbose generic subject-fitting instructions | Dark incorrect subject; rejected |
| 33 | Native mask, concise subject-first prompt, context 192 and blend 8 | Complete warmer mushroom in the original painted footprint; 1,525,188 protected pixels unchanged |
| 34 | Native Z-Image with a longer descriptive scene prompt | Mushroom cap still clipped; not promoted as the default regional fix |
| 35 | Real UI rectangle, context 256, blend 16 and explicit dusk/foreground notes | Complete softly lit mushroom with foreground flowers; 1,510,684 protected pixels unchanged |
| 36–37 | Separate cropped mushroom appearance reference, with descriptive and explicit reference prompts | Requested mushroom was not inserted reliably; upload feature removed before delivery |
| 38 | Original painted mask through the restarted native binary-mask backend | Reproduced 33, with all 1,525,188 protected pixels unchanged; user rejected the conspicuous yellow grass island and appearance mismatch |
| 39 | Real UI removal of the generated mushroom, rectangle, context 224 and blend 8 | Mushroom removed and flowers restored, with some faint boundary lighting/vegetation retained; all 1,510,684 protected pixels unchanged |
| 40–42 | Scene-specific notes, context 320/512 and 4/8 steps | Some better foreground continuity; more context or steps alone did not consistently improve integration |
| 43–49 | Generic integration instructions, subject/ground-cover variants and an extra low-strength harmonization pass | Ground-cover-specific prompts sometimes helped; generic prompts sometimes omitted the mushroom, and the second pass scarcely improved the grass island. No universal prompt or automatic extra pass was adopted |
| 50 | Native soft generation mask, concise prompt, context 320 and blend 16 | Small glowing mushroom emerged directly among continuing violet flowers; better softness and no distinct yellow grass island |
| 51 | Native soft mask with the old context 192 and blend 8 | Grass island remained. Soft masking alone is insufficient; the new defaults combine the wider context and transition |
| 52–53 | Final restarted code, omitted form controls to exercise defaults, two seeds | 52 reproduced 50 exactly, with all 1,525,188 protected pixels unchanged. Seed 3936008244 in 53 omitted the mushroom; integration improved in the successful edit but subject reliability remains unresolved |
| 54 | Real UI rectangle, default context 320 / blend 16, concise prompt and seed 3936008243 | Complete mushroom among continuing flowers, with all 1,510,422 protected pixels unchanged. Apply, Undo, Redo and Undo restored the source revision |
| 55–57 | Short addition instruction on both seeds and a descriptive subject prompt on the second seed | The action instruction omitted the subject on seed 1 and inserted it on seed 2; the descriptive prompt inserted a red mushroom on seed 2. No prompt was reliable across both seeds, so the provider was not changed to these templates |

Generations 35 and 54 were applied, undone, redone and undone again through real browser controls. Document dimensions remained 1254 × 1254 and the original revision was restored. Local diagnostic images and pixel counts are in `data/inpaint-diagnostic/`; they are generated validation artifacts rather than committed fixtures. `final-integration-compare.png` includes the previous result and both final default seeds, including the failed subject insertion; `final-soft-ui.png` shows the actual editor result.

Context visibility and padding changes were checked against canvas geometry. Settings were inspected at widths 1440, 390 and 320 without horizontal overflow; the modal keeps its heading and Done action visible while its controls scroll. The frontend build and 17 tests, plus backend syntax and 38 tests, pass.

One scene does not establish universal seamless blending or prompt compliance. In this fixture the second default seed failed to insert the subject, and prompt changes reversed which seed succeeded. Preview proposals before applying; use the existing proposal count control to compare alternatives. A broad fixed suite, repeated edits, reference identity, tiny/irregular selections, shadow boundaries and VRAM/latency measurements remain evaluation work. Automatic colour drift correction was added later; see the follow-up below. Separate appearance reference uploads are deferred after the failed local trials; the shipped image context is the padded rectangle from the current document.

## Repeated-seed reliability check

The user's follow-up questioned whether generation 50 was a lucky shot. The paired check declares eight fresh seeds before rendering: `7, 42, 1234, 20261007, 87654321, 314159265, 2718281828, 4287654321`. It holds the original source, painted mask, `Glowing mushroom` prompt, Klein 4B, 4 steps, guidance 1 and strength 1 fixed. Each seed runs with context 320 / blend 16 and context 192 / blend 8, interleaved through the API's single GPU worker. Both use the current native soft-mask pipeline, so this compares the combined context/blend settings rather than isolating binary versus soft sampling.

Visual review considers every result:

- **Promising:** a complete recognizable mushroom, continuous surrounding flowers, plausible contact/occlusion and local light, without a conspicuous separate patch or pedestal.
- **Weak:** a recognizable mushroom, but clipped geometry, a separate vegetation island, conspicuous appearance mismatch or a visible transition remains.
- **Failed subject:** the requested mushroom is absent or not recognizable.

These are qualitative judgments, not an automatic detector or a statistical claim about all images. Native dimensions and exact protected pixels are checked independently. All seeds and API records are retained; the tool never selects a winner or treats protected pixels as proof of visual quality.

All 16 renders completed (generations 58–73), at the original 1254 × 1254 dimensions, with zero changes to each image's 1,525,188 protected pixels. Visual judgments were recorded separately:

| Context / blend | Promising | Weak integration | Failed subject | Recognizable mushroom |
|---|---:|---:|---:|---:|
| 320 / 16 | 4 of 8 | 2 of 8 | 2 of 8 | 6 of 8 |
| 192 / 8 | 2 of 8 | 5 of 8 | 1 of 8 | 7 of 8 |

The broader context/transition showed more promising integration in this small paired sample, but less reliable subject insertion. It did not establish an overall reliable fix or isolate the feather's individual contribution. Retain the current controls and defaults as experimental; do not promote a specific seed, prompt template or this single scene's count as a general success rate. A stronger regional-editing route should meet both subject-compliance and integration criteria over multiple scenes before it replaces the current default.

Per-generation review:

| Seed | 320 / 16 | 192 / 8 |
|---|---|---|
| 7 | 58: promising curved mushroom and continuous flowers | 59: weak, conspicuous yellow grass patch |
| 42 | 60: failed subject, stem-like fragment without cap | 61: weak, separate vegetation island |
| 1234 | 62: weak, rigid bright stem and separate base | 63: weak, separate altered vegetation bed |
| 20261007 | 64: promising small mushroom, foreground flowers and local light | 65: weak, object at edge and yellow grass patch |
| 87654321 | 66: failed subject, changed vegetation only | 67: promising, flowers continue around the base; stylized cap remains |
| 314159265 | 68: promising violet cap among continuing flowers | 69: failed subject, changed vegetation only |
| 2718281828 | 70: promising dark cap, curved stem and local warm light | 71: promising small mushroom and foreground flowers; some altered grass |
| 4287654321 | 72: weak, distinct yellow grass mound and crisp stem | 73: weak, vivid stem and isolated grass at edge |

`data/inpaint-diagnostic/reliability-2026-10-07/results.json` retains all API jobs, seeds, controls, history records and protected-pixel checks. `review.json` retains every visual verdict and its reason. `current-all-seeds.png` and `small-context-all-seeds.png` display all eight outcomes per condition, including failures. This benchmark added no dependencies, checkpoints or changes to the running sampling code.

Run against the existing local API with a new output directory:

```powershell
.\backend\.venv\Scripts\python.exe backend\tools\benchmark_inpaint.py --source data\images\inputs\input-8312607659cd446c9486f0ad90fa1917.png --mask data\images\masks\mask-807c3a9295ef43d89702c3df202fd937.png --prompt "Glowing mushroom" --output data\inpaint-diagnostic\reliability-2026-10-07
```

The fixture paths are local diagnostic inputs; pass another saved source and matching mask to repeat the same protocol. Outputs and results live under the chosen local diagnostic directory, while the API also retains generation history. Reusing an existing output directory is refused to preserve previous evidence.

## Rightward placement and context geometry

The user noticed that many mushrooms appeared to the right of the selection. The original mask's bounding rectangle centre is `(254, 1055)` and its alpha centroid is approximately `(258.60, 1049.82)`. With context 320, the clamped crop is `(0, 604, 692, 1254)`, centred at `(346, 929)`: its horizontal centre is 92 native pixels to the right of the mask bounds centre. Left/bottom image edges remove context asymmetrically. Context 192 has a smaller horizontal difference, with crop centre `(282, 993)`.

Feather 16 shifts the alpha centroid slightly left, to x≈257.64. Resizing that mask to the model's 672 × 640 input and returning it to the native crop changes its centroid by less than 0.002 px per axis. These measurements do not support a crop/compositor translation error; the rendered subject's placement differs from the mask's registered location. `data/inpaint-diagnostic/position-2026-10-07/mask-position-overlay.png` shows the actual mask outline, mask bounds centre and context centre on four existing outputs.

A mechanistic paired trial kept the native context extent at 692 × 650 and model input at 672 × 640, but centred the context window at the selection: native bounds `(-92, 730, 600, 1380)`. Context beyond the image's left/bottom edges was reflected for this diagnostic only. The final result was mapped back to the original source coordinates; all 1,525,188 protected pixels remained identical. Seeds 3936008243, 7, 42 and 314159265 were selected from three known successful-subject cases and one failed case, so this is not a new unbiased reliability rate.

Generations 74–77 produced recognizable mushrooms closer to the mask centre in all four comparisons, including the previously incomplete seed 42. They also changed scale/style and often reintroduced a separate yellow/grass base. This supports context composition as a contributor to the placement bias, but the test also changes visible context and adds reflected pixels; it does not isolate crop centre as the sole cause or solve integration quality. The production crop/defaults were not changed. `centered-context-compare.png`, `centered-jobs.json` and `geometry.json` in that diagnostic directory retain all pairs and measurements.

**Follow-up (later on 7 October 2026).** In use, reference subjects kept landing to the right of selections near the left edge, partly outside the mask. Inpainting now uses the centred, mirrored context window in production (`center_selection` in `prepare_inpaint_workspace`; Expand keeps the clamped window). The visible Context rectangle still marks the protected area inside the image.

## Next model and sampling comparisons

Current recommendation, checked against primary sources on 7 October 2026:

| Candidate | Why compare it | Practical constraint |
|---|---|---|
| FLUX.1 Fill Dev | Dedicated image-plus-mask fill model; existing MosImage adapter and accessible metadata make it the first specialist baseline | 12B BF16 transformer weights alone are approximately 24 GB. The adapter's component-level `enable_model_cpu_offload()` does not make that entire component fit a 16 GB card. Add a measured supported quantized route or finer-grained offload before adopting it for interactive use; preserve the one-heavy-pipeline constraint. Model access is gated and the weights have a non-commercial license |
| FLUX.2 Klein 4B NVFP4 | Official lower-precision distilled checkpoint; paired native inpaint benchmark completed | The official NVFP4 transformer was the app Quality default until 9 October, when Klein 9B NVFP4 replaced it; 4B NVFP4 and BF16 remain selectable. The original isolated CUDA benchmark produced 12 edits at 6.26 s median versus 9.82 s BF16, with exact protected pixels and similar visual behavior. Transformer storage fell to 2.29 GiB, but whole-pipeline peak allocation stayed near 9.1 GiB. See the [complete experiment](../../benchmarks/NVFP4_RESULTS_2026-10-07.md) |
| FLUX.2 Klein Base 4B | Undistilled model with ordinary guidance and more sampling budget; same 4B class, Apache 2.0 weights | A different checkpoint, not merely extra steps on the current distilled 4B. The vendor example uses 50 steps and guidance 4.0. The later Anna/park follow-up measured 28/50-step renders at 21.745/39.090 s median, with no convincing improvement; broader regional quality remains unmeasured |
| LanPaint with Klein / Base | Alternative conditional sampler with explicit Klein examples; tests whether sampling rather than model size is the limiting factor | A separate implementation/integration experiment, with extra model evaluations and GPL-3.0 code. Neither its speed nor quality is established on this machine |
| Qwen-Image 2.1 | Unified editing, region annotations/separate masks, references and RGBA support | Its 7B figure describes the visual component, not the complete pipeline. Full-pipeline memory and an offloaded/quantized 16 GB route must be measured; weights use the Qwen Research License |
| Klein 9B | Larger general editing comparison | Still a general editor; a larger checkpoint alone does not establish precise masked placement. **Update, 9 October:** the official 9B NVFP4 transformer with a trimmed FP8 text encoder streamed from RAM fits the 16 GB card and is now the Quality default; it beat 4B on reference inserts, object swaps and two-person scenes at about 1.6× the time. Non-commercial licence. See the [model comparison](../../benchmarks/MODEL_COMPARISON_2026-10-09.md) |

Evaluate a selection-centred context strategy independently from model swaps. Keep the user-editable boundary, subject placement and scene context distinct; any extra editable space for shadows/glow must be visible and deliberate. After subject insertion and placement improve, test a narrow boundary/interaction refinement pass with the subject core protected. The previous whole-selection low-strength pass did not help, so do not silently add it or automatic color matching as a default.

Compare additions, replacements and removals across multiple scenes using declared seeds. Record subject presence, complete geometry, placement, integration, exact protected pixels, runtime and peak VRAM separately. The [new four-image suite](../../benchmarks/README.md) fixes sources, masks, prompts and three seeds per scene. Official Klein NVFP4 has now been benchmarked and integrated into the app with the same native CUDA loader; Qwen, LanPaint and larger Klein comparisons remain open. The observed BF16 Fill run below did not establish a practical 16 GB runtime.

## Observed BF16 Fill run and cancellation

After the research pass, the user selected Fill and started a three-proposal job at 28 steps per proposal. The weights downloaded and the pipeline began sampling. The first two steps took approximately 100 seconds each; a live sample showed GPU utilization 99%, 15,773 MiB of 16,303 MiB VRAM occupied, approximately 2.86 GiB free system RAM and 23,849 MiB of system pagefile usage. Pagefile usage is system-wide, not an isolated Fill allocation. This run demonstrates that the Windows runtime can begin sampling under substantial memory pressure; it does not validate a practical 16 GB interactive configuration or completed output quality.

The job was cancelled at the user's request without applying an image. The local supervisor replaced the worker and GPU usage returned near its idle level. The primary UI now exposes Cancel throughout loading and generation, retains a pending Cancelling state until cleanup/reconnection completes, and preserves the image and selection. Klein NVFP4 subsequently passed an isolated benchmark; a practical quantized Fill adapter remains separate, unfinished work.

## Research behind the changes

- [Official Klein 4B NVFP4 weights](https://huggingface.co/black-forest-labs/FLUX.2-klein-4b-nvfp4) and [file listing](https://huggingface.co/black-forest-labs/FLUX.2-klein-4b-nvfp4/tree/main) identify the lower-precision transformer now used by Quality; the original BF16 checkpoint remains selectable.

- [ComfyUI Inpaint Crop & Stitch](https://github.com/lquesada/ComfyUI-Inpaint-CropAndStitch) separates padded context cropping from returning generated pixels to the original image. MosImage adopts the workflow principle using its existing provider/compositor; no custom-node code or dependency was copied.
- [Diffusers native Klein inpaint implementation, 0.41.0](https://github.com/huggingface/diffusers/blob/v0.41.0/src/diffusers/pipelines/flux2/pipeline_flux2_klein_inpaint.py) supplies image-plus-mask sampling, strength and optional image references. Final pixel preservation is still enforced by MosImage.
- [Diffusers inpainting guide](https://huggingface.co/docs/diffusers/main/using-diffusers/inpaint) describes padded mask crops and softened transitions. A source overlay alone is insufficient evidence of seamless generation.
- [ComfyUI PS Composite](https://github.com/tenpel/ComfyUI-PS-Composite) uses surrounding source context, separate subject/halo masks and reduced denoise for integration. [InpaintCanvas](https://github.com/DenRakEiw/ComfyUI-InpaintCanvas) also explores padded context and color matching. These workflows informed experiments, but their extra node stacks, automatic color correction and prompt-expansion models were not installed or copied.
- Community reports about [high-resolution mask/reference editing](https://www.reddit.com/r/comfyui/comments/1wanvfs/inpainting_with_mask_reference_image_at_high/) and [Klein crop-and-stitch](https://www.reddit.com/r/comfyui/comments/1rdphws/crop_stitch_with_flux_2_klein/) suggested more surroundings, separate blending and careful color matching. These are anecdotal leads; parameter choices above follow local trials rather than a universal Reddit recipe.
- [LanPaint](https://github.com/scraed/LanPaint) remains a future sampling comparison. Its mask conventions differ from Diffusers; installing an additional moving custom-node stack was unnecessary for this fix.
- [FLUX.1 Fill model card](https://huggingface.co/black-forest-labs/FLUX.1-Fill-dev) describes the 12B specialist, gate/license and possible color/texture edge artifacts; it does not guarantee seamless results.
- [Klein Base 4B model card](https://huggingface.co/black-forest-labs/FLUX.2-klein-base-4B) describes the undistilled Apache model and its sampling/offload example. [Klein 9B](https://huggingface.co/black-forest-labs/FLUX.2-klein-9B) documents the larger model and hardware/license differences.
- [Qwen-Image 2.1 model card](https://huggingface.co/Qwen/Qwen-Image-2.1) describes region/reference editing and the visual component size. [Edit-2511](https://huggingface.co/Qwen/Qwen-Image-Edit-2511) remains an Apache alternative with a different architecture and memory profile.

## Follow-up: outward soft edge and colour drift correction

Users could still see where the mask had been. Two causes were addressed.

**Inward-only transition.** The old ramp started at zero on the painted boundary, so the original object's anti-aliased rim stayed in the result and the generated pixels reached full opacity only inside. The canvas now builds a signed chamfer distance field from the painted selection. Expand moves the boundary, then Soften adds an outward smoothstep. The overlay draws exactly the alpha that is uploaded. This matches Krita AI Diffusion's grow plus outward feather and the Klein guidance of a blur near width / 30 plus expansion for removals.

**VAE colour drift.** Klein's untouched latents still pass through the VAE, which shifts colour and saturation slightly. A patch can show a uniform cast next to exact source pixels. `correct_color_drift` compares generated and source pixels in the unselected ring, skipping 12 px beyond any visible alpha and transparent outpaint padding. It extrapolates the difference with normalized Gaussian convolution at 128 px working size, clamps it to ±24 levels and adds it to the generated crop before compositing. Only very low frequencies move. The correction is estimated from surroundings the model was meant to reproduce, so an intentional glow or material change inside the selection keeps its relative contrast. A 2048 px crop takes about 0.2 s on the CPU. `InpaintWorkspace.color_correction=False` disables it.

Object selection also uses SAM's three-mask output only for the first click. Refinement clicks ask for one mask, following the SAM authors' guidance for unambiguous prompts.

## Reference images

References put a specific person or object from a concept image into a selection. Diffusers 0.41's `Flux2KleinInpaintPipeline` accepts one `image_reference` or a list, and caps each near 1 MP. MosImage sends a single reference at most 1024 px on the long edge. Replace edits only append a short identity instruction to the prompt. Removal and Expand never receive a reference.

Local trial on `data/reference-trial/`, FLUX.2 Klein 4B NVFP4 in Balanced mode on the RTX 5070 Ti. The concepts were made with the Codex CLI image tool. The garden photo had a box selection beside the path and the prompt "sitting on a wooden bench beside the gravel path", with four seeds per variant.

| Variant | Warm time per image | Result |
|---|---|---|
| Text only, no reference | 5.5–6.1 s | Plausible person, no identity |
| Inpaint plus reference, studio concept | 7.8–8.4 s | 4/4 kept face, hair and clothing |
| Inpaint plus reference, busy market concept | 7.5–8.1 s | 4/4 usable |
| Same, SAM-isolated subject | 6.3–6.5 s | 3/4 usable; one figure broke apart |
| Full-frame multi-reference generation (`Flux2KleinPipeline`) | 10.3–12.4 s | Identity kept, but the whole garden was re-rendered |

The first run after a model switch adds about 4–5 s. Isolation was 15% faster but less reliable, so it is an optional toggle and the full image stays the default. It uses three SAM points on the vertical centre line, accepts masks between 3% and 85% of the frame, and places the crop on neutral grey.

A UI run with two proposals took 33 s including a cold model load. One proposal was clean. In the other, the face crossed the box edge and was cut by the blend. A subject must fit inside the selection. The hint asks for room for the whole body and its shadow, and two proposals are the default when a reference is picked.

### Several references

The pipeline crops every reference to the first one's size, so MosImage pads them to one shared size with neutral grey and names them in order in the prompt. Trial: two concepts (Anna, Tom), a wider box selection, four seeds per size.

| Size per reference | Warm time per image | Result |
|---|---|---|
| 768 × 1024 | 11.9–15.1 s | 4/4 kept both identities |
| 512 × 704 | 8.6–8.9 s | 4/4 kept both identities |

The smaller size kept both people just as well and is about 30% faster, so it is used whenever there is more than one reference; up to three are allowed. With start-up warm-up and cached prompt encoding, a UI run with two proposals and two references takes about 18 s, or 15 s when repeated. The contact sheet is `data/reference-trial/contact-sheet-multi.jpg`.

## Scale-aware placement research

**Status:** Research plus a completed one-scene local evaluation, 7 October 2026; see [the controlled Anna/park experiments](#controlled-annapark-experiments). No scale planner, new model provider or automatic crop policy has been implemented by this investigation. Existing defaults remain experimental.

### What the supplied garden example actually uses

Read-only inspection of the local API confirmed FLUX.2 Klein 4B NVFP4 with component CPU offload on the RTX 5070 Ti. Local generation 121 visually matches the supplied screenshot; 122 is its other proposal. Both use `sitting on a bench`, strength 1, an Anna reference, empty Scene notes and **32 px context**, rather than the 320 px default.

The original image is 1254 × 1254. Its selection bounds are `(850, 142, 1254, 690)`, or 404 × 548 px. The centred model window with context 32 is `(818, 110, 1286, 722)`, or 468 × 612 native px, resized to 448 × 608 for the model. The selection therefore occupies approximately 86% of the window's width and 90% of its height. Most of the garden's perspective and scale cues are absent. The original has no bench: the model invents both the seated person and the bench inside the same editable space.

Generations 119–120 use the same selection bounds with context 320. Their model window is 1044 × 1188 native px, resized to 896 × 1024. Selection coverage falls to approximately 39% width / 46% height. These are different seeds, so they are not a controlled comparison or evidence that context 320 fixes scale. Windows crossing image boundaries include mirrored scenery, which helps centre the selection but is not genuine additional scene information.

The current implementation has three distinct limits:

- `prepare_inpaint_workspace` supplies a local image, mask and fixed context padding. It has no semantic scene analysis, intended subject box, ground-contact anchor or relative-size constraint.
- `_inpaint_prompt` preserves reference identity and asks for matching perspective and complete subjects within the selection. That is a general instruction, without a concrete relationship to the path, flowers or horizon. Asking subjects to fit does not specify how much of the available area they should occupy.
- In the installed Diffusers Klein pipeline, the mask blends original/generated latents after scheduler steps. The transformer receives image/reference latents and text, but the mask is not a separately learned object-size constraint. The final compositor protects pixels; it cannot repair an implausible body size or pose.

This supports insufficient scene context and absent layout constraints as plausible contributors. It does not establish a unique cause or an objectively correct size in centimetres: an uncalibrated single image does not provide that information.

### Online comparison and limits

Primary sources were checked on 7 October 2026. These are vendor capabilities and research results, not local quality measurements.

| Approach | Relevant capability | Consequence for MosImage |
|---|---|---|
| FLUX.2 structural reference | BFL describes pose/layout guidance through reference images, including collages, and warns that structural matching is semantic rather than pixel-exact | First test a deliberately scaled layout guide with the existing Klein provider. Do not assume a ControlNet input or FLUX 3 box syntax exists in this local pipeline. [BFL pose and layout guide](https://docs.bfl.ai/guides/usecases_editing_controlnets) |
| Qwen3-VL-4B-Instruct as a planner | Its model card describes spatial grounding, viewpoints and occlusion reasoning | Candidate for reading the whole scene and proposing a target box, contact point and concise scene-specific instruction. It does not render the edit or guarantee physical dimensions. Local speed, memory, grounding accuracy and provider-switch cost need measurement. [Official model card](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct) |
| AnyDoor | Reference identity/detail conditioning with coarse target shape control | Evidence for explicit spatial conditioning as an alternative to general reference editing. It is a separate model/integration; its quality and Windows runtime have not been evaluated here. [CVPR 2024 paper](https://openaccess.thecvf.com/content/CVPR2024/papers/Chen_AnyDoor_Zero-shot_Object-level_Image_Customization_CVPR_2024_paper.pdf) |
| CatalogStitch | Adapts replacement masks to product aspect ratios and restores foreground occluders around several compositing models | Useful design evidence for preserving proportions and foreground overlap. Its mask calculation anchors on existing pixel dimensions; it does **not** estimate real-world size or seated human poses. Do not silently expand the user's editable boundary. [Paper and method](https://arxiv.org/html/2604.08836v1) |
| Qwen-Image 2.1 | Official release supports reference images, region annotations/separate masks and RGBA editing | Strong next local editor candidate, rather than a proven winner. Benchmark the complete quantized/offloaded pipeline on 16 GB; 7B refers to its visual component. Regional annotation support does not establish exact pixel protection or correct scale. [Official release](https://qwen.ai/blog?id=qwen-image-2.1), [model and memory guidance](https://huggingface.co/Qwen/Qwen-Image-2.1) |
| FLUX 3 Image via API | Documented source/target bounding boxes guide placement and scale, with reference-based editing | Most directly documented cloud layout-control candidate in this comparison. Boxes remain guidance and elements may exceed them; exact protection still belongs in the compositor. This is not the installed Klein model. No cloud requests or image uploads were made. [Official bounding-box documentation](https://docs.bfl.ai/flux_3/flux3_image_bounding_boxes) |

Klein 9B is another comparison, but simply increasing parameter count does not add a hard spatial constraint. Its BF16 model card quotes roughly 29 GB VRAM, beyond this machine's resident budget; a supported quantized/offloaded profile needs separate measurement. [Official 9B model card](https://huggingface.co/black-forest-labs/FLUX.2-klein-9B). That measurement is now in the [9 October model comparison](../../benchmarks/MODEL_COMPARISON_2026-10-09.md): 9B NVFP4 runs at a peak of 11 GiB with its encoder streamed from RAM.

### Recommended local implementation sequence

1. **Check the true mask footprint, then separate editable space from intended subject extent.** The mask-shape comparison below shows that the curved/soft boundary crosses baseline faces. Provide an opaque subject core and room for head, feet and interactions, with soft transitions outside that core. A diagnostic box fixes face clipping here but expands which pixels may change; expose that footprint to the user. Keep the painted/soft mask as the maximum area allowed to change. Add a visible, adjustable placement box and contact anchor for the subject or subject-plus-prop. Leave room inside the editable region for shadows and interaction. The box should express target size independently of the mask; it must not be presented as a guarantee from Klein.
2. **Use scene evidence deliberately.** Test a wider crop containing real perspective anchors while retaining enough subject pixels. If an automatic context choice is added, show its actual boundary and model-facing size. The paired garden comparison below is complete; next isolate layout guidance and repeat the crop comparison across other scenes. More context alone is not a universal solution.
3. **Add visual understanding only where it helps.** An optional local VLM reads a downscaled whole image, selection location, instruction and references, then returns validated structured placement/anchor/occlusion information. Cache it by source revision and relevant inputs; a crop-space transform maps native coordinates into the model window. Keep explicit user size instructions authoritative. Reject invalid/out-of-bounds proposals rather than silently broadening the edit. A full-image semantic analysis is different from adding the whole image as another generation reference, which previously failed in generation 31.
4. **Give the renderer a visible layout.** With Klein, compare a prepared structural reference against the baseline, specifying its layout role separately from Anna's identity. The current reference prompt treats every reference as a subject, so a layout guide needs its own typed role and correct numbering/token-budget handling. For unchanged poses, existing Layers + Transform + Blend in already provide manually controlled size as a useful baseline. Pasting a standing cutout and lightly harmonizing it will not reliably produce a seated pose.
5. **Benchmark another renderer if semantic guidance is insufficient.** Compare Qwen-Image 2.1 with the same scene/placement requirements, or an object-insertion specialist. Retain one heavy GPU pipeline at a time; a planner should run sequentially or on CPU and its loading/offloading overhead must count toward interactive latency. A narrow contact/shadow pass with the subject core protected is an experiment, not a default: the earlier whole-selection low-strength pass scarcely improved integration.

Validate with multiple scenes and declared seeds: the garden person-plus-bench example, a small foreground object, a more distant person, and a partially occluded object. Score subject identity, plausible relative size, placement/contact, requested pose, occlusion and seams separately. Record protected-pixel differences, effective subject resolution, end-to-end latency and peak GPU/system memory. Review every outcome. Automatic VLM judgments can help triage but cannot be the sole acceptance criterion.

The recommended direction is therefore **scene understanding plus explicit placement/layout, followed by bounded rendering**. The controlled experiment below supports the fast Klein runtime and a smaller crop for this scene. Automatic scale and general renderer superiority remain unverified.

## Controlled Anna/park experiments

**Status:** Completed local experiments on 7 October 2026; no production runtime, model registry, editor controls or crop defaults changed. The first 39 completed outputs were all rejected by the user for insufficient quality, with size, seated pose and ground contact as the priorities. The prompt-only follow-up below adds 18 renders (three reproduce the baseline exactly), for 57 completed outputs at that stage. Nine visual-guide renders and six paired mask-shape rerenders bring the total to 72. This is exploratory evidence for one scene, not proof of automatic physical scale.

### Protocol and failure accounting

The screenshot's original garden, soft mask and Anna studio reference were frozen under `data/park-scale-2026-10-07/`. Their SHA-256 hashes, selection bounds, full effective prompts, declared variants and seeds **7, 42, 20261007** are in `results.json`. The orange sphere is preserved. These local document assets are ignored by Git; the tracked four-image fixture suite remains unchanged.

The API cohort completed **18 images**: four context values (32, 192, 320, 512), strength 0.75 at context 320, and eight steps at context 320, each with three seeds. A longer Scene instruction at context 320 failed with CUDA OOM on its first seed; its other two seeds were not run. The first distilled BF16/context-320 attempt was interrupted when the user reported a computer crash; no image was produced, and its other two seeds were not run. Immediately before that interruption, GPU use was about 15.5 GB out of 16.3 GB and the first diffusion step took about 178 seconds. This observed condition is unsuitable for interactive use; it does not establish the crash's cause.

A separate connection failure occurred **before** the strength-0.75 job was submitted, was retained as a transport failure, and was retried only because it had no job ID. It is not an additional rendered image or a model failure. The initial API process was restarted after OOM. The fixture document and references were not edited or accepted into the canvas.

After the crash, available system RAM was only about 6–7 GiB. The follow-up therefore used a different, isolated runtime: load the BF16 text encoder directly onto CUDA, encode the three needed instructions, copy embeddings to CPU, release the encoder, then load **one** image renderer. PyTorch's CUDA allocator was capped at 65% of the device, with a 45-second render budget checked after steps and a 1.5-GiB available-RAM check. This is not a total desktop memory cap or a watchdog for an unresponsive CUDA call. No models run concurrently.

The follow-up completed **21 more images**: five distilled NVFP4 variants, Klein Base BF16 at 28 steps/CFG 4, and Klein Base BF16 at 50 steps/CFG 4, each with three seeds. It retained the context-32 crop and compared a 1024-px versus 512-px-long-edge Anna reference. It used cached model files only. Official repository metadata confirmed that the pinned Base checkpoint's text-encoder, tokenizer and VAE file hashes match the shared distilled components, and that the scheduler configurations are identical; see `base/components-check.json`.

Every completed full scene and enlarged insertion was manually reviewed, with identity, pose/body, relative size, contact, mask fit and seams recorded separately in `review.json`. Earlier relative ratings are preserved as research judgments; they do not override the user's rejection of all 39 images. These are subjective judgments; a hidden foot cannot establish correct ground contact. **All 39 outputs have native 1254 × 1254 dimensions and zero changed pixels where the frozen mask alpha is zero.** Pixel protection does not establish anatomical or scale correctness.

### Results

API timings below are ranges from the two later seeds; the first render may include model/prompt loading. These are exploratory wall times, not an isolated performance benchmark. Early renders also overlapped the Base checkpoint download.

| API variant | Later-seed time | Visual result across all three seeds |
|---|---:|---|
| Context 32, 4 steps, strength 1 | 6.8–8.1 s | 3/3 recognizable seated Anna and plausible adult bench; feet often hidden |
| Context 192 | 8.4–8.5 s | 3/3 Anna present, but lower body compressed and bench very low |
| Context 320 | 11.0–13.2 s | One missing subject; two incomplete lower bodies/weak contact |
| Context 512 | 10.9–11.1 s | Two missing subjects; one implausibly tiny figure |
| Context 320, strength 0.75 | 6.2–6.4 s | Two missing subjects; one partial insertion against the right edge |
| Context 320, 8 steps | 12.3–12.6 s | Better subject presence, but two seeds show head/selection overlap; no reliable fit |

The isolated follow-up timings include inference, VAE decoding, PNG saving and protected compositing, but **exclude** text encoding, renderer loading and one-step warming. Text loading/encoding for three instructions took 6.126 s. NVFP4 renderer loading took 2.802 s; Base loading took 6.431 s. Report these setup costs separately when estimating an interactive first edit.

| Isolated variant, context 32 | Median whole render | Peak PyTorch allocated during render | Visual result |
|---|---:|---:|---|
| NVFP4, 1024-px reference, 4 steps | 2.187 s | 3.515 GiB | 3/3 recognizable and coherent seated figures |
| NVFP4, 1024-px reference, 8 steps | 3.824 s | 3.515 GiB | 3/3 coherent; some improvement in legs/shoes |
| NVFP4, 512-px reference, 4 steps | 1.089 s | 3.146 GiB | 3/3 recognizable; Fast research baseline; subsequently rejected by the user |
| **NVFP4, 512-px reference, 8 steps** | **1.891 s** | **3.146 GiB** | **Fast research candidate; subsequently rejected by the user** |
| NVFP4, 512-px reference, short Scene instruction, 4 steps | 1.080 s | 3.146 GiB | 3/3 coherent; no clear general scale improvement |
| Klein Base BF16, 512-px reference, 28 steps/CFG 4 | 21.745 s | 8.088 GiB | 3/3 coherent; likeness somewhat weaker than NVFP4 |
| Klein Base BF16, 512-px reference, 50 steps/CFG 4 | 39.090 s | 8.088 GiB | Almost the same compositions as 28 steps; no persuasive advantage |

The shorter Scene instruction was: “Adult woman, seated naturally on a wooden garden bench beside the path, feet touching the ground.” It avoided the failed long-context/long-prompt combination, but crop, reference size and runtime also differ from that failed attempt. It does not isolate prompt length as the OOM cause.

The follow-up and API use the same source, mask, reference identity, seed cohort and nominal sampling settings, but the resident/encoder-separated path produces different images from the API path. Treat cross-runtime quality and latency as exploratory; the isolated cohort's within-runtime reference-size/step comparisons are the cleaner evidence. Investigate numerical/execution differences and repeat end-to-end UI timing before adopting the split runtime. Peak render allocation excludes the text phase, other applications and driver allocations.

### Product consequence and remaining work

More surroundings was **not** a scale fix here. The small crop leaves more effective subject pixels and makes the model render a more complete person-plus-bench. Eight distilled steps with a smaller reference remains a fast diagnostic baseline, but the user rejected its quality along with all other earlier results. A slower checkpoint is not justified by this example; the visual-guide comparison below improves fit; remaining comparisons need stronger control of pose and contact. The 512-px reference still needs testing on small faces, several people and references with busy backgrounds.

For an intuitive product, the user should select the location, choose Anna and describe the pose. Crop size, reference resolution, sampling steps and model placement should be chosen behind that interaction, with an adjustable visible size/placement preview when the user needs control. A mask remains the permission to change pixels; it does not specify the exact desired person's extent. The completed visual-guide experiment below tests an explicit placement/layout guide on this scene. Broader scene evaluation, exact pose/contact and a stable end-to-end speed budget remain open. A VLM planner, Qwen editor, Klein 9B and cloud box-guided editor have **not** been quality-tested in this cohort.

Review all results in `data/park-scale-2026-10-07/index.html`, generated by `backend/tools/summarize_park_reference.py`. Completed images, OOM, interruption, the transport failure and four deliberately unrun combinations remain visible. No test output was applied to the user's document and no model/runtime change was shipped.

Validation for this diagnostic change: frontend TypeScript/Vite build, backend app/tool compilation, diff whitespace check, 29 local documentation file links, frozen input hashes, all 39 native outputs and protected pixels, and browser checks for seed/variant selection, detail/original toggles and the interrupted case. The final viewer had no browser console errors. No production code was changed, so broader GPU/model regression suites were not rerun. Park tools reject the wrong working directory before importing configuration.

### Prompt-only follow-up after user rejection

**Completed, 7 October 2026.** The user asked to try better prompts. The first round declared four variants before running; a second round declared two additional variants after observing crossed legs and clipping. Each uses seeds **7, 42, 20261007**, the same frozen source/mask/Anna, context **32**, reference long edge **512**, **eight** distilled NVFP4 steps, guidance **1**, strength **1**, and the same compositor. Only the complete effective prompt changes. Each new cohort uses the encoder-separated runtime, a 65% PyTorch allocator cap and a **15-second** post-step render budget. This is a diagnostic runtime, not an app integration or a hard watchdog during a blocked GPU call.

`prompts/results.json` and `prompts-contact/results.json` retain all effective text, declared settings, pinned quantized/component revisions, setup costs and outcomes. The baseline's three PNGs are byte-identical to the previous `split-small8` outputs, giving a repeatable within-runtime comparison. The follow-up completed **18 renders**, including those three repeated baseline images, with no OOM/interruption. All have native **1254 × 1254** dimensions and **zero** changes where frozen mask alpha is zero.

| Prompt variant | Median render | Observation across all three seeds |
|---|---:|---|
| Original provider baseline | 1.937 s | Exactly repeats the rejected baseline; short-looking lower legs or hidden shoes remain |
| Natural anatomy, hips/thighs/knees/feet described | 1.878 s | Larger figures; two heads are clipped by the upper selection boundary |
| Relative crop positions for head/seat/soles | 1.934 s | More complete compositions, but crossed legs and raised feet contradict the requested placement |
| Perspective, support and contact shadows | 2.006 s | Shoes more visible in some outputs; crossed legs and raised feet still prevent convincing two-foot contact |
| Parallel thighs/shins and feet side by side | 1.970 s | More consistent parallel lower legs; shoes more visible on two seeds, but head/rim overlap and uncertain size/contact remain |
| Compact pose/identity/contact description | 1.944 s | Compact complete compositions; lower legs still appear short and shoes are hidden |

Peak PyTorch allocation during every render was **3.146 GiB**, excluding text encoding, driver allocations and other desktop applications. First-round text loading/encoding took **6.023 s**, renderer loading **2.382 s**; second-round encoding **6.968 s**, renderer loading **2.680 s**. Per-render times include inference, VAE decoding, PNG and compositing, but exclude these setup costs and one-step warming. Do not quote two seconds as measured first-edit app latency.

The manual review covers all 18 full scenes and enlarged insertions. The additional 15 images have not been accepted by the user. Text influences pose, but none of these templates reliably satisfies size, seated anatomy, ground contact and mask fit together. The parallel-leg wording is a useful comparison prompt, not a production fix. Keep the app prompt/defaults unchanged. The subsequent visual-guide and mask-shape comparisons are completed below.

### ComfyUI community leads, checked 7 October 2026

These are source-backed experiment leads. The basic identity/structural-reference route was subsequently tested below on one local fixture; the other routes remain untested locally. Existing crop/composite behavior is already present in MosImage; merely installing another crop node would not establish a new pose constraint.

| Technique | Why it addresses this example | Compatibility and test boundary |
|---|---|---|
| Separate seated layout/pose and identity references | A workflow creator uses a DAZ seated/body pose image separately from the face/character reference. A deliberately positioned dummy plus bench can encode head, hips and feet together. [Creator's face + pose walkthrough](https://www.reddit.com/r/comfyui/comments/1so8383/guide_complete_walkthrough_for_every_pipeline_in/) | Demonstrated workflow uses Klein 9B, not our 4B NVFP4. The later nine-render 4B comparison improves complete-body fit, but not exact pose/contact. MosImage's current multi-reference prompt labels every reference as a subject; a pose guide needs an explicit structural role instead |
| RefControl reference + depth/pose LoRA | Learned conditioning can provide geometry alongside Anna's identity. The author's catalog includes **4B reference-depth** and **9B DWPose/reference-pose**. [Author's catalog](https://github.com/thedeoxen/refcontrol), [4B depth card and bundled workflow](https://huggingface.co/thedeoxen/refcontrol-FLUX.2-klein-4B-reference-depth-lora) | The 4B model is for undistilled Base; the catalog documents fast distilled compatibility for 9B. Our 4B distilled NVFP4 LoRA path is unverified. A guide must include the desired seated person/bench geometry; a depth map of the empty garden alone cannot specify it |
| Smart Inpaint's reference-conditioned generation | Angelo injects the local crop as reference latents and zeros the editable latent at full denoise, with generous rectangles. This is a distinct generation-conditioning experiment. [Author's implementation and explanation](https://github.com/shootthesound/ComfyUI-Angelo#why-smart-inpaint-exists) | Its edit branch differs from our native Diffusers mask-mixing path; inspect exact conditioning before porting. The author's broad mask statement describes that workflow, not all inpainting models. Generous space can prevent clipping but does not guarantee body size |
| LanPaint with a small iteration budget | Alternative conditional sampling may improve consistency around the masked region. The Klein example uses two thinking iterations; start with 1–2 and record model calls and wall time. [Author's sampler](https://github.com/scraed/LanPaint) | General defaults add substantial computation; the author notes distilled-model limitations. No local quality/speed result exists, and it does not provide a hard seated pose or scale constraint |
| Preplaced collage / Put It Here with Qwen Edit | The workflow creator places/resizes a cutout on a canvas and asks the editor to integrate it. [Creator's Qwen 2511 workflow](https://www.reddit.com/r/comfyui/comments/1rc666c/qwen_2511_workflows_inpaint_and_put_it_here/). A [four-step Lightning adapter](https://huggingface.co/lightx2v/Qwen-Image-Edit-2511-Lightning) is a separate speed candidate | Four steps reduce sampling, not the large pipeline's weight footprint. Test quantized residency/latency first on 16 GB and verify the exact LoRA/checkpoint version. A standing cutout still needs a valid seated pose; it cannot define seated anatomy by resizing alone |

A [firsthand position thread](https://www.reddit.com/r/comfyui/comments/1uq37l8/inpaint_position_flux_klein/) suggests a colored box/circle guide, but the questioner reports off-centre placement even after trying it. Treat colored markers as a cheap ablation, not exact geometry.

The first **visual seated dummy/bench layout with the existing small renderer** comparison is now complete below. Learned depth/pose conditioning, a staged bench/person experiment and bounded alternate sampling/editors remain follow-up candidates. This ordering is our inference from the measured failures and community examples. Another proposed ablation is to establish the bench first, add Anna against its seat geometry, then refine only the foot/seat interaction boundary while protecting her core. That staged method has not been tested here; the older whole-selection low-strength refinement was ineffective.

The eventual interaction should remain simple: choose Anna, choose “Sitting”, drag/resize the placement preview and generate. Pose/depth preparation, crop size, model choice and bounded refinement belong behind that flow. Keep the editable footprint distinct from the placement preview; any additional space for feet/shadows must remain visible to the user. This is proposed product behavior, not an implemented control.

The viewer now retains **64** rows: 57 completed outputs, one OOM, one interruption, four skipped combinations and one pre-submission transport failure. No generated edit was applied. Final validation passed: frontend TypeScript/Vite build, app/tool compilation, diff whitespace, 32 local documentation file/anchor links, frozen hashes, an independent native-size/protected-pixel check of all 18 new renders, and browser controls for all 64 rows, prompt variants, seeds, detail/original and the interrupted BF16 case. All 57 successful outputs have one manual review. No production code changed; broader model regression suites were not rerun.

### Visual pose/layout guide comparison

**Completed, 7 October 2026.** After the user approved the nine-render proposal, the same source, soft mask and Anna reference were tested with no guide, a neutral-background pose guide and a scene-layout guide. All three conditions use seeds **7, 42, 20261007**, context **32**, eight NVFP4 steps, guidance **1**, strength **1**, the same native masking/compositor, and the existing encoder-separated runtime with a **15-second post-step budget**. No editor/provider defaults changed.

A single anonymous grey seated mannequin and wooden bench was generated using the **built-in imagegen tool**, then copied into the project. The exact asset-generation prompt is retained in `guides/generation.json`; the unchanged raw transparent PNG is `guides/dummy-original.png`. This is a supporting pose asset, not a cloud-generated final park edit. All nine final edits were rendered locally with Klein. One-time guide generation/preparation was not timed and is excluded from render latency; a product would need reusable/procedural pose presets or separately measured guide preparation.

`prepare_park_guides.py` prepares the neutral pose, placed-layout reference and full-scene placement preview. The manually chosen test box is **(866, 248, 1246, 610)**, from centre x 1056 / bottom y 610 and a 380 × 420 maximum. It is a size hypothesis, not calibrated physical dimensions. The first proposed bottom y 625 crossed 217 protected pixels at a bench leg; the guide was moved up before any inference. The final meaningful-alpha silhouette has zero overlap with protected pixels. This confirms that the actual soft mask's footprint matters beyond its bounding rectangle.

Both structural references contain exactly the same placed dummy geometry. The pose version has a neutral grey background; the layout version has the original garden context. Original scenery outside the source boundary is mirrored by the usual crop; the added dummy is not mirrored into the padding. Anna and both guides are prepared on the same **384 × 512** canvas, avoiding the pipeline's cropping of later reference images to the first reference size. The baseline uses the same prepared Anna image, so the three new conditions share identity pixels. This padding/resampling differs from the earlier prompt cohort: all three new baseline PNGs differ from the previous `prompt-parallel` PNGs, and cross-cohort differences cannot be attributed solely to a guide.

The guided conditions also use a role-aware prompt: studio photo supplies identity/clothing, grey mannequin supplies seated pose and bench geometry. They use **the same prompt**; their comparison isolates the structural reference's background. Guided versus baseline compares the combined extra-reference plus role-aware-prompt workflow, not an isolated image-only effect. No grey mannequin is intended in the final result.

| Condition | Median render | Peak PyTorch allocated during render | All-three-seed visual review |
|---|---:|---:|---|
| Prepared Anna only, parallel-leg prompt | 1.988 s | 3.146 GiB | Source vegetation overlaps the head/upper selection edge; shoes/contact remain inconsistent |
| Anna + neutral pose guide | 2.577 s | 3.147 GiB | All three complete figures fit, and the bench takes the guide's form. One crosses its legs; another renders bare feet instead of the white shoes |
| Anna + placed scene-layout guide | 2.573 s | 3.147 GiB | All three complete figures fit and retain white shoes. Two approximate the uncrossed pose; one crosses its legs and raises a foot. Sole/contact shadows and physical scale remain uncertain |

The within-cohort paired delta is roughly **0.59 seconds** median for either guide over the baseline. Setup cost was **6.451 s** for text loading/encoding, **2.427 s** for renderer loading and **0.792 s** for one-step warming, all excluded from per-render times. Peak allocation excludes the separate text phase, other applications and driver/library allocations. No OOM or interruption occurred.

All nine outputs are **1254 × 1254** with zero changes to the **1,397,802** pixels where frozen mask alpha is zero. `guide-tests/results.json` records reference paths, actual reference/model sizes, frozen auxiliary hashes, exact prompts, revisions and every outcome. `guides/manifest.json` records preparation geometry and the generated asset. Full scenes and enlarged insertions were reviewed separately; all nine remain unaccepted by the user.

**Conclusion:** the guide workflow improves complete-body fit on this fixture with modest additional local render time. Within this cohort, the scene-layout version is the exploratory candidate because shoes are more consistent than the neutral version. The following paired mask-shape test isolates another contributor to face clipping. Neither guide enforces the requested pose, proves correct physical scale, or establishes convincing two-foot contact across all seeds. Keep production defaults unchanged. The following mask-only comparison is complete. Bench-first/person-second construction, learned pose/depth conditioning and contact-focused refinement with the core protected remain untested.

The review now retains **73** rows: **66** completed outputs, one OOM, one interruption, four skipped combinations and one transport failure. It includes the actual input guides so users can distinguish a placement mockup from a rendered result. No edit was applied to the document and no new final-render checkpoint/provider was installed.


### Paired mask-shape comparison

**Completed, 7 October 2026, after the user reported missing faces.** The actual alpha contour's diagonal upper-left edge cuts through the baseline face. Its wide translucent rim retains/blends original foliage through the head. The compositor protects those pixels correctly, but this is unsuitable permission geometry for the generated seated figure. A bounding-box-only fit check had missed this failure.

The original alpha bounding box remains **(850, 142, 1254, 690)**. `benchmark_park_masks.py` creates a diagnostic rectangular mask within exactly those bounds, with a **16-px inward smooth edge** and opaque interior. Model crop **(818, 110, 1286, 722)**, effective **448 × 608** input, prompts, references, seeds **7, 42, 20261007**, eight NVFP4 steps, guidance 1, strength 1 and compositing are unchanged relative to `guide-baseline` / `guide-layout`. Only mask shape/alpha changes, used in both latent mixing and final compositing; this does not isolate those two mechanisms individually.

| Same-seed comparison | Median render | Observation over all three seeds |
|---|---:|---|
| Curved baseline → box baseline | 1.988 → **2.006 s** | Face/head now complete **3/3**. Two shoes and uncrossed legs present. Person/bench remain large; ground contact uncertain. Seed 7 replaces more background with grass and exposes a visible transition |
| Curved layout guide → box layout guide | 2.573 → **2.608 s** | All heads complete and intended smaller guide placement persists. Seed 42 still crosses its legs/raises feet; the box does not enforce exact pose or convincing contact |

The box intentionally permits **46,678** additional pixels inside the unchanged bounding rectangle to change. Its protected area is **1,351,124** pixels, compared with **1,397,802** originally protected pixels. Every output is **1254 × 1254** with **zero** changes outside the active box mask. However, **43,609–44,159 originally protected pixels change** per diagnostic output. These are expanded test permissions, not preservation of the original mask. The frozen mask/source and user's document remain unchanged; no test image was applied.

All six full scenes and enlarged insertions were reviewed. `mask-tests/results.json` records active mask path/hash, reference sizes, model input size, setup times and separate protection counts. Peak render allocation is **3.146–3.147 GiB**. Text setup **5.948 s**, renderer load **2.477 s**, warmup **0.711 s** are excluded from render medians. No OOM/interruption occurred. All six remain unaccepted by the user.

**Conclusion:** the curved mask is a demonstrated contributor to missing faces in this fixture. Fix permission geometry before another model/prompt sweep. Preserve a fully editable core for the complete head/body, then soften around it; do not use a narrow feathered silhouette as a desired subject-size constraint. An intuitive placement preview should show the actual editable footprint and check head/foot/bench fit against its opaque core. Expansion must be visible and adjustable. A box alone allows larger subjects/background replacement, so retain independent size/layout guidance. Anatomical pose and grounded shoes/bench remain open.

The viewer retains **79 rows / 72 completed outputs across 24 variants**, with one manual review per successful output. It defaults to curved-versus-box baseline, includes actual mask overlay and input guides, and separates changes outside the test mask from changes to originally protected pixels. Geometry/contact follow-ups remain proposals; production settings are unchanged.


Final validation for the guide/mask diagnostics: frontend TypeScript/Vite build; app/tool compilation; whitespace check; **36** local documentation file/anchor links; frozen input and auxiliary hashes; independent native-size/active-mask protection checks on all **15** new renders; **72** unique successful manual reviews. The box regeneration is identical, and invalid mask paths, missing/changed hashes and wrong working directory are rejected before model loading/output creation. Browser verification covered **79** rows, **26** declared variants (**24** successful), all new box outputs across three seeds, original/detail controls and the interrupted BF16 case. No browser console errors. No production code changed; broader model regression suites were not rerun.
