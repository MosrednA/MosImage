# Regional inpainting and blending

**Status:** Core editing path implemented locally, 7 October 2026; visual reliability remains experimental. Current behavior for the English Focus editor; linked from [README](README.md), [Focus validation](FOCUS_IMPLEMENTATION_2026-10-07.md) and the [v2 roadmap](V2_REFACTOR_PLAN.md).

## Root cause and correction

The previous Klein provider discarded `mask_path`, used ordinary image editing on the whole crop and clipped the result afterward. A generated mushroom could extend beyond the selection and be cut off, while recolored scenery remained inside the unused part of the mask. Exact protected pixels alone did not demonstrate correct masked generation.

The installed stable Diffusers 0.41.0 already provides `Flux2KleinInpaintPipeline`. MosImage now passes the real generation mask and edit strength to that pipeline. Native latent masking preserves surrounding context during sampling; the final native-resolution compositor independently enforces the user's pixel boundary. Task switches release the previous large pipeline, retaining BF16, TF32 and CPU offload on the RTX 5070 Ti profile. No additional dependencies or checkpoints were installed.

This native API is not the same as an inpaint-trained checkpoint. Inspection of the installed pipeline shows that the Klein transformer receives generated/source image latents and text; its mask blends source and generated latents after each scheduler step. The Fill pipeline instead passes masked image latents and the mask as additional transformer input. Klein's current route enforces a spatial editing boundary but does not give its general editing checkpoint the same explicit mask conditioning as a dedicated Fill model.

## Context, selection and blend

These controls have different responsibilities:

| Control | What it changes | Default |
|---|---|---|
| Painted selection | Where generated pixels may be returned | User selection |
| Expand | Grows (or shrinks) the selection before the edge ramp, shown live on the canvas | +4 native px |
| Soften | Outward smoothstep from the selection to the untouched image, shown live on the canvas | Auto: longest side / 40, 8–64 px |
| Surrounding context | How much nearby scenery the model sees around the selection's bounding rectangle | 320 native px per side |
| Scene notes | Optional lighting, perspective, material or foreground cues | Empty |

The crop clamps to image edges and is resized to a maximum model-facing edge of 1024, with dimensions divisible by 32. The canvas Context toggle shows its protected rectangle at the current zoom. Inpainting uses the exact chosen padding; no hidden proportional increase changes the displayed rectangle. A full-frame second reference is not added automatically.

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

## Next model and sampling comparisons

Current recommendation, checked against primary sources on 7 October 2026:

| Candidate | Why compare it | Practical constraint |
|---|---|---|
| FLUX.1 Fill Dev | Dedicated image-plus-mask fill model; existing MosImage adapter and accessible metadata make it the first specialist baseline | 12B BF16 transformer weights alone are approximately 24 GB. The adapter's component-level `enable_model_cpu_offload()` does not make that entire component fit a 16 GB card. Add a measured supported quantized route or finer-grained offload before adopting it for interactive use; preserve the one-heavy-pipeline constraint. Model access is gated and the weights have a non-commercial license |
| FLUX.2 Klein 4B NVFP4 | Official lower-precision distilled checkpoint; paired native inpaint benchmark completed | The official NVFP4 transformer is now the app Quality default; BF16 remains selectable. The original isolated CUDA benchmark produced 12 edits at 6.26 s median versus 9.82 s BF16, with exact protected pixels and similar visual behavior. Transformer storage fell to 2.29 GiB, but whole-pipeline peak allocation stayed near 9.1 GiB. See the [complete experiment](benchmarks/NVFP4_RESULTS_2026-10-07.md) |
| FLUX.2 Klein Base 4B | Undistilled model with ordinary guidance and more sampling budget; same 4B class, Apache 2.0 weights | A different checkpoint, not merely extra steps on the current distilled 4B. The vendor example uses 50 steps and guidance 4.0; local latency/VRAM and regional quality remain unmeasured |
| LanPaint with Klein / Base | Alternative conditional sampler with explicit Klein examples; tests whether sampling rather than model size is the limiting factor | A separate implementation/integration experiment, with extra model evaluations and GPL-3.0 code. Neither its speed nor quality is established on this machine |
| Qwen-Image 2.1 | Unified editing, region annotations/separate masks, references and RGBA support | Its 7B figure describes the visual component, not the complete pipeline. Full-pipeline memory and an offloaded/quantized 16 GB route must be measured; weights use the Qwen Research License |
| Klein 9B | Larger general editing comparison | Still a general editor; a larger checkpoint alone does not establish precise masked placement. The vendor card's ~29 GB VRAM profile exceeds this GPU, requiring quantization/offload; different license and encoder size |

Evaluate a selection-centred context strategy independently from model swaps. Keep the user-editable boundary, subject placement and scene context distinct; any extra editable space for shadows/glow must be visible and deliberate. After subject insertion and placement improve, test a narrow boundary/interaction refinement pass with the subject core protected. The previous whole-selection low-strength pass did not help, so do not silently add it or automatic color matching as a default.

Compare additions, replacements and removals across multiple scenes using declared seeds. Record subject presence, complete geometry, placement, integration, exact protected pixels, runtime and peak VRAM separately. The [new four-image suite](benchmarks/README.md) fixes sources, masks, prompts and three seeds per scene. Official Klein NVFP4 has now been benchmarked and integrated into the app with the same native CUDA loader; Qwen, LanPaint and larger Klein comparisons remain open. The observed BF16 Fill run below did not establish a practical 16 GB runtime.

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
