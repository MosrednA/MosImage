# Improving image quality across MosImage — 9 October 2026

Status: Historical
Last verified: 2026-10-11
Current state: [Guide](../guide.md) and [roadmap](../roadmap.md).

**Original scope:** Research snapshot completed before implementation. The investigation inspected the checkout, installed Diffusers 0.41.0, existing benchmark reports and two retained 9B outputs; checked primary online sources; ran a CPU workspace probe. It did not change the runtime or perform new diffusion renders or LoRA training.

**Implementation follow-up, 9 October:** the first two recommendations are implemented: 32 model-pixel generation overlap for expansion with the existing 16 native-pixel final blend, and separate opaque generation coverage / soft final alpha for inpaint. The [paired implementation report](../../benchmarks/EDIT_MASK_RESULTS_2026-10-09.md) retains 30 Klein 9B renders, protected-pixel checks and visual limitations. The shirt fringe remains; broader quality acceptance is open. Reversible LoRA residuals now work on the native NVFP4 transformer, and the [candidate evaluation](../../benchmarks/QUALITY_CANDIDATES_2026-10-09.md) records Consistency V2 and the specialized outpaint/detail follow-up. Remaining proposals below retain their original research context. Pipeline descriptions below record the pre-change snapshot; the [guide](../guide.md) describes current app behaviour.

## Recommendation

**Upscaling follow-up, 10 October:** the [bounded GPU comparison](../../benchmarks/UPSCALING_2026-10-10.md) measures x2plus and compact x4v3, PNG export cost, larger-halo tiling and coexistence with real 9B edits. Speed is sufficient, but **the user rejected all three displayed faces as too smooth**, including x2plus. The initial photo-export recommendation is withdrawn. Natural skin/beard texture and native-source enlargement must pass visual review before integration; no automatic upscale default was added.

**Texture-preservation follow-up, 10 October:** the [classical 2x study](../../benchmarks/UPSCALING_FIDELITY_2026-10-10.md) tests SwinIR Light/M and Swin2SR on six native-source and reconstruction cases, including one public-domain camera photo. Light keeps more beard/skin texture than the rejected reconstruction, but some enlarged texture appears stippled and missing natural detail is not guaranteed recoverable. FP32 and 512px tiles are the next recipe; the report separates warm standalone speed, cold initialization and actual Klein residency. Visual acceptance remains open and the existing Export is unchanged.

**SeedVR2 3B follow-up, 10 October:** the [27-output standalone test](../../benchmarks/SEEDVR2_3B_2026-10-10.md) adds visible skin, beard and cloth detail at about 9 seconds per one-megapixel native 2x input. The same seed is pixel-identical across repeats/processes. It also redraws eyes, fabric and small text; LAB colour correction helps tone but does not remove content changes. This is an optional extra-detail preview candidate, with user acceptance and Klein coexistence open. No product integration was added.

Keep Klein 9B distilled NVFP4 as the interactive baseline. Improve the editing workspace and compositing first, and evaluate one compatible consistency LoRA alongside that work. Outpainting, recolouring, removal and inserting a subject need different masks, context and acceptance criteria.

**Second-model follow-up, 10 October:** the [official eight-step Qwen Turbo quantization evaluation](../../benchmarks/QWEN_TURBO_QUANTS_2026-10-10.md) tests native NVFP4 and Unsloth GGUF, distinct from the earlier Viggle six-step model. It records alpha support, reference-resolution/speed tradeoffs, GPU/RAM pressure and complete new-prompt latency. The user prefers FLUX overall after the comparison; Qwen is not selected for integration and Klein 9B remains the Quality model.

| Priority | Work | Expected benefit | Evidence level |
|---|---|---|---|
| 1 | Resolution-aware outpaint overlap and crops | Better texture and structural continuation at large sizes | Overlap implemented and paired renders reviewed; staged crops remain proposed |
| 1 | Separate generation coverage from final blend coverage | Fewer source-colour fringes; complete subjects and shadows | Implemented and paired renders reviewed; shirt fringe remains |
| 2 | 9B Consistency LoRA compatibility and A/B test | Less colour drift and better preservation | Native NVFP4 compatibility verified; 36 outputs show preservation/compliance tradeoffs; optional V2 switch at 0.6 is off initially |
| 2 | Context/detail choices per operation | Useful subject detail without losing grounding | 18 current 9B outputs show finer local texture and possible invented materials; optional Fine detail switch is implemented for Replace |
| 3 | Dedicated outpaint adapter or FLUX Fill comparison | Alternative for difficult cases | FAL fills borders on Fast4 at about 6–7 seconds; Base was stopped for latency; Fill remains untested |
| 3 | Optional placement controls and targeted integration | Better pose, scale, occlusion and light | Earlier mixed guide results; specialist adapters have narrow scopes |

These are testing priorities, not promised quality gains.

## Existing evidence

The [9 October comparison](../../benchmarks/MODEL_COMPARISON_2026-10-09.md) retained 216 outputs across six arms, twelve cases and three seeds. Its visual review preferred 9B over 4B for creative edits and identity scenes. Qwen-Image 2.1 turbo was slower and had masked-edit failures; it is not a justified replacement for every tool. Encoder variants gave no material improvement in that suite.

That report leaves a blue shirt fringe open across models. I inspected the retained 9B seed-42 shirt crop and Anna insertion: the shirt keeps a visible blue edge, while the insertion is substantially more coherent. This is a two-image spot check, not a new cohort or a proven diagnosis of the fringe.

The [8 October study](../../benchmarks/INPAINT_QUALITY_2026-10-08.md) used 4B. Roomy masks fixed clipped heads; pose, likeness and ground contact still varied. More context, references or distilled steps were not consistently better. Its 32 px context / 16 px edge / ordinary 512 px reference recipe is a starting point for that insertion fixture, not a universal replacement for current defaults. Crop upscaling improved available detail but did not reliably solve identity or pose.

## Outpaint: tone seams and structural seams

At the research snapshot, [services.py](../../backend/app/services.py) created transparent margins, flattened to neutral grey for inference, and used Klein inpaint. The generation mask grew 16 native pixels into the original. [compositing.py](../../backend/app/compositing.py) applied low-frequency colour correction, carried the overlap's tone offset into the new area, and blended across the edge. Original pixels beyond the blend band stayed protected; the band itself could change.

This addresses tone mismatch. It cannot guarantee continuation of a fence, horizon, branch, perspective line or fine texture. A blurred mismatch can still be a structural mismatch.

The model crop is capped at 1024 on its longest edge. A single-side expansion can use a smaller crop; expanding four sides makes the selection bounding box cover the full canvas. Installed Klein inpaint also caps source area near one megapixel. Raising the app limit alone does not establish higher-resolution inference.

### New CPU workspace probe

Used the pre-change `prepare_inpaint_workspace`, context ratio 0.5, minimum context 128, generation growth 16 and `match_edges=True`:

| Expanded canvas | Expansion | Model input | 16 native px expressed at model resolution |
|---|---|---|---:|
| 1024 × 512 | Left only | 512 × 512 | 16 px |
| 2048 × 1024 | Left only | 1024 × 1024 | 16 px |
| 4096 × 2048 | Left only | 1024 × 1024 | 8 px |
| 1024 × 512 | Four sides | 1024 × 512 | 16 px |
| 2048 × 1024 | Four sides | 1024 × 512 | 8 px |
| 4096 × 2048 | Four sides | 1024 × 512 | 4 px |

The probe uses uniform synthetic pixels and reads saved model inputs/masks, not neural outputs. JSON and scratch images are local under `data/quality-research-2026-10-09/`. This confirms scale-dependent overlap loss, not that it causes every visible seam.

Test independently:

1. Define generation overlap in model pixels and map back to native coordinates. Compare 16/32/64 model-pixel candidates. Keep final compositing coverage separate: wider inference overlap need not permit wider final edits.
2. Compare one-shot expansion with staged edges or overlapping strips at useful resolution, including corners and wide extensions. Staging can accumulate drift and create joins of its own.
3. Review raw, colour-corrected and final outputs separately; compare correction on/off.
4. If needed, offer a targeted seam-repair proposal with a visible editable band. Earlier generic low-strength integration passes did not reliably improve insertion, so a second pass must justify its extra time.

Keep the current 16 px final blend contract while testing larger generation overlap. Show any larger final repair footprint to the user.

## Inpaint: generation and blending need distinct footprints

An exact cut-out helps isolate a layer. Generating a replacement needs fully editable room for its complete silhouette, interactions and contact shadows.

The frontend supplies the final soft selection and sets backend `blend_feather=0`. Before implementation, the backend retained those soft values; Klein used them during denoising, and final compositing used coverage again. This intentionally retained old pixels near a soft edge and was a hypothesis for the shirt fringe. The paired implementation trial shows that separating generation coverage alone does not remove that fringe. It was not proof of accidental double compositing. The layer path already stores a binary footprint because the proposal contains the blend; preserve that fix in [selectionPatch](../../frontend/src/layers/bitmaps.ts).

| Operation | Generation footprint | Final integration |
|---|---|---|
| Recolour/material | Cover the full object, including its outline | Narrow transition at the real boundary; verify old colour disappears |
| Insert/replace | Roomy opaque core around the full subject, hands, feet and contact | Soften outside the subject; explicitly include support/shadow area |
| Remove | Include object, cast shadow and reflection | Reconstruct surface continuity without residual silhouette |
| Blend a layer | Include local interactions with the scene | Preserve identity while allowing displayed light/contact changes |

Review perspective, scale, occlusion, shadow direction, ground contact, sharpness, noise and depth of field separately. Colour matching solves only part of natural integration.

Preview the actual editable footprint. Do not silently expand the final selection to repair a face, hair or shadow. Only the main image needs a generation mask; references remain unmasked. Optional reference crops can remove distractions. Automatic background isolation should not become the default without evidence: the existing reference trial preferred ordinary photos.

Compare native crops with optional upscaling to 768/1024, and context 32/96/320 px where relevant. Broader context can help perspective but dilute small-subject detail; a universally tight crop loses scene information. Whole-image edits currently render around one megapixel and resize back, so 4K export dimensions do not imply 4K generated detail.

## LoRA shortlist

| Candidate | Use | Compatibility and decision |
|---|---|---|
| [dx8152 / Klein 9B Consistency](https://huggingface.co/dx8152/Flux2-Klein-9B-Consistency) | Preservation, colour cast and saturation | First adapter to test. Metadata names distilled Klein 9B; author reports V2 colour/detail improvements. MosImage benefit is unverified. Pin the file/revision. |
| [fal / Klein 4B outpaint](https://huggingface.co/fal/flux-2-klein-4B-outpaint-lora) | Extend images from green borders | Separate 4B workflow, not a 9B adapter. SDK example uses the **4B Base edit** endpoint, while model metadata names 4B without clearly distinguishing the variant. Confirm the intended checkpoint. Pure-green input is its published recipe; our grey masked input differs. |
| [RefControl 9B reference depth](https://huggingface.co/thedeoxen/refcontrol-FLUX.2-klein-9B-reference-depth-lora) | Identity plus controlled structure/pose | Uses undistilled **9B Base**, depth and a specific reference order. Optional placement workflow. The page's automatically generated Stable Diffusion ControlNet snippet is not valid Klein integration guidance; inspect the author's workflow. |
| [Sun Direction 9B](https://huggingface.co/eric-venti-seeds/Sun-Direction-Lora-Flux2Klein9B) | Outdoor light direction | Narrow experiment: author requires overcast input/preparation and a sphere reference; interiors were not trained. Not a general insertion-harmonization solution. |
| [9B 360 ERP outpaint](https://huggingface.co/nomadoor/flux-2-klein-9B-360-erp-outpaint-lora) | 360-degree panoramas | Exclude from ordinary Expand comparison. Expects projected references on a 2:1 canvas; author recommends Base and reports remaining seam difficulty. |

A subject LoRA can help repeatedly depict a person, product or style. It does not automatically repair joins, geometry or light. Evaluate downloaded adapters before custom training. An integration adapter would need paired examples of desired placement/harmonization; ordinary subject photos alone do not teach those operations.

BFL distinguishes [interactive distilled weights from Base models for training](https://github.com/black-forest-labs/flux2). The [9B Base recipe](https://huggingface.co/black-forest-labs/FLUX.2-klein-base-9B) uses 50 steps and guidance 4.0. Base is a different checkpoint, not an increased step count on the current model. Earlier local Base tests did not convincingly solve the park fixture; defer a broad sweep until a specific adapter requires it.

## Runtime integration

Installed Klein generation/inpaint expose `Flux2LoraLoaderMixin`, but `peft` is absent from the current environment. [nvfp4_runtime.py](../../backend/app/models/nvfp4_runtime.py) replaces quantized projections with `Nvfp4Linear`, an `nn.Module` using packed buffers rather than standard `nn.Linear`. The app has no adapter-loading lifecycle.

A loader call alone is insufficient. Prefer a supported quantization/PEFT route if it preserves the proven runtime. For custom layers, [PEFT documents experimental custom-module dispatch](https://huggingface.co/docs/peft/developer_guides/custom_models). A feasibility experiment could retain the native NVFP4 matmul and add a small BF16 LoRA residual. This is proposed engineering, not verified compatibility. Do not silently dequantize the transformer or fuse adapters into calibrated packed weights.

Verify weight-name conversion, targeted projections, scale/alpha, zero-scale equivalence, disable/unload, shared generation/inpaint components, cancellation and native kernel use. Invalidate prompt caches when modifying the text encoder; transformer-only adapters can reuse unchanged embeddings. Record adapter/model hashes, weight, prompts, seed, dimensions, masks and context. Start with one adapter at a time.

[Diffusers offers Klein LoRA training scripts and memory optimizations](https://github.com/huggingface/diffusers/blob/main/examples/dreambooth/README_flux2.md). This does not prove 9B training fits our 16 GB Windows profile. Measure training separately after inference evaluation. Preserve one-model residency and evaluate adapter and underlying-model licences together.

## Generation, faces and alternative models

Keep generation at the distilled four-step baseline. Evaluate natural texture and faithful editing independently: adding detail can make preservation worse. Match reference preprocessing/aspect ratios between app and benchmark, and test one to three references without assuming more is better.

For Fix faces, compare identity, lighting and texture with the original at normal viewing size. Existing 768-area rendering and global crop colour shift can sharpen a face while changing likeness or local light. Keep restoration as an optional proposal.

[FLUX Fill is dedicated to inpaint/outpaint](https://huggingface.co/docs/diffusers/api/pipelines/flux), making a quantized local route a useful later control for difficult removal/structure. The earlier BF16 transformer offload route did not fit this card; another profile requires measured Windows compatibility, VRAM, latency and quality. The existing Qwen comparison does not justify a broad new sweep. Klein KV is primarily a latency candidate, not an established quality gain.

## Next experiment and release gates

Use the **real 9B app provider**, streamed FP8 encoder and compositor. Freeze fixtures and seeds 7, 42, 20261009, 11 and 123 before review; retain every failure and raw crop. Add consented real-photo fixtures where identity matters: synthetic tests cannot establish a general success rate.

Run bounded stages:

1. Mask/compositing: shirt recolour, removal with shadow, small replacement and person insertion. Change one mask/context/resolution choice at a time.
2. Outpaint: garden, architecture/kitchen and portrait; single-side/four-side, 25%/50% expansion and multiple native resolutions. Compare fixed/adaptive overlap before staged generation.
3. Consistency adapter: baseline and independently tested weights, initially 0.3/0.6/1.0 as experimental values. Do not stack adapters.
4. If needed, compare the author's 4B outpaint recipe and a compatible Fill profile with the best 9B route; include model-switch/load time.

Score compliance, identity, anatomy/geometry, integration and seams separately. Inspect fit-to-screen and 100% views. Edge-discontinuity metrics alone can reward blur. Check protected pixels outside each declared footprint, proposal/applied/exported consistency, dimensions, warm end-to-end latency, cold loading, allocated/reserved VRAM and process RAM. Separate renderer-only timings from UI latency.

Advance a recipe after consistent visual gains across retained seeds without material regressions. Confirm Apply, layer masks, Undo/Redo, expansion offsets and export in the editor before defaults change. Unit tests protect contracts; they cannot prove natural-looking output.

The first implementation slice is **separate mask coverage plus resolution-aware outpaint overlap**, alongside a small **9B Consistency adapter feasibility test**. Automatic pose planning, global relighting and custom training remain later options.

## Investigation validation

At the research snapshot, the six CPU workspace cases completed, existing compositing/outpaint tests passed (23 tests) with isolated data and temporary directories, and frontend `npm run build` and backend `compileall` passed. All 35 local link targets and 11 local heading anchors in the three changed documents resolved; whitespace checks passed. Those checks validated the investigation and existing contracts, not proposed visual gains. The report, docs index and roadmap were the only changes at that stage; scratch probe assets remain under ignored `data/`. The follow-up report records implementation validation separately.
