# MosImage v2 — Refactor and Upgrade Plan

Status: Active
Last verified: 2026-10-11
Review by: 2026-11-11

**Target:** Windows desktop, local-first\
**Primary hardware:** Ryzen 7 9700X, 32 GB RAM, RTX 5070 Ti 16 GB VRAM

**Editing masks, 9 October 2026:** resolution-aware expansion overlap and separate generation/final blend coverage are implemented. Expansion regenerates 32 model pixels inside the original while the final blend remains limited to 16 native pixels; inpaint generates fully inside the displayed soft footprint and composites with its original alpha. The [paired implementation report](../benchmarks/EDIT_MASK_RESULTS_2026-10-09.md) retains 30 Klein 9B renders with exact protected pixels. Visual gains are limited and the shirt fringe remains. The [quality investigation](notes/2026-10-09-image-quality-research.md) preserves the original proposals; LoRA compatibility and selected detail crops now have separate evaluation evidence, while staged expansion and operation-specific mask recipes remain proposed.
**Plan date:** July 16, 2026

**Quality candidates, 9 October 2026:** reversible NVFP4/BF16 LoRA residuals and a local comparison runner are implemented; 78 cohort images and two equivalence renders are complete. Consistency V2 trades preservation against edit compliance. FAL outpaint works experimentally on distilled Fast4 in about 6–7 seconds; the 157–180-second Base route was stopped by the user's latency requirement. Selected detail crops show more local texture at about 8 seconds, with possible invented materials. All 60 selected outputs preserve protected RGBA pixels. See the [candidate report](../benchmarks/QUALITY_CANDIDATES_2026-10-09.md). Optional Consistency, Specialized outpaint and Fine detail editor switches are implemented and initially off; broader real-photo quality validation remains open. The [guide](guide.md#runtime-modes) describes their scope and model switching.

**Current behaviour:** [Guide](guide.md) and [architecture](architecture.md). [Focus implementation and validation](notes/2026-10-07-focus-implementation.md) preserves the dated delivery log through 10 October. [Model research and product direction](notes/2026-10-07-direction.md) preserves the initial research snapshot and longer-term proposals.

## Implementation status

**Quantized Klein on a Mac, 11 October 2026:** Klein's transformer and text encoder can run in MLX with 8-bit or 4-bit weights (`MOSIMAGE_APPLE_WEIGHTS`), inside the same diffusers pipelines, so masked edits, references and every LoRA keep working. On an M5 Max, 8-bit lowers Quality's peak from 36–42 GiB to 24–30 GiB and 4-bit to 18–24 GiB, at 4–7% more time. The default follows unified memory (BF16 from 64 GB, 8-bit from 48 GB, else 4-bit), and GPU caches are released after every job. Next: smaller VAE tiles on Apple silicon, which would cut another 5–10 GiB from every peak. See the [comparison](../benchmarks/MLX_KLEIN_2026-10-11.md).

**Small Macs, 11 October 2026:** a Mac without room for Quality (below 36 GB with 4-bit weights) now opens the editor on Fast, the model it warms up, and decodes the VAE in 512 px tiles: one 1024² decode takes 1.3 GiB instead of 6.5 GiB. Decoded only, as generated images are, 512 px tiles measure 26–31 dB against untiled, with faint tile-shaped colour shifts, so larger Macs keep the untiled decode. See the [note](notes/2026-10-11-apple-vae-tiles.md).

**macOS port, 11 October 2026:** MosImage runs on Apple silicon. On macOS the engine selects Metal (MPS) with bfloat16 and offers the BF16 Klein 9B (Quality) and 4B (Fast), since NVFP4 needs Blackwell CUDA; Consistency, the detail enhancer and Specialized outpaint work as on Windows. `start.sh` sets up and supervises the engine like `start.cmd`, and `npm run tauri:build:mac` installs `MosImage.app` in `/Applications`. The UI was checked in WebKit, where `createImageBitmap` dithering turned face and object selections into full-canvas noise masks (fixed by decoding through `<img>`) and the canvas lacks `filter` (layer footprints blur in script). On an M5 Max a warm 1024² Quality generation takes 7 s, a full-frame edit 14.5 s. See the [guide](guide.md#macos-apple-silicon).

**Quality and editor updates, 10 October 2026:** Enhance adds detail with Subtle/Strong settings and native-resolution tiling; Fix faces defaults to Faithful, offers Redraw and shows detected faces for manual picking. Crop (C) changes the canvas without AI; Shift+C opens Expand, and desktop Export opens Save As. The Background is an ordinary layer, with multi-selection, Merge down, Merge visible, Flatten image and Select pixels. Expansion uses the visible layer composite and preserves separate layer placement. See the [guide](guide.md#frontend) for current controls and the [quality/speed evaluation](notes/2026-10-10-quality-speed-sweep.md) for measured tradeoffs. Enhance is implemented; generative 2x upscaling remains experimental.

**SAM 3 selection and refinement evaluation completed, 10 October 2026:** [seven frozen default point/text cases](../benchmarks/SAM3_SELECTION_2026-10-10.md) compare the existing SAM 2.1 Tiny/Grounding DINO CPU routes against official SAM 3 on CPU FP32 and GPU BF16. The [real-photo follow-up](../benchmarks/SAM3_REFINEMENT_2026-10-10.md) adds six calibration and twelve separate pedestrian evaluation cases, equal correction clicks/boxes, mask feedback and absent-concept controls. Feedback mean annotation IoU is 0.929 for SAM 3 versus 0.925 for current selection; incomplete/ambiguous annotations limit ranking. SAM 3 rejects absent `astronaut` on all twelve street photos, where current default text selection returns ordinary people. Warmed GPU new-image selection takes about 0.12 s for points and 0.16 s for text in the follow-up, after a 16–18 s first load/inference; CUDA tensor peaks are 1.20/2.12 GiB per separately loaded route. Targeted prompts or positive boxes recover original poster/brooch misses. All 593 follow-up masks were verified and 276 repetitions were identical. Retain the CPU default pending broader annotation-complete object, soft-cutout and loaded-Klein acceptance; no provider switch is integrated.

**Second-model evaluation completed, 10 October 2026:** [official Qwen 2.1 Turbo NVFP4/GGUF](../benchmarks/QWEN_TURBO_QUANTS_2026-10-10.md) has isolated GPU and complete prompt-to-file evidence on the 16 GB Blackwell card. Native NVFP4 with 512 conditioning fits, but full-resolution reference caching can exhaust VRAM, the streamed two-reference case has RAM pressure, and local edits show blending/reference-scale limitations. **The user prefers FLUX overall; Qwen is not selected for provider/UI integration.** Klein 9B remains the Quality model; all benchmark evidence is retained.

**Upscaling experiment, 10 October 2026:** x2plus and compact x4v3 run natively with optional isolated dependencies. Warm processing takes about 0.18–0.34 seconds on 1024² inputs at the tested 32-pixel tile halo; larger PNG saves dominate export time. Both coexist with Quality 9B without reload and preserve deterministic Klein outputs. **The user rejected all three displayed face variants as too smooth, including x2plus; the photo-export recommendation is withdrawn.** Preserve pores, small wrinkles and individual beard hairs before accepting another recipe. AI Export and post-render crop integration remain proposed, pending visual acceptance. See the [upscaling evidence](../benchmarks/UPSCALING_2026-10-10.md), including cold starts, tile overlap and remaining acceptance work.

**Classical 2x follow-up, 10 October 2026:** [SwinIR Light/M and Swin2SR evidence](../benchmarks/UPSCALING_FIDELITY_2026-10-10.md) covers six native-source and reconstruction cases, including a real camera image. Light preserves more source beard/skin texture and takes about 2.7 seconds for a warmed 1024² input with 512px tiles (2.9 seconds including PNG save). Beside actual Klein 9B it takes about 3–5 seconds after releasing unused CUDA cache; retaining that cache instead caused 24–26-second later calls. Klein stays loaded and the subsequent edit is pixel-identical in both runs. Fine stippled texture still needs visual acceptance. BF16 gives no speed gain. These are local research results, not an implemented upscale toggle, cache-policy change or accepted default.

Latest editing-mask evidence: [9 October paired implementation comparison](../benchmarks/EDIT_MASK_RESULTS_2026-10-09.md), alongside the [8 October mask/reference investigation](../benchmarks/INPAINT_QUALITY_2026-10-08.md). These are reproducible experiments; general inpainting-quality acceptance remains open.

**SeedVR2 3B test, 10 October 2026:** [27 retained outputs](../benchmarks/SEEDVR2_3B_2026-10-10.md) compare the same six sources, native 2x and reconstruction, two seeds/repeats on three faces, and LAB versus no colour correction. More skin/beard detail comes with changed eyes, fabric and small text. Warm native 2x takes about 9 seconds for the one-megapixel man and 3.5–3.7 seconds for 512² portraits. Observed CUDA allocation peaks at 8.9 GiB (11.3 GiB reserved), with Klein unloaded. The same seed is pixel-identical across repeats and independent processes. This is a manual/Export preview candidate; user acceptance, loaded-Klein coexistence and UI integration remain open.

**Layer masks, 10 October 2026:** Photoshop-style mask thumbnails in the layer row (click paints, Shift switches off, Alt views alone), a mask brush that paints any grey from white to black with Softness and Opacity (B and E on a selected layer, X swaps), Layer and Mask groups in the panel, a red `\` overlay of hidden parts, and Add mask / Mask from selection. Sliders no longer swallow shortcuts. See [implementation and validation](notes/2026-10-07-focus-implementation.md#photoshop-style-layer-masks-10-october-2026).

**Selection bar and variants, 10 October 2026:** a floating bar beside each selection (Generate, Remove, Faces, Enhance, To layer, Mask layer, invert, deselect), the mode following the selection, proposals kept as switchable variants of one layer (`,` and `.`) or split into layers, and Photoshop tool keys (W, V/Ctrl+T, H, Ctrl+D). See [implementation and validation](notes/2026-10-07-focus-implementation.md#selection-bar-variants-and-photoshop-shortcuts-10-october-2026).

**Layer workflow, 9 October 2026:** Generate in an open document adds a full image layer fitted to the existing canvas. Replace and Remove always add a selected-region layer, including empty or hidden-only stacks. Apply preserves the background, opens Layers and selects Transform; preview placement, mask footprints, undo/redo and persistence are verified. New starts a separate document; Expand retains its background/dimension workflow. See [implementation and validation](notes/2026-10-07-focus-implementation.md#ai-results-as-layers-9-october-2026).
**Quality model, 9 October 2026:** FLUX.2 Klein 9B NVFP4 replaces 4B NVFP4 as the Quality default. Its Qwen3-8B encoder is trimmed to 28 layers, stored as FP8 in pinned RAM and streamed to the GPU per prompt, so transformer and VAE stay resident within 16 GB. 4B NVFP4 and BF16 remain selectable. See the [model comparison](../benchmarks/MODEL_COMPARISON_2026-10-09.md).

**Klein-only model lineup, 9 October 2026:** Z-Image Turbo and the FLUX.1 Fill adapter are removed. FLUX.2 Klein now runs every AI tool, including expansion: Quality is 9B NVFP4 and Fast is 4B NVFP4, with 4B BF16 in Advanced controls. Expansion carries the tone offset of the redrawn band into the new area and crossfades across it, which removed a visible 4-level step at the original edge in the local garden, kitchen and portrait trials. Selection stays on the CPU models.

**Canvas-edge inpainting, 9 October 2026:** Centred crops extend mask coverage into the mirrored picture context, preserving replacement coverage at selected canvas edges and corners. The final editable footprint stays unchanged. See [regression evidence](notes/2026-10-07-focus-implementation.md#canvas-edge-inpaint-masks-9-october-2026).

| Phase | Status | Delivered |
|---|---|---|
| 2 — Real canvas | Complete | Native-resolution masks; open, paste, drag/drop, save/export; move/pan, zoom, fit, actual size; brush, erase, rectangle, lasso, grow, shrink, feather, invert, clear; mask undo/redo. |
| 3 — Documents and revisions | Complete | Persistent v2 documents, revisions, alternatives, metadata autosave, document undo/redo, close/reopen, and non-destructive v1-generation import. |
| 4 — Object selection | Partial | SAM 2.1 Tiny point selection and positive/negative refinement, cached CPU image embeddings, and manual mask correction. SAM 3 CPU/GPU, real-photo refinement, threshold and negative-control evaluation are complete; integration awaits broader object, soft-cutout and loaded-Klein validation. |
| 5 — Inpainting | Core implemented; quality validation open | Native mask-conditioned Klein default, visible padded context rectangles, soft latent masks, inward edge blending, scene notes, sequential alternatives, in-place preview, explicit accept, and protected-pixel regression coverage. An eight-seed check found missing subjects and weak integrations. A fixed four-scene suite and 24-output BF16/NVFP4 comparison now exist: NVFP4 reduced median edit time by 36%, with similar visual limitations. NVFP4 app integration is complete, including generation, masked edits and the BF16 comparison option. Broad editing-quality validation remains open; see the [experiment](../benchmarks/NVFP4_RESULTS_2026-10-07.md). |
| 6 — Generation and presets | Complete | FLUX.2 Klein 9B NVFP4 Quality, 4B NVFP4 Fast and selectable 4B BF16, product-first controls, Advanced technical settings, aspect presets, and one-to-four sequential alternatives. Krea 2 Turbo remains unintegrated pending a measured 16 GB quantized path and the license-required safety controls. |
| 7 — Outpainting | Complete for standard edge expansion | Draggable edges, format presets, exact margins, native masked Klein expansion of the visible layers with seam matching, protected RGBA compositing, sequential proposals and revision dimension undo/redo. Arbitrary interior transparent regions and a broad quality benchmark remain follow-up work. |
| 8 — Desktop packaging and recovery | Partial | Tauri builds the executable and MSI/NSIS installers; the shell starts or reuses the checkout engine, supervises restart and shutdown, and exports through Save As. A self-contained Python/CUDA package, download/disk-space UI and first-install recovery acceptance remain open. |

Phases 4 and 8 are partially delivered; phase 5 still needs visual reliability validation. The English canvas-first Focus shell applies to new generation, selection, editing, expansion, and proposal review. Phase completion here refers to each phase's own exit criterion, not the full v2 definition of done. Sections below describe the target design where it extends beyond this implementation table.

**Runtime maintenance, 8 October 2026:** Named SQLite migrations/WAL, bounded upload reads, image-only file serving, finished-job retention, typed revision conflicts, serialized CPU subject isolation and Library deletion are implemented. The dated review fixed dtype reuse in the then-active Z-Image provider and cleaned unused layer assets across revision history. See [maintenance validation and limits](notes/2026-10-07-focus-implementation.md#runtime-refactor-validation-8-october-2026); broader editing-quality and phase 8 work remain open.

**Phase 5 follow-up, 7 October 2026:** [Scale-aware placement research](notes/2026-10-07-inpainting.md#scale-aware-placement-research) diagnoses the supplied garden/reference edit and proposes separate subject placement geometry, deliberate scene context, optional local visual planning and controlled renderer comparisons. The [controlled Anna/park tests](notes/2026-10-07-inpainting.md#controlled-annapark-experiments) retain 72 completed outputs, including 18 prompt renders, nine visual-guide renders and six paired mask-shape rerenders. The user rejected all earlier 39 images for size, seated pose and ground contact. Smaller-crop NVFP4 remains a fast diagnostic baseline; Base at 28/50 steps offers no convincing advantage. Explicit parallel-leg prompts improve pose but do not reliably solve fit, scale or contact. The visual seated-layout comparison with role-aware identity/structure references is complete: whole-body fit improves at about 2.6 s per local render, while exact pose/scale/contact remain open. The curved/soft mask crosses baseline faces; a same-crop box restores complete faces in 3/3 seeds while expanding the visible edit footprint. Prioritize an opaque subject core, independent placement geometry and mask-fit preview before broader model sweeps. Bench-first/person-second construction and learned pose/depth conditioning remain candidates; see the [community leads](notes/2026-10-07-inpainting.md#comfyui-community-leads-checked-7-october-2026). Automatic scale/layout planning remains proposed, and the encoder-separated runtime requires wider quality and end-to-end UI validation before integration.

## 1. Objective

Evolve the existing MosImage prototype into a focused local AI canvas editor.

The core interaction should be:

> Open or generate an image, select an area or object, describe the desired change, and generate it locally.

MosImage v2 should retain the working runtime, model-provider, job, storage, API, React, and Tauri foundations. The primary refactor is from a generation form with an image preview into a document-oriented editor with an interactive canvas.

This is an incremental upgrade of the current repository, not a clean rewrite.

## 2. Product principles

1. **The selection expresses intent.** When a selection exists, generation affects that selection.
2. **Pixels outside the selection are protected.** Inpainting must not silently modify unrelated areas.
3. **Models are implementation details.** Model loading and GPU placement should be automatic.
4. **Every operation is reversible.** Generated changes, mask edits, and accepted alternatives support undo and redo.
5. **Simple by default.** Technical inference controls belong in an Advanced panel.
6. **Local-first.** Images and prompts remain on the machine unless a future cloud provider is explicitly enabled.
7. **Keep the application runnable.** Each migration phase should leave a usable build.

## 3. Current foundation

The current checkout provides:

- React and TypeScript frontend.
- Vite development and production builds.
- Tauri desktop shell with checkout-engine supervision and Windows installer builds.
- FastAPI backend.
- Typed request and response schemas.
- Model provider interface and registry.
- FLUX.2 Klein 9B/4B NVFP4 and 4B BF16 generation, edit, inpaint and outpaint pipelines.
- CUDA, BF16, TF32, CPU-offload, and one-pipeline residency controls.
- A serialized single-worker GPU job queue.
- Local image, mask, and output storage.
- SQLite generation metadata and history.
- Startup scripts tuned for the target machine.

These are the current foundations to retain. The initial v1 Z-Image provider and later FLUX Fill adapter have been removed; their earlier use is preserved in the dated implementation notes.

## 4. Main differences from v1

| Area | MosImage v1 | MosImage v2 |
|---|---|---|
| Main UI | Generation form and result preview | Interactive canvas and contextual actions |
| Work modes | Separate text-to-image and inpaint tabs | One editor whose action follows canvas state |
| Selection | White brush and clear | SAM, brush, erase, lasso, rectangle, add/subtract, invert, grow, shrink, feather |
| Segmentation | None | Click, box, and text-prompt segmentation |
| Inpainting | Full image and mask passed to model | Context crop, mask-aware generation, protected compositing |
| Generation | One result | Multiple alternatives with in-place preview |
| History | Flat generation list | Documents, revisions, undo, redo, and alternatives |
| Model controls | Exposed in the main sidebar | Automatic presets; technical settings under Advanced |
| Progress | Polling with coarse job state | Real progress events and cancellation |
| Desktop runtime | Tauri scaffold plus separately launched API | Packaged desktop app supervising a Python sidecar |
| Outpainting | None | Canvas extension and generated continuation |

## 5. Target user experience

### Primary tools

- Move and zoom
- Object Select
- Mask Brush
- Lasso or rectangle selection
- Erase Mask

### Context-sensitive generation

- Empty canvas: **Generate image**
- Image with selection: **Fill selection**
- Image without selection: **Create variation**
- Selection touching empty canvas space: **Extend image**

### Object selection

- Hover previews a segmentation mask.
- Click selects an object.
- Shift-click adds to the selection.
- Alt-click subtracts from the selection.
- A text prompt can select concepts such as `all windows`, `the person in red`, or `the background`.
- Manual mask tools refine imperfect automatic masks.

### Default controls

- Prompt
- Variation count
- Creativity or strength
- Feather
- Generate or Cancel

### Advanced controls

- Model preset
- Seed
- Steps
- Guidance
- Negative prompt
- Runtime placement
- Output resolution

## 6. Target model stack

### Generation

**Quality default:** FLUX.2 Klein 9B Distilled NVFP4 at four steps and guidance 1, with its FP8 text encoder streamed from RAM on the target 16 GB GPU.

**Fast:** FLUX.2 Klein 4B Distilled NVFP4 at four steps and guidance 1. Its Apache 2.0 weights permit commercial use; the 9B weights and detail-enhancer adapter are non-commercial.

**Advanced comparison:** FLUX.2 Klein 4B BF16 for GPUs without native NVFP4 support. All three providers support generation and editing; Enhance requires Quality 9B.

The original plan proposed Klein 4B as the default, Z-Image Turbo as Fast and Krea 2 Turbo as an optional quality candidate. The Klein-only lineup supersedes those choices: Z-Image is removed, Krea is unintegrated, and the later Qwen comparison did not lead to integration. Preserve the [original direction research](notes/2026-10-07-direction.md) and [model comparison](../benchmarks/MODEL_COMPARISON_2026-10-09.md) as decision evidence.

### Segmentation

**Current:** Meta SAM 2.1 Tiny for point selection on CPU. **Evaluated candidate:** Meta SAM 3 for concept, box and text selection. The [default comparison](../benchmarks/SAM3_SELECTION_2026-10-10.md) shows GPU speed gains with mixed quality. The [refinement follow-up](../benchmarks/SAM3_REFINEMENT_2026-10-10.md) finds similar refined person-mask agreement and better rejection of an absent astronaut concept; annotation limits and broader object/runtime acceptance prevent a default provider decision.

- Support point, box, mask, example, and text prompts.
- Cache image embeddings by document revision and image content hash.
- Unload before heavy generation when required by VRAM pressure.

### Inpainting and outpainting

**Current default:** Native mask-conditioned FLUX.2 Klein editing and expansion through Diffusers 0.41.0. Quality uses 9B NVFP4; Fast uses 4B NVFP4, with 4B BF16 selectable.

- Inpaint uses a visible context crop, opaque generation coverage and the displayed soft selection for protected final compositing.
- Expansion regenerates 32 model pixels inside the original and blends over only 16 native pixels. Optional Specialized outpaint uses the distilled Fast4 FAL adapter.
- Broad scene and seam acceptance remains open; see the [paired mask evaluation](../benchmarks/EDIT_MASK_RESULTS_2026-10-09.md) and [quality/speed follow-up](notes/2026-10-10-quality-speed-sweep.md).

FLUX.1 Fill Dev was an initial specialist candidate. Its adapter is removed; introducing a different provider would require a new quality, latency and licensing decision.

## 7. Target architecture

```text
Tauri desktop shell
├── React editor UI
│   ├── document state
│   ├── viewport state
│   ├── mask state
│   └── inference job state
├── GPU-accelerated canvas
└── supervised Python sidecar
    ├── FastAPI or versioned local IPC
    ├── model manager
    ├── generation providers
    ├── segmentation provider
    ├── inpaint compositor
    ├── serialized GPU jobs
    └── SQLite and local assets
```

### Frontend boundaries

Refactor the current monolithic `frontend/src/App.tsx` into:

```text
frontend/src/
├── app/
│   ├── App.tsx
│   └── EditorShell.tsx
├── canvas/
│   ├── EditorCanvas.tsx
│   ├── viewport.ts
│   ├── coordinates.ts
│   ├── compositor.ts
│   └── overlays/
├── documents/
│   ├── documentTypes.ts
│   ├── documentReducer.ts
│   ├── history.ts
│   └── persistence.ts
├── masks/
│   ├── maskTypes.ts
│   ├── maskOperations.ts
│   ├── BrushTool.ts
│   ├── LassoTool.ts
│   └── ObjectSelectTool.ts
├── generation/
│   ├── PromptPanel.tsx
│   ├── AlternativesStrip.tsx
│   └── generationState.ts
├── api/
│   ├── client.ts
│   └── types.ts
└── styles/
```

Folder names may be adjusted to established conventions during implementation. Avoid adding abstractions before they have a concrete owner or use.

### Backend boundaries

Retain the current structure and add focused modules:

```text
backend/app/
├── models/
│   ├── base.py
│   ├── registry.py
│   ├── z_image.py
│   ├── flux2_klein.py
│   ├── flux_fill.py
│   └── segmentation/
│       └── sam3.py
├── documents.py
├── masks.py
├── compositing.py
├── jobs.py
├── services.py
├── storage.py
└── main.py
```

Keep provider-specific runtime code behind typed provider interfaces.

## 8. Document and revision model

The current `generations` table should remain readable during migration. Add a document-oriented schema rather than overloading flat generation records.

```text
Document
├── id
├── name
├── canvas_width
├── canvas_height
├── color_profile
├── current_revision_id
├── created_at
└── updated_at

Revision
├── id
├── document_id
├── parent_revision_id
├── operation_type
├── composite_image_path
├── mask_image_path
├── operation_parameters
├── created_at
└── selected_alternative_id

Alternative
├── id
├── revision_id
├── seed
├── model_name
├── output_image_path
└── metadata
```

### Migration approach

- Keep the v1 `generations` table initially.
- New documents use the v2 schema.
- Provide a one-time action to open a v1 result as a new v2 document.
- Do not destructively migrate or delete existing generated images.

## 9. Image-space correctness

The v1 mask canvas is fixed at 1024×1024 while source images may have other aspect ratios. V2 must use image-space coordinates independent of viewport size.

Requirements:

- The mask buffer dimensions match the underlying document or source image.
- Zoom and pan affect only viewport transforms.
- Pointer coordinates are transformed into image coordinates before editing masks.
- Letterboxing is excluded from mask coordinates.
- Mask export preserves source dimensions exactly.
- High-resolution masks may render through lower-resolution viewport textures, but editing remains image-space accurate.

Add coordinate-transform and non-square-image regression tests before integrating SAM.

## 10. Inpainting pipeline

Replace direct full-image inpainting with an editor-grade pipeline:

1. Validate that a non-empty mask exists.
2. Calculate the mask bounding box.
3. Expand the crop to include sufficient context.
4. Grow the internal generation mask beneath the visible boundary.
5. Run the fill model on the contextual crop.
6. Feather the final compositing mask.
7. Composite the generated pixels only through the protected mask.
8. Compare the pre- and post-operation buffers outside the mask.
9. Save the result as a new revision.
10. Preserve all generated alternatives until the document is closed or explicitly cleaned up.

The protected-pixel guarantee applies to the internal RGBA composite. Export encoding may change compressed file bytes without changing visible pixel values.

## 11. API evolution

Preserve existing v1 routes during the transition and introduce versioned v2 routes.

```text
GET    /api/v2/health
GET    /api/v2/models
POST   /api/v2/models/load
POST   /api/v2/models/unload

POST   /api/v2/documents
GET    /api/v2/documents/{id}
POST   /api/v2/documents/{id}/revisions
POST   /api/v2/documents/{id}/undo
POST   /api/v2/documents/{id}/redo

POST   /api/v2/generate
POST   /api/v2/inpaint
POST   /api/v2/outpaint

POST   /api/v2/segment/point
POST   /api/v2/segment/box
POST   /api/v2/segment/text

GET    /api/v2/jobs/{id}
GET    /api/v2/jobs/{id}/events
POST   /api/v2/jobs/{id}/cancel
```

Use strict Pydantic models at every request boundary. Reject malformed dimensions, masks, model names, and operation types.

## 12. Job and model management changes

Current delivery already supports cooperative cancellation plus an eight-second supervised worker replacement fallback. The primary dock exposes Cancel throughout queued/loading/generation work; running cancellation remains `cancelling` until cleanup finishes, and the UI recovers from the replacement worker losing its in-memory job. The streamed stages below remain target scope.

Retain the serialized GPU queue and extend it with:

- Cooperative cancellation between inference steps.
- Meaningful progress stages rather than only `0.05` and `1.0`.
- Server-sent events or another simple streamed event mechanism.
- Explicit queued, loading, encoding, generating, decoding, compositing, cancelling, completed, cancelled, and failed states.
- Automatic model selection based on operation and quality preset.
- Preloading based on the active editor tool.
- Clean CUDA out-of-memory recovery.
- Verified unloading and cache cleanup between incompatible pipelines.

Do not introduce concurrent heavy GPU jobs on the 16 GB target profile.

## 13. Refactor sequence

### Phase 0 — Stabilize v1

- Record current generation and inpaint behavior on the target machine.
- Capture load time, inference time, peak VRAM, and output dimensions.
- Add backend syntax checks and frontend build verification.
- Add a minimal automated test harness.
- Preserve representative inputs, masks, and expected metadata as fixtures without committing private user content.

**Exit criterion:** Current text-to-image and manual inpainting behavior is reproducible and covered by smoke tests.

### Phase 1 — Extract frontend boundaries

- Split API types and client functions out of the current files.
- Split status, prompt controls, history, and preview into focused components.
- Introduce separate document, viewport, mask, and job state.
- Preserve the existing v1 screens until the new editor shell can replace them.

**Exit criterion:** No user-visible feature change; the frontend builds with smaller, owned modules.

### Phase 2 — Introduce the real canvas

- Add open, paste, drag-and-drop, save, and export.
- Implement pan, zoom, fit, and actual-size controls.
- Store source images and masks at their native dimensions.
- Add brush, eraser, rectangle, lasso, clear, invert, expand, contract, and feather operations.
- Add undo and redo for mask edits.
- Replace `MaskCanvas` after parity is reached.

**Exit criterion:** Non-square images and masks remain aligned at every zoom level and after export.

### Phase 3 — Add documents and revisions

- Add v2 document, revision, and alternative storage.
- Implement non-destructive operation history.
- Add document undo and redo.
- Allow a v1 history item to open as a v2 document.
- Add autosave of document metadata.

**Exit criterion:** A generated or imported image can be edited, undone, redone, closed, and reopened.

### Phase 4 — Add SAM 3 segmentation

- Extend the completed person/refinement benchmark to broader annotation-complete objects, meaningful negative concepts and soft cutouts; prove loaded-Klein memory/scheduling before adding a SAM 3 provider and model lifecycle. Retain the current CPU selector meanwhile.
- Implement point, box, and text segmentation routes.
- Cache embeddings by revision content hash.
- Add hover preview, click selection, add, subtract, and text concept selection.
- Ensure manual tools can refine SAM masks.

**Exit criterion:** Common objects can be selected with one click and corrected without leaving the canvas.

### Phase 5 — Upgrade inpainting

- Use the implemented Klein mask-conditioned route; a specialist replacement requires separate evaluation.
- Implement contextual crop extraction.
- Keep user-controlled mask growth separate from inward blending and protected compositing. Show the padded context rectangle.
- Generate multiple alternatives.
- Preview alternatives in place before accepting one.
- Add protected-pixel regression tests.

**Exit criterion:** Replacing an object does not modify pixels outside the protected selection.

### Phase 6 — Upgrade generation and presets

- Use the implemented Klein 9B NVFP4 Quality, 4B NVFP4 Fast and advanced 4B BF16 providers.
- Add Fast and Quality product presets.
- Hide raw model and runtime controls under Advanced.
- Add aspect-ratio presets and multiple alternatives.
- Evaluate Krea 2 only after the primary providers are stable.

**Exit criterion:** A user can generate a new image without choosing a model or configuring inference terminology.

### Phase 7 — Add outpainting

- Support resizing the document canvas beyond the current image.
- Detect transparent or empty canvas regions.
- Build overlap masks and contextual crops.
- Generate extensions and composite them as revisions.

**Exit criterion:** A user can extend any canvas edge while preserving the existing image.

### Phase 8 — Desktop packaging and recovery

**Delivered:** The shell runs the checkout Python environment as a controlled component, supervises startup/restart/shutdown, reuses an existing engine and builds Windows installers. Models download on first use; documents and revisions persist. The Python/CUDA environment and models are not bundled into the installer.

**Remaining:**

- Package or provision Python, CUDA PyTorch and runtime dependencies for a fresh machine.
- Add disk-space, model-download progress and validation UI.
- Validate interrupted downloads, crash recovery and document restoration through the installed app on a clean profile.

**Exit criterion:** A new user can install MosImage and complete the core workflow without manually starting a backend.

## 14. Testing strategy

### Frontend unit tests

- Canvas-to-image and image-to-canvas coordinate transforms.
- Mask add, subtract, invert, expand, contract, and feather.
- Document reducer and revision traversal.
- Undo and redo.
- Context-sensitive action selection.

### Backend unit tests

- Strict schema validation.
- Provider registry selection.
- Context crop calculation.
- Mask normalization and dimension validation.
- Protected compositing.
- Model placement decisions.
- Job cancellation state transitions.

### Integration tests

- Worker startup, shutdown, crash, and restart.
- Model download, load, unload, and switching.
- Generate, segment, inpaint, outpaint, cancel, and export workflows.
- CUDA out-of-memory fallback.
- Opening v1 history as a v2 document.

### Visual regression tests

- Mask alignment on square, landscape, and portrait images.
- Selection overlays at multiple zoom levels.
- Feathered inpaint boundaries.
- Crop reinsertion without seams.
- Outpaint overlap regions.

### Fixed model evaluation set

Include non-private fixtures covering:

- People and hair
- Animals
- Products
- Vehicles
- Buildings and windows
- Foregrounds and backgrounds
- Small and overlapping objects
- Object removal and replacement
- Material and color changes
- Poster and text generation

Record segmentation quality, protected-pixel changes, latency, peak VRAM, and subjective edit quality.

## 15. Acceptance criteria

- The application runs locally on the target Windows machine.
- Images can be opened, pasted, dropped, generated, saved, and exported.
- Users can select objects by click, box, text, lasso, rectangle, or brush.
- Masks support add, subtract, erase, invert, expand, contract, and feather.
- Inpainting offers alternatives and supports cancellation.
- Pixels outside the protected mask remain unchanged in the internal composite.
- Every accepted AI operation supports undo and redo.
- Documents survive application restart.
- Models load, unload, and switch automatically.
- The primary workflow exposes no node graph or diffusion terminology.
- A first-time user can generate an image and replace one object without documentation.

## 16. Explicitly out of scope for v2 MVP

- Full Photoshop-compatible layers and blend modes.
- Vector drawing and typography tools.
- Node graphs and arbitrary inference pipelines.
- LoRA or full-model training.
- Video editing.
- Cloud generation providers.
- Multi-user collaboration.
- Mobile support.
- Plugin marketplace.

## 17. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Frontend rewrite destabilizes working generation | Extract boundaries first and keep v1 screens until canvas parity. |
| Model switching feels slow | Preload based on active tool and benchmark validated quantized formats. |
| Inpainting changes unrelated pixels | Use crop generation plus explicit protected-mask compositing. |
| SAM edges are imperfect | Pair segmentation with fast manual correction tools. |
| Large images exhaust VRAM | Use contextual crops, tiled decoding, and controlled fallback presets. |
| Windows Python and CUDA packaging is fragile | Pin a tested runtime and supervise it as an application component. |
| New models exceed 16 GB | Preserve single-pipeline residency and validate FP8, NVFP4, or offload profiles. |
| Licensing prevents redistribution | Track licenses per provider and separate optional model downloads from application code. |
| Technical controls recreate Invoke complexity | Keep product presets in the main UI and raw controls under Advanced. |

## 18. Historical first implementation slice

This section records the original starting slice. The canvas, compositing, revisions and edge expansion have since been delivered; see the implementation status above. Remaining roadmap work includes the rest of phase 4 and self-contained installation/recovery in the partially delivered phase 8. The October product-direction proposal evaluates additional follow-up work.

The first vertical slice should avoid model expansion and prove the editor architecture using the existing Z-Image provider:

1. Refactor frontend state into document, viewport, mask, and job domains.
2. Add a real image-space canvas.
3. Fix non-square mask alignment.
4. Add brush, erase, lasso, invert, expand, and feather.
5. Send the resulting image and mask through the existing inpaint API.
6. Composite the result only through the mask.
7. Add undo and redo for the accepted edit.

This creates the v2 product skeleton before introducing SAM 3 or additional large models.

## 19. Definition of done

MosImage v2 is complete when a non-technical user can:

1. Launch the installed desktop application.
2. Generate or open an image.
3. Click an object to select it.
4. Correct the mask if necessary.
5. Describe a replacement.
6. Preview several alternatives.
7. Accept, undo, or redo the edit.
8. Export the final image.

The user should not need to know which model was used or understand samplers, schedulers, VAEs, CFG values, quantization, GPU placement, or node graphs.
