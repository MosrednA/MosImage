# Separate edit masks and resolution-aware expansion — 9 October 2026

Status: implemented and validated locally. This delivers the first two recommendations from the [quality investigation](../docs/notes/2026-10-09-image-quality-research.md). The mask contracts are verified; general image-quality acceptance remains open. No LoRA was loaded or trained.

## Delivered behaviour

- Inpaint gives Klein opaque generation coverage wherever the final selection has nonzero alpha, retaining resampling coverage when reducing the crop. Final compositing uses the original soft alpha and requested inward feather. Pixels outside that footprint remain exact. The reference image stays unmasked.
- Expand grows the generation mask **after resizing**, by **32 model pixels** inside the source. This avoids the old 16 native-pixel overlap shrinking to four model pixels on a 4096 × 2048 four-side canvas. The final crossfade still reaches only **16 native pixels** inside the original. Wider regeneration does not authorise wider final edits.
- Colour-drift reference pixels exclude the wider regenerated overlap. The original seam tone matching remains, using the final native edge band.
- Mask growth uses repeated 3 × 3 square dilation. It produces the same generation, blend and colour-reference pixels as the former large MaxFilter, with less CPU work.

The service selects a named [outpaint workspace](../backend/app/compositing.py) or the inpaint workspace through the existing provider interface. API fields, model steps, prompts, reference handling and editor layer application remain unchanged. The [guide](../docs/guide.md) describes the current controls.

## Paired GPU protocol

[benchmark_edit_masks.py](../backend/tools/benchmark_edit_masks.py) runs the **real app Klein 9B NVFP4 provider**, including the streamed FP8 text encoder, job queue and final compositor. Every case uses seeds **7, 42, 20261009**, four distilled steps, guidance 1 and strength 1. Before/after runs alternate per seed on one resident provider: **30 retained renders**, with no retries or excluded visual failures.

| Case | Native output | Evaluation |
|---|---|---|
| Kitchen removal | 1254 × 1254 | Mug, shadow and reflection removal; marble continuity |
| Shirt | 1254 × 1254 | Blue shirt to burgundy knitted sweater; colour fringe |
| Anna | 1254 × 1254 | Person/reference insertion; complete figure and contact |
| Garden, right expansion | 1894 × 1254 | +640 right; texture, plants and path continuity |
| Kitchen, four-side expansion | 2048 × 2048 | 1536² source, +256 on each side; corners and structure |

The inpaint fixtures and ordinary Anna reference come from the retained [model comparison](MODEL_COMPARISON_2026-10-09.md). Expansion uses the versioned v1 garden/kitchen sources. The protocol records package versions, model revisions, input hashes, requests, reference order and exact final prompts. Source, model input, generation mask, raw output and native composite are retained per render.

Before reproduces the previous production workspace: soft inpaint generation alpha, or 16 native pixels of expansion overlap with colour references outside the final blend mask. After uses the delivered recipe. Outpaint changes overlap and its colour-reference exclusion together; this cohort does not isolate their individual contributions. Crop/context settings are held fixed.

All local evidence is under [data/edit-mask-validation-2026-10-09](../data/edit-mask-validation-2026-10-09/):

- [Full gallery](../data/edit-mask-validation-2026-10-09/index.html), with native PNGs and raw crops.
- [Protocol](../data/edit-mask-validation-2026-10-09/protocol.json) and [all measurements](../data/edit-mask-validation-2026-10-09/results.json).
- Comparison sheets: [shirt](../data/edit-mask-validation-2026-10-09/shirt-comparison.jpg), [removal](../data/edit-mask-validation-2026-10-09/kitchen-comparison.jpg), [Anna](../data/edit-mask-validation-2026-10-09/anna-comparison.jpg), [garden expansion](../data/edit-mask-validation-2026-10-09/garden-right-comparison.jpg), [four-side expansion](../data/edit-mask-validation-2026-10-09/kitchen-four-sides-comparison.jpg). Left is before; right is after; rows follow the declared seed order.

These generated assets remain ignored local data. Reproduction requires the retained model-comparison assets and cached model weights; they are not downloaded by this tool.

## Visual review

All five comparison sheets and native-resolution detail crops were inspected. This is an unblinded review of synthetic fixtures, not a general success rate.

| Case | Finding across the retained seeds |
|---|---|
| Kitchen removal | Both recipes remove the mug and reconstruct marble. No consistent visual advantage from separating coverage in this fixture. |
| Shirt | Knitted material is generated, but the blue outline/collar/cuff fringe remains in all three after outputs. Full generation coverage alone does not fix colour retained by the final soft blend or an insufficient object footprint. |
| Anna | Both recipes retain a complete seated person. Hands/bench pose varies, including a changed hand placement at seed 42. No consistent identity or ground-contact improvement is established. |
| Garden expansion | The wider overlap gives smoother local plant/gravel continuation in the reviewed joins, most visibly at seed 42. New vegetation and fencing still vary by seed; it does not establish structural correctness for other scenes. |
| Four-side kitchen | Both recipes continue the kitchen reasonably at viewing size. Native detail remains softer in the generated margins; cabinet/counter geometry varies. No consistent large visual gain is established. |

**Protected pixels: zero changed pixels in all 30 outputs**, measured outside each declared final composite footprint, including alpha for expansion. Regression tests additionally verify downscaled overlap, resolution-independent native blend coverage, exact protected RGBA pixels and exclusion of regenerated pixels from colour-drift references.

## Runtime measurements

RTX 5070 Ti 16 GB, Ryzen 7 9700X, 32 GB RAM. Times below include inference and final compositing, **excluding workspace preparation and UI/API time**. The first kitchen-before call took **55.18 s**, including cold model loading, and is excluded from warmed timing summaries. Its baseline median uses two samples; other cells use three. Other applications were running, and prompt caches are shared between paired calls; these timings do not establish a speedup.

| Case | Before median | After median | After range |
|---|---:|---:|---:|
| Kitchen removal | 5.30 s | 5.20 s | 5.20–5.27 s |
| Shirt | 7.23 s | 7.22 s | 7.20–7.36 s |
| Anna | 3.60 s | 3.63 s | 3.60–3.69 s |
| Garden expansion | 7.98 s | 8.22 s | 7.60–9.38 s |
| Four-side kitchen | 8.73 s | 8.97 s | 8.60–10.67 s |

The whole cohort peaked at **10.98 GiB PyTorch allocation / 11.04 GiB reserved**. These are allocator measurements, not total card usage or process RAM. New-prompt encoding can peak above a cached repeat, so before/after allocation differences are not evidence that the mask change saves VRAM.

The initial cohort used the old large MaxFilter during workspace preparation. It cost median 4.54 s after for garden and 6.95 s after for the four-side kitchen. The delivered dilation optimisation was then measured independently, three preparations per variant, and checked byte-for-byte against the cohort's generation masks and the original growth implementation's blend/reference masks:

| Workspace | Before recipe, optimised | After recipe, optimised |
|---|---:|---:|
| Garden expansion | 1.11 s | 1.20 s |
| Four-side kitchen | 2.16 s | 1.88 s |

[Optimised workspace measurements](../data/edit-mask-validation-2026-10-09/optimized-workspaces.json) retain every sample. This CPU equivalence check does not rerender or change the visual cohort. Compositing remains part of the renderer/compositor timing above.

## Real editor validation

An isolated API/database and Vite editor ran two additional real 9B jobs at seed 42. The user's document library was not used.

1. Open garden, expand +640 right, generate and Apply. Undo restores 1254²; Redo restores 1894 × 1254. The exported PNG matches the proposal in **every decoded RGBA pixel**, and every original pixel beyond the 16 px final band remains exact.
2. Draw a Box selection around the sphere, with Expand +4, automatic Soften 47 and context 320; replace it with a glowing mushroom. The uploaded selection contains **63,584 partial-alpha pixels**. Apply adds one cropped AI layer above the unchanged background with a binary footprint mask. Export matches the soft-composited proposal in **every decoded RGBA pixel**, with zero protected-pixel changes. Undo removes the layer; Redo restores it. Reloading and reopening through Library retains the layer and expanded dimensions, and exports the identical image again.

Evidence: [pixel checks](../data/edit-mask-ui-validation-2026-10-09/pixel-proof.json), [expanded export](../data/edit-mask-ui-validation-2026-10-09/expanded-export.png), [selected export](../data/edit-mask-ui-validation-2026-10-09/selected-export.png), [reopened export](../data/edit-mask-ui-validation-2026-10-09/reopened-export.png). These two jobs prove the changed service/editor path; they are separate from the paired quality cohort. A transient revoked-preview blob error occurred after expansion Apply; exports and persisted pixels were unaffected.

## Completion and remaining work

Validation: **135 backend tests and 77 frontend tests pass**, along with Python compilation and the frontend production build. All **92 local file links / 19 heading links** in the eight affected documents resolve, and whitespace checks pass. The temporary API and Vite servers are stopped; `devports list --json` returns no Node listeners. Restart the backend or desktop app to load this backend change; a desktop executable rebuild is not needed.

The separation and scale contract are delivered. The shirt needs a separately evaluated outline/transition recipe, and reliable pose, lighting, contact and high-resolution structure remain open. Larger cohorts, real photos, 16/32/64 overlap ablations, staged expansion and LoRA compatibility are not established by this slice.
